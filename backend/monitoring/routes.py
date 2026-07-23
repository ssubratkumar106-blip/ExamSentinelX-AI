"""
backend/monitoring/routes.py — AI Monitoring REST API
======================================================
ROUTES:
    POST /monitor/frame                    → Analyze a webcam frame
    POST /monitor/browser-event           → Browser-only events (tab switch, fullscreen) [NEW]
    GET  /monitor/violations/<session_id> → Get violation list
    GET  /monitor/session/<session_id>/audit → Full session audit with trust score [NEW]
    GET  /monitor/live-feed               → Admin live violations feed page [NEW]
"""

from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from datetime import datetime, timezone
import json

from backend.extensions import db
from database.models import ExamSession, ViolationLog
from ai.detector import get_or_create_detector

monitoring_bp = Blueprint('monitoring', __name__)


# ── Per-type cooldown for REST endpoint (mirrors socket cooldown) ─────────────
_COOLDOWN_SECONDS = 6.0
_rest_cooldown: dict[int, dict[str, float]] = {}


def _rest_cooldown_ok(session_id: int, violation_type: str) -> bool:
    """Independent cooldown tracker for REST-based frame analysis."""
    import time
    registry = _rest_cooldown.setdefault(session_id, {})
    last = registry.get(violation_type, 0.0)
    if time.time() - last >= _COOLDOWN_SECONDS:
        registry[violation_type] = time.time()
        return True
    return False


# ── Violation alert messages ───────────────────────────────────────────────────
_ALERT_MESSAGES = {
    'face_absent':         'WARNING: Please ensure your face is visible on camera!',
    'multiple_persons':    'ALERT: Multiple persons detected! Only you should be in frame.',
    'looking_away':        'WARNING: Please look at the screen during the exam.',
    'phone_detected':      'ALERT: Mobile phone detected! Put it away immediately.',
    'suspicious_object':   'WARNING: Suspicious item detected. Please remove unauthorized materials.',
    'suspicious_behavior': 'WARNING: Suspicious behavior detected by AI classifier.',
    'talking_to_others':   'ALERT: Talking detected! Please remain silent.',
    'tab_switch':          'ALERT: Tab switch or window minimization detected!',
    'fullscreen_exit':     'ALERT: Fullscreen mode exited! Please return to fullscreen.',
}


@monitoring_bp.route('/frame', methods=['POST'])
@login_required
def analyze_frame():
    """
    Receive and analyze a webcam frame.

    Request Body (JSON):
        {
            "session_id": 42,
            "frame": "data:image/jpeg;base64,/9j/4AAQ..."
        }

    Response includes trust_score (new field from ExamGuard architecture).
    """
    data = request.get_json()
    session_id = data.get('session_id')
    frame_b64  = data.get('frame')

    if not session_id or not frame_b64:
        return jsonify({'error': 'Missing session_id or frame'}), 400

    # Verify session belongs to this user
    session = ExamSession.query.get(session_id)
    if not session or session.student_id != current_user.id:
        return jsonify({'error': 'Invalid session'}), 403

    if session.status not in ('active', 'flagged'):
        return jsonify({'error': 'Session not active', 'session_status': session.status}), 400

    # Get or create detector for this session
    from pathlib import Path
    capture_dir = str(Path(__file__).parent.parent.parent / 'captures' / 'evidence')
    detector = get_or_create_detector(
        session_id=session_id,
        capture_dir=capture_dir
    )

    # Run AI analysis
    result = detector.analyze_frame(frame_b64, trust_score=session.trust_score)

    # If violation detected → save to database (with per-type cooldown)
    if result.has_violation and result.violation_type:
        if _rest_cooldown_ok(session_id, result.violation_type):
            violation = ViolationLog(
                session_id=session_id,
                violation_type=result.violation_type,
                confidence=result.confidence,
                timestamp=datetime.now(timezone.utc),
                screenshot_path=result.details.get('screenshot_path'),
                details=json.dumps(result.details)
            )
            db.session.add(violation)

            # Update session violation count, trust score, risk score
            session.total_violations += 1
            session.update_trust_score(result.violation_type)   # NEW: trust decay
            session.calculate_risk_score()

            # Auto-flag only on very high violation counts
            if session.total_violations >= 25 or session.risk_score >= 9.5:
                session.status = 'flagged'

            db.session.commit()

            # Broadcast to proctor live feed via SocketIO
            try:
                from backend.extensions import socketio
                socketio.emit('live_violation', {
                    'kind': 'violation',
                    'session_id': session_id,
                    'candidate_name': (session.student.full_name
                                       or session.student.username),
                    'violation_type': result.violation_type,
                    'violation_display': violation.violation_display,
                    'severity': violation.severity,
                    'confidence': round(result.confidence, 3),
                    'trust_score': session.trust_score,
                    'risk_score': round(session.risk_score, 2),
                    'timestamp': datetime.utcnow().isoformat(),
                }, to='admin_live', namespace='/')
            except Exception:
                pass  # Live broadcast failure is non-fatal

            return jsonify({
                'has_violation': True,
                'violation_type': result.violation_type,
                'confidence': round(result.confidence, 3),
                'face_count': result.face_count,
                'head_direction': result.head_direction or 'forward',
                'detected_objects': result.detected_objects or [],
                'total_violations': session.total_violations,
                'risk_score': round(session.risk_score, 2),
                'trust_score': session.trust_score,             # NEW
                'alert_message': _ALERT_MESSAGES.get(
                    result.violation_type or '', 'Suspicious activity detected!'),
                'annotated_frame': result.annotated_frame_b64,
                'session_status': session.status
            })

    # No violation
    return jsonify({
        'has_violation': False,
        'violation_type': None,
        'confidence': 0.0,
        'face_count': result.face_count,
        'head_direction': result.head_direction or 'forward',
        'detected_objects': result.detected_objects or [],
        'annotated_frame': result.annotated_frame_b64,
        'total_violations': session.total_violations,
        'risk_score': round(session.risk_score, 2),
        'trust_score': session.trust_score,                     # NEW
        'session_status': session.status
    })


@monitoring_bp.route('/browser-event', methods=['POST'])
@login_required
def report_browser_event():
    """
    Browser-only events that video cannot see:
      - tab_switch     : candidate switched tabs / minimized window / window blur
      - fullscreen_exit: candidate pressed Escape or exited fullscreen mode

    This is the REST equivalent of the 'browser_event' SocketIO handler.
    The frontend calls this via fetch() as a reliable fallback.

    Request Body (JSON):
        {
            "session_id": 42,
            "event_type": "tab_switch" | "fullscreen_exit" | "talking_to_others",
            "detail": "Candidate switched tabs"
        }
    """
    data       = request.get_json()
    session_id = data.get('session_id')
    event_type = data.get('event_type')
    detail     = data.get('detail', '')

    if not session_id or event_type not in ('tab_switch', 'fullscreen_exit', 'talking_to_others'):
        return jsonify({'error': 'Invalid event_type or session_id'}), 400

    session = ExamSession.query.get(session_id)
    if not session or session.student_id != current_user.id:
        return jsonify({'error': 'Invalid session'}), 403

    if session.status not in ('active', 'flagged'):
        return jsonify({'status': 'ignored', 'reason': 'session not active'}), 200

    # Per-type cooldown: don't log the same browser event twice in 6s
    if not _rest_cooldown_ok(session_id, event_type):
        return jsonify({'status': 'throttled'}), 200

    violation = ViolationLog(
        session_id=session_id,
        violation_type=event_type,
        confidence=1.0,
        timestamp=datetime.now(timezone.utc),
        details=json.dumps({'detail': detail, 'source': 'browser_js_rest'})
    )
    db.session.add(violation)
    session.total_violations += 1
    session.update_trust_score(event_type)
    session.calculate_risk_score()
    db.session.commit()

    # Also broadcast to proctor live feed
    try:
        from backend.extensions import socketio
        socketio.emit('live_violation', {
            'kind': 'violation',
            'session_id': session_id,
            'candidate_name': session.student.full_name or session.student.username,
            'violation_type': event_type,
            'violation_display': violation.violation_display,
            'severity': violation.severity,
            'confidence': 1.0,
            'detail': detail,
            'trust_score': session.trust_score,
            'risk_score': round(session.risk_score, 2),
            'timestamp': datetime.utcnow().isoformat(),
        }, to='admin_live', namespace='/')
    except Exception:
        pass

    return jsonify({
        'status': 'recorded',
        'trust_score': session.trust_score,
        'total_violations': session.total_violations
    })


@monitoring_bp.route('/violations/<int:session_id>', methods=['GET'])
@login_required
def get_violations(session_id):
    """Get all violations for a session (for the student summary panel)."""
    session = ExamSession.query.get_or_404(session_id)

    if session.student_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403

    violations = ViolationLog.query.filter_by(
        session_id=session_id
    ).order_by(ViolationLog.timestamp.desc()).all()

    return jsonify({
        'session_id': session_id,
        'total': len(violations),
        'trust_score': session.trust_score,
        'violations': [
            {
                'id': v.id,
                'type': v.violation_type,
                'display': v.violation_display,
                'confidence': round(v.confidence or 0, 3),
                'timestamp': v.timestamp.isoformat(),
                'severity': v.severity
            }
            for v in violations
        ]
    })


@monitoring_bp.route('/session/<int:session_id>/audit', methods=['GET'])
@login_required
def session_audit(session_id):
    """
    Full audit of a session: metadata + complete violation history with
    trust score trajectory. Admin or session owner only.

    This implements the ExamGuard-style session audit endpoint.
    """
    session = ExamSession.query.get_or_404(session_id)

    if session.student_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403

    violations = ViolationLog.query.filter_by(
        session_id=session_id
    ).order_by(ViolationLog.timestamp.asc()).all()

    # Reconstruct trust score trajectory for the timeline chart
    trajectory_score = 100.0
    trajectory = []
    for v in violations:
        penalty = ExamSession.TRUST_PENALTIES.get(v.violation_type, 5.0)
        trajectory_score = max(0.0, trajectory_score - penalty)
        trajectory.append({
            'timestamp': v.timestamp.isoformat(),
            'violation_type': v.violation_type,
            'violation_display': v.violation_display,
            'severity': v.severity,
            'confidence': round(v.confidence or 0, 3),
            'trust_score_after': round(trajectory_score, 1),
            'snapshot_path': v.screenshot_path,
            'details': v.details,
        })

    # Violation type summary
    type_counts: dict[str, int] = {}
    for v in violations:
        type_counts[v.violation_type] = type_counts.get(v.violation_type, 0) + 1

    return jsonify({
        'session': {
            'id': session.id,
            'candidate_name': session.student.full_name or session.student.username,
            'candidate_email': session.student.email,
            'exam_name': session.exam.title,
            'started_at': session.started_at.isoformat(),
            'ended_at': session.ended_at.isoformat() if session.ended_at else None,
            'status': session.status,
            'duration': session.duration_display,
            'trust_score': session.trust_score,
            'risk_score': round(session.risk_score, 2),
            'total_violations': session.total_violations,
            'score': session.score,
        },
        'violation_summary': type_counts,
        'violation_timeline': trajectory,
    })


@monitoring_bp.route('/active-sessions', methods=['GET'])
@login_required
def active_sessions():
    """
    List all currently active exam sessions with their trust scores.
    Admin only. Used by the proctor live dashboard.
    """
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403

    sessions = ExamSession.query.filter_by(status='active').order_by(
        ExamSession.started_at.desc()
    ).all()

    return jsonify({
        'active_count': len(sessions),
        'sessions': [
            {
                'id': s.id,
                'candidate_name': s.student.full_name or s.student.username,
                'exam_name': s.exam.title,
                'started_at': s.started_at.isoformat(),
                'duration': s.duration_display,
                'trust_score': s.trust_score,
                'risk_score': round(s.risk_score, 2),
                'total_violations': s.total_violations,
                'status': s.status,
            }
            for s in sessions
        ]
    })
