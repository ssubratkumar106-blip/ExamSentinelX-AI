"""
backend/monitoring/socket_events.py — Real-time WebSocket Events
================================================================
PURPOSE: Handle real-time bidirectional communication using Socket.IO.

EVENTS HANDLED:
    Client → Server:
        'connect'            → Client connected (start session tracking)
        'disconnect'         → Client left (mark session)
        'frame_analysis'     → Send frame for AI analysis
        'exam_heartbeat'     → Periodic ping to confirm student still active
        'browser_event'      → Tab switch / fullscreen exit from JS (NEW)
        'join_admin_room'    → Proctor joins live admin feed room (NEW)

    Server → Client:
        'analysis_result'    → Send AI detection result back
        'alert'              → Push important alerts
        'session_update'     → Update violation count in UI
        'live_violation'     → Broadcast violation to all proctors (NEW)
        'trust_score_update' → Push updated trust score to student (NEW)
"""

from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import current_user
from datetime import datetime
import json
import base64
import logging

logger = logging.getLogger(__name__)

# ── Cooldown Registry ───────────────────────────────────────────────────────
# Per-session, per-type cooldown: {session_id: {violation_type: last_raised_ts}}
# Prevents spamming the same violation type within cooldown window.
_COOLDOWN_SECONDS = 6.0
_cooldown_registry: dict[int, dict[str, float]] = {}


def _cooldown_ok(session_id: int, violation_type: str) -> bool:
    """
    Check if enough time has passed since the last raise of this violation type.
    Returns True if we can raise it again (and updates the timestamp).
    Each violation type has its own independent timer per session.
    """
    import time
    registry = _cooldown_registry.setdefault(session_id, {})
    last = registry.get(violation_type, 0.0)
    if time.time() - last >= _COOLDOWN_SECONDS:
        registry[violation_type] = time.time()
        return True
    return False


def register_socket_events(socketio: SocketIO):
    """
    Register all Socket.IO event handlers.
    Called once from app.py during app creation.
    """

    @socketio.on('connect')
    def handle_connect():
        """Called when any client connects via WebSocket."""
        logger.info("WebSocket connected")
        emit('connected', {
            'message': 'Connected to ExamSentinelX AI monitoring',
            'timestamp': datetime.utcnow().isoformat()
        })

    @socketio.on('disconnect')
    def handle_disconnect():
        """Called when client disconnects."""
        logger.info("WebSocket disconnected")

    @socketio.on('join_session')
    def handle_join_session(data):
        """
        Student joins their exam session room.
        Rooms allow targeting specific sessions with Socket.IO messages.

        Data: { "session_id": 42 }
        """
        session_id = data.get('session_id')
        if session_id:
            room = f'session_{session_id}'
            join_room(room)
            emit('joined', {
                'room': room,
                'session_id': session_id,
                'message': 'Monitoring started'
            })
            logger.info(f"Client joined session room {room}")

    @socketio.on('leave_session')
    def handle_leave_session(data):
        """Student leaves their session room (exam ended)."""
        session_id = data.get('session_id')
        if session_id:
            room = f'session_{session_id}'
            leave_room(room)
            # Clean up cooldown state for this session
            _cooldown_registry.pop(session_id, None)

    @socketio.on('join_admin_room')
    def handle_join_admin_room():
        """
        Proctor/admin joins the 'admin_live' room to receive real-time
        violation broadcasts from all active candidate sessions.

        This implements the ExamGuard-style live proctor feed.
        """
        join_room('admin_live')
        emit('joined_admin', {'message': 'Joined live proctor feed'})
        logger.info("Admin joined live proctor feed")

    @socketio.on('browser_event')
    def handle_browser_event(data):
        """
        Browser-only signals that video cannot detect:
          - Tab switch / window minimization  (event_type='tab_switch')
          - Fullscreen exit                   (event_type='fullscreen_exit')
          - Window blur                       (event_type='tab_switch')

        Data: {
            "session_id": 42,
            "event_type": "tab_switch" | "fullscreen_exit",
            "detail": "Candidate switched tabs"
        }

        This mirrors the ExamGuard approach of reporting client-side
        browser events via a dedicated REST/socket endpoint so that
        events invisible to the camera are still captured.
        """
        session_id  = data.get('session_id')
        event_type  = data.get('event_type')  # 'tab_switch' | 'fullscreen_exit' | 'talking_to_others'
        detail      = data.get('detail', '')

        if not session_id or event_type not in ('tab_switch', 'fullscreen_exit', 'talking_to_others'):
            return

        # Per-type cooldown: don't spam DB if browser fires repeatedly
        if not _cooldown_ok(session_id, event_type):
            return

        try:
            from database.models import ExamSession, ViolationLog
            from backend.extensions import db

            session = ExamSession.query.get(session_id)
            if not session or session.status not in ('active', 'flagged'):
                return

            # Persist violation
            violation = ViolationLog(
                session_id=session_id,
                violation_type=event_type,
                confidence=1.0,
                timestamp=datetime.utcnow(),
                details=json.dumps({'detail': detail, 'source': 'browser_js'})
            )
            db.session.add(violation)
            session.total_violations += 1
            session.update_trust_score(event_type)
            session.calculate_risk_score()
            db.session.commit()

            logger.info(f"Browser event '{event_type}' for session {session_id} "
                        f"— trust_score={session.trust_score:.1f}")

            # Emit trust score update back to the student
            emit('trust_score_update', {
                'trust_score': session.trust_score,
                'total_violations': session.total_violations,
                'risk_score': round(session.risk_score, 2),
                'violation_type': event_type,
            })

            # Broadcast to all proctors watching the live feed
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
            }, to='admin_live')

        except Exception as e:
            logger.error(f"Browser event handler error: {e}")

    @socketio.on('frame_analysis')
    def handle_frame_analysis(data):
        """
        Receive a webcam frame from the browser, run AI analysis,
        and emit the result back to the same client.

        Data: {
            "session_id": 42,
            "frame": "data:image/jpeg;base64,...",
            "timestamp": "2024-01-01T12:00:00"
        }

        This is the CORE real-time event of the entire system.
        """
        session_id = data.get('session_id')
        frame_b64  = data.get('frame')

        if not session_id or not frame_b64:
            emit('error', {'message': 'Invalid frame data'})
            return

        try:
            from database.models import ExamSession, ViolationLog
            from backend.extensions import db
            from ai.detector import get_or_create_detector

            # Verify session exists (no nested app_context — already in context)
            session = ExamSession.query.get(session_id)
            if not session or session.status not in ('active', 'flagged'):
                emit('session_ended', {'session_id': session_id})
                return

            # Get or create detector for this session
            detector = get_or_create_detector(session_id=session_id)

            # Run AI analysis
            result = detector.analyze_frame(frame_b64, trust_score=session.trust_score)

            # Save violation to database (with per-type cooldown)
            if result.has_violation and result.violation_type:
                if _cooldown_ok(session_id, result.violation_type):
                    violation = ViolationLog(
                        session_id=session_id,
                        violation_type=result.violation_type,
                        confidence=result.confidence,
                        timestamp=datetime.utcnow(),
                        screenshot_path=result.details.get('screenshot_path'),
                        details=json.dumps(result.details)
                    )
                    db.session.add(violation)
                    session.total_violations += 1
                    session.update_trust_score(result.violation_type)
                    session.calculate_risk_score()
                    db.session.commit()

                    # Broadcast to proctors
                    socketio.emit('live_violation', {
                        'kind': 'violation',
                        'session_id': session_id,
                        'candidate_name': (session.student.full_name
                                           or session.student.username),
                        'violation_type': result.violation_type,
                        'violation_display': violation.violation_display,
                        'severity': violation.severity,
                        'confidence': round(result.confidence, 3),
                        'detail': result.details.get('detail', ''),
                        'trust_score': session.trust_score,
                        'risk_score': round(session.risk_score, 2),
                        'timestamp': datetime.utcnow().isoformat(),
                    }, to='admin_live')

            # Emit result back to THIS client
            emit('analysis_result', {
                'has_violation': result.has_violation,
                'violation_type': result.violation_type,
                'confidence': round(result.confidence, 3),
                'face_count': result.face_count,
                'head_direction': result.head_direction or 'forward',
                'detected_objects': result.detected_objects or [],
                'total_violations': session.total_violations,
                'risk_score': round(session.risk_score, 2),
                'trust_score': session.trust_score,
                'annotated_frame': result.annotated_frame_b64,
                'timestamp': result.timestamp,
                'session_status': session.status
            })

        except Exception as e:
            logger.error(f"Socket frame_analysis error: {e}")
            emit('error', {'message': 'Analysis failed'})

    @socketio.on('exam_heartbeat')
    def handle_heartbeat(data):
        """
        Periodic ping from student to confirm they are still active.
        If no heartbeat for >30s, flag the session.
        """
        emit('heartbeat_ack', {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'active'
        })
