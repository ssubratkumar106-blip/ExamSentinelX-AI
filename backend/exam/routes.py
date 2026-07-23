"""
backend/exam/routes.py — Exam Session Management
==================================================
ROUTES:
    GET  /exam/dashboard        → Student dashboard
    GET  /exam/available        → List available exams
    POST /exam/start/<exam_id>  → Start an exam session
    GET  /exam/take/<session_id>→ Exam taking interface
    POST /exam/answer           → Submit an answer
    POST /exam/submit/<session_id> → End exam session
    GET  /exam/history          → Student's past exams
"""

from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, jsonify)
from flask_login import login_required, current_user
from datetime import datetime, timezone
import random

from backend.extensions import db
from database.models import Exam, ExamSession, Question, StudentAnswer, ViolationLog

exam_bp = Blueprint('exam', __name__)


@exam_bp.route('/dashboard')
@login_required
def student_dashboard():
    """Student's main dashboard showing their exam history and stats."""
    if current_user.is_admin:
        return redirect(url_for('admin.dashboard'))

    # Get student's sessions
    sessions = ExamSession.query.filter_by(
        student_id=current_user.id
    ).order_by(ExamSession.started_at.desc()).limit(10).all()

    # Get available exams (not yet taken today)
    available_exams = Exam.query.filter_by(is_active=True).all()

    # Compute stats
    total_sessions = ExamSession.query.filter_by(student_id=current_user.id).count()
    completed = ExamSession.query.filter_by(
        student_id=current_user.id, status='completed'
    ).count()
    flagged = ExamSession.query.filter_by(
        student_id=current_user.id, status='flagged'
    ).count()

    stats = {
        'total_exams': total_sessions,
        'completed': completed,
        'flagged': flagged,
        'total_violations': current_user.total_violations,
    }

    return render_template('student/dashboard.html',
                           sessions=sessions,
                           available_exams=available_exams,
                           stats=stats)


@exam_bp.route('/start/<int:exam_id>', methods=['POST'])
@login_required
def start_exam(exam_id):
    """Create a new exam session and redirect to exam taking page."""
    if current_user.is_admin:
        return jsonify({'error': 'Admins cannot take exams'}), 403

    exam = Exam.query.get_or_404(exam_id)
    if not exam.is_active:
        flash('This exam is currently not available.', 'error')
        return redirect(url_for('exam.student_dashboard'))

    # Check if already has active session for this exam
    active_session = ExamSession.query.filter_by(
        student_id=current_user.id,
        exam_id=exam_id,
        status='active'
    ).first()

    if active_session:
        return redirect(url_for('exam.take_exam', session_id=active_session.id))

    # Create new session
    session = ExamSession(
        student_id=current_user.id,
        exam_id=exam_id,
        started_at=datetime.utcnow(),
        status='active',
        ip_address=request.remote_addr
    )
    db.session.add(session)
    db.session.commit()

    return redirect(url_for('exam.take_exam', session_id=session.id))


@exam_bp.route('/take/<int:session_id>')
@login_required
def take_exam(session_id):
    """The main exam-taking interface with webcam monitoring."""
    session = ExamSession.query.get_or_404(session_id)

    # Security: only the session owner can access
    if session.student_id != current_user.id:
        flash('Access denied.', 'error')
        return redirect(url_for('exam.student_dashboard'))

    if session.status != 'active':
        flash('This exam session is no longer active.', 'warning')
        return redirect(url_for('exam.student_dashboard'))

    exam = session.exam

    # Shuffle question order — seed with session_id for consistency
    # (same student refreshing gets same order, but different sessions differ)
    all_questions = exam.questions.order_by(Question.order).all()
    rng = random.Random(session.id)  # deterministic per-session shuffle
    questions = all_questions[:]
    rng.shuffle(questions)

    # Get already-answered questions for this session
    answered = {
        a.question_id: a.selected_answer
        for a in session.answers
    }

    # ── Calculate actual time remaining ───────────────────────────────────────
    # Accounts for page refreshes — time already spent is deducted
    from datetime import timezone
    now_utc = datetime.now(timezone.utc)
    started = session.started_at
    # Make started_at timezone-aware if it isn't already
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    elapsed_seconds = int((now_utc - started).total_seconds())
    total_seconds = exam.duration_minutes * 60
    time_remaining = max(0, total_seconds - elapsed_seconds)

    # Auto-submit if time already expired
    if time_remaining == 0 and session.status == 'active':
        session.status = 'completed'
        session.ended_at = datetime.utcnow()
        session.calculate_risk_score()
        db.session.commit()
        flash('Time is up! Your exam was auto-submitted.', 'warning')
        return redirect(url_for('exam.student_dashboard'))

    return render_template('student/exam.html',
                           session=session,
                           exam=exam,
                           questions=questions,
                           answered=answered,
                           time_remaining=time_remaining)


@exam_bp.route('/answer', methods=['POST'])
@login_required
def submit_answer():
    """Save a student's answer to a question (auto-save)."""
    data = request.get_json()
    session_id = data.get('session_id')
    question_id = data.get('question_id')
    selected = data.get('selected_answer')

    session = ExamSession.query.get_or_404(session_id)
    if session.student_id != current_user.id or session.status != 'active':
        return jsonify({'error': 'Invalid session'}), 403

    # Check if already answered
    existing = StudentAnswer.query.filter_by(
        session_id=session_id, question_id=question_id
    ).first()

    if existing:
        existing.selected_answer = selected
        existing.answered_at = datetime.utcnow()
    else:
        answer = StudentAnswer(
            session_id=session_id,
            question_id=question_id,
            selected_answer=selected
        )
        db.session.add(answer)

    db.session.commit()
    return jsonify({'status': 'saved'})


@exam_bp.route('/submit/<int:session_id>', methods=['POST'])
@login_required
def submit_exam(session_id):
    """End the exam session, calculate score, generate report."""
    session = ExamSession.query.get_or_404(session_id)

    if session.student_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403

    if session.status != 'active':
        return jsonify({'error': 'Session already ended'}), 400

    # Calculate score
    correct = 0
    total_marks = 0
    for answer in session.answers:
        if answer.is_correct:
            correct += answer.question.marks
        total_marks += answer.question.marks

    session.score = correct
    session.ended_at = datetime.utcnow()
    session.calculate_risk_score()

    # Determine final status
    violations = session.total_violations
    if violations >= 5 or session.risk_score >= 7.0:
        session.status = 'flagged'
    else:
        session.status = 'completed'

    db.session.commit()

    from ai.detector import cleanup_detector
    cleanup_detector(session_id)

    flash(f'Exam submitted! Score: {correct}/{total_marks}', 'success')
    return jsonify({
        'status': 'submitted',
        'score': correct,
        'total': total_marks,
        'session_status': session.status,
        'redirect': url_for('exam.student_dashboard')
    })


@exam_bp.route('/history')
@login_required
def exam_history():
    """Show student's complete exam history."""
    sessions = ExamSession.query.filter_by(
        student_id=current_user.id
    ).order_by(ExamSession.started_at.desc()).all()

    return render_template('student/history.html', sessions=sessions)
