"""
backend/admin/routes.py — Admin Dashboard
==========================================
ROUTES:
    GET  /admin/dashboard                    → Overview stats
    GET  /admin/students                     → All students list
    GET  /admin/session/<id>                 → Session detail view
    GET  /admin/reports/<session_id>         → Download PDF report
    GET  /admin/exam/create                  → Create exam form
    POST /admin/exam/create                  → Submit new exam + questions
    GET  /admin/exam/<id>/questions          → Manage exam questions
    POST /admin/exam/<id>/question/add       → Add a question to exam
    POST /admin/question/<id>/delete         → Delete a question
    POST /admin/user/<id>/toggle             → Activate/deactivate user
"""

import json as json_mod
from flask import (Blueprint, render_template, request,
                   redirect, url_for, flash, jsonify, send_file)
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime, timedelta, timezone

from backend.extensions import db
from database.models import User, Exam, Question, ExamSession, ViolationLog

admin_bp = Blueprint('admin', __name__)


def admin_required(f):
    """
    Decorator that restricts routes to admin users only.
    Example usage: @admin_required on a route function.
    """
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('exam.student_dashboard'))
        return f(*args, **kwargs)
    return decorated_function


@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """
    Admin overview dashboard with:
    - Total students, exams, sessions
    - Recent violations
    - Flagged sessions
    - Activity charts data
    """
    # ── Summary Statistics ─────────────────────────────────────────────────────
    total_students = User.query.filter_by(role='student').count()
    total_exams = Exam.query.count()
    total_sessions = ExamSession.query.count()
    total_violations = ViolationLog.query.count()
    flagged_sessions = ExamSession.query.filter_by(status='flagged').count()
    active_sessions = ExamSession.query.filter_by(status='active').count()

    stats = {
        'total_students': total_students,
        'total_exams': total_exams,
        'total_sessions': total_sessions,
        'total_violations': total_violations,
        'flagged_sessions': flagged_sessions,
        'active_sessions': active_sessions,
    }

    # ── Recent Flagged Sessions ────────────────────────────────────────────────
    flagged = ExamSession.query.filter_by(
        status='flagged'
    ).order_by(ExamSession.started_at.desc()).limit(10).all()

    # ── Recent Violations ──────────────────────────────────────────────────────
    recent_violations = ViolationLog.query.order_by(
        ViolationLog.timestamp.desc()
    ).limit(20).all()

    # ── Violation Type Distribution (for charts) ───────────────────────────────
    violation_types = db.session.execute(
        db.text("""
            SELECT violation_type, COUNT(*) as count
            FROM violation_logs
            GROUP BY violation_type
            ORDER BY count DESC
        """)
    ).fetchall()

    chart_data = {
        'labels': [row[0].replace('_', ' ').title() for row in violation_types],
        'values': [row[1] for row in violation_types]
    }

    # ── Daily Violations (last 7 days) ─────────────────────────────────────────
    daily_data = []
    for i in range(6, -1, -1):
        day = datetime.now(timezone.utc) - timedelta(days=i)
        count = ViolationLog.query.filter(
            ViolationLog.timestamp >= day.replace(hour=0, minute=0, second=0),
            ViolationLog.timestamp < day.replace(hour=23, minute=59, second=59)
        ).count()
        daily_data.append({'date': day.strftime('%b %d'), 'count': count})

    return render_template('admin/dashboard.html',
                           stats=stats,
                           flagged=flagged,
                           recent_violations=recent_violations,
                           chart_data=chart_data,
                           daily_data=daily_data)


@admin_bp.route('/students')
@admin_required
def students():
    """List all students with their session stats."""
    all_students = User.query.filter_by(role='student').order_by(
        User.created_at.desc()
    ).all()

    student_data = []
    for student in all_students:
        sessions = student.sessions.all()
        student_data.append({
            'user': student,
            'total_sessions': len(sessions),
            'flagged_sessions': sum(1 for s in sessions if s.status == 'flagged'),
            'total_violations': sum(s.total_violations for s in sessions),
            'avg_risk': round(
                sum(s.risk_score for s in sessions) / len(sessions), 2
            ) if sessions else 0.0
        })

    return render_template('admin/students.html', students=student_data)


@admin_bp.route('/session/<int:session_id>')
@admin_required
def session_detail(session_id):
    """Detailed view of a specific exam session with all violations."""
    session = ExamSession.query.get_or_404(session_id)
    violations = session.violations.order_by(ViolationLog.timestamp.asc()).all()

    violation_timeline = []
    for v in violations:
        violation_timeline.append({
            'type': v.violation_display,
            'confidence': round(v.confidence or 0, 2),
            'time': v.timestamp.strftime('%H:%M:%S'),
            'severity': v.severity,
            'screenshot': v.screenshot_path
        })

    return render_template('admin/session_detail.html',
                           session=session,
                           violations=violations,
                           timeline=violation_timeline)


@admin_bp.route('/reports/<int:session_id>')
@admin_required
def generate_report(session_id):
    """Generate and download PDF report for a session."""
    from backend.reports.generator import generate_session_report

    session = ExamSession.query.get_or_404(session_id)
    pdf_path = generate_session_report(session)

    if pdf_path:
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=f'ExamSentinelX_Report_Session_{session_id}.pdf',
            mimetype='application/pdf'
        )

    flash('Failed to generate report.', 'error')
    return redirect(url_for('admin.session_detail', session_id=session_id))


@admin_bp.route('/exam/create', methods=['GET', 'POST'])
@admin_required
def create_exam():
    """Create a new exam with questions (questions submitted as JSON)."""
    if request.method == 'POST':
        title       = request.form.get('title', '').strip()
        description = request.form.get('description', '').strip()
        subject     = request.form.get('subject', '').strip()
        duration    = int(request.form.get('duration_minutes', 60))
        total_marks = int(request.form.get('total_marks', 100))

        if not title:
            flash('Exam title is required.', 'error')
            return redirect(url_for('admin.create_exam'))

        exam = Exam(
            title=title,
            description=description,
            subject=subject,
            duration_minutes=duration,
            total_marks=total_marks,
            created_by=current_user.id,
            is_active=True
        )
        db.session.add(exam)
        db.session.flush()  # Get exam.id without full commit

        # ── Parse and insert questions from JSON payload ─────────────────────
        questions_json = request.form.get('questions_json', '[]')
        try:
            q_data = json_mod.loads(questions_json)
        except (ValueError, TypeError):
            q_data = []

        inserted = 0
        for i, q in enumerate(q_data):
            if not all(k in q for k in ('text', 'option_a', 'option_b',
                                         'option_c', 'option_d', 'correct_answer')):
                continue  # skip malformed questions
            question = Question(
                exam_id=exam.id,
                question_text=q['text'],
                option_a=q['option_a'],
                option_b=q['option_b'],
                option_c=q['option_c'],
                option_d=q['option_d'],
                correct_answer=q['correct_answer'].upper(),
                marks=int(q.get('marks', 1)),
                order=i
            )
            db.session.add(question)
            inserted += 1

        db.session.commit()

        flash(f'Exam "{title}" created with {inserted} question(s)!', 'success')
        return redirect(url_for('admin.manage_questions', exam_id=exam.id))

    return render_template('admin/create_exam.html')


@admin_bp.route('/exam/<int:exam_id>/questions', methods=['GET'])
@admin_required
def manage_questions(exam_id):
    """View and manage questions for an existing exam."""
    exam = Exam.query.get_or_404(exam_id)
    questions = Question.query.filter_by(exam_id=exam_id).order_by(Question.order).all()
    return render_template('admin/manage_questions.html', exam=exam, questions=questions)


@admin_bp.route('/exam/<int:exam_id>/question/add', methods=['POST'])
@admin_required
def add_question(exam_id):
    """Add a single question to an existing exam (from manage_questions page)."""
    exam = Exam.query.get_or_404(exam_id)

    question_text  = request.form.get('question_text', '').strip()
    option_a       = request.form.get('option_a', '').strip()
    option_b       = request.form.get('option_b', '').strip()
    option_c       = request.form.get('option_c', '').strip()
    option_d       = request.form.get('option_d', '').strip()
    correct_answer = request.form.get('correct_answer', '').upper().strip()
    marks          = int(request.form.get('marks', 1))

    if not all([question_text, option_a, option_b, option_c, option_d, correct_answer]):
        flash('All question fields are required.', 'error')
        return redirect(url_for('admin.manage_questions', exam_id=exam_id))

    if correct_answer not in ('A', 'B', 'C', 'D'):
        flash('Correct answer must be A, B, C, or D.', 'error')
        return redirect(url_for('admin.manage_questions', exam_id=exam_id))

    order = Question.query.filter_by(exam_id=exam_id).count()
    q = Question(
        exam_id=exam_id,
        question_text=question_text,
        option_a=option_a,
        option_b=option_b,
        option_c=option_c,
        option_d=option_d,
        correct_answer=correct_answer,
        marks=marks,
        order=order
    )
    db.session.add(q)
    db.session.commit()
    flash('Question added successfully.', 'success')
    return redirect(url_for('admin.manage_questions', exam_id=exam_id))


@admin_bp.route('/question/<int:question_id>/delete', methods=['POST'])
@admin_required
def delete_question(question_id):
    """Delete a question from an exam."""
    question = Question.query.get_or_404(question_id)
    exam_id  = question.exam_id
    db.session.delete(question)
    db.session.commit()
    flash('Question deleted.', 'success')
    return redirect(url_for('admin.manage_questions', exam_id=exam_id))


@admin_bp.route('/api/stats')
@admin_required
def api_stats():
    """JSON API for live dashboard updates."""
    active_sessions = ExamSession.query.filter_by(status='active').count()
    recent_violations = ViolationLog.query.filter(
        ViolationLog.timestamp >= datetime.now(timezone.utc) - timedelta(minutes=5)
    ).count()

    return jsonify({
        'active_sessions': active_sessions,
        'recent_violations': recent_violations,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


@admin_bp.route('/exams')
@admin_required
def exams():
    """List all exams with session counts."""
    all_exams = Exam.query.order_by(Exam.created_at.desc()).all()
    exam_data = []
    for exam in all_exams:
        sessions = ExamSession.query.filter_by(exam_id=exam.id).all()
        exam_data.append({
            'exam': exam,
            'total_sessions': len(sessions),
            'flagged_sessions': sum(1 for s in sessions if s.status == 'flagged'),
            'avg_score': round(
                sum(s.score or 0 for s in sessions) / len(sessions), 1
            ) if sessions else 0.0,
            'creator': User.query.get(exam.created_by)
        })
    return render_template('admin/exams.html', exam_data=exam_data)


@admin_bp.route('/users')
@admin_required
def users():
    """List all users (admin + students)."""
    all_users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin/users.html', users=all_users)


@admin_bp.route('/user/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    """Activate or deactivate a user account."""
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        flash('You cannot deactivate your own account.', 'error')
        return redirect(url_for('admin.users'))
    user.is_active = not user.is_active
    db.session.commit()
    status = 'activated' if user.is_active else 'deactivated'
    flash(f'User {user.username} has been {status}.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/exam/<int:exam_id>/toggle', methods=['POST'])
@admin_required
def toggle_exam(exam_id):
    """Activate or deactivate an exam."""
    exam = Exam.query.get_or_404(exam_id)
    exam.is_active = not exam.is_active
    db.session.commit()
    status = 'activated' if exam.is_active else 'deactivated'
    flash(f'Exam "{exam.title}" has been {status}.', 'success')
    return redirect(url_for('admin.exams'))


@admin_bp.route('/exam/<int:exam_id>/delete', methods=['POST'])
@admin_required
def delete_exam(exam_id):
    """Delete an exam and all its sessions/violations."""
    exam = Exam.query.get_or_404(exam_id)
    title = exam.title
    # Cascade delete handled by ORM relationships
    db.session.delete(exam)
    db.session.commit()
    flash(f'Exam "{title}" deleted successfully.', 'success')
    return redirect(url_for('admin.exams'))
