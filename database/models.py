"""
database/models.py — SQLAlchemy ORM Database Models
====================================================
PURPOSE: Define all database tables as Python classes.

WHAT IS ORM?
    ORM = Object-Relational Mapper.
    Instead of writing raw SQL, we define Python classes that 
    map to database tables. SQLAlchemy then generates the SQL for us.

TABLES DEFINED:
    1. User         — Students and admins
    2. Exam         — Exam definitions
    3. Question     — Exam questions
    4. ExamSession  — When a student takes an exam
    5. ViolationLog — Each detected cheating event
    6. StudentAnswer — Student's answers during exam

RELATIONSHIPS:
    User ──< ExamSession ──< ViolationLog
    Exam ──< ExamSession
    Exam ──< Question ──< StudentAnswer ──> ExamSession
"""

from datetime import datetime, timezone
from typing import Optional
from flask_login import UserMixin
from backend.extensions import db


# ═══════════════════════════════════════════════════════════════════════════════
# USER MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class User(UserMixin, db.Model):
    """
    Represents both students and administrators.
    
    Flask-Login's UserMixin provides default implementations of:
    - is_authenticated (True if logged in)
    - is_active (True if account enabled)
    - get_id() (returns str version of user ID)
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    full_name = db.Column(db.String(120), nullable=False, default='')
    role = db.Column(db.String(10), nullable=False, default='student')  # 'student' or 'admin'
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    profile_image = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # ── OAuth & OTP Fields ─────────────────────────────────────────────────────
    auth_provider = db.Column(db.String(20), default='local', nullable=False) # 'local', 'google', 'microsoft'
    oauth_id = db.Column(db.String(120), unique=True, nullable=True)
    otp_code = db.Column(db.String(6), nullable=True)
    otp_expiry = db.Column(db.DateTime, nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    # back_populates creates a two-way link: user.sessions & session.student
    sessions = db.relationship('ExamSession', back_populates='student', lazy='dynamic')
    created_exams = db.relationship('Exam', back_populates='creator', lazy='dynamic')

    @property
    def is_admin(self):
        """Check if user has admin role."""
        return self.role == 'admin'

    @property
    def total_violations(self):
        """Total violations across all sessions."""
        total = 0
        for session in self.sessions:
            total += session.total_violations
        return total

    def __repr__(self):
        return f'<User {self.username} ({self.role})>'


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class Exam(db.Model):
    """
    Represents an exam that can be taken by students.
    Admins create exams; students take them.
    """
    __tablename__ = 'exams'

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    subject = db.Column(db.String(100), nullable=True)
    duration_minutes = db.Column(db.Integer, default=60, nullable=False)
    total_marks = db.Column(db.Integer, default=100)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    scheduled_at = db.Column(db.DateTime, nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    creator = db.relationship('User', back_populates='created_exams')
    questions = db.relationship('Question', back_populates='exam', lazy='dynamic',
                                 cascade='all, delete-orphan')
    sessions = db.relationship('ExamSession', back_populates='exam', lazy='dynamic')

    def __init__(self,
                 title: str,
                 created_by: int,
                 description: str = '',
                 subject: str = '',
                 duration_minutes: int = 60,
                 total_marks: int = 100,
                 is_active: bool = True,
                 scheduled_at: Optional[datetime] = None):
        """Explicit __init__ so type checkers can validate constructor calls."""
        self.title = title
        self.created_by = created_by
        self.description = description
        self.subject = subject
        self.duration_minutes = duration_minutes
        self.total_marks = total_marks
        self.is_active = is_active
        self.scheduled_at = scheduled_at

    @property
    def question_count(self):
        return self.questions.count()

    def __repr__(self):
        return f'<Exam {self.title}>'


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class Question(db.Model):
    """Multiple-choice question belonging to an exam."""
    __tablename__ = 'questions'

    id = db.Column(db.Integer, primary_key=True)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    option_a = db.Column(db.String(500), nullable=False)
    option_b = db.Column(db.String(500), nullable=False)
    option_c = db.Column(db.String(500), nullable=False)
    option_d = db.Column(db.String(500), nullable=False)
    correct_answer = db.Column(db.String(1), nullable=False)  # 'A', 'B', 'C', or 'D'
    marks = db.Column(db.Integer, default=1)
    order = db.Column(db.Integer, default=0)

    # ── Relationships ──────────────────────────────────────────────────────────
    exam = db.relationship('Exam', back_populates='questions')
    answers = db.relationship('StudentAnswer', back_populates='question', lazy='dynamic')

    def __init__(self,
                 exam_id: int,
                 question_text: str,
                 option_a: str,
                 option_b: str,
                 option_c: str,
                 option_d: str,
                 correct_answer: str,
                 marks: int = 1,
                 order: int = 0):
        """Explicit __init__ so type checkers can validate constructor calls."""
        self.exam_id = exam_id
        self.question_text = question_text
        self.option_a = option_a
        self.option_b = option_b
        self.option_c = option_c
        self.option_d = option_d
        self.correct_answer = correct_answer
        self.marks = marks
        self.order = order

    def __repr__(self):
        return f'<Question {self.id}: {self.question_text[:50]}...>'


# ═══════════════════════════════════════════════════════════════════════════════
# EXAM SESSION MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ExamSession(db.Model):
    """
    Represents one instance of a student taking an exam.
    
    This is the CORE table for the proctoring system:
    - Tracks when exam started/ended
    - Stores total violation count
    - Calculates risk score
    - Links to all violation events
    """
    __tablename__ = 'exam_sessions'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    exam_id = db.Column(db.Integer, db.ForeignKey('exams.id'), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    status = db.Column(db.String(20), default='active', nullable=False)
    # status options: 'active', 'completed', 'flagged', 'terminated'
    total_violations = db.Column(db.Integer, default=0, nullable=False)
    risk_score = db.Column(db.Float, default=0.0, nullable=False)
    trust_score = db.Column(db.Float, default=100.0, nullable=False)  # 100→0 decay per violation
    score = db.Column(db.Integer, nullable=True)  # Exam score after completion
    ip_address = db.Column(db.String(50), nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    student = db.relationship('User', back_populates='sessions')
    exam = db.relationship('Exam', back_populates='sessions')
    violations = db.relationship('ViolationLog', back_populates='session',
                                  lazy='dynamic', cascade='all, delete-orphan')
    answers = db.relationship('StudentAnswer', back_populates='session',
                               lazy='dynamic', cascade='all, delete-orphan')

    def __init__(self, **kwargs):
        super(ExamSession, self).__init__(**kwargs)
        if getattr(self, 'trust_score', None) is None:
            self.trust_score = 100.0
        if getattr(self, 'risk_score', None) is None:
            self.risk_score = 0.0
        if getattr(self, 'total_violations', None) is None:
            self.total_violations = 0

    @property
    def duration_seconds(self):
        """How long the session lasted in seconds."""
        if self.ended_at and self.started_at:
            return int((self.ended_at - self.started_at).total_seconds())
        elif self.started_at:
            return int((datetime.now(timezone.utc).replace(tzinfo=None) - self.started_at).total_seconds())
        return 0

    @property
    def duration_display(self):
        """Human-readable duration: '45 min 23 sec'"""
        secs = self.duration_seconds
        return f"{secs // 60} min {secs % 60} sec"

    # ── Trust Score Penalties (type-specific, from ExamGuard architecture) ───────
    TRUST_PENALTIES = {
        'phone_detected':    20.0,
        'multiple_persons':  15.0,
        'face_absent':        5.0,
        'looking_away':       3.0,
        'suspicious_object': 10.0,
        'suspicious_behavior': 8.0,
        'talking_to_others': 12.0,
        'tab_switch':        10.0,
        'fullscreen_exit':   10.0,
    }

    def update_trust_score(self, violation_type: str) -> float:
        """
        Deduct penalty from trust_score (100 → 0) for the given violation type.
        Returns the new trust_score.
        """
        penalty = self.TRUST_PENALTIES.get(violation_type, 5.0)
        self.trust_score = max(0.0, self.trust_score - penalty)
        return self.trust_score

    def calculate_risk_score(self):
        """
        Calculate a risk score from 0.0 to 10.0.
        
        Scoring logic:
        - Each violation adds weight based on type
        - Capped at 10.0 (maximum risk)
        """
        weights = {
            'phone_detected': 3.0,
            'multiple_persons': 3.0,
            'face_absent': 2.0,
            'looking_away': 1.0,
            'suspicious_object': 2.5,
            'talking_to_others': 2.0,
            'tab_switch': 1.5,
            'fullscreen_exit': 1.5,
        }
        score = 0.0
        for v in self.violations:
            weight = weights.get(v.violation_type, 1.0)
            score += weight * (v.confidence or 0.5)
        self.risk_score = min(10.0, score)
        return self.risk_score

    def __repr__(self):
        return f'<ExamSession student={self.student_id} exam={self.exam_id} status={self.status}>'


# ═══════════════════════════════════════════════════════════════════════════════
# VIOLATION LOG MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class ViolationLog(db.Model):
    """
    Records each detected cheating/abnormal activity event.
    
    This is the PRIMARY output of the AI detection module.
    Every time the AI detects something suspicious, a row is inserted here.
    
    VIOLATION TYPES (from the IEEE paper + ExamGuard extension):
        - 'face_absent'       → No face detected (student left)
        - 'multiple_persons'  → More than one person visible
        - 'looking_away'      → Head pose outside allowed range
        - 'phone_detected'    → Mobile phone visible (YOLOv8)
        - 'suspicious_object' → Book, tablet, or other banned item
        - 'suspicious_behavior' → CNN classifier flagged anomaly
        - 'talking_to_others' → Lip movement / audio anomaly
        - 'tab_switch'        → Browser tab switched or window minimized (JS)
        - 'fullscreen_exit'   → Candidate exited fullscreen mode (JS)
    """
    __tablename__ = 'violation_logs'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('exam_sessions.id'), nullable=False)
    violation_type = db.Column(db.String(50), nullable=False, index=True)
    confidence = db.Column(db.Float, nullable=True)  # AI detection confidence 0.0–1.0
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    screenshot_path = db.Column(db.String(500), nullable=True)  # Path to saved frame
    details = db.Column(db.Text, nullable=True)  # JSON string with extra info
    is_reviewed = db.Column(db.Boolean, default=False)  # Admin reviewed this

    # ── Relationships ──────────────────────────────────────────────────────────
    session = db.relationship('ExamSession', back_populates='violations')

    def __init__(self,
                 session_id: int,
                 violation_type: str,
                 confidence: Optional[float] = None,
                 timestamp: Optional[datetime] = None,
                 screenshot_path: Optional[str] = None,
                 details: Optional[str] = None,
                 is_reviewed: bool = False):
        """Explicit __init__ so type checkers can validate constructor calls."""
        self.session_id = session_id
        self.violation_type = violation_type
        self.confidence = confidence
        if timestamp is not None:
            self.timestamp = timestamp
        self.screenshot_path = screenshot_path
        self.details = details
        self.is_reviewed = is_reviewed

    @property
    def severity(self):
        """Return human-readable severity: 'High', 'Medium', 'Low'"""
        high   = ['phone_detected', 'multiple_persons', 'talking_to_others']
        medium = ['suspicious_object', 'face_absent', 'suspicious_behavior',
                  'tab_switch', 'fullscreen_exit']
        if self.violation_type in high:
            return 'High'
        elif self.violation_type in medium:
            return 'Medium'
        return 'Low'

    @property
    def violation_display(self):
        """Human-readable violation name."""
        names = {
            'face_absent': 'Face Not Visible',
            'multiple_persons': 'Multiple Persons Detected',
            'looking_away': 'Looking Away',
            'phone_detected': 'Mobile Phone Detected',
            'suspicious_object': 'Suspicious Object Detected',
            'suspicious_behavior': 'Suspicious Behavior (AI)',
            'talking_to_others': 'Talking Detected',
            'tab_switch': 'Tab / Window Switch',
            'fullscreen_exit': 'Exited Fullscreen',
        }
        return names.get(self.violation_type, self.violation_type.replace('_', ' ').title())

    def __repr__(self):
        return f'<ViolationLog {self.violation_type} @ {self.timestamp}>'


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT ANSWER MODEL
# ═══════════════════════════════════════════════════════════════════════════════
class StudentAnswer(db.Model):
    """Records a student's answer to a specific question during an exam session."""
    __tablename__ = 'student_answers'

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.Integer, db.ForeignKey('exam_sessions.id'), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    selected_answer = db.Column(db.String(1), nullable=True)  # 'A', 'B', 'C', 'D' or None
    answered_at = db.Column(db.DateTime, default=datetime.utcnow)
    time_spent_seconds = db.Column(db.Integer, default=0)

    # ── Relationships ──────────────────────────────────────────────────────────
    session = db.relationship('ExamSession', back_populates='answers')
    question = db.relationship('Question', back_populates='answers')

    @property
    def is_correct(self):
        """Check if the selected answer matches the correct answer."""
        if self.selected_answer and self.question:
            return self.selected_answer.upper() == self.question.correct_answer.upper()
        return False

    def __repr__(self):
        return f'<StudentAnswer Q{self.question_id}: {self.selected_answer}>'
