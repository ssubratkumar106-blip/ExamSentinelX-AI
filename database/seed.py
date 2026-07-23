"""
database/seed.py — Database Seeder
====================================
PURPOSE: Populate the database with initial data for testing.

WHAT IT CREATES:
    1. Admin user account
    2. Sample student accounts
    3. Sample exam with questions
    4. Some violation log entries for demo

This runs automatically when the app starts for the first time.
If data already exists, it skips to avoid duplicates.
"""

import os
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash
from backend.extensions import db
from database.models import User, Exam, Question, ExamSession, ViolationLog


def seed_database():
    """
    Seed the database with initial demo data.
    Checks if data exists first to prevent duplicates.
    """
    # Check if already seeded
    if User.query.first() is not None:
        return  # Already seeded, skip

    print("  [*] Seeding database with initial data...")

    # ── Create Admin User ──────────────────────────────────────────────────────
    admin = User(
        username=os.getenv('ADMIN_USERNAME', 'admin'),
        email=os.getenv('ADMIN_EMAIL', 'admin@examguard.ai'),
        password_hash=generate_password_hash(
            os.getenv('ADMIN_PASSWORD', 'Admin@123456')
        ),
        full_name='System Administrator',
        role='admin',
        is_active=True
    )
    db.session.add(admin)

    # ── Create Sample Students ─────────────────────────────────────────────────
    students_data = [
        {'username': 'john_doe', 'email': 'john@student.edu',
         'full_name': 'John Doe', 'password': 'Student@123'},
        {'username': 'jane_smith', 'email': 'jane@student.edu',
         'full_name': 'Jane Smith', 'password': 'Student@123'},
        {'username': 'bob_wilson', 'email': 'bob@student.edu',
         'full_name': 'Bob Wilson', 'password': 'Student@123'},
        {'username': 'alice_brown', 'email': 'alice@student.edu',
         'full_name': 'Alice Brown', 'password': 'Student@123'},
    ]

    students = []
    for s in students_data:
        student = User(
            username=s['username'],
            email=s['email'],
            full_name=s['full_name'],
            password_hash=generate_password_hash(s['password']),
            role='student',
            is_active=True
        )
        db.session.add(student)
        students.append(student)

    db.session.flush()  # Get IDs without committing

    # ── Create Sample Exam ─────────────────────────────────────────────────────
    exam = Exam(
        title='Computer Science Fundamentals',
        description='A comprehensive test covering data structures, algorithms, and OOP concepts.',
        subject='Computer Science',
        duration_minutes=60,
        total_marks=50,
        created_by=admin.id,
        is_active=True
    )
    db.session.add(exam)
    db.session.flush()

    # ── Create Questions ───────────────────────────────────────────────────────
    questions_data = [
        {
            'question_text': 'What is the time complexity of binary search?',
            'option_a': 'O(n)', 'option_b': 'O(log n)',
            'option_c': 'O(n²)', 'option_d': 'O(1)',
            'correct_answer': 'B', 'marks': 5
        },
        {
            'question_text': 'Which data structure uses LIFO (Last In First Out) principle?',
            'option_a': 'Queue', 'option_b': 'Array',
            'option_c': 'Stack', 'option_d': 'Linked List',
            'correct_answer': 'C', 'marks': 5
        },
        {
            'question_text': 'What does OOP stand for?',
            'option_a': 'Open Oriented Programming', 'option_b': 'Object Oriented Programming',
            'option_c': 'Operator Object Programming', 'option_d': 'Object Output Processing',
            'correct_answer': 'B', 'marks': 5
        },
        {
            'question_text': 'Which sorting algorithm has the best average time complexity?',
            'option_a': 'Bubble Sort', 'option_b': 'Selection Sort',
            'option_c': 'QuickSort', 'option_d': 'Insertion Sort',
            'correct_answer': 'C', 'marks': 5
        },
        {
            'question_text': 'What is a primary key in a relational database?',
            'option_a': 'The first column in a table',
            'option_b': 'A unique identifier for each record',
            'option_c': 'A foreign key reference',
            'option_d': 'The largest value in a column',
            'correct_answer': 'B', 'marks': 5
        },
        {
            'question_text': 'In Python, what is a list comprehension?',
            'option_a': 'A type of loop',
            'option_b': 'A shorthand way to create lists',
            'option_c': 'A sorting algorithm',
            'option_d': 'A data type',
            'correct_answer': 'B', 'marks': 5
        },
        {
            'question_text': 'What does CPU stand for?',
            'option_a': 'Central Processing Unit',
            'option_b': 'Computer Processing Unit',
            'option_c': 'Core Program Utility',
            'option_d': 'Central Program Unit',
            'correct_answer': 'A', 'marks': 5
        },
        {
            'question_text': 'Which HTTP method is used to send data to a server?',
            'option_a': 'GET', 'option_b': 'DELETE',
            'option_c': 'POST', 'option_d': 'HEAD',
            'correct_answer': 'C', 'marks': 5
        },
        {
            'question_text': 'What is recursion in programming?',
            'option_a': 'A function calling another function',
            'option_b': 'A loop that never ends',
            'option_c': 'A function calling itself',
            'option_d': 'A variable referencing itself',
            'correct_answer': 'C', 'marks': 5
        },
        {
            'question_text': 'Which layer of the OSI model handles routing?',
            'option_a': 'Data Link Layer',
            'option_b': 'Transport Layer',
            'option_c': 'Network Layer',
            'option_d': 'Session Layer',
            'correct_answer': 'C', 'marks': 5
        },
    ]

    for i, q_data in enumerate(questions_data):
        q = Question(
            exam_id=exam.id,
            question_text=q_data['question_text'],
            option_a=q_data['option_a'],
            option_b=q_data['option_b'],
            option_c=q_data['option_c'],
            option_d=q_data['option_d'],
            correct_answer=q_data['correct_answer'],
            marks=q_data['marks'],
            order=i + 1
        )
        db.session.add(q)

    # ── Create Sample Completed Sessions (for demo dashboard) ──────────────────
    sample_session = ExamSession(
        student_id=students[0].id,
        exam_id=exam.id,
        started_at=datetime.utcnow() - timedelta(hours=2),
        ended_at=datetime.utcnow() - timedelta(hours=1),
        status='flagged',
        total_violations=3,
        risk_score=6.5,
        score=35
    )
    db.session.add(sample_session)
    db.session.flush()

    # Sample violations for demo
    violations_data = [
        {
            'violation_type': 'phone_detected',
            'confidence': 0.87,
            'timestamp': datetime.utcnow() - timedelta(hours=1, minutes=45),
            'details': '{"class": "cell phone", "bbox": [120, 80, 200, 150]}'
        },
        {
            'violation_type': 'looking_away',
            'confidence': 0.92,
            'timestamp': datetime.utcnow() - timedelta(hours=1, minutes=30),
            'details': '{"yaw": 45.2, "pitch": 10.1, "direction": "left"}'
        },
        {
            'violation_type': 'face_absent',
            'confidence': 1.0,
            'timestamp': datetime.utcnow() - timedelta(hours=1, minutes=15),
            'details': '{"duration_seconds": 5}'
        },
    ]

    for v_data in violations_data:
        v = ViolationLog(
            session_id=sample_session.id,
            violation_type=v_data['violation_type'],
            confidence=v_data['confidence'],
            timestamp=v_data['timestamp'],
            details=v_data['details']
        )
        db.session.add(v)

    # ── Commit Everything ──────────────────────────────────────────────────────
    db.session.commit()
    print("  [OK] Database seeded successfully!")
    print("     [Admin]   username: admin       / password: Admin@123456")
    print("     [Student] username: john_doe    / password: Student@123")
