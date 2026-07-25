import unittest
from backend.app import create_app
from backend.extensions import db
from database.models import User, Exam, ExamSession, Question
from database.seed import seed_database
import json

class PlatformHealthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config['TESTING'] = True
        cls.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        cls.app.config['WTF_CSRF_ENABLED'] = False

        with cls.app.app_context():
            db.create_all()
            seed_database() # This adds admin and some sample data

    @classmethod
    def tearDownClass(cls):
        with cls.app.app_context():
            db.session.remove()
            db.drop_all()

    def get_client(self, email):
        client = self.app.test_client()
        with self.app.app_context():
            user = User.query.filter_by(email=email).first()
            if user:
                with client.session_transaction() as sess:
                    sess['_user_id'] = str(user.id)
                    sess['_fresh'] = True
        return client

    def test_01_admin_workflow(self):
        print("\n--- Testing Admin Workflow ---")
        client = self.get_client('admin@examguard.ai')
        
        # 2. Access Admin Dashboard
        response = client.get('/admin/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'System overview', response.data)
        print("[OK] Admin Dashboard Loaded Perfectly")

        # 3. Access Admin Students List
        response = client.get('/admin/students')
        self.assertEqual(response.status_code, 200)
        print("[OK] Admin Students View Loaded Perfectly")

    def test_02_student_workflow(self):
        print("\n--- Testing Student Workflow ---")
        client = self.get_client('john@student.edu')

        # 2. Access Student Dashboard
        response = client.get('/exam/dashboard')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Available Exams', response.data)
        print("[OK] Student Dashboard Loaded Perfectly")

        # 3. Start Exam (Find an exam ID)
        with self.app.app_context():
            exam = Exam.query.first()
            exam_id = exam.id

        response = client.post(f'/exam/start/{exam_id}', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Exam Navigation', response.data)
        print("[OK] Exam Session Started Perfectly")

        # Get session ID
        with self.app.app_context():
            # Get the ACTIVE session we just created, not the seeded one
            session = ExamSession.query.filter_by(
                exam_id=exam_id, status='active'
            ).order_by(ExamSession.started_at.desc()).first()
            session_id = session.id
            question = session.exam.questions.first()
            question_id = question.id
            total_marks = session.exam.total_marks

        # 4. Submit Answer
        response = client.post('/exam/answer', json={
            'session_id': session_id,
            'question_id': question_id,
            'selected_answer': 'A'
        })
        self.assertEqual(response.status_code, 200)
        print("[OK] Answer Submitted Perfectly")

        # 5. Submit Exam
        response = client.post(f'/exam/submit/{session_id}')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'submitted')
        self.assertEqual(data['total'], total_marks) # Verify our recent fix!
        print(f"[OK] Exam Graded Perfectly! Expected Total: {total_marks}, Output Total: {data['total']}")

if __name__ == '__main__':
    unittest.main(verbosity=2)
