import base64
from backend.app import create_app
from backend.extensions import db
from database.models import User, Exam, ExamSession

app = create_app()

with app.app_context():
    admin = User.query.filter_by(role='admin').first()
    if not admin:
        print("No admin found")
    else:
        # Create dummy session
        exam = Exam.query.first()
        sess = ExamSession(student_id=admin.id, exam_id=exam.id, status='active')
        db.session.add(sess)
        db.session.commit()

        client = app.test_client()
        with client.session_transaction() as sess_dict:
            # Manually login user in test client
            sess_dict['_user_id'] = str(admin.id)
            sess_dict['_fresh'] = True
            sess_dict['_id'] = "123"

        print("Sending request")
        res = client.post('/monitor/frame', json={
            'session_id': sess.id,
            'frame': 'data:image/jpeg;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
        })
        print(res.status_code)
        try:
            print(res.json)
        except Exception as e:
            print("Not JSON:", e)
            print(res.data.decode()[:200])

        # Cleanup
        db.session.delete(sess)
        db.session.commit()
