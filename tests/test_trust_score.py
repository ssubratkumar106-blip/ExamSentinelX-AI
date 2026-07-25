import pytest
from database.models import ExamSession, ViolationLog

def test_trust_score_initialization():
    session = ExamSession()
    assert session.trust_score == 100.0
    assert session.total_violations == 0
    assert session.risk_score == 0.0

def test_trust_score_decay():
    session = ExamSession()
    
    # Phone detected should deduct 20
    new_score = session.update_trust_score('phone_detected')
    assert new_score == 80.0
    assert session.trust_score == 80.0
    
    # Tab switch should deduct 10
    new_score = session.update_trust_score('tab_switch')
    assert new_score == 70.0
    
    # Face absent should deduct 5
    new_score = session.update_trust_score('face_absent')
    assert new_score == 65.0

def test_trust_score_floor():
    session = ExamSession()
    
    # Repeated violations should not drop score below 0
    for _ in range(10):
        session.update_trust_score('phone_detected')
        
    assert session.trust_score == 0.0

def test_risk_score_calculation():
    session = ExamSession()
    
    # Add some mock violations
    v1 = ViolationLog(session_id=1, violation_type='phone_detected', confidence=1.0)
    v2 = ViolationLog(session_id=1, violation_type='tab_switch', confidence=1.0)
    
    session.violations = [v1, v2]
    
    # Phone (3.0 * 1.0) + Tab switch (1.5 * 1.0) = 4.5
    risk = session.calculate_risk_score()
    assert risk == 4.5
    assert session.risk_score == 4.5
