/**
 * exam.js — Exam Taking Logic
 * ============================
 * Manages the exam interface: timer, auto-save answers, progress, submission.
 */

// ── State ──────────────────────────────────────────────────────────────────────
let sessionId     = null;
let examId        = null;
let durationMin   = 60;
let timeLeft      = 0;
let timerInterval = null;
let answeredCount = 0;
let totalQuestions = 0;
let isSubmitting  = false;

// ── Initialize on DOM ready ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const dataEl = document.getElementById('session-data');
  if (!dataEl) return;

  sessionId   = parseInt(dataEl.dataset.sessionId);
  examId      = parseInt(dataEl.dataset.examId);
  durationMin = parseInt(dataEl.dataset.duration);

  // Use server-computed remaining time (correct on page refresh)
  const serverRemaining = parseInt(dataEl.dataset.timeRemaining);
  timeLeft = isNaN(serverRemaining) ? durationMin * 60 : serverRemaining;

  // Count all question cards in DOM
  totalQuestions = document.querySelectorAll('.q-card').length;

  // Count pre-answered questions (from server-rendered state)
  document.querySelectorAll('input[type="radio"]:checked').forEach(() => answeredCount++);
  updateProgress();

  // Initialize AI monitoring FIRST so sessionId is set before webcam fires
  if (typeof initMonitoring === 'function') initMonitoring(sessionId);

  // Start countdown timer
  startTimer();

  // Initialize browser monitoring for tab/window switches
  initBrowserMonitoring();

  // Prevent accidental page close during exam
  window.addEventListener('beforeunload', (e) => {
    if (!isSubmitting) {
      e.preventDefault();
      e.returnValue = 'Your exam session is active. Are you sure you want to leave?';
    }
  });

  console.log(`[Exam] Session ${sessionId} | ${totalQuestions} questions | ${timeLeft}s remaining`);
});

// ── Timer ──────────────────────────────────────────────────────────────────────
function startTimer() {
  const timerEl = document.getElementById('exam-timer');
  if (!timerEl) return;

  timerInterval = setInterval(() => {
    timeLeft = Math.max(0, timeLeft - 1);

    const mins = Math.floor(timeLeft / 60);
    const secs = timeLeft % 60;
    timerEl.textContent = `${pad(mins)}:${pad(secs)}`;

    // Visual warning states
    timerEl.className = 'timer-value';
    if (timeLeft <= 300) timerEl.classList.add('warn');    // 5 min — yellow
    if (timeLeft <= 60)  timerEl.classList.add('danger');  // 1 min — red pulse

    // Alerts at key moments
    if (timeLeft === 300) showTimerWarning('5 minutes remaining — please review your answers.');
    if (timeLeft === 60)  showTimerWarning('1 minute remaining! Submit now.');
    if (timeLeft <= 0) {
      clearInterval(timerInterval);
      showTimerWarning('Time is up! Submitting your exam...');
      setTimeout(() => submitExam(), 2500);
    }
  }, 1000);
}

function pad(n) { return String(n).padStart(2, '0'); }

function showTimerWarning(message) {
  const overlay = document.getElementById('alert-overlay');
  const title   = document.getElementById('alert-title');
  const msg     = document.getElementById('alert-message');
  if (overlay && msg) {
    if (title) title.textContent = '⏰ Timer';
    msg.textContent = message;
    overlay.style.display      = 'block';
    overlay.style.background   = 'rgba(245,158,11,0.1)';
    overlay.style.borderColor  = 'rgba(245,158,11,0.35)';
    setTimeout(() => { if (overlay) overlay.style.display = 'none'; }, 4000);
  }
}

// ── Answer saving ──────────────────────────────────────────────────────────────
async function saveAnswer(sid, questionId, selectedAnswer, questionIndex) {
  try {
    const res = await fetch('/exam/answer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sid,
        question_id: questionId,
        selected_answer: selectedAnswer
      })
    });

    if (res.ok) {
      markAnswered(questionIndex);
      updateProgress();
      console.log(`[Exam] Saved Q${questionId} = ${selectedAnswer}`);
    }
  } catch (err) {
    // Silent fail — don't interrupt student
    console.warn('[Exam] Save failed (offline?):', err.message);
  }
}

function markAnswered(index) {
  const card = document.getElementById(`question-${index}`);
  if (card && !card.classList.contains('answered')) {
    card.classList.add('answered');
    answeredCount++;
  }
  const btn = document.getElementById(`qnav-${index}`);
  if (btn) btn.classList.add('answered');
}

function updateProgress() {
  const bar  = document.getElementById('progress-bar');
  const text = document.getElementById('progress-text');
  const pct  = totalQuestions > 0 ? (answeredCount / totalQuestions) * 100 : 0;
  if (bar)  bar.style.width   = `${pct}%`;
  if (text) text.textContent  = `${answeredCount} / ${totalQuestions}`;
}

// ── Browser Monitoring ─────────────────────────────────────────────────────────
function initBrowserMonitoring() {
  // Tab switch or minimization
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && !isSubmitting) {
      reportBrowserEvent('tab_switch', 'Candidate switched to another tab or minimized the window.');
    }
  });

  // Window loses focus
  window.addEventListener('blur', () => {
    if (!isSubmitting) {
      reportBrowserEvent('tab_switch', 'Exam window lost focus.');
    }
  });

  // Fullscreen exit
  document.addEventListener('fullscreenchange', () => {
    if (!document.fullscreenElement && !isSubmitting) {
      reportBrowserEvent('fullscreen_exit', 'Candidate exited fullscreen mode.');
    }
  });
}

async function reportBrowserEvent(eventType, detail) {
  if (!sessionId || isSubmitting) return;
  try {
    const res = await fetch('/monitor/browser-event', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        event_type: eventType,
        detail: detail
      })
    });
    // The response will be processed by the backend.
    // Real-time HUD updates will arrive via the frame_analysis loop or socket broadcasts,
    // but we log here for client-side debugging.
    if (res.ok) {
      console.log(`[Exam] Browser event recorded: ${eventType}`);
    }
  } catch (err) {
    console.warn(`[Exam] Failed to report browser event: ${err.message}`);
  }
}

// ── Navigation ─────────────────────────────────────────────────────────────────
function scrollToQuestion(index) { scrollToQ(index); }
function scrollToQ(index) {
  const card = document.getElementById(`question-${index}`);
  if (card) card.scrollIntoView({ behavior: 'smooth', block: 'start' });

  // Update nav highlight
  document.querySelectorAll('.qnav-btn').forEach((b, i) => {
    b.classList.toggle('current', i === index);
  });
}

// ── Submission ─────────────────────────────────────────────────────────────────
async function submitExam() {
  if (isSubmitting) return;

  const unanswered = totalQuestions - answeredCount;
  if (unanswered > 0 && timeLeft > 30) {
    const ok = window.confirm(
      `You have ${unanswered} unanswered question(s).\n\nSubmit anyway?`
    );
    if (!ok) return;
  }

  isSubmitting = true;
  clearInterval(timerInterval);

  // Stop AI monitoring and webcam
  if (typeof stopMonitoring === 'function') stopMonitoring();
  if (typeof stopWebcam === 'function')     stopWebcam();

  // Update all submit buttons
  ['submit-exam-btn', 'submit-exam-btn-bottom'].forEach(id => {
    const btn = document.getElementById(id);
    if (btn) { btn.textContent = 'Submitting...'; btn.disabled = true; }
  });

  try {
    const res    = await fetch(`/exam/submit/${sessionId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' }
    });
    const result = await res.json();

    if (result.redirect) {
      alert(`Exam submitted!\n\nScore: ${result.score} / ${result.total}\n\nRedirecting to your dashboard...`);
      window.location.href = result.redirect;
    } else if (result.error) {
      throw new Error(result.error);
    }
  } catch (err) {
    console.error('[Exam] Submission error:', err);
    alert('Submission failed. Please contact your administrator.');
    isSubmitting = false;
    ['submit-exam-btn', 'submit-exam-btn-bottom'].forEach(id => {
      const btn = document.getElementById(id);
      if (btn) { btn.textContent = id.includes('bottom') ? 'Submit Exam' : 'Submit'; btn.disabled = false; }
    });
  }
}
