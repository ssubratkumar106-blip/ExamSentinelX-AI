/**
 * monitor.js — Real-time AI Monitoring
 * ======================================
 * Sends webcam frames to /monitor/frame every ~1.5s.
 * Processes analysis_result and updates the AI HUD.
 *
 * FIXED:
 *   1. CSS vars updated to match production design system
 *   2. All status fields always update (face, direction, objects, risk)
 *   3. isAnalyzing lock with timeout watchdog
 *   4. Exponential backoff on consecutive errors
 *   5. handleViolation null-safe for all DOM elements
 *   6. addViolationToList uses correct .event-item class
 */

// ── State ──────────────────────────────────────────────────────────────────────
let monitorSessionId  = null;
let isMonitoring      = false;
let totalViolations   = 0;
let alertTimeout      = null;
let isAnalyzing       = false;
let analyzeStartTime  = 0;
const ANALYZE_TIMEOUT_MS = 20000;
let consecutiveErrors = 0;
let framesSent        = 0;
let lastSuccessTime   = 0;

// Violation display config
const VIOLATION_CONFIG = {
  phone_detected:      { icon: '📱', label: 'Phone Detected',      level: 'danger'  },
  multiple_persons:    { icon: '👥', label: 'Multiple Persons',    level: 'danger'  },
  face_absent:         { icon: '🚫', label: 'Face Not Visible',    level: 'warning' },
  looking_away:        { icon: '👀', label: 'Looking Away',        level: 'warning' },
  suspicious_object:   { icon: '📚', label: 'Suspicious Object',   level: 'warning' },
  suspicious_behavior: { icon: '🤔', label: 'Suspicious Behavior', level: 'warning' },
  talking_to_others:   { icon: '🗣️', label: 'Talking Detected',   level: 'danger'  },
  tab_switch:          { icon: '💻', label: 'Tab Switched',       level: 'warning' },
  fullscreen_exit:     { icon: '🪟', label: 'Exited Fullscreen',  level: 'warning' },
};

// CSS vars in new design system
const COLOR = {
  success: 'var(--success)',
  warning: 'var(--warning)',
  danger:  'var(--danger)',
  muted:   'var(--text-3)',
  info:    'var(--info)',
};

/** Initialize monitoring — called from exam.js */
function initMonitoring(sid) {
  monitorSessionId  = sid;
  isMonitoring      = true;
  consecutiveErrors = 0;
  lastSuccessTime   = Date.now();
  console.log(`[Monitor] ✅ Started session ${monitorSessionId}`);
  setStatusBadge('ACTIVE', 'badge-info');

  // Watchdog: force-reset stuck isAnalyzing flag
  setInterval(() => {
    if (isAnalyzing && (Date.now() - analyzeStartTime) > ANALYZE_TIMEOUT_MS) {
      console.warn('[Monitor] ⚠ Timeout — resetting isAnalyzing lock');
      isAnalyzing = false;
    }
  }, 2000);
}

/** Stop monitoring */
function stopMonitoring() {
  isMonitoring = false;
  isAnalyzing  = false;
  setStatusBadge('STOPPED', 'badge-muted');
  console.log('[Monitor] 🛑 Stopped');
}

/** Update the monitoring status badge */
function setStatusBadge(text, cls) {
  const el = document.getElementById('monitoring-status');
  if (el) { el.textContent = text; el.className = `badge ${cls}`; }
}

/**
 * Called by webcam.js every time a frame is captured.
 * Sends frame to backend /monitor/frame endpoint.
 */
async function onFrameCaptured(frameB64) {
  if (!isMonitoring || !monitorSessionId || isAnalyzing) {
    if (!monitorSessionId) console.warn('[Monitor] sessionId not set yet');
    return;
  }

  // Backoff on repeated errors
  if (consecutiveErrors >= 5) {
    if (Date.now() - lastSuccessTime > 30000) {
      consecutiveErrors = 0;
    } else {
      return; // Still in backoff period
    }
  }

  isAnalyzing      = true;
  analyzeStartTime = Date.now();
  framesSent++;

  const controller   = new AbortController();
  const fetchTimeout = setTimeout(() => controller.abort(), 25000);

  try {
    const response = await fetch('/monitor/frame', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: monitorSessionId, frame: frameB64 }),
      signal: controller.signal
    });
    clearTimeout(fetchTimeout);

    if (!response.ok) {
      if (response.status === 400) {
        try {
          const err = await response.json();
          if (err.session_status === 'completed' || err.session_status === 'ended') {
            console.log('[Monitor] Session ended — stopping.');
            stopMonitoring();
            return;
          }
          // Flagged but keep monitoring
          console.warn('[Monitor] Session flagged — continuing.');
          isAnalyzing = false;
          return;
        } catch { stopMonitoring(); return; }
      }
      if (response.status === 403) { stopMonitoring(); return; }
      throw new Error(`HTTP ${response.status}`);
    }

    const result = await response.json();
    consecutiveErrors = 0;
    lastSuccessTime   = Date.now();

    console.log(`[Monitor] Frame #${framesSent}: faces=${result.face_count} dir=${result.head_direction} viol=${result.has_violation ? result.violation_type : 'none'}`);
    processResult(result);

  } catch (err) {
    clearTimeout(fetchTimeout);
    if (err.name === 'AbortError') {
      console.warn('[Monitor] ⏱ Frame timed out');
    } else if (err.message?.includes('Failed to fetch')) {
      console.warn('[Monitor] 🔌 Server unreachable');
    } else {
      console.error('[Monitor] Error:', err.message);
    }
    consecutiveErrors++;
    setStatusBadge(
      consecutiveErrors >= 5 ? 'ERROR' : 'SYNCING',
      consecutiveErrors >= 5 ? 'badge-danger' : 'badge-warning'
    );
  } finally {
    isAnalyzing = false;
  }
}

/** Process AI result and update ALL HUD elements */
function processResult(result) {
  updateHUD(result);

  if (result.annotated_frame) updateAnnotatedFrame(result.annotated_frame);

  if (result.has_violation && result.violation_type) {
    handleViolation(result);
    setStatusBadge('⚠ ALERT', 'badge-danger');
  } else {
    setStatusBadge('ACTIVE', 'badge-info');
  }

  if (result.session_status === 'flagged') showFlaggedWarning(result.total_violations);
}

/** Update every field in the AI HUD panel */
function updateHUD(result) {
  // — Face count —
  const faceEl = document.getElementById('face-count');
  if (faceEl && result.face_count !== undefined) {
    const n = result.face_count;
    faceEl.textContent = n === 0 ? 'No Face' : `${n} Face${n > 1 ? 's' : ''}`;
    faceEl.style.color = n === 0 ? COLOR.danger : n > 1 ? COLOR.warning : COLOR.success;
  }

  // — Head direction —
  const dirEl = document.getElementById('head-direction');
  if (dirEl && result.head_direction) {
    const d = result.head_direction;
    dirEl.textContent = d.charAt(0).toUpperCase() + d.slice(1);
    dirEl.style.color = d.toLowerCase() === 'forward' ? COLOR.success : COLOR.warning;
  }

  // — Detected objects —
  const objEl = document.getElementById('detected-objects');
  if (objEl) {
    if (result.detected_objects && result.detected_objects.length > 0) {
      objEl.textContent = result.detected_objects.join(', ');
      objEl.style.color = COLOR.warning;
    } else {
      objEl.textContent = 'Clear';
      objEl.style.color = COLOR.success;
    }
  }

  // — Violation count —
  const violEl = document.getElementById('violation-count');
  if (violEl && result.total_violations !== undefined) {
    totalViolations      = result.total_violations;
    violEl.textContent   = totalViolations;
    violEl.style.color   =
      totalViolations > 5 ? COLOR.danger :
      totalViolations > 2 ? COLOR.warning : COLOR.success;
  }

  // — Risk score + bar —
  const riskEl  = document.getElementById('risk-score');
  const riskBar = document.getElementById('risk-bar-fill');
  if (result.risk_score !== undefined) {
    const r = parseFloat(result.risk_score);
    if (riskEl)  riskEl.textContent = `${r.toFixed(1)} / 10`;
    if (riskBar) {
      riskBar.style.width      = `${(r / 10) * 100}%`;
      riskBar.style.background =
        r >= 7 ? COLOR.danger :
        r >= 4 ? COLOR.warning : COLOR.success;
    }
  }

  // — Trust score —
  const trustEl = document.getElementById('trust-score');
  if (trustEl && result.trust_score !== undefined) {
    const ts = parseFloat(result.trust_score);
    trustEl.textContent = `${Math.round(ts)}%`;
    trustEl.style.color =
      ts <= 50 ? COLOR.danger :
      ts <= 80 ? COLOR.warning : COLOR.success;
  }
}

/** Show alert overlay and log to event list */
function handleViolation(result) {
  const cfg = VIOLATION_CONFIG[result.violation_type] || {
    icon: '⚠️', label: result.violation_type, level: 'warning'
  };

  // Alert overlay
  const overlay = document.getElementById('alert-overlay');
  const msgEl   = document.getElementById('alert-message');
  if (overlay && msgEl) {
    const titleEl = document.getElementById('alert-title');
    if (titleEl) titleEl.textContent = `${cfg.icon} ${cfg.label}`;
    msgEl.innerHTML = `<strong>${cfg.icon} ${cfg.label}</strong><br>` +
                      (result.alert_message || 'Anomaly detected by AI engine.');

    const isDanger = cfg.level === 'danger';
    overlay.style.cssText = `
      display:block;
      background: ${isDanger ? 'rgba(239,68,68,0.08)' : 'rgba(245,158,11,0.08)'};
      border-color: ${isDanger ? 'rgba(239,68,68,0.35)' : 'rgba(245,158,11,0.35)'};
    `;
    if (alertTimeout) clearTimeout(alertTimeout);
    alertTimeout = setTimeout(() => { if (overlay) overlay.style.display = 'none'; }, 5000);
  }

  // Add to event log
  addEventToLog(cfg, result);

  // Alert beep
  try {
    const ctx  = new (window.AudioContext || window.webkitAudioContext)();
    const osc  = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain); gain.connect(ctx.destination);
    osc.frequency.value = cfg.level === 'danger' ? 1100 : 700;
    osc.type = 'sine';
    gain.gain.setValueAtTime(0.1, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
    osc.start(); osc.stop(ctx.currentTime + 0.5);
  } catch { /* silent fail */ }
}

/** Add violation entry to the events list */
function addEventToLog(cfg, result) {
  const list = document.getElementById('violations-list');
  if (!list) return;

  // Remove placeholder
  list.querySelector('[data-placeholder]')?.remove();

  const now  = new Date().toLocaleTimeString('en-US', { hour12: false });
  const conf = result.confidence ? Math.round(result.confidence * 100) : '?';

  const item = document.createElement('div');
  item.className = 'event-item';
  item.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;">
      <span style="font-weight:600;font-size:0.8rem;">${cfg.icon} ${cfg.label}</span>
      <span style="color:var(--text-3);font-size:0.72rem;white-space:nowrap;">${now}</span>
    </div>
    <div style="color:var(--text-3);margin-top:3px;font-size:0.72rem;">
      Confidence: <span style="color:var(--text-1);font-weight:600;">${conf}%</span>
    </div>
  `;
  list.insertBefore(item, list.firstChild);

  // Cap at 15 items
  const items = list.querySelectorAll('.event-item');
  if (items.length > 15) items[items.length - 1].remove();
}

/** Show warning when session is flagged */
function showFlaggedWarning(count) {
  const overlay = document.getElementById('alert-overlay');
  const msgEl   = document.getElementById('alert-message');
  if (overlay && msgEl) {
    overlay.style.display     = 'block';
    overlay.style.background  = 'rgba(239,68,68,0.15)';
    overlay.style.borderColor = 'rgba(239,68,68,0.5)';
    msgEl.innerHTML = `<strong>Session Flagged</strong><br>
      ${count} violation(s) recorded. This session is under review.`;
  }
}

// Request notification permission
document.addEventListener('DOMContentLoaded', () => {
  if ('Notification' in window && Notification.permission === 'default') {
    Notification.requestPermission();
  }
});
