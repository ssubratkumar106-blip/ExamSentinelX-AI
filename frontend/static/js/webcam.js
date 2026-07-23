/**
 * webcam.js — Webcam Access and Frame Capture
 * ============================================
 * FIXES:
 *   1. Frame capture works regardless of scroll position
 *   2. Canvas-based capture doesn't rely on element visibility
 *   3. Stream auto-restarts if it drops (browser tab focus change)
 *   4. Intersection observer replaced with direct video state check
 *   5. Capture interval never stops unless explicitly called
 */

// ── State ──────────────────────────────────────────────────────────────────────
let webcamStream      = null;
let captureInterval   = null;
let lastFrameTime     = 0;
let frameCount        = 0;
let streamRestartCount = 0;

const CAPTURE_INTERVAL_MS = 1500; // Every 1.5s — allows CNN time to process on CPU

/**
 * Start the webcam stream.
 * Auto-called on page load, or when user clicks "Enable Camera".
 */
async function startWebcam() {
  const video   = document.getElementById('webcam-feed');
  const overlay = document.getElementById('camera-overlay');

  if (!video) return; // Not on exam page

  // If stream already active, just make sure capture is running
  if (webcamStream && webcamStream.active) {
    if (!captureInterval) startFrameCapture();
    if (overlay) overlay.style.display = 'none';
    return;
  }

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: 'user',
        width:  { ideal: 640 },
        height: { ideal: 480 },
        frameRate: { ideal: 15 }
      },
      audio: true
    });

    video.srcObject = webcamStream;

    await new Promise((resolve, reject) => {
      video.onloadedmetadata = resolve;
      video.onerror = reject;
      setTimeout(reject, 5000); // 5s timeout
    });

    await video.play();
    if (overlay) overlay.style.display = 'none';

    startFrameCapture();
    startAudioDetection();
    console.log('[Webcam] Started successfully');

    // Watch for stream ending (e.g., camera unplugged)
    webcamStream.getVideoTracks()[0].addEventListener('ended', () => {
      console.warn('[Webcam] Stream track ended — attempting restart');
      webcamStream = null;
      setTimeout(startWebcam, 2000);
    });

  } catch (error) {
    console.error('[Webcam] Error:', error);
    handleCameraError(error);
  }
}

/** Stop the webcam and all capture. */
function stopWebcam() {
  if (captureInterval) {
    clearInterval(captureInterval);
    captureInterval = null;
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
  }
  const video = document.getElementById('webcam-feed');
  if (video) video.srcObject = null;
  console.log('[Webcam] Stopped');
}

/**
 * Start periodic frame capture.
 * Capture happens from the video element directly — not affected by scroll.
 * The canvas is hidden but always captures the correct frame.
 */
function startFrameCapture() {
  if (captureInterval) clearInterval(captureInterval);

  captureInterval = setInterval(() => {
    // Always capture — scroll position doesn't matter
    const frame = captureFrame();
    if (frame && typeof onFrameCaptured === 'function') {
      onFrameCaptured(frame);
    }

    // FPS display
    const now = Date.now();
    if (lastFrameTime > 0) {
      const fps = Math.round(1000 / (now - lastFrameTime));
      const el = document.getElementById('fps-display');
      if (el) el.textContent = `${fps} fps`;
    }
    lastFrameTime = now;
    frameCount++;
  }, CAPTURE_INTERVAL_MS);

  console.log(`[Webcam] Capture started (every ${CAPTURE_INTERVAL_MS}ms)`);
}

/**
 * Capture current video frame to base64 JPEG.
 * Works even when video element is scrolled out of view —
 * canvas drawImage pulls from the video stream, not the rendered DOM.
 */
function captureFrame() {
  const video  = document.getElementById('webcam-feed');
  const canvas = document.getElementById('capture-canvas');

  if (!video || !canvas) return null;
  if (!video.srcObject || video.readyState < 2) return null;
  if (video.videoWidth === 0 || video.videoHeight === 0) return null;

  // Set canvas to video dimensions (only once)
  if (canvas.width !== video.videoWidth) {
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
  }

  const ctx = canvas.getContext('2d');
  // Draw video frame — this works regardless of scroll position
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Return as JPEG base64 (0.75 quality = good balance)
  return canvas.toDataURL('image/jpeg', 0.75);
}

/** Toggle between raw and AI-annotated feed. */
let showAnnotated = false;
function toggleAnnotatedView() {
  const raw       = document.getElementById('webcam-feed');
  const annotated = document.getElementById('annotated-feed');
  const btn       = document.getElementById('toggle-view-btn');
  showAnnotated   = !showAnnotated;

  if (showAnnotated && annotated && annotated.src) {
    raw.style.display = 'none';
    annotated.style.display = 'block';
    if (btn) btn.textContent = 'Raw view';
  } else {
    if (raw) raw.style.display = 'block';
    if (annotated) annotated.style.display = 'none';
    if (btn) btn.textContent = 'Toggle AI view';
    showAnnotated = false;
  }
}

/** Update annotated frame from AI result. */
function updateAnnotatedFrame(frameB64) {
  const el = document.getElementById('annotated-feed');
  if (el && frameB64) el.src = frameB64;
}

/** Handle camera errors with clear messages. */
function handleCameraError(error) {
  const overlay = document.getElementById('camera-overlay');
  let msg = 'Camera access failed.';

  if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
    msg = 'Camera permission denied. Allow camera access in your browser and refresh.';
  } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
    msg = 'No camera found. Please connect a webcam.';
  } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
    msg = 'Camera is being used by another app. Close it and retry.';
  } else if (error.name === 'OverconstrainedError') {
    msg = 'Camera constraints not met. Try refreshing.';
  } else if (error.message?.includes('timeout')) {
    msg = 'Camera took too long to respond. Check connections.';
  }

  if (overlay) {
    overlay.innerHTML = `
      <div style="font-size:1.8rem;margin-bottom:8px;">⚠️</div>
      <p style="font-size:0.78rem;color:var(--danger);text-align:center;padding:0 8px;margin:0 0 10px;line-height:1.5;">${msg}</p>
      <button onclick="startWebcam()" class="btn btn-outline btn-sm">Retry Camera</button>
    `;
    overlay.style.display = 'flex';
  }
}

// Auto-start webcam when page loads (with small delay for DOM readiness)
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('webcam-feed')) {
    setTimeout(startWebcam, 600);
  }
});

// Restart capture if page becomes visible after being hidden (tab switch)
document.addEventListener('visibilitychange', () => {
  if (!document.hidden && webcamStream && webcamStream.active) {
    if (!captureInterval) {
      console.log('[Webcam] Tab refocused — restarting capture');
      startFrameCapture();
    }
  }
});

// --- Audio & Speech Detection ---
let speechRecognizer = null;
let audioContext = null;
let analyser = null;
let microphone = null;
let audioDetectionInterval = null;

async function startAudioDetection() {
  try {
    if (!webcamStream) return;
    
    // 1. Microphone Volume Detection
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    microphone = audioContext.createMediaStreamSource(webcamStream);
    microphone.connect(analyser);
    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    audioDetectionInterval = setInterval(() => {
      analyser.getByteFrequencyData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
      }
      let average = sum / bufferLength;
      if (average > 35) { // Threshold for sound
        if (typeof monitorSessionId !== 'undefined' && monitorSessionId) {
          fetch('/monitor/browser-event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: monitorSessionId, event_type: 'talking_to_others', detail: 'Microphone picked up sound/talking' })
          }).catch(()=>{});
        }
      }
    }, 2000);

    // 2. Speech Recognition (Web Speech API)
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      speechRecognizer = new SpeechRecognition();
      speechRecognizer.continuous = true;
      speechRecognizer.interimResults = false;
      
      speechRecognizer.onresult = (event) => {
        const transcript = event.results[event.results.length - 1][0].transcript;
        if (transcript.trim().length > 0 && typeof monitorSessionId !== 'undefined' && monitorSessionId) {
          fetch('/monitor/browser-event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: monitorSessionId, event_type: 'talking_to_others', detail: `Speech detected: "${transcript}"` })
          }).catch(()=>{});
        }
      };
      
      speechRecognizer.onerror = (e) => console.warn('SpeechRecognition error:', e);
      speechRecognizer.onend = () => {
        if (webcamStream && webcamStream.active) {
          try { speechRecognizer.start(); } catch(e){}
        }
      };
      
      speechRecognizer.start();
      console.log('[Audio] Speech Recognition started');
    }
  } catch (err) {
    console.warn('[Audio] Audio detection failed:', err);
  }
}

