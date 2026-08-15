(() => {
  'use strict';

  const micBtn = document.getElementById('micBtn');
  const micStatus = document.getElementById('micStatus');
  const waveform = document.getElementById('waveform');
  const pipelineSteps = [...document.querySelectorAll('.pipeline__step')];
  const guardBanner = document.getElementById('guardBanner');
  const resultSection = document.getElementById('result');
  const transcriptText = document.getElementById('transcriptText');
  const answerText = document.getElementById('answerText');
  const confidenceFill = document.getElementById('confidenceFill');
  const confidenceValue = document.getElementById('confidenceValue');
  const sourcesToggle = document.getElementById('sourcesToggle');
  const sourcesList = document.getElementById('sourcesList');
  const sourcesCount = document.getElementById('sourcesCount');
  const latencyRow = document.getElementById('latencyRow');
  const errorBox = document.getElementById('errorBox');
  const connDot = document.getElementById('connDot');

  const STEP_IDS = ['stt', 'guard-in', 'retrieval', 'generation', 'guard-out'];
  // Rough relative weight of each stage, used only to pace the optimistic
  // progress animation while we wait for the single /api/ask response.
  const STEP_WEIGHTS = { stt: 0.35, 'guard-in': 0.05, retrieval: 0.1, generation: 0.4, 'guard-out': 0.1 };

  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;
  let isBusy = false;
  let progressTimer = null;
  let sttProvider = '';

  function setPipelineState(stateByStep) {
    pipelineSteps.forEach((el) => {
      const step = el.dataset.step;
      el.classList.remove('is-active', 'is-done', 'is-blocked');
      const state = stateByStep[step];
      if (state) el.classList.add(state);
    });
  }

  function resetPipeline() {
    setPipelineState({});
  }

  function runOptimisticProgress() {
    let elapsed = 0;
    const totalGuessMs = 6000;
    resetPipeline();
    let currentIdx = 0;
    setPipelineState({ [STEP_IDS[0]]: 'is-active' });

    progressTimer = setInterval(() => {
      elapsed += 150;
      const targetFraction = elapsed / totalGuessMs;
      let acc = 0;
      let idx = 0;
      for (let i = 0; i < STEP_IDS.length; i++) {
        acc += STEP_WEIGHTS[STEP_IDS[i]];
        if (targetFraction < acc) { idx = i; break; }
        idx = i;
      }
      if (idx !== currentIdx) {
        const done = {};
        for (let i = 0; i < idx; i++) done[STEP_IDS[i]] = 'is-done';
        done[STEP_IDS[idx]] = 'is-active';
        setPipelineState(done);
        currentIdx = idx;
      }
    }, 150);
  }

  function stopOptimisticProgress() {
    if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
  }

  function finalizePipeline(data) {
    stopOptimisticProgress();
    const done = {};
    STEP_IDS.forEach((s) => { done[s] = 'is-done'; });

    if (data.error) {
      // Best-effort: we don't know exactly which stage failed from the
      // shape alone beyond what's in the error string, so mark stt+guard
      // done (we got a transcript) and flag generation as the failure point
      // when transcript is present, else STT itself.
      if (data.error.startsWith('stt_failed')) {
        done['stt'] = 'is-blocked';
        done['guard-in'] = done['retrieval'] = done['generation'] = done['guard-out'] = undefined;
      } else if (data.error.startsWith('generation_failed')) {
        done['generation'] = 'is-blocked';
        done['guard-out'] = undefined;
      }
    } else if (data.guard_flags?.input_unsafe || data.guard_flags?.off_topic) {
      done['guard-in'] = 'is-blocked';
      done['retrieval'] = done['generation'] = done['guard-out'] = undefined;
    } else if (data.guard_flags?.output_grounded === false) {
      done['guard-out'] = 'is-blocked';
    }
    setPipelineState(done);
  }

  function setMicUI() {
    micBtn.classList.toggle('is-recording', isRecording);
    micBtn.classList.toggle('is-busy', isBusy);
    micBtn.setAttribute('aria-pressed', String(isRecording));
    waveform.classList.toggle('is-live', isRecording);
    if (isBusy) micStatus.textContent = 'Thinking…';
    else if (isRecording) micStatus.textContent = 'Listening — tap to stop';
    else micStatus.textContent = 'Tap to ask a question';
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks = [];
      const mimeType = MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : '';
      mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
      mediaRecorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        sendAudio(blob);
      };
      mediaRecorder.start();
      isRecording = true;
      setMicUI();
    } catch (err) {
      showError('Microphone access failed: ' + (err?.message || err));
    }
  }

  function stopRecording() {
    if (mediaRecorder && isRecording) {
      isRecording = false;
      mediaRecorder.stop();
      setMicUI();
    }
  }

  function showError(message) {
    errorBox.hidden = false;
    errorBox.textContent = message;
  }

  function hideError() {
    errorBox.hidden = true;
    errorBox.textContent = '';
  }

  function clearResults() {
    resultSection.hidden = true;
    guardBanner.hidden = true;
    guardBanner.classList.remove('is-danger');
    sourcesList.hidden = true;
    sourcesToggle.setAttribute('aria-expanded', 'false');
  }

  async function sendAudio(blob) {
    isBusy = true;
    setMicUI();
    hideError();
    clearResults();
    runOptimisticProgress();

    const form = new FormData();
    form.append('audio', blob, 'question.webm');

    try {
      const resp = await fetch('/api/ask', { method: 'POST', body: form });
      const data = await resp.json();
      connDot.style.background = 'var(--accent)';
      finalizePipeline(data);
      renderResult(data);
    } catch (err) {
      stopOptimisticProgress();
      resetPipeline();
      showError('Request failed: ' + (err?.message || err));
    } finally {
      isBusy = false;
      setMicUI();
    }
  }

  function renderResult(data) {
    if (data.error) {
      showError(data.error);
      return;
    }

    resultSection.hidden = false;

    if (data.guard_flags?.off_topic) {
      guardBanner.hidden = false;
      guardBanner.textContent = `Out of scope — query similarity ${fmtScore(data.guard_flags.off_topic_score)} was below threshold`;
    } else if (data.guard_flags?.input_unsafe) {
      guardBanner.hidden = false;
      guardBanner.classList.add('is-danger');
      guardBanner.textContent = 'Flagged by the input guardrail — not answered';
    } else if (data.guard_flags?.output_grounded === false) {
      guardBanner.hidden = false;
      guardBanner.classList.add('is-danger');
      guardBanner.textContent = `Low grounding confidence (${fmtScore(data.guard_flags.output_grounding_score)}) — answer may be incomplete`;
    }

    transcriptText.textContent = data.transcript ? `"${data.transcript}"` : '—';
    answerText.textContent = data.answer || '—';

    const conf = typeof data.confidence === 'number' ? data.confidence : 0;
    confidenceFill.style.width = `${Math.round(conf * 100)}%`;
    confidenceValue.textContent = `confidence ${fmtScore(conf)}`;

    const sources = data.sources || [];
    sourcesCount.textContent = String(sources.length);
    sourcesList.innerHTML = '';
    sources.forEach((s) => {
      const li = document.createElement('li');
      li.className = 'source-item';
      li.innerHTML = `
        <div class="source-item__meta">
          <span class="tag">${escapeHtml(s.strategy)}</span>
          <span>${escapeHtml(s.chunk_id)}</span>
          <span>score ${fmtScore(s.score)}</span>
        </div>
        <p class="source-item__text">${escapeHtml(s.text)}</p>
      `;
      sourcesList.appendChild(li);
    });

    const lat = data.stage_latency_ms || data.latency_breakdown || {};
    let total = data.total_latency_ms || lat.total_ms;
    const sttVal = lat.stt ?? lat.stt_ms;
    if (sttProvider && sttProvider.toLowerCase().includes('elevenlabs') && typeof sttVal === 'number') {
      total = Math.max(0, total - sttVal);
    }
    const cells = [
      ['STT', sttVal],
      ['Retrieval', lat.retrieval ?? lat.retrieval_ms],
      ['Generation', lat.generation ?? lat.generation_ms],
      ['Total', total],
    ];
    latencyRow.innerHTML = cells.map(([label, ms]) => `
      <div class="latency-cell">
        <span class="latency-cell__value ${label === 'Retrieval' && ms != null && ms < 200 ? 'is-fast' : ''}">${ms != null ? Math.round(ms) : '—'}</span>
        <span class="latency-cell__label">${label}</span>
      </div>
    `).join('');
  }

  function fmtScore(n) {
    if (typeof n !== 'number' || Number.isNaN(n)) return '—';
    return n.toFixed(2);
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str ?? '';
    return div.innerHTML;
  }

  sourcesToggle.addEventListener('click', () => {
    const expanded = sourcesToggle.getAttribute('aria-expanded') === 'true';
    sourcesToggle.setAttribute('aria-expanded', String(!expanded));
    sourcesList.hidden = expanded;
  });

  micBtn.addEventListener('click', () => {
    if (isBusy) return;
    if (isRecording) stopRecording();
    else startRecording();
  });

  // Quick reachability check so the status dot reflects backend health.
  fetch('/api/health').then((r) => {
    connDot.style.background = r.ok ? 'var(--accent)' : 'var(--danger)';
  }).catch(() => { connDot.style.background = 'var(--danger)'; });

  fetch('/api/config').then(r => r.json()).then(config => {
    sttProvider = config.stt_provider || '';
  }).catch(() => {});
})();
