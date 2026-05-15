/* GLDIS — Frontend Application Logic v2 */
'use strict';

const API = '';  // same-origin

// ── Utilities ───────────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json', ...opts.headers },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || err.error || res.statusText);
  }
  return res.json();
}

function setStatus(el, msg, type = 'info') {
  el.textContent = msg;
  el.className = `status-box ${type}`;
  el.classList.remove('hidden');
}

function badgeHTML(status) {
  const map = { ready:'badge-ready', processing:'badge-processing', failed:'badge-failed', uploaded:'badge-uploaded' };
  return `<span class="badge ${map[status] || 'badge-uploaded'}">${status}</span>`;
}

function timeAgo(isoStr) {
  if (!isoStr) return '';
  const s = Math.floor((Date.now() - new Date(isoStr)) / 1000);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s/60)}m ago`;
  return `${Math.floor(s/3600)}h ago`;
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}-content`).classList.add('active');
    if (btn.dataset.tab === 'improvements') loadImprovements();
    if (btn.dataset.tab === 'feedback')     loadFeedbackHistory();
    if (btn.dataset.tab === 'draft')        loadDocumentsForDraft();
  });
});

// ── Status check ────────────────────────────────────────────────────────────
async function checkStatus() {
  try {
    const data = await api('/api/status');
    const dot = document.querySelector('.status-dot');
    dot.className = 'status-dot ok';
    const vlmNote = data.vlm_enabled ? '· VLM ✓' : '';
    document.getElementById('statusText').textContent =
      `${data.indexed_vectors} vectors ${vlmNote} · ${data.openai_configured ? 'OpenAI ✓' : 'Mock mode'}`;
  } catch {
    document.querySelector('.status-dot').className = 'status-dot err';
    document.getElementById('statusText').textContent = 'Server unreachable';
  }
}
checkStatus();

// ── Pipeline stage helper ────────────────────────────────────────────────────
const STAGES = ['stage-upload', 'stage-ocr', 'stage-chunk', 'stage-index', 'stage-done'];
function setStage(idx) {
  STAGES.forEach((id, i) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('active', 'done');
    if (i < idx)  el.classList.add('done');
    if (i === idx) el.classList.add('active');
  });
}
function resetStages() { STAGES.forEach(id => { const el = document.getElementById(id); if (el) el.classList.remove('active','done'); }); }

// ── TAB 1: Upload & Process ──────────────────────────────────────────────────
let selectedFile = null;
const fileInput    = document.getElementById('fileInput');
const dropzone     = document.getElementById('dropzone');
const btnUpload    = document.getElementById('btnUpload');
const uploadStatus = document.getElementById('uploadStatus');

function onFileSelect(file) {
  selectedFile = file;
  document.querySelector('.drop-text').innerHTML =
    `📄 <strong>${file.name}</strong> (${(file.size/1024).toFixed(1)} KB)`;
  btnUpload.disabled = false;
  resetStages();
}

fileInput.addEventListener('change', () => fileInput.files[0] && onFileSelect(fileInput.files[0]));
dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('keydown', e => e.key === 'Enter' && fileInput.click());
dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
dropzone.addEventListener('drop', e => {
  e.preventDefault(); dropzone.classList.remove('drag-over');
  const f = e.dataTransfer.files[0]; if (f) onFileSelect(f);
});

btnUpload.addEventListener('click', async () => {
  if (!selectedFile) return;
  btnUpload.disabled = true;
  resetStages();
  setStage(0);
  setStatus(uploadStatus, '⏳ Uploading...', 'loading');

  try {
    const fd = new FormData();
    fd.append('file', selectedFile);
    const up = await fetch('/api/documents/upload', { method:'POST', body:fd });
    if (!up.ok) throw new Error((await up.json()).detail || 'Upload failed');
    const data = await up.json();

    setStage(1);
    setStatus(uploadStatus, `✓ Uploaded (${data.document_id.slice(0,8)}…). Processing pipeline…`, 'info');

    const proc = await api(`/api/documents/${data.document_id}/process/sync`, { method:'POST' });

    setStage(2);
    await sleep(200);
    setStage(3);
    await sleep(200);
    setStage(4);

    const engines = proc.ocr_engines_used?.join(', ') || '—';
    setStatus(uploadStatus,
      `✓ Done! ${proc.chunks_created} chunks · ${proc.entities_extracted} entities · engines: ${engines}`,
      'success');

    loadDocuments();
    checkStatus();
  } catch(e) {
    setStatus(uploadStatus, `✗ ${e.message}`, 'error');
    resetStages();
  } finally {
    btnUpload.disabled = false;
  }
});

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function loadDocuments() {
  const list = document.getElementById('docList');
  try {
    const docs = await api('/api/documents');
    if (!docs.length) { list.innerHTML = '<p class="muted">No documents yet.</p>'; return; }
    list.innerHTML = docs.map(d => `
      <div class="doc-item" data-id="${d.document_id}" onclick="selectDoc('${d.document_id}')">
        <div>
          <div class="doc-name">${d.filename}</div>
          <div class="doc-meta">${d.page_count||0} pages · ${timeAgo(d.upload_time)}</div>
        </div>
        ${badgeHTML(d.status)}
      </div>`).join('');
  } catch(e) {
    list.innerHTML = `<p class="muted">Error: ${e.message}</p>`;
  }
}

function selectDoc(id) {
  const sel = document.getElementById('draftDocSelect');
  if (sel) { sel.value = id; }
}

document.getElementById('btnRefreshDocs').addEventListener('click', loadDocuments);
loadDocuments();

// ── TAB 2: Generate Draft ────────────────────────────────────────────────────
let activeDraftId = null;
let activeEvidence = [];

async function loadDocumentsForDraft() {
  const sel = document.getElementById('draftDocSelect');
  try {
    const docs = await api('/api/documents');
    const ready = docs.filter(d => d.status === 'ready');
    sel.innerHTML = '<option value="">— select a processed document —</option>' +
      ready.map(d => `<option value="${d.document_id}">${d.filename}</option>`).join('');
  } catch {}
}

document.getElementById('btnGenerate').addEventListener('click', async () => {
  const docId  = document.getElementById('draftDocSelect').value;
  const query  = document.getElementById('draftQuery').value.trim() || undefined;
  const topK   = parseInt(document.getElementById('draftTopK').value) || 5;
  const btn    = document.getElementById('btnGenerate');
  const status = document.getElementById('draftStatus');
  const output = document.getElementById('draftOutput');

  if (!docId) { setStatus(status, '⚠ Select a document first.', 'error'); return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Generating…';
  setStatus(status, '⏳ Retrieving evidence and generating grounded draft…', 'loading');
  output.innerHTML = '<p class="muted">Working…</p>';

  try {
    const draft = await api('/api/drafts/generate', {
      method: 'POST',
      body: JSON.stringify({ document_id:docId, query, top_k:topK }),
    });

    activeDraftId = draft.draft_id;
    activeEvidence = draft.evidence_chunks || [];

    // Render draft with sentence-level highlighting
    renderDraftWithGrounding(output, draft.generated_text, activeEvidence, draft.citations || []);

    // Badge + feedback pre-fill
    const draftIdEl = document.getElementById('draftId');
    draftIdEl.textContent = draft.draft_id.slice(0,8) + '…';
    draftIdEl.classList.remove('hidden');
    document.getElementById('fbDraftId').value  = draft.draft_id;
    document.getElementById('fbOriginal').value = draft.generated_text;

    // Grounding bar
    const pct = Math.round((draft.grounding_score || 0) * 100);
    document.getElementById('groundingBar').style.width = pct + '%';
    document.getElementById('groundingLabel').textContent =
      `${pct}% sentences grounded · ${draft.citations?.length || 0} inline citations`;
    document.getElementById('groundingPanel').classList.remove('hidden');
    document.getElementById('groundingLegend').classList.remove('hidden');

    // Evidence panel
    renderEvidence(activeEvidence);

    // Action buttons
    const loadBtn = document.getElementById('btnLoadToFeedback');
    const copyBtn = document.getElementById('btnCopyDraft');
    loadBtn.style.display = 'inline-flex';
    copyBtn.style.display = 'inline-flex';

    status.classList.add('hidden');
  } catch(e) {
    setStatus(status, `✗ ${e.message}`, 'error');
    output.innerHTML = '<p class="muted">Generation failed.</p>';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Draft';
  }
});

// ── Sentence-level grounding rendering ──────────────────────────────────────
function renderDraftWithGrounding(container, text, evidenceChunks, citations) {
  const citedChunkIds = new Set(citations.map(c => c.chunk_id));
  const evidenceTexts = evidenceChunks.map(c => c.text.toLowerCase());

  // Split into sentences (preserve newlines)
  const lines = text.split('\n');
  let html = '';
  lines.forEach(line => {
    if (!line.trim()) { html += '\n'; return; }
    // Basic sentence splitter
    const sentences = line.match(/[^.!?]+[.!?]?/g) || [line];
    sentences.forEach(raw => {
      const s = raw.trim();
      if (!s) return;
      const cssClass = classifyGrounding(s, evidenceTexts, citations);
      html += `<span class="sentence ${cssClass}" title="${cssClass.replace('-sentence','').replace(/-/g,' ')}">${escHtml(s)} </span>`;
    });
    html += '\n';
  });
  container.innerHTML = html;

  // Click handler: highlight corresponding evidence
  container.querySelectorAll('.sentence').forEach(el => {
    el.addEventListener('click', () => onSentenceClick(el, evidenceChunks, citations));
  });
}

function classifyGrounding(sentence, evidenceTexts, citations) {
  const s = sentence.toLowerCase();
  // Check if any evidence chunk contains substantial overlap
  const THRESHOLD = 0.25;
  for (const ev of evidenceTexts) {
    if (jaccardSim(s, ev) >= THRESHOLD) return 'grounded-sentence';
  }
  // Check for [Source: ...] citation tag
  if (/\[source:/i.test(sentence)) return 'grounded-sentence';
  // Very short lines, headings, etc.
  if (sentence.length < 20) return '';
  // Check hedge words
  if (/\b(may|possibly|likely|appears|seems|unclear|not found|unavailable)\b/i.test(sentence))
    return 'uncertain-sentence';
  return 'ungrounded-sentence';
}

function jaccardSim(a, b) {
  const setA = new Set(a.split(/\s+/).filter(w => w.length > 3));
  const setB = new Set(b.split(/\s+/).filter(w => w.length > 3));
  if (!setA.size || !setB.size) return 0;
  let inter = 0;
  setA.forEach(w => { if (setB.has(w)) inter++; });
  return inter / (setA.size + setB.size - inter);
}

function onSentenceClick(el, evidenceChunks, citations) {
  // Remove previous active
  document.querySelectorAll('.sentence.active-sentence').forEach(e => e.classList.remove('active-sentence'));
  el.classList.add('active-sentence');

  // Find best matching evidence chunk
  const s = el.textContent.toLowerCase();
  let bestIdx = -1, bestSim = 0;
  evidenceChunks.forEach((chunk, i) => {
    const sim = jaccardSim(s, chunk.text.toLowerCase());
    if (sim > bestSim) { bestSim = sim; bestIdx = i; }
  });

  if (bestIdx >= 0) {
    const evEls = document.querySelectorAll('.evidence-chunk');
    evEls.forEach(e => e.classList.remove('highlighted'));
    if (evEls[bestIdx]) {
      evEls[bestIdx].classList.add('highlighted');
      evEls[bestIdx].scrollIntoView({ behavior:'smooth', block:'nearest' });
    }
  }
}

function renderEvidence(chunks) {
  if (!chunks.length) return;
  const panel = document.getElementById('evidencePanel');
  const list  = document.getElementById('evidenceList');
  list.innerHTML = chunks.map((c, i) => `
    <div class="evidence-chunk" id="ev-chunk-${i}">
      <div class="evidence-meta">
        #${i+1} · chunk: ${c.chunk_id.slice(0,8)}… · page ${c.page || '?'} · score ${c.score.toFixed(3)}
        ${c.section ? `· §${c.section.slice(0,40)}` : ''}
      </div>
      <div class="evidence-text">${escHtml(c.text.slice(0, 350))}${c.text.length > 350 ? '…' : ''}</div>
    </div>`).join('');
  panel.classList.remove('hidden');
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Load into feedback tab
document.getElementById('btnLoadToFeedback').addEventListener('click', () => {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-feedback').classList.add('active');
  document.getElementById('tab-feedback-content').classList.add('active');
  loadFeedbackHistory();
});

// Copy draft
document.getElementById('btnCopyDraft').addEventListener('click', async () => {
  const text = document.getElementById('fbOriginal').value;
  try { await navigator.clipboard.writeText(text); } catch {}
});

// ── TAB 3: Feedback ──────────────────────────────────────────────────────────

// Live diff preview
function computeDiff(original, edited) {
  const origWords = original.split(/\s+/);
  const editWords = edited.split(/\s+/);
  // Simple LCS-based diff
  let html = '';
  let i = 0, j = 0;
  while (i < origWords.length || j < editWords.length) {
    if (i < origWords.length && j < editWords.length && origWords[i] === editWords[j]) {
      html += escHtml(origWords[i]) + ' '; i++; j++;
    } else if (j < editWords.length && (i >= origWords.length || origWords[i] !== editWords[j])) {
      html += `<span class="diff-ins">+${escHtml(editWords[j])}</span> `; j++;
    } else {
      html += `<span class="diff-del">-${escHtml(origWords[i])}</span> `; i++;
    }
  }
  return html;
}

let diffTimeout;
document.getElementById('fbEdited').addEventListener('input', () => {
  clearTimeout(diffTimeout);
  diffTimeout = setTimeout(() => {
    const orig   = document.getElementById('fbOriginal').value.trim();
    const edited = document.getElementById('fbEdited').value.trim();
    const panel  = document.getElementById('diffPreview');
    const content= document.getElementById('diffPreviewContent');
    if (!orig || !edited) { panel.classList.add('hidden'); return; }
    content.innerHTML = computeDiff(orig, edited);
    panel.classList.remove('hidden');
  }, 300);
});

document.getElementById('btnSubmitFeedback').addEventListener('click', async () => {
  const draftId  = document.getElementById('fbDraftId').value.trim();
  const original = document.getElementById('fbOriginal').value.trim();
  const edited   = document.getElementById('fbEdited').value.trim();
  const fbType   = document.getElementById('fbType').value;
  const comment  = document.getElementById('fbComment').value.trim() || undefined;
  const status   = document.getElementById('fbStatus');

  if (!draftId || !original || !edited) {
    setStatus(status, '⚠ Fill in Draft ID, original draft, and edited draft.', 'error');
    return;
  }

  setStatus(status, '⏳ Submitting feedback…', 'loading');
  try {
    const rec = await api('/api/feedback', {
      method: 'POST',
      body: JSON.stringify({
        draft_id: draftId,
        original_draft: original,
        edited_draft: edited,
        feedback_type: fbType,
        reviewer_comment: comment,
      }),
    });
    setStatus(status,
      `✓ Feedback recorded (${rec.feedback_id.slice(0,8)}…). System learned from this edit.`,
      'success');
    document.getElementById('fbEdited').value = '';
    document.getElementById('diffPreview').classList.add('hidden');
    loadFeedbackHistory();
  } catch(e) {
    setStatus(status, `✗ ${e.message}`, 'error');
  }
});

async function loadFeedbackHistory() {
  const el = document.getElementById('feedbackHistory');
  try {
    const items = await api('/api/feedback/history?limit=10');
    if (!items.length) { el.innerHTML = '<p class="muted">No feedback submitted yet.</p>'; return; }
    el.innerHTML = items.map(f => `
      <div class="feedback-item">
        <div class="feedback-item-header">
          <span class="feedback-type">${f.feedback_type}</span>
          <span class="feedback-time">${timeAgo(f.created_at)}</span>
        </div>
        ${f.reviewer_comment ? `<p style="font-size:12px;color:#a0b0c0;margin-bottom:6px">${f.reviewer_comment}</p>` : ''}
        <pre class="feedback-diff">${escHtml((f.diff_summary||'').slice(0,400))}</pre>
      </div>`).join('');
  } catch(e) {
    el.innerHTML = `<p class="muted">Error: ${e.message}</p>`;
  }
}

document.getElementById('btnRefreshFeedback').addEventListener('click', loadFeedbackHistory);

// ── TAB 4: Improvements ─────────────────────────────────────────────────────
async function loadImprovements() {
  const statsEl   = document.getElementById('improvementStats');
  const rulesEl   = document.getElementById('styleRules');
  const trendEl   = document.getElementById('trendChart');
  const fewShotEl = document.getElementById('fewShotList');

  try {
    const [data, examples] = await Promise.all([
      api('/api/feedback/improvements'),
      api('/api/feedback/examples?limit=5').catch(() => []),
    ]);

    const s = data.statistics;
    statsEl.innerHTML = `
      <div class="stat-item"><div class="stat-value">${s.total_edits}</div><div class="stat-label">Operator Edits</div></div>
      <div class="stat-item"><div class="stat-value">${s.few_shot_examples}</div><div class="stat-label">Few-Shot Examples</div></div>
      <div class="stat-item"><div class="stat-value">${data.active_style_rules?.length||0}</div><div class="stat-label">Style Rules</div></div>
      <div class="stat-item"><div class="stat-value">${((s.avg_example_quality||0)*100).toFixed(0)}%</div><div class="stat-label">Avg Quality</div></div>
    `;

    rulesEl.innerHTML = data.active_style_rules?.length
      ? data.active_style_rules.map(r => `<div class="rule-item">• ${escHtml(r)}</div>`).join('')
      : '<p class="muted">No rules learned yet. Submit feedback to start learning.</p>';

    // Trend chart
    const trend = data.improvement_trend || [];
    if (trend.length) {
      const maxSim = Math.max(...trend.map(t => t.similarity||0), 0.01);
      trendEl.innerHTML = trend.slice(-20).map((t, i) => {
        const h = Math.round(((t.similarity||0)/maxSim)*58);
        const pct = ((t.similarity||0)*100).toFixed(0);
        return `<div class="trend-bar-wrap">
          <div class="trend-bar-outer">
            <div class="trend-bar-inner" style="height:${h}px" title="Edit ${i+1}: ${pct}% similarity"></div>
          </div>
          <div class="trend-bar-label">${i+1}</div>
        </div>`;
      }).join('');
    } else {
      trendEl.innerHTML = '<p class="muted-sm">Submit at least 2 feedback edits to see the trend.</p>';
    }

    // Few-shot examples
    fewShotEl.innerHTML = examples.length
      ? examples.map(ex => `
          <div class="few-shot-item">
            <div class="few-shot-meta">
              <span>Type: ${ex.feedback_type}</span>
              <span>Quality: ${((ex.quality_score||0)*100).toFixed(0)}%</span>
              <span>Used: ${ex.use_count||0}×</span>
            </div>
            <div class="few-shot-preview">${escHtml((ex.corrected_draft_preview||'').slice(0,200))}…</div>
          </div>`).join('')
      : '<p class="muted">No examples yet. Submit high-quality feedback to build the pool.</p>';

  } catch(e) {
    statsEl.innerHTML = `<p class="muted">Error: ${e.message}</p>`;
  }
}

document.getElementById('btnRefreshImprovements').addEventListener('click', loadImprovements);
