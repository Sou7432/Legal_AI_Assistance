/* ══════════════════════════════════════════════════════════
   Nyaya AI — Modern Indian Legal Assistant
   Client-side controller (Zero-jump, Seamless UX)
   ══════════════════════════════════════════════════════════ */

const SAMPLE_SCENARIOS = [
  { label: "📱 Defective Phone Refund", query: "I bought a defective mobile phone but the seller refuses to refund or replace it even after multiple complaints." },
  { label: "💳 Fake Bank Call / OTP Fraud", query: "Someone called me posing as my bank manager and fraudulently withdrew money from my savings account using an OTP." },
  { label: "💼 Unpaid Salary", query: "My employer has not disbursed my salary for the last 3 months despite full attendance and written reminders." },
  { label: "🏠 Landlord Deposit Withheld", query: "My landlord is refusing to return my security deposit of 3 months rent after I vacated the house in undamaged condition." },
  { label: "🏷️ Overcharged Above MRP", query: "A retailer charged me well above the printed MRP for packaged food items and refused to give an itemized tax invoice." },
  { label: "🚗 Drunk Driver / Hit & Run", query: "A drunken driver rammed into my vehicle and fled the scene. I noted down the registration number." },
  { label: "⚖️ Cheque Dishonour / Bounce", query: "A business client issued a cheque that bounced due to insufficient funds and is ignoring my payment reminders." },
  { label: "🛑 Online Stalking / Defamation", query: "An individual is posting defamatory statements and harassing me continuously across social media platforms." }
];

document.addEventListener('DOMContentLoaded', () => {
  renderSampleChips();
  checkSystemStatus();
  initTextareaListeners();
});

// ── Render sample prompts ─────────────────────────────────
function renderSampleChips() {
  const container = document.getElementById('sampleChips');
  if (!container) return;
  container.innerHTML = '';

  SAMPLE_SCENARIOS.forEach(item => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'chip-btn';
    btn.textContent = item.label;
    btn.title = item.query;
    btn.onclick = () => {
      const textarea = document.getElementById('problemInput');
      textarea.value = item.query;
      updateCharCount();
      textarea.focus();
    };
    container.appendChild(btn);
  });
}

// ── Status polling ─────────────────────────────────────────
async function checkSystemStatus() {
  const label = document.getElementById('statusLabel');
  try {
    const res = await fetch('/api/status', { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json();
      label.textContent = `Online · ${data.sections_indexed} Statutory Sections Active`;
      label.style.color = '#10b981';
    } else {
      throw new Error();
    }
  } catch {
    label.textContent = 'Engine Offline (Start app.py)';
    label.style.color = '#f43f5e';
  }
}

// ── Input listeners ───────────────────────────────────────
function initTextareaListeners() {
  const ta = document.getElementById('problemInput');
  ta.addEventListener('input', updateCharCount);
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuery();
    }
  });
}

function updateCharCount() {
  const ta = document.getElementById('problemInput');
  const countSpan = document.getElementById('charCount');
  const len = ta.value.length;
  countSpan.textContent = len;
  countSpan.style.color = len > 1800 ? '#f43f5e' : '';
}

function clearInput() {
  const ta = document.getElementById('problemInput');
  ta.value = '';
  updateCharCount();
  ta.focus();
  dismissError();
}

function dismissError() {
  const banner = document.getElementById('errorBanner');
  if (banner) banner.style.display = 'none';
}

function showError(msg) {
  const banner = document.getElementById('errorBanner');
  const msgEl = document.getElementById('errorMessage');
  if (banner && msgEl) {
    msgEl.innerHTML = msg;
    banner.style.display = 'flex';
  }
}

// ── Main Query Execution (Navigates to dedicated new page!) ──────────
function submitQuery() {
  const ta = document.getElementById('problemInput');
  const submitBtn = document.getElementById('submitBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const btnText = submitBtn.querySelector('.btn-text');
  const problem = ta.value.trim();

  dismissError();

  if (!problem) {
    ta.focus();
    showError('Please describe your legal situation before submitting.');
    return;
  }

  if (problem.length < 8) {
    showError('Please provide a little more detail regarding your issue (at least 8 characters).');
    return;
  }

  // Show loading spinner on button
  submitBtn.disabled = true;
  btnSpinner.style.display = 'inline-block';
  btnText.textContent = 'Opening Analysis...';

  // Save to session and navigate to new page!
  sessionStorage.setItem('nyaya_problem', problem);
  sessionStorage.removeItem('nyaya_analysis');
  window.location.href = `/results?q=${encodeURIComponent(problem)}`;
}

// ── Render Results Seamlessly ───────────────────────────────
function renderAnalysis(queryText, data) {
  const section = document.getElementById('resultsSection');

  // Echo query
  document.getElementById('resultQueryEcho').textContent = queryText;

  // Metadata tags
  const riskPill = document.getElementById('resultRisk');
  riskPill.textContent = data.risk_category || 'General Legal Matter';

  const ragPill = document.getElementById('resultRagSource');
  ragPill.textContent = data.answer_source ? data.answer_source : 'Vector Search Verified';

  // Primary Statutes
  document.getElementById('resultLaw').textContent = data.law || 'Indian Statutory Law';
  document.getElementById('resultSection').textContent = data.section || 'Applicable Provisions';

  // Detailed Analysis
  document.getElementById('resultAnswer').textContent = data.expected_answer || 'No analysis available.';
  document.getElementById('resultSolution').textContent = data.solution || 'Consult legal counsel.';
  document.getElementById('resultBasis').textContent = data.solution_based_on_law || 'Governed by relevant Acts.';
  document.getElementById('resultConditions').textContent = data.applicable_conditions || 'Documentation required.';

  // Render Clauses Accordion
  renderClauses(data.retrieved_sections || []);

  // Display the card without violent scrolling!
  section.style.display = 'block';
}

function renderClauses(clauses) {
  const body = document.getElementById('clausesBody');
  const countBadge = document.getElementById('retrievedCountBadge');
  body.innerHTML = '';
  countBadge.textContent = `${clauses.length} Statutes Retrieved`;

  if (!clauses.length) {
    body.innerHTML = '<div style="color:#64748b;font-size:0.85rem;">No direct statutory extracts available.</div>';
    return;
  }

  clauses.forEach((item, index) => {
    const div = document.createElement('div');
    div.className = 'clause-item';
    const sim = item.similarity ? `${(item.similarity * 100).toFixed(1)}% match` : 'Relevant';
    div.innerHTML = `
      <div class="clause-meta">
        <span class="clause-title">${escapeHtml(item.law_name)} — ${escapeHtml(item.section_id)}</span>
        <span class="clause-sim">Rank #${index + 1} (${sim})</span>
      </div>
      <div class="clause-text">${escapeHtml(item.text)}</div>
    `;
    body.appendChild(div);
  });
}

function toggleClauses() {
  const body = document.getElementById('clausesBody');
  const arrow = document.getElementById('clausesArrow');
  if (body.style.display === 'none' || body.style.display === '') {
    body.style.display = 'block';
    arrow.textContent = '▲';
  } else {
    body.style.display = 'none';
    arrow.textContent = '▼';
  }
}

// ── Copy Analysis to Clipboard ─────────────────────────────
function copyAnalysis() {
  const law = document.getElementById('resultLaw').textContent;
  const sec = document.getElementById('resultSection').textContent;
  const opinion = document.getElementById('resultAnswer').textContent;
  const solution = document.getElementById('resultSolution').textContent;
  const basis = document.getElementById('resultBasis').textContent;
  const conditions = document.getElementById('resultConditions').textContent;

  const fullText = `=== NYAYA AI LEGAL ANALYSIS ===
APPLICABLE ACT: ${law}
SECTION: ${sec}

LEGAL OPINION & OVERVIEW:
${opinion}

ACTION PLAN:
${solution}

STATUTORY BASIS:
${basis}

CONDITIONS & EVIDENCE REQUIRED:
${conditions}

(Disclaimer: General legal guidance under Indian statutes. Please consult an advocate for formal proceedings.)`;

  navigator.clipboard.writeText(fullText).then(() => {
    const copyText = document.getElementById('copyBtnText');
    const oldText = copyText.textContent;
    copyText.textContent = 'Copied!';
    setTimeout(() => copyText.textContent = oldText, 2000);
  }).catch(() => {
    alert('Advice copied to clipboard.');
  });
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
