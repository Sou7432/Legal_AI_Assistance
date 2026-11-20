/* ══════════════════════════════════════════════════════════
   Nyaya AI — Results Page Controller
   ══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  displayReportDate();
  loadAndRenderReport();
});

function displayReportDate() {
  const dateEl = document.getElementById('reportDate');
  if (dateEl) {
    const now = new Date();
    dateEl.textContent = `Generated on ${now.toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })}`;
  }
}

async function loadAndRenderReport() {
  const loadingCard = document.getElementById('loadingCard');
  const errorCard = document.getElementById('errorCard');
  const reportContent = document.getElementById('reportContent');

  // Check URL param first, then sessionStorage
  const urlParams = new URLSearchParams(window.location.search);
  let problem = urlParams.get('q');

  if (!problem) {
    problem = sessionStorage.getItem('nyaya_problem');
  }

  if (!problem || !problem.trim()) {
    loadingCard.style.display = 'none';
    showError("No legal question or situation provided. Please return to the consultation page.");
    return;
  }

  problem = problem.trim();

  // Check if we already have the pre-fetched analysis in sessionStorage for instant rendering
  const cachedAnalysisStr = sessionStorage.getItem('nyaya_analysis');
  const cachedProblem = sessionStorage.getItem('nyaya_problem');

  if (cachedAnalysisStr && cachedProblem === problem) {
    try {
      const data = JSON.parse(cachedAnalysisStr);
      renderData(problem, data);
      return;
    } catch (e) {
      // Fallback to fetch
    }
  }

  // Fetch analysis from server
  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ problem }),
      signal: AbortSignal.timeout(30000)
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Server returned error (${res.status})`);
    }

    const data = await res.json();
    // Cache it
    sessionStorage.setItem('nyaya_analysis', JSON.stringify(data));
    sessionStorage.setItem('nyaya_problem', problem);

    renderData(problem, data);

  } catch (err) {
    loadingCard.style.display = 'none';
    showError(err.message || 'Failed to complete legal analysis.');
  }
}

function renderData(queryText, data) {
  const loadingCard = document.getElementById('loadingCard');
  const reportContent = document.getElementById('reportContent');

  loadingCard.style.display = 'none';

  // Echo query
  document.getElementById('resultQueryEcho').textContent = queryText;

  // Metadata tags
  document.getElementById('resultRisk').textContent = data.risk_category || 'General Matter';
  document.getElementById('resultRagSource').textContent = data.answer_source || 'Vector Verified';

  // Primary Statutes
  document.getElementById('resultLaw').textContent = data.law || 'Indian Statutory Law';
  document.getElementById('resultSection').textContent = data.section || 'Applicable Provisions';

  // Detailed Analysis
  document.getElementById('resultAnswer').textContent = data.expected_answer || 'No analysis available.';
  document.getElementById('resultSolution').textContent = data.solution || 'Consult legal counsel.';
  document.getElementById('resultBasis').textContent = data.solution_based_on_law || 'Governed by relevant Acts.';
  document.getElementById('resultConditions').textContent = data.applicable_conditions || 'Documentation required.';

  // Render Clauses
  renderClauses(data.retrieved_sections || []);

  reportContent.style.display = 'block';
}

function renderClauses(clauses) {
  const body = document.getElementById('clausesBody');
  const countBadge = document.getElementById('retrievedCountBadge');
  body.innerHTML = '';
  countBadge.textContent = `${clauses.length} Statutes Retrieved`;

  if (!clauses.length) {
    body.innerHTML = '<div style="color:#64748b;font-size:0.85rem;">No statutory extracts available.</div>';
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

function copyAnalysis() {
  const law = document.getElementById('resultLaw').textContent;
  const sec = document.getElementById('resultSection').textContent;
  const opinion = document.getElementById('resultAnswer').textContent;
  const solution = document.getElementById('resultSolution').textContent;
  const basis = document.getElementById('resultBasis').textContent;
  const conditions = document.getElementById('resultConditions').textContent;

  const fullText = `=== NYAYA AI LEGAL ANALYSIS REPORT ===
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

function showError(msg) {
  const errCard = document.getElementById('errorCard');
  const msgEl = document.getElementById('errorMessage');
  if (errCard && msgEl) {
    msgEl.textContent = msg;
    errCard.style.display = 'flex';
  }
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
