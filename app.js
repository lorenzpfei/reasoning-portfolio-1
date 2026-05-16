'use strict';

// ===== Constants =====
const GOAL           = 100;
const THETA          = 1e-9;
const GAMMA          = 1.0;
const MAX_SWEEPS     = 200000;
const SNAPSHOT_SWEEPS = [1, 2, 3, 10, 32];

const SWEEP_COLORS = {
    1:     { color: '#3b82f6', width: 1.5, label: 'Sweep 1' },
    2:     { color: '#f59e0b', width: 1.5, label: 'Sweep 2' },
    3:     { color: '#10b981', width: 1.5, label: 'Sweep 3' },
    10:    { color: '#8b5cf6', width: 1.5, label: 'Sweep 10' },
    32:    { color: '#ec4899', width: 1.5, label: 'Sweep 32' },
    final: { color: '#1e293b', width: 2.5, label: 'Final' },
};

const AN_SNAPSHOT_COLORS = ['#f59e0b', '#10b981', '#8b5cf6', '#ec4899'];

let currentZoom = 100;
const ZOOM_MIN  = 50;
const ZOOM_MAX  = 250;
const ZOOM_STEP = 10;

// ===== Zoom =====
function applyZoom() {
    document.body.style.zoom = currentZoom + '%';
    document.getElementById('zoom-level').textContent = currentZoom + '%';
}
function zoomIn()  { if (currentZoom < ZOOM_MAX) { currentZoom = Math.min(currentZoom + ZOOM_STEP, ZOOM_MAX); applyZoom(); } }
function zoomOut() { if (currentZoom > ZOOM_MIN) { currentZoom = Math.max(currentZoom - ZOOM_STEP, ZOOM_MIN); applyZoom(); } }

// ===== Value Iteration =====
function runValueIteration(ph, targetReward, stepReward, opts) {
    const useStrict = opts && opts.useStrictEquality;
    const cap       = (opts && opts.maxSweeps) ? opts.maxSweeps : MAX_SWEEPS;
    const gamma     = (opts && opts.gamma != null) ? opts.gamma : GAMMA;

    const V           = new Float64Array(GOAL + 1);
    const history     = [];
    const roundLog    = [];
    const snapshotSet = new Set(SNAPSHOT_SWEEPS);

    let sweeps = 0, converged = false;

    while (sweeps < cap) {
        let delta = 0;

        for (let s = 1; s < GOAL; s++) {
            const oldV = V[s];
            let best   = -Infinity;
            const maxA = Math.min(s, GOAL - s);

            for (let a = 1; a <= maxA; a++) {
                const rWin = (s + a === GOAL) ? targetReward : stepReward;
                const ev   = ph * (rWin + gamma * V[s + a]) + (1.0 - ph) * (stepReward + gamma * V[s - a]);
                if (ev > best) best = ev;
            }

            V[s] = best;
            const d = useStrict ? (V[s] !== oldV ? 1 : 0) : Math.abs(V[s] - oldV);
            if (d > delta) delta = d;
        }

        sweeps++;
        converged = useStrict ? (delta === 0) : (delta < THETA);
        roundLog.push({ sweep: sweeps, delta, v25: V[25], v50: V[50], v75: V[75], v99: V[99] });

        if (converged) {
            if (!history.some(h => h.sweep === sweeps))
                history.push({ sweep: sweeps, V: V.slice(), isFinal: true });
            break;
        }
        if (snapshotSet.has(sweeps))
            history.push({ sweep: sweeps, V: V.slice(), isFinal: false });
    }

    return { V, sweeps, converged, history, roundLog };
}

// ===== Policy Extraction =====
function extractPolicy(V, ph, targetReward, stepReward, convention, gamma) {
    const naiveLargest = convention === 'naive-largest';
    gamma = (gamma != null) ? gamma : GAMMA;
    const policy = new Int32Array(GOAL + 1);

    for (let s = 1; s < GOAL; s++) {
        let best = -Infinity, bestA = 1;
        const maxA = Math.min(s, GOAL - s);

        for (let a = 1; a <= maxA; a++) {
            const rWin = (s + a === GOAL) ? targetReward : stepReward;
            const ev   = ph * (rWin + gamma * V[s + a]) + (1.0 - ph) * (stepReward + gamma * V[s - a]);
            if (naiveLargest) {
                if (ev >= best - 1e-12) { best = ev; bestA = a; }
            } else {
                if (ev > best + 1e-9) { best = ev; bestA = a; }
            }
        }
        policy[s] = bestA;
    }
    return policy;
}

// ===== Gambler's Ruin Win Probability =====
function flatBetWinProb(betSize, s, goal, ph) {
    if (betSize <= 0) return 0;
    const rho = (1 - ph) / ph;
    const sN  = s    / betSize;
    const gN  = goal / betSize;
    if (Math.abs(rho - 1) < 1e-12) return sN / gN;
    const num = Math.pow(rho, sN) - 1;
    const den = Math.pow(rho, gN) - 1;
    if (!isFinite(den) || den === 0) return 0;
    return Math.max(0, Math.min(1, num / den));
}

function formatProbPct(p) {
    const pct = p * 100;
    if (pct >= 1)     return pct.toFixed(2) + '%';
    if (pct >= 0.001) return pct.toFixed(5) + '%';
    if (pct >= 1e-9)  return pct.toFixed(11) + '%';
    return pct.toExponential(3) + '%';
}

// Returns { abstract, filled } HTML strings for the formula expandable
function makeBellmanHtml(betSize, s, goal, ph) {
    const prob  = flatBetWinProb(betSize, s, goal, ph);
    const rho   = (1 - ph) / ph;
    const sWin  = s + betSize;
    const sLose = s - betSize;

    if (sWin === goal && sLose === 0) {
        return {
            abstract: `\\(Q(${s},\\, a=${betSize}) = p_h \\cdot V(${sWin}) + (1-p_h) \\cdot V(${sLose})\\)`,
            filled:   `\\(= ${ph} \\cdot 1 + ${(1-ph).toFixed(1)} \\cdot 0\\) = ${formatProbPct(prob)}`,
        };
    }

    const sN = +(s    / betSize).toFixed(2);
    const gN = +(goal / betSize).toFixed(2);
    return {
        abstract: `$$V(s) = \\frac{\\rho^{${sN}} - 1}{\\rho^{${gN}} - 1}, \\quad \\rho = \\frac{1-p_h}{p_h}$$`,
        filled:   `\\(\\rho = ${rho.toFixed(3)},\\quad \\frac{${rho.toFixed(3)}^{${sN}}-1}{${rho.toFixed(3)}^{${gN}}-1}\\) = ${formatProbPct(prob)}`,
    };
}

// ===== KaTeX Helper =====
function renderKaTeX(el) {
    if (!window.renderMathInElement) return;
    renderMathInElement(el || document.body, {
        delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '\\(', right: '\\)', display: false },
        ],
        ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
        throwOnError: false,
    });
}

// ===== Stats Cards =====
function updateStats(sweeps, V, policy) {
    document.getElementById('stat-sweeps').textContent = sweeps;
    document.getElementById('stat-v50').textContent    = V[50].toFixed(3);
    document.getElementById('stat-p50').textContent    = '$' + policy[50];
    document.getElementById('stat-p25').textContent    = '$' + policy[25];
}

// ===== Value Iteration Log =====
function updateLog(roundLog) {
    const logBody = document.getElementById('log-body');
    const showIdx = new Set();
    for (let i = 0; i < Math.min(5, roundLog.length); i++) showIdx.add(i);
    for (let i = 0; i < roundLog.length; i += 5) showIdx.add(i);
    showIdx.add(roundLog.length - 1);

    const sortedIdx = Array.from(showIdx).sort((a, b) => a - b);
    let html = `<table class="log-table"><thead><tr>
        <th>Sweep</th><th>Delta</th><th>V($25)</th><th>V($50)</th><th>V($75)</th><th>V($99)</th>
    </tr></thead><tbody>`;

    let prevShown = -1;
    for (const idx of sortedIdx) {
        if (prevShown !== -1 && idx - prevShown > 1)
            html += `<tr class="log-separator"><td colspan="6">&middot;&middot;&middot;</td></tr>`;

        const r = roundLog[idx], isFinal = (idx === roundLog.length - 1);
        let rowClass = '';
        if (isFinal)            rowClass = 'log-row-final';
        else if (r.sweep <= 3)  rowClass = 'log-row-early';
        else if (r.sweep <= 10) rowClass = 'log-row-mid';
        else if (r.sweep <= 32) rowClass = 'log-row-late';

        html += `<tr class="${rowClass}">
            <td>${isFinal ? `${r.sweep}<span class="log-conv-tag">conv</span>` : r.sweep}</td>
            <td>${r.delta.toExponential(3)}</td>
            <td>${r.v25.toFixed(4)}</td><td>${r.v50.toFixed(4)}</td>
            <td>${r.v75.toFixed(4)}</td><td>${r.v99.toFixed(4)}</td>
        </tr>`;
        prevShown = idx;
    }
    html += `</tbody></table>`;
    logBody.innerHTML = html;
    logBody.scrollTop = logBody.scrollHeight;
}

// ===========================
// ===== ANALYSIS TAB ========
// ===========================
let valueChart         = null;
let analysisPolicyChart = null;
let analysisGhost       = null;   // { V, label }
let analysisSnapshots   = [];     // up to 4 saved runs: { V, label }
let analysisCurrentHistory = null;

const ANALYSIS_LABELS = Array.from({ length: GOAL - 1 }, (_, i) => i + 1);

function initAnalysisCharts() {
    // Value function chart
    const vCtx = document.getElementById('value-chart').getContext('2d');
    valueChart = new Chart(vCtx, {
        type: 'line',
        data: { labels: ANALYSIS_LABELS, datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
            scales: {
                x: { title: { display: true, text: 'Capital (s)', font: { size: 12, weight: 'bold' } }, ticks: { maxTicksLimit: 20 } },
                y: { title: { display: true, text: 'Value V(s)', font: { size: 12, weight: 'bold' } } }
            }
        }
    });

    // Policy chart
    const pCtx = document.getElementById('an-policy-chart').getContext('2d');
    analysisPolicyChart = new Chart(pCtx, {
        type: 'bar',
        data: {
            labels: ANALYSIS_LABELS,
            datasets: [{
                label: 'Stake π(s)',
                data: new Array(GOAL - 1).fill(0),
                backgroundColor: 'rgba(16, 185, 129, 0.70)',
                borderColor: 'rgb(5, 150, 105)',
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 0.85,
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
            scales: {
                x: { title: { display: true, text: 'Capital (s)', font: { size: 12, weight: 'bold' } }, ticks: { maxTicksLimit: 20 } },
                y: { title: { display: true, text: 'Stake (a)', font: { size: 12, weight: 'bold' } }, min: 0, ticks: { precision: 0 } }
            }
        }
    });
}

function buildValueDatasets() {
    const datasets = [];

    if (analysisGhost) {
        datasets.push({
            label: analysisGhost.label,
            data: Array.from({ length: GOAL - 1 }, (_, i) => analysisGhost.V[i + 1]),
            borderColor: 'rgba(148, 163, 184, 0.40)',
            borderWidth: 1.5,
            borderDash: [4, 3],
            pointRadius: 0,
            fill: false,
            tension: 0,
            order: 100,
        });
    }

    analysisSnapshots.forEach((snap, i) => {
        datasets.push({
            label: snap.label,
            data: Array.from({ length: GOAL - 1 }, (_, j) => snap.V[j + 1]),
            borderColor: AN_SNAPSHOT_COLORS[i % AN_SNAPSHOT_COLORS.length],
            borderWidth: 1.5,
            borderDash: [6, 2],
            pointRadius: 0,
            fill: false,
            tension: 0,
            order: 50 - i,
        });
    });

    if (analysisCurrentHistory) {
        for (const snap of analysisCurrentHistory) {
            const key = snap.isFinal ? 'final' : snap.sweep;
            const cfg = SWEEP_COLORS[key] || SWEEP_COLORS.final;
            datasets.push({
                label: snap.isFinal ? `Final (sweep ${snap.sweep})` : (cfg.label || `Sweep ${snap.sweep}`),
                data: Array.from({ length: GOAL - 1 }, (_, i) => snap.V[i + 1]),
                borderColor: cfg.color,
                borderWidth: cfg.width,
                pointRadius: 0,
                fill: false,
                tension: 0,
                order: 1,
            });
        }
    }

    return datasets;
}

function updateValueChart() {
    valueChart.data.datasets = buildValueDatasets();
    valueChart.update();
    buildValueLegend();
}

function buildValueLegend() {
    const el = document.getElementById('value-legend');
    el.innerHTML = '';

    const entries = [];
    if (analysisGhost)
        entries.push({ label: analysisGhost.label, color: 'rgba(148,163,184,0.6)', dashed: true });
    analysisSnapshots.forEach((s, i) =>
        entries.push({ label: s.label, color: AN_SNAPSHOT_COLORS[i % 4], dashed: true }));
    if (analysisCurrentHistory) {
        for (const snap of analysisCurrentHistory) {
            const key = snap.isFinal ? 'final' : snap.sweep;
            const cfg = SWEEP_COLORS[key] || SWEEP_COLORS.final;
            entries.push({ label: snap.isFinal ? `Final (sweep ${snap.sweep})` : (cfg.label || `Sweep ${snap.sweep}`), color: cfg.color, width: cfg.width, dashed: false });
        }
    }

    for (const e of entries) {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const sw = document.createElement('span');
        sw.className = 'legend-swatch';
        if (e.dashed) {
            sw.style.background = 'transparent';
            sw.style.borderTop  = `2px dashed ${e.color}`;
            sw.style.height     = '0';
            sw.style.display    = 'inline-block';
        } else {
            sw.style.background = e.color;
            sw.style.height     = (e.width >= 2 ? '3px' : '2px');
        }
        const lbl = document.createElement('span');
        lbl.textContent = e.label;
        item.appendChild(sw);
        item.appendChild(lbl);
        el.appendChild(item);
    }
}

function makeSnapshotLabel(ph, stepReward) {
    return `Snapshot — ph=${ph.toFixed(2)}, step=${stepReward.toFixed(1)}`;
}

// ===== Main Analysis Run =====
function calculate() {
    const phRaw = parseFloat(document.getElementById('ph-input').value);
    const ph    = isNaN(phRaw) ? 0.4 : Math.max(0, Math.min(1, phRaw));

    const trRaw        = parseFloat(document.getElementById('target-reward').value);
    const targetReward = isNaN(trRaw) ? 1.0 : trRaw;

    const srRaw      = parseFloat(document.getElementById('step-reward').value);
    const stepReward = isNaN(srRaw) ? 0.0 : srRaw;

    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Running…';

    const t0 = performance.now();
    const { V, sweeps, converged, history, roundLog } = runValueIteration(ph, targetReward, stepReward);
    const policy = extractPolicy(V, ph, targetReward, stepReward, 'epsilon-smallest');
    const ms     = (performance.now() - t0).toFixed(1);

    analysisCurrentHistory = history;
    calculate._lastResult  = { V: V.slice(), ph, targetReward, stepReward };

    updateValueChart();

    // Update policy chart
    analysisPolicyChart.data.datasets[0].data = Array.from({ length: GOAL - 1 }, (_, i) => policy[i + 1]);
    analysisPolicyChart.update();

    updateStats(sweeps, V, policy);
    updateLog(roundLog);

    statusEl.textContent = converged
        ? `Converged in ${sweeps} sweep${sweeps !== 1 ? 's' : ''} — ${ms} ms — ph = ${ph.toFixed(2)}`
        : `Cap reached after ${sweeps} sweeps — ${ms} ms`;
}

function saveSnapshot() {
    const res = calculate._lastResult;
    if (!res) return;
    if (analysisSnapshots.length >= 4) analysisSnapshots.shift();
    analysisSnapshots.push({ V: res.V, label: makeSnapshotLabel(res.ph, res.stepReward) });
    updateValueChart();
}

function initAnalysisGhost() {
    const { V } = runValueIteration(0.4, 1.0, 0.0);
    analysisGhost = { V: V.slice(), label: 'Default ghost (ph=0.40)' };
}

// ===========================
// ===== GUESSING TAB ========
// ===========================
let revealChart  = null;
let revealShown  = false;

function getGuessPh() {
    const raw = parseFloat(document.getElementById('guess-ph-input').value);
    return (isNaN(raw) || raw < 0.01 || raw > 0.99) ? 0.4 : raw;
}

function addGuessRow() {
    const container = document.getElementById('guess-bets-container');
    const row = document.createElement('div');
    row.className = 'bet-row';

    const nameInput        = document.createElement('input');
    nameInput.type         = 'text';
    nameInput.className    = 'bet-name';
    nameInput.placeholder  = 'Name';

    const guessInput       = document.createElement('input');
    guessInput.type        = 'number';
    guessInput.className   = 'bet-guess';
    guessInput.placeholder = 'Stake';
    guessInput.min         = '1';
    guessInput.max         = '49';
    guessInput.step        = '1';

    const removeBtn        = document.createElement('button');
    removeBtn.className    = 'remove-bet-btn';
    removeBtn.type         = 'button';
    removeBtn.textContent  = '−';
    removeBtn.title        = 'Remove';
    removeBtn.addEventListener('click', () => {
        if (container.children.length > 1) row.remove();
    });

    row.appendChild(nameInput);
    row.appendChild(guessInput);
    row.appendChild(removeBtn);
    container.appendChild(row);
    guessInput.focus();
}

function revealGuessing() {
    const ph          = getGuessPh();
    const nameInputs  = document.querySelectorAll('#guess-bets-container .bet-name');
    const guessInputs = document.querySelectorAll('#guess-bets-container .bet-guess');

    // Compute optimal via value iteration
    const { V }      = runValueIteration(ph, 1.0, 0.0);
    const policy     = extractPolicy(V, ph, 1.0, 0.0, 'epsilon-smallest');
    const optimalBet = policy[50];
    const optProb    = V[50];

    const bets = [];
    for (let i = 0; i < guessInputs.length; i++) {
        const v = parseInt(guessInputs[i].value, 10);
        if (!isNaN(v) && v >= 1) {
            const name = nameInputs[i] ? nameInputs[i].value.trim() || `Player ${i + 1}` : `Player ${i + 1}`;
            bets.push({ name, bet: v });
        }
    }

    const entries = bets.map(b => ({ ...b, prob: flatBetWinProb(b.bet, 50, 100, ph) }));

    // Build probability table — column header shows current p_h
    const tableWrap = document.getElementById('reveal-table-wrap');
    let html = `<table class="reveal-table"><thead><tr>
        <th>Name</th><th>Bet</th><th>Win probability \\(p_h = ${ph.toFixed(2)}\\)</th>
    </tr></thead><tbody>`;

    for (const e of entries) {
        const note   = (e.bet === 50) ? 'one flip, direct' : 'flat-bet strategy';
        const rowCls = e.bet === optimalBet ? 'reveal-row reveal-row-optimal' : 'reveal-row';
        html += `<tr class="${rowCls}">
            <td>${e.name}</td>
            <td class="reveal-bet-cell">$${e.bet}</td>
            <td class="reveal-prob-cell">
                <span class="reveal-pct">${formatProbPct(e.prob)}</span>
                <span class="reveal-note">${note}</span>
            </td>
        </tr>`;
    }

    const optNote = (optimalBet === 50) ? 'one flip, direct' : 'value iteration';
    html += `<tr class="reveal-row reveal-row-optimal">
        <td>Optimal <span class="reveal-check">&#10003;</span></td>
        <td class="reveal-bet-cell">$${optimalBet}</td>
        <td class="reveal-prob-cell">
            <span class="reveal-pct">${formatProbPct(optProb)}</span>
            <span class="reveal-note">${optNote}</span>
        </td>
    </tr>`;

    html += `</tbody></table>`;
    tableWrap.innerHTML = html;

    // Horizontal log-scale bar chart
    const allEntries  = [
        ...entries.map(e => ({ ...e, isOptimalBet: e.bet === optimalBet })),
        { name: 'Optimal', bet: optimalBet, prob: optProb, isOptimal: true, isOptimalBet: true },
    ];
    const chartLabels = allEntries.map(e => e.isOptimal ? 'Optimal ★' : `${e.name} ($${e.bet})`);
    const chartData   = allEntries.map(e => Math.max(e.prob, 1e-100));
    const chartBg     = allEntries.map(e =>
        (e.isOptimal || e.isOptimalBet) ? 'rgba(16, 185, 129, 0.75)' : 'rgba(59, 130, 246, 0.65)');

    if (revealChart) { revealChart.destroy(); revealChart = null; }
    const ctx = document.getElementById('reveal-chart').getContext('2d');
    revealChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: chartLabels,
            datasets: [{
                data: chartData,
                backgroundColor: chartBg,
                borderColor: chartBg.map(c => c.replace('0.75','1').replace('0.65','1')),
                borderWidth: 1,
                barPercentage: 0.6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false, animation: { duration: 600, easing: 'easeOutQuart' },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: ctx => ' ' + formatProbPct(ctx.parsed.x) } }
            },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: { display: false },
                    ticks: {
                        callback: val => {
                            if (val <= 0) return '';
                            const lv = Math.log10(val);
                            const exp = Math.round(lv);
                            if (Math.abs(lv - exp) > 0.01) return '';
                            const pct = val * 100;
                            if (pct >= 1)    return pct.toFixed(0) + '%';
                            if (pct >= 0.01) return pct.toFixed(2) + '%';
                            return pct.toExponential(0) + '%';
                        }
                    }
                },
                y: { ticks: { font: { size: 11 } } }
            }
        }
    });

    revealShown = true;
    document.getElementById('reveal-results').style.display = '';
    document.getElementById('reveal-family-callout').style.display = '';
    renderKaTeX(document.getElementById('reveal-results'));
}

// ===========================
// ===== DEV INSIGHTS ========
// ===========================
const di = { broken1Chart: null, fixed1Chart: null, broken2Chart: null, fixed2Chart: null };

function makeValueLineChart(canvasId) {
    const ctx    = document.getElementById(canvasId).getContext('2d');
    const labels = Array.from({ length: GOAL - 1 }, (_, i) => i + 1);
    return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
            scales: {
                x: { title: { display: true, text: 'Capital (s)', font: { size: 11 } }, ticks: { maxTicksLimit: 20 } },
                y: { title: { display: true, text: 'Value V(s)', font: { size: 11 } } }
            }
        }
    });
}

function makePolicyBarChart(canvasId) {
    const ctx    = document.getElementById(canvasId).getContext('2d');
    const labels = Array.from({ length: GOAL - 1 }, (_, i) => i + 1);
    return new Chart(ctx, {
        type: 'bar',
        data: { labels, datasets: [] },
        options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { display: false }, tooltip: { mode: 'index', intersect: false } },
            scales: {
                x: { title: { display: true, text: 'Capital (s)', font: { size: 11 } }, ticks: { maxTicksLimit: 20 } },
                y: { title: { display: true, text: 'Stake (a)', font: { size: 11 } }, min: 0, ticks: { precision: 0 } }
            }
        }
    });
}

function renderSingleValueLine(chart, V, color) {
    chart.data.datasets = [{
        data: Array.from({ length: GOAL - 1 }, (_, i) => V[i + 1]),
        borderColor: color, borderWidth: 2, pointRadius: 0, fill: false, tension: 0,
    }];
    chart.update();
}

function renderSinglePolicyBars(chart, policy, color) {
    chart.data.datasets = [{
        data: Array.from({ length: GOAL - 1 }, (_, i) => policy[i + 1]),
        backgroundColor: color,
        borderColor: color.replace('0.70', '1'),
        borderWidth: 1, barPercentage: 1.0, categoryPercentage: 0.85,
    }];
    chart.update();
}

function diShow(areaId, statusId, text) {
    document.getElementById(areaId).style.display = '';
    document.getElementById(statusId).textContent = text;
}

function runDiBroken1() {
    document.getElementById('di-broken1-btn').disabled = true;
    const { V, sweeps, converged } = runValueIteration(0.4, 1.0, 0.0, { useStrictEquality: true, maxSweeps: 2000 });
    if (!di.broken1Chart) di.broken1Chart = makeValueLineChart('di-broken1-chart');
    renderSingleValueLine(di.broken1Chart, V, '#e11d48');
    diShow('di-broken1-area', 'di-broken1-status',
        converged
            ? `Converged after ${sweeps} sweeps (floating point happened to match exactly).`
            : `Did not converge — hit cap of ${sweeps} sweeps. Values look correct; only the termination is broken.`);
    document.getElementById('di-broken1-btn').disabled = false;
}

function runDiFixed1() {
    document.getElementById('di-fixed1-btn').disabled = true;
    const { V, sweeps, converged } = runValueIteration(0.4, 1.0, 0.0);
    if (!di.fixed1Chart) di.fixed1Chart = makeValueLineChart('di-fixed1-chart');
    renderSingleValueLine(di.fixed1Chart, V, '#10b981');
    diShow('di-fixed1-area', 'di-fixed1-status',
        converged
            ? `Converged after ${sweeps} sweep${sweeps !== 1 ? 's' : ''} using epsilon = 1e-9.`
            : `Did not converge after ${sweeps} sweeps.`);
    document.getElementById('di-fixed1-btn').disabled = false;
}

function runDiBroken2() {
    document.getElementById('di-broken2-btn').disabled = true;
    const { V, sweeps } = runValueIteration(0.4, 1.0, 0.0);
    const policy = extractPolicy(V, 0.4, 1.0, 0.0, 'naive-largest');
    if (!di.broken2Chart) di.broken2Chart = makePolicyBarChart('di-broken2-chart');
    renderSinglePolicyBars(di.broken2Chart, policy, 'rgba(234, 88, 12, 0.70)');
    diShow('di-broken2-area', 'di-broken2-status',
        `Policy after ${sweeps} sweeps — naive argmax (largest bet wins on ties). No characteristic peaks.`);
    document.getElementById('di-broken2-btn').disabled = false;
}

function runDiFixed2() {
    document.getElementById('di-fixed2-btn').disabled = true;
    const { V, sweeps } = runValueIteration(0.4, 1.0, 0.0);
    const policy = extractPolicy(V, 0.4, 1.0, 0.0, 'epsilon-smallest');
    if (!di.fixed2Chart) di.fixed2Chart = makePolicyBarChart('di-fixed2-chart');
    renderSinglePolicyBars(di.fixed2Chart, policy, 'rgba(16, 185, 129, 0.70)');
    diShow('di-fixed2-area', 'di-fixed2-status',
        `Policy after ${sweeps} sweeps — smallest bet on tie. Characteristic peaks at $25, $50, $75 (matches Figure 4.3).`);
    document.getElementById('di-fixed2-btn').disabled = false;
}

// ===========================
// ===== GAMMA EXPERIMENT ====
// ===========================
let diGammaValueChart  = null;
let diGammaPolicyChart = null;

function runGammaExperiment() {
    const gammaVal = parseFloat(document.getElementById('di-gamma-input').value);
    const gamma    = isNaN(gammaVal) ? 1.0 : Math.max(0.1, Math.min(1.0, gammaVal));

    const { V, sweeps } = runValueIteration(0.4, 1.0, 0.0, { gamma });
    const policy        = extractPolicy(V, 0.4, 1.0, 0.0, 'epsilon-smallest', gamma);

    if (!diGammaValueChart)  diGammaValueChart  = makeValueLineChart('di-gamma-value-chart');
    if (!diGammaPolicyChart) diGammaPolicyChart = makePolicyBarChart('di-gamma-policy-chart');

    renderSingleValueLine(diGammaValueChart,   V,      '#3b82f6');
    renderSinglePolicyBars(diGammaPolicyChart, policy, 'rgba(59, 130, 246, 0.70)');

    let summaryHtml;
    if (gamma >= 0.99) {
        summaryHtml = `With \\(\\gamma \\approx 1\\), the agent treats all future wins equally. The policy matches the book&rsquo;s Figure 4.3.`;
    } else if (gamma >= 0.85) {
        summaryHtml = `With \\(\\gamma = ${gamma}\\), future wins are slightly discounted. The agent starts preferring to finish faster &ndash; bets get a little larger.`;
    } else {
        summaryHtml = `With \\(\\gamma = ${gamma}\\), a win many flips from now is worth much less than winning soon. The agent becomes aggressive: it would rather take a big risk now than wait through many uncertain rounds.`;
    }

    const summaryEl = document.getElementById('di-gamma-summary');
    summaryEl.innerHTML = summaryHtml;

    const resultsEl = document.getElementById('di-gamma-results');
    resultsEl.style.display = '';
    renderKaTeX(resultsEl);
}

// ===== Mode Switching =====
function switchMode(mode) {
    const modes   = ['presentation', 'micro', 'analysis', 'labdev'];
    const sidebar = document.getElementById('sidebar');

    for (const m of modes) {
        document.getElementById('tab-' + m).classList.remove('active');
        const v = document.getElementById('view-' + m);
        if (v) v.style.display = 'none';
    }

    document.getElementById('tab-' + mode).classList.add('active');
    document.getElementById('view-' + mode).style.display = 'flex';

    // Sidebar only for analysis
    sidebar.style.display = (mode === 'analysis') ? '' : 'none';

    if (mode === 'analysis') {
        setTimeout(() => {
            if (valueChart) valueChart.resize();
            if (analysisPolicyChart) analysisPolicyChart.resize();
        }, 0);
    }
}

// ===== Sidebar sync: slider <-> number input =====
function wireSidebar() {
    const slider  = document.getElementById('ph-slider');
    const phInput = document.getElementById('ph-input');

    slider.addEventListener('input', () => { phInput.value = parseFloat(slider.value).toFixed(2); });
    phInput.addEventListener('input', () => {
        const val = parseFloat(phInput.value);
        if (!isNaN(val)) slider.value = Math.max(0, Math.min(1, val));
    });
    phInput.addEventListener('change', () => {
        let val = parseFloat(phInput.value);
        if (isNaN(val)) val = 0.4;
        val = Math.max(0, Math.min(1, val));
        phInput.value = val.toFixed(2);
        slider.value  = val;
    });
}

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    wireSidebar();

    document.getElementById('tab-presentation').addEventListener('click', () => switchMode('presentation'));
    document.getElementById('tab-micro').addEventListener('click',        () => switchMode('micro'));
    document.getElementById('tab-analysis').addEventListener('click',     () => switchMode('analysis'));
    document.getElementById('tab-labdev').addEventListener('click',       () => switchMode('labdev'));

    document.getElementById('zoom-in').addEventListener('click',  zoomIn);
    document.getElementById('zoom-out').addEventListener('click', zoomOut);

    document.getElementById('calculate-btn').addEventListener('click',     calculate);
    document.getElementById('save-snapshot-btn').addEventListener('click', saveSnapshot);

    // Guessing tab
    document.getElementById('guess-add-btn').addEventListener('click', addGuessRow);
    document.getElementById('reveal-btn').addEventListener('click',    revealGuessing);
    document.getElementById('guess-ph-input').addEventListener('input', () => {
        if (revealShown) revealGuessing();
    });
    const firstRemove = document.querySelector('#guess-bets-container .remove-bet-btn');
    if (firstRemove) {
        firstRemove.addEventListener('click', () => {
            const container = document.getElementById('guess-bets-container');
            if (container.children.length > 1) firstRemove.closest('.bet-row').remove();
        });
    }

    // Dev Insights
    document.getElementById('di-broken1-btn').addEventListener('click', runDiBroken1);
    document.getElementById('di-fixed1-btn').addEventListener('click',  runDiFixed1);
    document.getElementById('di-broken2-btn').addEventListener('click', runDiBroken2);
    document.getElementById('di-fixed2-btn').addEventListener('click',  runDiFixed2);

    // Gamma experiment
    document.getElementById('di-gamma-run-btn').addEventListener('click', runGammaExperiment);

    // Init analysis: ghost + auto-run with defaults
    initAnalysisCharts();
    initAnalysisGhost();
    calculate();  // pre-populate with defaults so chart is ready when tab opens

    switchMode('presentation');

    // Initial KaTeX render for all static math on page
    renderKaTeX(document.body);
});
