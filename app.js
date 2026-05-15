'use strict';

// ===== Constants =====
const GOAL       = 100;
const THETA      = 1e-9;
const GAMMA      = 1.0;
const MAX_SWEEPS = 200000;

const SNAPSHOT_SWEEPS = [1, 2, 3, 10, 32];

const SWEEP_COLORS = {
    1:       { color: '#3b82f6', width: 1.5, label: 'Sweep 1' },
    2:       { color: '#f59e0b', width: 1.5, label: 'Sweep 2' },
    3:       { color: '#10b981', width: 1.5, label: 'Sweep 3' },
    10:      { color: '#8b5cf6', width: 1.5, label: 'Sweep 10' },
    32:      { color: '#ec4899', width: 1.5, label: 'Sweep 32' },
    final:   { color: '#1e293b', width: 2.5, label: 'Final' },
};

let valueChart  = null;
let policyChart = null;
let currentZoom = 100;
const ZOOM_MIN  = 50;
const ZOOM_MAX  = 250;
const ZOOM_STEP = 10;

// ===== Zoom Controls =====
function applyZoom() {
    document.body.style.zoom = currentZoom + '%';
    document.getElementById('zoom-level').textContent = currentZoom + '%';
}

function zoomIn() {
    if (currentZoom < ZOOM_MAX) {
        currentZoom = Math.min(currentZoom + ZOOM_STEP, ZOOM_MAX);
        applyZoom();
    }
}

function zoomOut() {
    if (currentZoom > ZOOM_MIN) {
        currentZoom = Math.max(currentZoom - ZOOM_STEP, ZOOM_MIN);
        applyZoom();
    }
}

// ===== Value Iteration =====
function runValueIteration(ph, targetReward, stepReward) {
    const V = new Float64Array(GOAL + 1);

    const history  = [];
    const roundLog = [];

    const snapshotSet = new Set(SNAPSHOT_SWEEPS);
    let finalSweepNum = -1;

    let sweeps = 0;
    let converged = false;

    while (sweeps < MAX_SWEEPS) {
        let delta = 0;

        for (let s = 1; s < GOAL; s++) {
            const oldV = V[s];
            let   best = -Infinity;
            const maxA = Math.min(s, GOAL - s);

            for (let a = 1; a <= maxA; a++) {
                const rWin = (s + a === GOAL) ? targetReward : stepReward;
                const ev   = ph * (rWin + V[s + a]) + (1.0 - ph) * (stepReward + V[s - a]);
                if (ev > best) best = ev;
            }

            V[s]    = best;
            const d = Math.abs(V[s] - oldV);
            if (d > delta) delta = d;
        }

        sweeps++;
        converged = (delta < THETA);

        roundLog.push({ sweep: sweeps, delta, v25: V[25], v50: V[50], v75: V[75], v99: V[99] });

        if (converged) {
            finalSweepNum = sweeps;
            if (snapshotSet.has(sweeps)) {
                history.push({ sweep: sweeps, V: V.slice(), isFinal: true });
            } else {
                if (!history.some(h => h.sweep === sweeps)) {
                    history.push({ sweep: sweeps, V: V.slice(), isFinal: true });
                }
            }
            break;
        }

        if (snapshotSet.has(sweeps)) {
            history.push({ sweep: sweeps, V: V.slice(), isFinal: false });
        }
    }

    return { V, sweeps, history, roundLog };
}

// ===== Policy Extraction =====
function extractPolicy(V, ph, targetReward, stepReward) {
    const policy = new Int32Array(GOAL + 1);

    for (let s = 1; s < GOAL; s++) {
        let best  = -Infinity;
        let bestA = 1;
        const maxA = Math.min(s, GOAL - s);

        for (let a = 1; a <= maxA; a++) {
            const rWin = (s + a === GOAL) ? targetReward : stepReward;
            const ev   = ph * (rWin + V[s + a]) + (1.0 - ph) * (stepReward + V[s - a]);

            if (ev > best + 1e-9) {
                best  = ev;
                bestA = a;
            }
        }

        policy[s] = bestA;
    }

    return policy;
}

// ===== Build Value Chart Legend =====
function buildValueLegend(history) {
    const legendEl = document.getElementById('value-legend');
    legendEl.innerHTML = '';

    for (const snap of history) {
        const key = snap.isFinal ? 'final' : snap.sweep;
        const cfg = SWEEP_COLORS[key] || SWEEP_COLORS.final;

        const item = document.createElement('span');
        item.className = 'legend-item';

        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = cfg.color;
        swatch.style.height = (cfg.width >= 2 ? '3px' : '2px');

        const label = document.createElement('span');
        label.textContent = snap.isFinal
            ? `Final (sweep ${snap.sweep})`
            : cfg.label || `Sweep ${snap.sweep}`;

        item.appendChild(swatch);
        item.appendChild(label);
        legendEl.appendChild(item);
    }
}

// ===== Chart Initialisation =====
function initCharts() {
    const labels = Array.from({ length: GOAL - 1 }, (_, i) => i + 1);
    const zeros  = new Array(GOAL - 1).fill(0);

    const vCtx = document.getElementById('value-chart').getContext('2d');
    valueChart = new Chart(vCtx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'V(s)',
                data: zeros.slice(),
                borderColor: '#3b82f6',
                borderWidth: 1.5,
                pointRadius: 0,
                fill: false,
                tension: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Capital (s)',
                        font: { size: 12, weight: 'bold' }
                    },
                    ticks: { maxTicksLimit: 20 }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value V(s)',
                        font: { size: 12, weight: 'bold' }
                    }
                }
            }
        }
    });

    const pCtx = document.getElementById('policy-chart').getContext('2d');
    const defaultBg     = new Array(GOAL - 1).fill('rgba(16, 185, 129, 0.70)');
    const defaultBorder = new Array(GOAL - 1).fill('rgb(5, 150, 105)');

    policyChart = new Chart(pCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Stake pi(s)',
                data: zeros.slice(),
                backgroundColor: defaultBg,
                borderColor: defaultBorder,
                borderWidth: 1,
                barPercentage: 1.0,
                categoryPercentage: 0.85,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Capital (s)',
                        font: { size: 12, weight: 'bold' }
                    },
                    ticks: { maxTicksLimit: 20 }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Stake (a)',
                        font: { size: 12, weight: 'bold' }
                    },
                    min: 0,
                    ticks: { precision: 0 }
                }
            }
        }
    });
}

// ===== Update Value Chart =====
function updateValueChart(history) {
    const labels = Array.from({ length: GOAL - 1 }, (_, i) => i + 1);

    const datasets = history.map(snap => {
        const key = snap.isFinal ? 'final' : snap.sweep;
        const cfg = SWEEP_COLORS[key] || SWEEP_COLORS.final;
        return {
            label: snap.isFinal ? `Final (sweep ${snap.sweep})` : (cfg.label || `Sweep ${snap.sweep}`),
            data: Array.from({ length: GOAL - 1 }, (_, i) => snap.V[i + 1]),
            borderColor: cfg.color,
            borderWidth: cfg.width,
            pointRadius: 0,
            fill: false,
            tension: 0,
        };
    });

    valueChart.data.labels   = labels;
    valueChart.data.datasets = datasets;
    valueChart.update();
}

// ===== Results Summary =====
function showResultsSummary(policy, startCapital) {
    const optimalStake = policy[startCapital];
    const resultsDiv   = document.getElementById('results-summary');
    const contentEl    = document.getElementById('results-content');

    let html = `<div class="optimal-stake">Optimal stake at $${startCapital}: <strong>$${optimalStake}</strong></div>`;

    const guessInputs = document.querySelectorAll('.bet-guess');
    const nameInputs  = document.querySelectorAll('.bet-name');
    const bets = [];

    for (let i = 0; i < guessInputs.length; i++) {
        const guess = parseInt(guessInputs[i].value, 10);
        if (!isNaN(guess)) {
            const name = nameInputs[i] ? nameInputs[i].value.trim() || `Player ${i + 1}` : `Player ${i + 1}`;
            bets.push({ name, guess });
        }
    }

    if (bets.length === 0) {
        html += '<div class="bet-result incorrect">No audience bets submitted.</div>';
    } else {
        html += '<div class="bets-results">';
        for (const bet of bets) {
            if (bet.guess === optimalStake) {
                html += `<div class="bet-result correct">${bet.name} guessed $${bet.guess} — spot on!</div>`;
            } else {
                html += `<div class="bet-result incorrect">${bet.name} guessed $${bet.guess} — not optimal.</div>`;
            }
        }
        html += '</div>';
    }

    contentEl.innerHTML = html;
    resultsDiv.style.display = '';
}

// ===== Update Stats Cards =====
function updateStats(sweeps, V, policy) {
    document.getElementById('stat-sweeps').textContent = sweeps;
    document.getElementById('stat-v50').textContent    = V[50].toFixed(3);
    document.getElementById('stat-p50').textContent    = '$' + policy[50];
    document.getElementById('stat-p25').textContent    = '$' + policy[25];
}

// ===== Policy Chart Highlight =====
function updatePolicyChartHighlight(startCapital) {
    const bgColors     = new Array(GOAL - 1).fill('rgba(16, 185, 129, 0.70)');
    const borderColors = new Array(GOAL - 1).fill('rgb(5, 150, 105)');

    const idx = startCapital - 1;
    bgColors[idx]     = 'rgba(234, 88, 12, 0.85)';
    borderColors[idx] = 'rgb(194, 65, 12)';

    policyChart.data.datasets[0].backgroundColor = bgColors;
    policyChart.data.datasets[0].borderColor      = borderColors;
    policyChart.update();
}

// ===== Value Iteration Log (light design) =====
function updateLog(roundLog, sweeps) {
    const logBody = document.getElementById('log-body');

    const showIdx = new Set();
    for (let i = 0; i < Math.min(5, roundLog.length); i++) showIdx.add(i);
    for (let i = 0; i < roundLog.length; i += 5) showIdx.add(i);
    showIdx.add(roundLog.length - 1);

    const sortedIdx = Array.from(showIdx).sort((a, b) => a - b);

    let html = `<table class="log-table">
        <thead><tr>
            <th>Sweep</th>
            <th>Delta</th>
            <th>V($25)</th>
            <th>V($50)</th>
            <th>V($75)</th>
            <th>V($99)</th>
        </tr></thead><tbody>`;

    let prevShown = -1;

    for (const idx of sortedIdx) {
        if (prevShown !== -1 && idx - prevShown > 1) {
            html += `<tr class="log-separator"><td colspan="6">&middot;&middot;&middot;</td></tr>`;
        }

        const r       = roundLog[idx];
        const isFinal = (idx === roundLog.length - 1);

        let rowClass;
        if (isFinal)         rowClass = 'log-row-final';
        else if (r.sweep <= 3)  rowClass = 'log-row-early';
        else if (r.sweep <= 10) rowClass = 'log-row-mid';
        else if (r.sweep <= 32) rowClass = 'log-row-late';
        else                    rowClass = '';

        const sweepCell = isFinal
            ? `${r.sweep}<span class="log-conv-tag">conv</span>`
            : r.sweep;

        html += `<tr class="${rowClass}">
            <td>${sweepCell}</td>
            <td>${r.delta.toExponential(3)}</td>
            <td>${r.v25.toFixed(4)}</td>
            <td>${r.v50.toFixed(4)}</td>
            <td>${r.v75.toFixed(4)}</td>
            <td>${r.v99.toFixed(4)}</td>
        </tr>`;

        prevShown = idx;
    }

    html += `</tbody></table>`;
    logBody.innerHTML = html;
    logBody.scrollTop = logBody.scrollHeight;
}

// ===== Dynamic Bet Row Management =====
function createBetRow() {
    const row = document.createElement('div');
    row.className = 'bet-row';

    const nameInput = document.createElement('input');
    nameInput.type        = 'text';
    nameInput.className   = 'bet-name';
    nameInput.placeholder = 'Name';

    const guessInput = document.createElement('input');
    guessInput.type        = 'number';
    guessInput.className   = 'bet-guess';
    guessInput.placeholder = 'Stake';
    guessInput.min         = '1';
    guessInput.max         = '99';
    guessInput.step        = '1';

    const removeBtn = document.createElement('button');
    removeBtn.className   = 'remove-bet-btn';
    removeBtn.type        = 'button';
    removeBtn.textContent = '-';
    removeBtn.title       = 'Remove bet';
    removeBtn.addEventListener('click', () => {
        const container = document.getElementById('bets-container');
        if (container.children.length > 1) {
            row.remove();
        }
    });

    row.appendChild(nameInput);
    row.appendChild(guessInput);
    row.appendChild(removeBtn);
    return row;
}

function addBetRow() {
    const container = document.getElementById('bets-container');
    const row = createBetRow();
    container.appendChild(row);
    row.querySelector('.bet-guess').focus();
}

// ===== Hero Updates =====
function updateHeroCapital(value) {
    const heroCapital = document.getElementById('hero-capital');
    const parsed = parseInt(value, 10);
    heroCapital.textContent = '$' + (isNaN(parsed) ? '—' : parsed);
}

function updateHeroPh(value) {
    const heroPh = document.getElementById('hero-ph');
    if (!heroPh) return;
    const parsed = parseFloat(value);
    heroPh.textContent = isNaN(parsed) ? '—' : (parsed * 100).toFixed(0) + '%';
}

// ===== Mode Switching =====
function switchMode(mode) {
    const tabs = ['presentation', 'micro', 'analysis'];
    tabs.forEach(t => {
        document.getElementById('tab-' + t).classList.remove('active');
        const view = document.getElementById('view-' + t);
        if (view) view.style.display = 'none';
    });

    document.getElementById('tab-' + mode).classList.add('active');
    document.getElementById('view-' + mode).style.display = 'flex';

    const audienceSection = document.getElementById('audience-section');
    audienceSection.style.display = (mode === 'presentation') ? '' : 'none';

    if (mode === 'presentation') {
        setTimeout(() => { if (policyChart) policyChart.resize(); }, 0);
    } else if (mode === 'analysis') {
        setTimeout(() => { if (valueChart) valueChart.resize(); }, 0);
    }
}

// ===== Main Calculate & Plot =====
function calculate() {
    const phRaw = parseFloat(document.getElementById('ph-input').value);
    const ph    = isNaN(phRaw) ? 0.4 : Math.max(0, Math.min(1, phRaw));

    const trRaw        = parseFloat(document.getElementById('target-reward').value);
    const targetReward = isNaN(trRaw) ? 1.0 : trRaw;

    const srRaw      = parseFloat(document.getElementById('step-reward').value);
    const stepReward = isNaN(srRaw) ? 0.0 : srRaw;

    const scRaw        = parseInt(document.getElementById('start-capital').value, 10);
    const startCapital = isNaN(scRaw) ? 50 : Math.max(1, Math.min(99, scRaw));

    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Running value iteration…';

    const t0 = performance.now();
    const { V, sweeps, history, roundLog } = runValueIteration(ph, targetReward, stepReward);
    const policy = extractPolicy(V, ph, targetReward, stepReward);
    const ms     = (performance.now() - t0).toFixed(1);

    // Value chart (analysis tab)
    updateValueChart(history);
    buildValueLegend(history);

    // Policy chart (guessing tab) — show card, then populate
    document.getElementById('policy-card').style.display = '';
    const policyData = Array.from({ length: GOAL - 1 }, (_, i) => policy[i + 1]);
    policyChart.data.datasets[0].data = policyData;
    updatePolicyChartHighlight(startCapital);

    // Stats (analysis tab)
    updateStats(sweeps, V, policy);

    // Log (analysis tab)
    updateLog(roundLog, sweeps);

    // Results summary (guessing tab)
    showResultsSummary(policy, startCapital);

    statusEl.textContent =
        `Converged in ${sweeps} sweep${sweeps !== 1 ? 's' : ''} — ${ms} ms — p_h = ${ph.toFixed(2)}`;
}

// ===== Add remove button to initial bet row =====
function initBetRows() {
    const container   = document.getElementById('bets-container');
    const existingRow = container.querySelector('.bet-row');
    if (existingRow) {
        const removeBtn = document.createElement('button');
        removeBtn.className   = 'remove-bet-btn';
        removeBtn.type        = 'button';
        removeBtn.textContent = '-';
        removeBtn.title       = 'Remove bet';
        removeBtn.addEventListener('click', () => {
            if (container.children.length > 1) {
                existingRow.remove();
            }
        });
        existingRow.appendChild(removeBtn);
    }
}

// ===== Event Wiring =====
document.addEventListener('DOMContentLoaded', () => {
    const slider  = document.getElementById('ph-slider');
    const phInput = document.getElementById('ph-input');

    slider.addEventListener('input', () => {
        phInput.value = parseFloat(slider.value).toFixed(2);
        updateHeroPh(slider.value);
    });

    phInput.addEventListener('input', () => {
        const val = parseFloat(phInput.value);
        if (!isNaN(val)) {
            slider.value = Math.max(0, Math.min(1, val));
            updateHeroPh(val);
        }
    });

    phInput.addEventListener('change', () => {
        let val = parseFloat(phInput.value);
        if (isNaN(val)) val = 0.4;
        val = Math.max(0, Math.min(1, val));
        phInput.value = val.toFixed(2);
        slider.value  = val;
        updateHeroPh(val);
    });

    const startCapitalInput = document.getElementById('start-capital');
    startCapitalInput.addEventListener('input',  () => updateHeroCapital(startCapitalInput.value));
    startCapitalInput.addEventListener('change', () => updateHeroCapital(startCapitalInput.value));

    document.getElementById('tab-presentation').addEventListener('click', () => switchMode('presentation'));
    document.getElementById('tab-micro').addEventListener('click',        () => switchMode('micro'));
    document.getElementById('tab-analysis').addEventListener('click',     () => switchMode('analysis'));

    document.getElementById('calculate-btn').addEventListener('click', calculate);
    document.getElementById('add-bet-btn').addEventListener('click',   addBetRow);
    document.getElementById('zoom-in').addEventListener('click',       zoomIn);
    document.getElementById('zoom-out').addEventListener('click',      zoomOut);

    initBetRows();
    initCharts();

    updateHeroCapital(startCapitalInput.value);
    updateHeroPh(phInput.value);

    switchMode('presentation');
});
