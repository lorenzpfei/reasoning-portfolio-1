'use strict';

// ===== Constants =====
const GOAL      = 100;    // terminal winning state
const THETA     = 1e-9;   // convergence threshold
const GAMMA     = 1.0;    // discount factor
const MAX_SWEEPS = 200000; // safety ceiling

// Chart.js instances (kept alive; only data is swapped on recalculation)
let valueChart  = null;
let policyChart = null;
let currentZoom = 100;   // percent
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
let lastHighlightIdx = -1;

// ===== Value Iteration =====
// In-place (Gauss-Seidel) sweeps until max delta < THETA.
function runValueIteration(ph, targetReward, stepReward) {
    const V = new Float64Array(GOAL + 1);

    let sweeps = 0;
    while (sweeps < MAX_SWEEPS) {
        let delta = 0;

        for (let s = 1; s < GOAL; s++) {
            const oldV  = V[s];
            let   best  = -Infinity;
            const maxA  = Math.min(s, GOAL - s);

            for (let a = 1; a <= maxA; a++) {
                const rWin = (s + a === GOAL) ? targetReward : stepReward;
                const ev   = ph * (rWin + V[s + a]) + (1.0 - ph) * (stepReward + V[s - a]);
                if (ev > best) best = ev;
            }

            V[s]      = best;
            const d   = Math.abs(V[s] - oldV);
            if (d > delta) delta = d;
        }

        sweeps++;
        if (delta < THETA) break;
    }

    return { V, sweeps };
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
                backgroundColor: 'rgba(59, 130, 246, 0.08)',
                borderWidth: 2.5,
                pointRadius: 0,
                fill: true,
                tension: 0.15,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { boxWidth: 14, font: { size: 12 } }
                },
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
    const defaultBg = new Array(GOAL - 1).fill('rgba(16, 185, 129, 0.70)');
    const defaultBorder = new Array(GOAL - 1).fill('rgb(5, 150, 105)');

    policyChart = new Chart(pCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Stake π(s)',
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
                legend: {
                    position: 'top',
                    labels: { boxWidth: 14, font: { size: 12 } }
                },
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

// ===== Results Summary =====
function showResultsSummary(policy, startCapital) {
    const optimalStake = policy[startCapital];
    const resultsDiv   = document.getElementById('results-summary');
    const contentEl    = document.getElementById('results-content');

    let html = `<div class="optimal-stake">The mathematical optimal stake at $${startCapital} is: <strong>${optimalStake}</strong></div>`;

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
                html += `<div class="bet-result correct">${bet.name} guessed ${bet.guess} — spot on!</div>`;
            } else {
                html += `<div class="bet-result incorrect">${bet.name} guessed ${bet.guess} — not the optimal stake.</div>`;
            }
        }
        html += '</div>';
    }

    contentEl.innerHTML = html;
    resultsDiv.style.display = '';
}

// ===== Update Policy Chart with Highlight at chosen capital =====
function updatePolicyChartHighlight(startCapital) {
    const bgColors     = new Array(GOAL - 1).fill('rgba(16, 185, 129, 0.70)');
    const borderColors = new Array(GOAL - 1).fill('rgb(5, 150, 105)');

    // Reset previous highlight if any
    const idx = startCapital - 1;
    bgColors[idx]     = 'rgba(234, 88, 12, 0.85)';
    borderColors[idx] = 'rgb(194, 65, 12)';

    policyChart.data.datasets[0].backgroundColor = bgColors;
    policyChart.data.datasets[0].borderColor = borderColors;
    policyChart.update();
}

// ===== Dynamic Bet Row Management =====
function createBetRow() {
    const row = document.createElement('div');
    row.className = 'bet-row';

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.className = 'bet-name';
    nameInput.placeholder = 'Name';

    const guessInput = document.createElement('input');
    guessInput.type = 'number';
    guessInput.className = 'bet-guess';
    guessInput.placeholder = 'Stake';
    guessInput.min = '1';
    guessInput.max = '99';
    guessInput.step = '1';

    const removeBtn = document.createElement('button');
    removeBtn.className = 'remove-bet-btn';
    removeBtn.type = 'button';
    removeBtn.textContent = '−';
    removeBtn.title = 'Remove bet';
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

// ===== Main Calculate & Plot =====
function calculate() {
    const phRaw = parseFloat(document.getElementById('ph-input').value);
    const ph    = isNaN(phRaw) ? 0.4 : Math.max(0, Math.min(1, phRaw));

    const trRaw        = parseFloat(document.getElementById('target-reward').value);
    const targetReward = isNaN(trRaw) ? 1.0 : trRaw;

    const srRaw      = parseFloat(document.getElementById('step-reward').value);
    const stepReward = isNaN(srRaw) ? 0.0 : srRaw;

    const scRaw       = parseInt(document.getElementById('start-capital').value, 10);
    const startCapital = isNaN(scRaw) ? 50 : Math.max(1, Math.min(99, scRaw));

    const statusEl = document.getElementById('status');
    statusEl.textContent = 'Running value iteration…';

    const t0             = performance.now();
    const { V, sweeps }  = runValueIteration(ph, targetReward, stepReward);
    const policy         = extractPolicy(V, ph, targetReward, stepReward);
    const ms             = (performance.now() - t0).toFixed(1);

    const valueData  = Array.from({ length: GOAL - 1 }, (_, i) => V[i + 1]);
    const policyData = Array.from({ length: GOAL - 1 }, (_, i) => policy[i + 1]);

    valueChart.data.datasets[0].data = valueData;
    valueChart.update();

    policyChart.data.datasets[0].data = policyData;
    updatePolicyChartHighlight(startCapital);

    showResultsSummary(policy, startCapital);

    statusEl.textContent =
        `Converged in ${sweeps} sweep${sweeps !== 1 ? 's' : ''} — ${ms} ms — p_h = ${ph.toFixed(2)}`;
}

// ===== Add remove buttons to initial bet row on load =====
function initBetRows() {
    const container = document.getElementById('bets-container');
    const existingRow = container.querySelector('.bet-row');
    if (existingRow) {
        const removeBtn = document.createElement('button');
        removeBtn.className = 'remove-bet-btn';
        removeBtn.type = 'button';
        removeBtn.textContent = '−';
        removeBtn.title = 'Remove bet';
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
    });

    phInput.addEventListener('input', () => {
        const val = parseFloat(phInput.value);
        if (!isNaN(val)) {
            slider.value = Math.max(0, Math.min(1, val));
        }
    });

    phInput.addEventListener('change', () => {
        let val = parseFloat(phInput.value);
        if (isNaN(val)) val = 0.4;
        val = Math.max(0, Math.min(1, val));
        phInput.value = val.toFixed(2);
        slider.value  = val;
    });

    document.getElementById('calculate-btn').addEventListener('click', calculate);
    document.getElementById('add-bet-btn').addEventListener('click', addBetRow);
    document.getElementById('zoom-in').addEventListener('click', zoomIn);
    document.getElementById('zoom-out').addEventListener('click', zoomOut);

    initBetRows();
    initCharts();
});