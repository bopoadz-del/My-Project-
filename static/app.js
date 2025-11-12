// MSSDPPG Simulator - Main Application Logic (v4 Modular)

// Chart instances
let powerBreakdownChart, powerChart, flywheelChart;
let statusPollInterval = null;

// Scenario configuration
let scenarioConfig = {};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadScenarios();
    initializeCharts();
    addLog('System initialized', 'info');
});

// Load Scenarios
function loadScenarios() {
    fetch('/api/scenarios')
        .then(r => r.json())
        .then(scenarios => {
            scenarioConfig = scenarios;
            console.log('Available scenarios:', scenarios);
            updateScenarioInfo();
            addLog('Scenarios loaded', 'info');
        })
        .catch(err => {
            console.error('Failed to load scenarios:', err);
            addLog('Failed to load scenarios', 'error');
        });
}

// Update Scenario Info (when scenario changes)
function updateScenarioInfo() {
    const scenario = document.getElementById('scenario').value;
    const config = scenarioConfig[scenario];

    if (config) {
        // Update n_pendulums slider range
        const slider = document.getElementById('n_pendulums');
        const min = config.n_pendulums_min || 1;
        const max = config.n_pendulums_max || 48;
        const current = parseInt(slider.value);

        slider.min = min;
        slider.max = max;

        // Reset to default if current is out of range
        if (current < min || current > max) {
            slider.value = config.n_pendulums_default || min;
        }

        updatePendulumInfo();
        addLog(`Scenario: ${config.display_name}`, 'info');
    }
}

// Update Pendulum Info (when n_pendulums changes)
function updatePendulumInfo() {
    const n = parseInt(document.getElementById('n_pendulums').value);
    document.getElementById('n_pendulums_display').textContent = n;

    // Calculate total generators: 2 per pendulum + 2 ground (FW/RV)
    const totalGens = 2 * n + 2;

    const scenario = document.getElementById('scenario').value;
    const config = scenarioConfig[scenario];
    const scenarioName = config ? config.display_name : scenario;

    document.getElementById('scenario_info').textContent =
        `${n} pendulums → ${totalGens} generators (${scenarioName})`;
}

// Initialize Charts
function initializeCharts() {
    const chartConfig = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 0 },
        plugins: {
            legend: {
                labels: { color: '#a0a0a0', font: { size: 11 } }
            }
        },
        scales: {
            y: {
                ticks: { color: '#a0a0a0', font: { size: 10 } },
                grid: { color: 'rgba(78, 205, 196, 0.1)' }
            },
            x: {
                ticks: { color: '#a0a0a0', font: { size: 10 } },
                grid: { color: 'rgba(78, 205, 196, 0.1)' }
            }
        }
    };

    // Power Breakdown Chart (Bar chart)
    const breakdownCtx = document.getElementById('powerBreakdownChart').getContext('2d');
    powerBreakdownChart = new Chart(breakdownCtx, {
        type: 'bar',
        data: {
            labels: ['Hinge 1', 'Hinge 2', 'Ground FW', 'Ground RV'],
            datasets: [{
                label: 'Avg Power (kW)',
                data: [0, 0, 0, 0],
                backgroundColor: [
                    'rgba(78, 205, 196, 0.7)',
                    'rgba(149, 225, 211, 0.7)',
                    'rgba(68, 160, 141, 0.7)',
                    'rgba(255, 107, 107, 0.7)'
                ],
                borderColor: [
                    '#4ECDC4',
                    '#95E1D3',
                    '#44A08D',
                    '#FF6B6B'
                ],
                borderWidth: 1
            }]
        },
        options: {
            ...chartConfig,
            indexAxis: 'y'
        }
    });

    // Total Power Chart (Line chart)
    const powerCtx = document.getElementById('powerChart').getContext('2d');
    powerChart = new Chart(powerCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Total Power (kW)',
                data: [],
                borderColor: '#4ECDC4',
                backgroundColor: 'rgba(78, 205, 196, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: chartConfig
    });

    // Flywheel RPM Chart (Dual line)
    const flywheelCtx = document.getElementById('flywheelChart').getContext('2d');
    flywheelChart = new Chart(flywheelCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Flywheel FW (RPM)',
                    data: [],
                    borderColor: '#95E1D3',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Flywheel RV (RPM)',
                    data: [],
                    borderColor: '#FF6B6B',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: chartConfig
    });
}

// Polling for simulation status
function pollSimulationStatus() {
    fetch('/api/sim-status')
        .then(r => r.json())
        .then(status => {
            updateProgress(status.progress);

            if (!status.running) {
                // Simulation finished
                clearInterval(statusPollInterval);
                statusPollInterval = null;

                fetch('/api/sim-result')
                    .then(r => r.json())
                    .then(result => {
                        if (!result.error) {
                            addLog('Simulation completed!', 'info');
                            updateResultsDisplay(result);
                        } else {
                            addLog(`Error: ${result.error}`, 'error');
                        }
                    })
                    .catch(err => {
                        console.error('Result fetch error:', err);
                        addLog('Error fetching results', 'error');
                    });

                const runBtn = document.getElementById('runBtn');
                runBtn.disabled = false;
                runBtn.textContent = '▶ Run Simulation';
                runBtn.classList.remove('running');
            }
        })
        .catch(err => {
            console.error('Poll error:', err);
        });
}

// Start Simulation
function startSimulation() {
    const scenario = document.getElementById('scenario').value;
    const n_pendulums = parseInt(document.getElementById('n_pendulums').value);
    const duration = parseInt(document.getElementById('duration').value);
    const assist = document.getElementById('assist').value;

    const params = {
        scenario: scenario,
        duration: duration,
        n_pendulums: n_pendulums,
        assist: assist
    };

    const totalGens = 2 * n_pendulums + 2;
    addLog(`Starting simulation: ${scenario} (${n_pendulums} pendulums, ${totalGens} generators, ${duration}h)`, 'info');

    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running...';
    runBtn.classList.add('running');

    // Clear charts
    if (powerChart) {
        powerChart.data.labels = [];
        powerChart.data.datasets[0].data = [];
    }
    if (flywheelChart) {
        flywheelChart.data.labels = [];
        flywheelChart.data.datasets[0].data = [];
        flywheelChart.data.datasets[1].data = [];
    }

    fetch('/api/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params)
    })
    .then(r => r.json())
    .then(res => {
        if (res.error) {
            addLog(`Error: ${res.error}`, 'error');
            runBtn.disabled = false;
            runBtn.textContent = '▶ Run Simulation';
            runBtn.classList.remove('running');
        } else {
            // Start polling for status updates
            addLog('Simulation started, polling status...', 'info');
            statusPollInterval = setInterval(pollSimulationStatus, 500);
        }
    })
    .catch(err => {
        addLog(`Network error: ${err.message}`, 'error');
        runBtn.disabled = false;
        runBtn.textContent = '▶ Run Simulation';
        runBtn.classList.remove('running');
    });
}

// Update Results Display
function updateResultsDisplay(result) {
    // Update stats panel
    document.getElementById('stat_pendulums').textContent = result.n_pendulums || '--';
    document.getElementById('stat_generators').textContent = result.total_generators || '--';
    document.getElementById('stat_avg_power').textContent = (result.power?.total_avg_kW || 0).toFixed(2);
    document.getElementById('stat_peak_power').textContent = (result.power?.total_peak_kW || 0).toFixed(2);
    document.getElementById('stat_energy').textContent = (result.power?.energy_kWh || 0).toFixed(2);
    document.getElementById('stat_temp').textContent = (result.coil_temp_max_C || 0).toFixed(1);

    // Update power breakdown chart
    if (powerBreakdownChart) {
        powerBreakdownChart.data.datasets[0].data = [
            result.power?.hinge1_avg_kW || 0,
            result.power?.hinge2_avg_kW || 0,
            result.power?.ground_fw_avg_kW || 0,
            result.power?.ground_rv_avg_kW || 0
        ];
        powerBreakdownChart.update('none');
    }

    addLog(`Results: Avg ${(result.power?.total_avg_kW || 0).toFixed(2)} kW, Peak ${(result.power?.total_peak_kW || 0).toFixed(2)} kW`, 'info');
}

// Update Progress
function updateProgress(progress) {
    document.getElementById('progressFill').style.width = progress + '%';
    document.getElementById('progressText').textContent = `${progress}% Complete`;
}

// Logging
function addLog(message, type = 'info') {
    const logArea = document.getElementById('logArea');
    const p = document.createElement('p');
    p.className = `log-${type}`;
    const time = new Date().toLocaleTimeString();
    p.textContent = `[${time}] ${message}`;
    logArea.appendChild(p);
    logArea.scrollTop = logArea.scrollHeight;

    // Keep only last 20 logs
    while (logArea.children.length > 20) {
        logArea.firstChild.remove();
    }
}

// Camera controls
function resetCamera() {
    if (window.resetThreeCamera) {
        window.resetThreeCamera();
    }
    addLog('Camera reset', 'info');
}

function toggleWireframe() {
    if (window.toggleWireframeMode) {
        window.toggleWireframeMode();
    }
    addLog('Wireframe toggled', 'info');
}

function toggleRotation() {
    if (window.toggleAutoRotate) {
        window.toggleAutoRotate();
    }
    addLog('Auto-rotation toggled', 'info');
}
