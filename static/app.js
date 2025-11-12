// MSSDPPG Simulator - Main Application Logic

// Chart instances
let powerChart, windChart, angleChart;
let statusPollInterval = null;

// Data storage
const data = {
    time: [],
    power: [],
    wind: [],
    theta1: [],
    theta2: [],
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeCharts();
    loadScenarios();
    addLog('System initialized', 'info');
});

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

    // Power Chart
    const powerCtx = document.getElementById('powerChart').getContext('2d');
    powerChart = new Chart(powerCtx, {
        type: 'line',
        data: {
            labels: data.time,
            datasets: [{
                label: 'Power (kW)',
                data: data.power,
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

    // Wind Chart
    const windCtx = document.getElementById('windChart').getContext('2d');
    windChart = new Chart(windCtx, {
        type: 'line',
        data: {
            labels: data.time,
            datasets: [{
                label: 'Wind (m/s)',
                data: data.wind,
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: chartConfig
    });

    // Angle Chart
    const angleCtx = document.getElementById('angleChart').getContext('2d');
    angleChart = new Chart(angleCtx, {
        type: 'line',
        data: {
            labels: data.time,
            datasets: [
                {
                    label: 'θ1 (rad)',
                    data: data.theta1,
                    borderColor: '#95E1D3',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5
                },
                {
                    label: 'θ2 (rad)',
                    data: data.theta2,
                    borderColor: '#AA96DA',
                    backgroundColor: 'transparent',
                    fill: false,
                    tension: 0.4,
                    pointRadius: 0,
                    borderWidth: 1.5
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

            if (status.frames && status.frames.length > 0) {
                status.frames.forEach(frame => {
                    updateFrameData(frame);
                });
                updateCharts();
            }

            if (!status.running) {
                // Simulation finished
                clearInterval(statusPollInterval);
                fetch('/api/sim-result')
                    .then(r => r.json())
                    .then(result => {
                        if (result.summary) {
                            addLog('Simulation completed!', 'info');
                            updateStats(result.summary);
                        } else if (result.error) {
                            addLog(`Error: ${result.error}`, 'error');
                        }
                    });

                document.getElementById('runBtn').disabled = false;
                document.getElementById('runBtn').textContent = '▶ Run Simulation';
                document.getElementById('runBtn').classList.remove('running');
            }
        })
        .catch(err => {
            console.error('Poll error:', err);
        });
}

// Load Scenarios
function loadScenarios() {
    fetch('/api/scenarios')
        .then(r => r.json())
        .then(scenarios => {
            console.log('Available scenarios:', scenarios);
        });
}

// Start Simulation
function startSimulation() {
    const params = {
        scenario: document.getElementById('scenario').value,
        duration: document.getElementById('duration').value,
        mode: document.getElementById('mode').value,
        control: document.getElementById('control').value,
        assist: document.getElementById('assist').value,
    };

    addLog(`Starting simulation: ${params.scenario} (${params.duration})`, 'info');

    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;
    runBtn.textContent = '⏳ Running...';
    runBtn.classList.add('running');

    // Clear data
    data.time = [];
    data.power = [];
    data.wind = [];
    data.theta1 = [];
    data.theta2 = [];

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
            addLog('Polling simulation status...', 'info');
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

// Update Frame Data
function updateFrameData(frame) {
    const t_hours = frame.t / 3600;
    data.time.push(t_hours.toFixed(2));
    data.power.push((Math.random() * 5 + 2).toFixed(2)); // Placeholder
    data.wind.push(frame.wind.toFixed(1));
    data.theta1.push(frame.theta1.toFixed(3));
    data.theta2.push(frame.theta2.toFixed(3));

    // Keep only last 100 points
    const maxPoints = 100;
    if (data.time.length > maxPoints) {
        data.time.shift();
        data.power.shift();
        data.wind.shift();
        data.theta1.shift();
        data.theta2.shift();
    }

    // Update 3D visualization
    if (window.updateVisualization) {
        window.updateVisualization(frame);
    }
}

// Update Charts
function updateCharts() {
    if (powerChart) {
        powerChart.data.labels = data.time;
        powerChart.data.datasets[0].data = data.power;
        powerChart.update('none');
    }

    if (windChart) {
        windChart.data.labels = data.time;
        windChart.data.datasets[0].data = data.wind;
        windChart.update('none');
    }

    if (angleChart) {
        angleChart.data.labels = data.time;
        angleChart.data.datasets[0].data = data.theta1;
        angleChart.data.datasets[1].data = data.theta2;
        angleChart.update('none');
    }
}

// Update Progress
function updateProgress(progress) {
    document.getElementById('progressFill').style.width = progress + '%';
    document.getElementById('progressText').textContent = `${progress}% Complete`;
}

// Update Stats
function updateStats(summary) {
    if (summary) {
        document.querySelectorAll('.info-panel span').forEach((el, i) => {
            const values = [
                summary.avg_kW?.toFixed(2) || '--',
                summary.peak_kW?.toFixed(2) || '--',
                summary.energy_kWh?.toFixed(2) || '--',
                summary.coil_Tmax_C?.toFixed(1) || '--'
            ];
            el.textContent = values[i];
        });
    }
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
