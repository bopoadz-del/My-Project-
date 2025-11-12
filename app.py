"""
MSSDPPG Web UI with Real-time 3D Visualization
Flask backend for simulation and data streaming
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import os
import json
import threading
from datetime import datetime
import time
from queue import Queue

from MSSDPPG_UltraRealistic_v2 import (
    SCENARIOS, standard_wind_profile, load_wind_csv,
    Pendulum2D, Pendulum3D, run_one
)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'mssdppg-secret-key'

# Simulation data queue for polling-based updates
sim_queue = Queue()
sim_result = {}

# Global simulation state
sim_state = {
    'running': False,
    'progress': 0,
    'current_frame': 0,
    'total_frames': 0,
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scenarios')
def get_scenarios():
    """Get available scenarios"""
    return jsonify({
        name: {
            'display_name': scenario.name,
            'L1': scenario.L1,
            'L2': scenario.L2,
            'n_pendulums': scenario.n_pendulums,
            'color': scenario.color
        }
        for name, scenario in SCENARIOS.items()
    })

@app.route('/api/config')
def get_config():
    """Get simulation configuration options"""
    return jsonify({
        'durations': ['6h', '12h'],
        'modes': ['2d', '3d', 'dual'],
        'controls': ['lock', 'pushpull'],
        'assist': ['on', 'off'],
        'scenarios': list(SCENARIOS.keys())
    })

@app.route('/api/simulate', methods=['POST'])
def start_simulation():
    """Start a new simulation"""
    global sim_result
    if sim_state['running']:
        return jsonify({'error': 'Simulation already running'}), 400

    params = request.json
    sim_state['running'] = True
    sim_state['progress'] = 0

    # Run simulation in background thread
    thread = threading.Thread(
        target=run_simulation_background,
        args=(params,)
    )
    thread.daemon = True
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/sim-status')
def sim_status():
    """Get current simulation status and data"""
    return jsonify({
        'running': sim_state['running'],
        'progress': sim_state['progress'],
        'frame_count': sim_state['current_frame'],
        'frames': list(sim_queue.queue) if not sim_queue.empty() else []
    })

@app.route('/api/sim-result')
def sim_result_endpoint():
    """Get simulation results"""
    return jsonify(sim_result)

def run_simulation_background(params):
    """Run simulation and stream data via queue"""
    global sim_result
    try:
        scenario_key = params.get('scenario', '4x40ft')
        duration_h = int(params.get('duration', '6').replace('h', ''))
        sim_mode = params.get('mode', '2d')
        control = params.get('control', 'lock')
        assist = params.get('assist', 'on') == 'on'

        duration_s = duration_h * 3600

        # Load wind profile
        wind_file = params.get('windfile', '')
        if wind_file and os.path.exists(wind_file):
            t_wind, v_wind = load_wind_csv(wind_file)
        else:
            t_wind, v_wind = standard_wind_profile(duration_s)

        S = SCENARIOS[scenario_key]

        # Run simulation with data streaming
        sim_result = run_simulation_with_streaming(
            sim_mode, S, duration_s, control, assist,
            dict(offset=0.12, k_phi=10.0, c_phi=0.6, Km_phi=3.0),
            t_wind, v_wind
        )

    except Exception as e:
        sim_result = {'error': str(e)}
    finally:
        sim_state['running'] = False

def run_simulation_with_streaming(sim_mode, S, duration_s, control_mode, assist,
                                  spatial_params, t_wind, v_wind, pend_2d=None, pend_3d=None):
    """Run simulation and stream frame data"""

    from scipy.integrate import solve_ivp

    results = {'frames': [], 'summary': {}}

    if sim_mode == '2d':
        pend = Pendulum2D(S, control_mode=control_mode, assist=assist)
        y0 = [0.15, 0.0, 0.0, 0.0]
        eom_func = pend.eom
    elif sim_mode == '3d':
        pend = Pendulum3D(S, control_mode=control_mode, assist=assist, **spatial_params)
        y0 = [0.15, 0.0, 0.0, 0.0, pend.phi0, pend.wphi0]
        eom_func = pend.eom3d
    else:
        # For 'dual' mode, just run 2D for streaming
        pend = Pendulum2D(S, control_mode=control_mode, assist=assist)
        y0 = [0.15, 0.0, 0.0, 0.0]
        eom_func = pend.eom

    t_all = []
    y_all = [[] for _ in range(len(y0))]

    # Integrate
    t = 0.0
    y = np.array(y0, dtype=float)
    dt_step = 1.0  # 1 second per frame for visualization
    frame_count = 0

    while t < duration_s:
        # Simulate next second
        t_next = min(t + dt_step, duration_s)

        # Wind interpolation
        v_wind_curr = float(np.interp(t, t_wind, v_wind))
        pend.v_wind = v_wind_curr
        pend.dt_local = dt_step

        # Simple Euler step (for visualization; full sim uses LSODA)
        dydt = eom_func(t, y)
        y_new = y + np.array(dydt) * (t_next - t)
        y = y_new

        # Store frame
        frame = {
            't': float(t),
            'theta1': float(y[0]),
            'omega1': float(y[1]),
            'theta2': float(y[2]),
            'omega2': float(y[3]),
            'wind': float(v_wind_curr),
            'frame': frame_count
        }

        if sim_mode in ['3d', 'dual']:
            frame['phi'] = float(y[4] if len(y) > 4 else 0.0)

        results['frames'].append(frame)

        # Store frame every 10 seconds
        if frame_count % 10 == 0:
            sim_state['progress'] = int((t / duration_s) * 100)
            sim_state['current_frame'] = frame_count
            sim_queue.put(frame)

        t = t_next
        frame_count += 1

    # Calculate summary
    if pend.P_upper_hist:
        P_system = (np.array(pend.P_upper_hist) + np.array(pend.P_lower_hist) +
                   np.array(pend.P_shaft_hist) * 0.95 * 0.88) * S.n_pendulums / 1000.0
        results['summary'] = {
            'avg_kW': float(np.mean(P_system)),
            'peak_kW': float(np.max(P_system)),
            'energy_kWh': float(np.trapz(P_system) / 3600.0 if len(P_system) > 1 else 0),
            'coil_Tmax_C': float(np.max(pend.T_coil_hist) - 273.15) if pend.T_coil_hist else 0,
        }

    return results

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
