"""
MSSDPPG Web UI with Real-time 3D Visualization
Flask backend for v4 modular multi-pendulum simulator
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import os
import json
import threading
from datetime import datetime
import time

# Import v4 modular simulator
from MSSDPPG_Modular_v4 import (
    SCENARIOS, run_simulation
)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'mssdppg-secret-key'

# Global simulation state
sim_state = {
    'running': False,
    'progress': 0,
    'current_result': None,
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/scenarios')
def get_scenarios():
    """Get available scenarios with modular parameters"""
    return jsonify({
        name: {
            'display_name': scenario.name,
            'L1': float(scenario.L1),
            'L2': float(scenario.L2),
            'n_pendulums_default': scenario.n_pendulums,
            'n_pendulums_min': 1,
            'n_pendulums_max': scenario.n_pendulums * 2,  # Allow scaling up
            'color': scenario.color,
            'total_generators_default': 2 * scenario.n_pendulums + 2,
        }
        for name, scenario in SCENARIOS.items()
    })

@app.route('/api/config')
def get_config():
    """Get simulation configuration options"""
    return jsonify({
        'durations': [1, 2, 3, 6, 12],
        'controls': ['adaptive'],
        'assist': ['on', 'off'],
        'scenarios': list(SCENARIOS.keys())
    })

@app.route('/api/simulate', methods=['POST'])
def start_simulation():
    """Start a new modular simulation"""
    global sim_state

    if sim_state['running']:
        return jsonify({'error': 'Simulation already running'}), 400

    params = request.json
    sim_state['running'] = True
    sim_state['progress'] = 0

    # Run simulation in background thread
    thread = threading.Thread(
        target=run_simulation_background,
        args=(params,),
        daemon=True
    )
    thread.start()

    return jsonify({'status': 'started'})

@app.route('/api/sim-status')
def sim_status():
    """Get current simulation status"""
    return jsonify({
        'running': sim_state['running'],
        'progress': sim_state['progress'],
    })

@app.route('/api/sim-result')
def sim_result_endpoint():
    """Get simulation results"""
    if sim_state['current_result']:
        return jsonify(sim_state['current_result'])
    return jsonify({'error': 'No results available'}), 404

def run_simulation_background(params):
    """Run modular simulation in background"""
    global sim_state

    try:
        scenario_key = params.get('scenario', 'tower')
        duration_h = int(params.get('duration', 6))
        n_pendulums = int(params.get('n_pendulums', None) or SCENARIOS[scenario_key].n_pendulums)
        assist = params.get('assist', 'on') == 'on'

        # Update progress
        sim_state['progress'] = 10

        # Run simulation
        results, t, y, P = run_simulation(
            scenario_key=scenario_key,
            duration_h=duration_h,
            n_pendulums=n_pendulums,
            control_mode='adaptive',
            assist=assist
        )

        # Format results for UI
        formatted_results = {
            'scenario': scenario_key,
            'n_pendulums': results['n_pendulums'],
            'total_generators': 2 * results['n_pendulums'] + 2,
            'duration_h': duration_h,

            # Power breakdown
            'power': {
                'total_avg_kW': results['P_avg_kW'],
                'total_peak_kW': results['P_peak_kW'],
                'energy_kWh': results['E_kWh'],

                # Per-generator breakdown
                'hinge1_avg_kW': results['P_hinge1_avg'],
                'hinge2_avg_kW': results['P_hinge2_avg'],
                'ground_fw_avg_kW': results['P_ground_fw_avg'],
                'ground_rv_avg_kW': results['P_ground_rv_avg'],
            },

            # Thermal
            'coil_temp_max_C': results['coil_T_max'],

            # Flywheel
            'flywheel_fw_rpm_avg': results['rpm_fw_avg'],
            'flywheel_rv_rpm_avg': results['rpm_rv_avg'],

            # Metadata
            'timestamp': datetime.now().isoformat(),
        }

        sim_state['current_result'] = formatted_results
        sim_state['progress'] = 100

    except Exception as e:
        sim_state['current_result'] = {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        print(f"Simulation error: {e}")

    finally:
        sim_state['running'] = False

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
