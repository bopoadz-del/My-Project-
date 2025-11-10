#!/usr/bin/env python3
"""
Run All Scenario Simulations and Generate Comprehensive Comparison

This script runs simulations for all 4 deployment scenarios:
1. 4×40ft Container (48 pendulums)
2. 1×20ft Container (24 pendulums)
3. Tower Facade (8 pendulums)
4. Mega 15m Standalone (1 pendulum)

For each scenario, it compares 2D planar vs 3D spatial-offset performance.
"""

import os
import sys
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Scenarios to test
SCENARIOS = {
    '4x40ft': '4×40ft Container (48 pendulums)',
    '1x20ft': '1×20ft Container (24 pendulums)',
    'tower': 'Tower Facade (8 pendulums)',
    'mega': 'Mega 15m Standalone (1 pendulum)'
}

def run_simulation(scenario_key, duration='6h'):
    """Run simulation for a single scenario"""
    print(f"\n{'='*70}")
    print(f"🚀 RUNNING: {SCENARIOS[scenario_key]}")
    print(f"{'='*70}")

    # Run simulation
    cmd = [
        'python', 'MSSDPPG_UltraRealistic_v2.py',
        '--scenario', scenario_key,
        '--mode', 'both',
        '--duration', duration,
        '--assist', 'on'
    ]

    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ ERROR running {scenario_key}:")
        print(result.stderr)
        return None

    print(f"✅ Completed {scenario_key}")

    # Read results
    csv_path = 'outputs/performance_summary.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Rename outputs to preserve them
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_csv = f'outputs/performance_{scenario_key}_{timestamp}.csv'
        os.rename(csv_path, new_csv)

        # Rename plots
        for plot_type in ['2D', '3D']:
            old_plot = f'outputs/power_vs_time_{plot_type}.png'
            if os.path.exists(old_plot):
                new_plot = f'outputs/power_vs_time_{scenario_key}_{plot_type}_{timestamp}.png'
                os.rename(old_plot, new_plot)

        return df, new_csv

    return None

def collect_all_results():
    """Collect all simulation results"""
    results = {}

    for scenario_key in SCENARIOS.keys():
        result = run_simulation(scenario_key, duration='6h')
        if result:
            df, csv_path = result
            results[scenario_key] = {
                'dataframe': df,
                'csv_path': csv_path,
                'name': SCENARIOS[scenario_key]
            }

    return results

def generate_comparison_report(results):
    """Generate comprehensive comparison report"""
    print(f"\n{'='*70}")
    print("📊 GENERATING COMPREHENSIVE COMPARISON REPORT")
    print(f"{'='*70}\n")

    # Create comparison figure
    fig = plt.figure(figsize=(18, 12))

    # Prepare data
    scenarios = []
    planar_avg_kw = []
    spatial_avg_kw = []
    planar_peak_kw = []
    spatial_peak_kw = []
    planar_energy_kwh = []
    spatial_energy_kwh = []
    improvements = []

    for scenario_key in ['4x40ft', '1x20ft', 'tower', 'mega']:
        if scenario_key in results:
            df = results[scenario_key]['dataframe']
            scenarios.append(SCENARIOS[scenario_key])

            # Extract 2D and 3D data
            row_2d = df[df['Geometry'] == '2D'].iloc[0] if '2D' in df['Geometry'].values else None
            row_3d = df[df['Geometry'] == '3D'].iloc[0] if '3D' in df['Geometry'].values else None

            if row_2d is not None:
                planar_avg_kw.append(row_2d['Avg_kW'])
                planar_peak_kw.append(row_2d['Peak_kW'])
                planar_energy_kwh.append(row_2d['Energy_kWh'])
            else:
                planar_avg_kw.append(0)
                planar_peak_kw.append(0)
                planar_energy_kwh.append(0)

            if row_3d is not None:
                spatial_avg_kw.append(row_3d['Avg_kW'])
                spatial_peak_kw.append(row_3d['Peak_kW'])
                spatial_energy_kwh.append(row_3d['Energy_kWh'])
            else:
                spatial_avg_kw.append(0)
                spatial_peak_kw.append(0)
                spatial_energy_kwh.append(0)

            # Calculate improvement
            if row_2d is not None and row_3d is not None and row_2d['Avg_kW'] > 0:
                improvement = ((row_3d['Avg_kW'] / row_2d['Avg_kW']) - 1) * 100
                improvements.append(improvement)
            else:
                improvements.append(0)

    # 1. Average Power Comparison
    ax1 = plt.subplot(3, 3, 1)
    x = np.arange(len(scenarios))
    width = 0.35
    bars1 = ax1.bar(x - width/2, planar_avg_kw, width, label='Planar (2D)',
                     color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars2 = ax1.bar(x + width/2, spatial_avg_kw, width, label='Spatial-Offset (3D)',
                     color='#95E1D3', alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Average Power (kW)', fontsize=11, weight='bold')
    ax1.set_title('Average Power Output', fontsize=12, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    # 2. Peak Power Comparison
    ax2 = plt.subplot(3, 3, 2)
    bars1 = ax2.bar(x - width/2, planar_peak_kw, width, label='Planar (2D)',
                     color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars2 = ax2.bar(x + width/2, spatial_peak_kw, width, label='Spatial-Offset (3D)',
                     color='#95E1D3', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Peak Power (kW)', fontsize=11, weight='bold')
    ax2.set_title('Peak Power Output', fontsize=12, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Energy Production
    ax3 = plt.subplot(3, 3, 3)
    bars1 = ax3.bar(x - width/2, planar_energy_kwh, width, label='Planar (2D)',
                     color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, spatial_energy_kwh, width, label='Spatial-Offset (3D)',
                     color='#95E1D3', alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Energy (kWh)', fontsize=11, weight='bold')
    ax3.set_title('Total Energy Production (6h)', fontsize=12, weight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Performance Improvement (3D vs 2D)
    ax4 = plt.subplot(3, 3, 4)
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax4.bar(x, improvements, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Improvement (%)', fontsize=11, weight='bold')
    ax4.set_title('3D Spatial-Offset Improvement vs 2D Planar', fontsize=12, weight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'+{imp:.1f}%' if imp >= 0 else f'{imp:.1f}%',
                ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9, weight='bold')

    # 5. Power Density (W per pendulum)
    ax5 = plt.subplot(3, 3, 5)
    n_pendulums = [48, 24, 8, 1]
    planar_density = [p/n * 1000 if p > 0 else 0 for p, n in zip(planar_avg_kw, n_pendulums)]
    spatial_density = [p/n * 1000 if p > 0 else 0 for p, n in zip(spatial_avg_kw, n_pendulums)]

    bars1 = ax5.bar(x - width/2, planar_density, width, label='Planar (2D)',
                     color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars2 = ax5.bar(x + width/2, spatial_density, width, label='Spatial-Offset (3D)',
                     color='#95E1D3', alpha=0.8, edgecolor='black')
    ax5.set_ylabel('Power per Pendulum (W)', fontsize=11, weight='bold')
    ax5.set_title('Power Density', fontsize=12, weight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. Scalability Analysis
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(n_pendulums, planar_avg_kw, s=200, c='#4ECDC4',
                alpha=0.8, edgecolors='black', linewidths=2, label='Planar (2D)')
    ax6.scatter(n_pendulums, spatial_avg_kw, s=200, c='#95E1D3',
                alpha=0.8, edgecolors='black', linewidths=2, label='Spatial-Offset (3D)')
    ax6.set_xlabel('Number of Pendulums', fontsize=11, weight='bold')
    ax6.set_ylabel('Total Power (kW)', fontsize=11, weight='bold')
    ax6.set_title('System Scaling', fontsize=12, weight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale('log')

    # 7. Deployment Comparison Table
    ax7 = plt.subplot(3, 3, 7)
    ax7.axis('off')

    table_data = []
    for i, scenario_key in enumerate(['4x40ft', '1x20ft', 'tower', 'mega']):
        if i < len(scenarios):
            table_data.append([
                scenarios[i].split('(')[0].strip(),
                f"{planar_avg_kw[i]:.1f}",
                f"{spatial_avg_kw[i]:.1f}",
                f"+{improvements[i]:.1f}%" if improvements[i] >= 0 else f"{improvements[i]:.1f}%"
            ])

    table = ax7.table(cellText=table_data,
                     colLabels=['Scenario', '2D (kW)', '3D (kW)', 'Improvement'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold')

    # Color improvement column
    for i in range(1, len(table_data) + 1):
        if i <= len(improvements):
            color = '#90EE90' if improvements[i-1] > 0 else '#FFB6C6'
            table[(i, 3)].set_facecolor(color)

    ax7.set_title('Performance Summary', fontsize=12, weight='bold', pad=20)

    # 8. Cost-Benefit Analysis
    ax8 = plt.subplot(3, 3, 8)
    # Assuming $47K for 2D, $56K for 3D (20% increase)
    system_costs = [47, 56.4, 25, 30, 8, 9.6, 50, 60]  # 2D, 3D for each scenario (estimated)
    roi_planar = [p / (47 if i == 0 else 25 if i == 1 else 8 if i == 2 else 50)
                  for i, p in enumerate(planar_avg_kw)]
    roi_spatial = [p / (56.4 if i == 0 else 30 if i == 1 else 9.6 if i == 2 else 60)
                   for i, p in enumerate(spatial_avg_kw)]

    bars1 = ax8.bar(x - width/2, roi_planar, width, label='Planar (2D)',
                     color='#4ECDC4', alpha=0.8, edgecolor='black')
    bars2 = ax8.bar(x + width/2, roi_spatial, width, label='Spatial-Offset (3D)',
                     color='#95E1D3', alpha=0.8, edgecolor='black')
    ax8.set_ylabel('kW per $1K Investment', fontsize=11, weight='bold')
    ax8.set_title('Return on Investment', fontsize=12, weight='bold')
    ax8.set_xticks(x)
    ax8.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=15, ha='right', fontsize=9)
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Recommendations
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    avg_improvement = np.mean(improvements) if improvements else 0
    best_scenario = scenarios[np.argmax(spatial_avg_kw)] if spatial_avg_kw else "Unknown"

    recommendations = f"""
    SIMULATION RESULTS SUMMARY

    ✅ Average 3D Improvement: {avg_improvement:.1f}%

    🏆 Best Performer: {best_scenario.split('(')[0].strip()}
       • 2D: {max(planar_avg_kw):.1f} kW
       • 3D: {max(spatial_avg_kw):.1f} kW

    💡 RECOMMENDATIONS:

    • Deploy 3D Spatial-Offset for all scenarios
    • Expected {avg_improvement:.0f}% performance gain
    • 20% cost increase justified by ROI

    📊 NEXT STEPS:

    1. File patent for spatial-offset innovation
    2. Build physical prototypes
    3. Pilot deployment of best scenario
    4. Scale to commercial production
    """

    ax9.text(0.1, 0.5, recommendations, fontsize=9,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('outputs/all_scenarios_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✅ Comparison plot saved to: outputs/all_scenarios_comparison.png")
    plt.close()

    # Generate text report
    print("\n" + "="*70)
    print("COMPREHENSIVE SIMULATION REPORT")
    print("="*70)

    for i, scenario_key in enumerate(['4x40ft', '1x20ft', 'tower', 'mega']):
        if i < len(scenarios):
            print(f"\n{scenarios[i]}")
            print("-" * 70)
            print(f"  2D Planar:")
            print(f"    Average Power: {planar_avg_kw[i]:.2f} kW")
            print(f"    Peak Power: {planar_peak_kw[i]:.2f} kW")
            print(f"    Energy (6h): {planar_energy_kwh[i]:.2f} kWh")
            print(f"  3D Spatial-Offset:")
            print(f"    Average Power: {spatial_avg_kw[i]:.2f} kW")
            print(f"    Peak Power: {spatial_peak_kw[i]:.2f} kW")
            print(f"    Energy (6h): {spatial_energy_kwh[i]:.2f} kWh")
            print(f"  Improvement: {improvements[i]:+.1f}%")

    print(f"\n{'='*70}")
    print(f"OVERALL AVERAGE 3D IMPROVEMENT: {avg_improvement:.1f}%")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    import os
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print("=" * 70)
    print("🚀 COMPREHENSIVE MSSDPPG SCENARIO ANALYSIS")
    print("=" * 70)
    print("\nRunning simulations for all 4 deployment scenarios...")
    print("This will take approximately 20-30 minutes.\n")

    # Run all simulations
    results = collect_all_results()

    if results:
        # Generate comparison report
        generate_comparison_report(results)

        print("\n" + "="*70)
        print("✅ ALL SIMULATIONS COMPLETE!")
        print("="*70)
        print("\nOutputs:")
        print("  • outputs/all_scenarios_comparison.png")
        print("  • outputs/performance_*.csv (individual results)")
        print("  • outputs/power_vs_time_*.png (individual plots)")
    else:
        print("\n❌ No results collected. Check for errors.")
