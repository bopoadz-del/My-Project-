# MSSDPPG Ultra-Realistic v2

Dual-geometry (2D Planar + 3D Spatial-Offset) long-run simulator for Modularized Self‑Sustained Double Pendulum Power Generation (MSSDPPG).

## Features
- **Geometry modes**: Planar (2D) baseline, Spatial Offset (3D) optional
- **Control modes**: Lock–Release (default), Magnetic Push–Pull
- **Assist toggle**: On/Off to test natural decay (no sustainability aid)
- **Wind profile**: Standard 0–20 m/s sinusoid + noise, or CSV import
- **Durations**: 6h (default) or 12h endurance
- **Solver**: LSODA adaptive with hourly chunking for stability
- **Outputs**: PNG plots + CSV summary in `outputs/`, log in `logs/`

## Quick Start
```bash
pip install -r requirements.txt
python MSSDPPG_UltraRealistic_v2.py
```

## CLI Options
```bash
# Default 6h planar, assist ON, lock-release, standard wind (0–20 m/s)
python MSSDPPG_UltraRealistic_v2.py

# 12h endurance
python MSSDPPG_UltraRealistic_v2.py --duration 12h

# Spatial-only
python MSSDPPG_UltraRealistic_v2.py --mode spatial

# Dual run (2D + 3D)
python MSSDPPG_UltraRealistic_v2.py --mode both

# Disable assist (natural decay / no sustain)
python MSSDPPG_UltraRealistic_v2.py --assist off

# Use CSV wind profile (two columns: time_seconds, wind_mps)
python MSSDPPG_UltraRealistic_v2.py --windfile examples/wind_profile_standard.csv

# Switch control to Magnetic Push–Pull
python MSSDPPG_UltraRealistic_v2.py --control pushpull
```

## Outputs
- `outputs/performance_summary.csv`
- `outputs/power_vs_time_2D.png` (if 2D run)
- `outputs/power_vs_time_3D.png` (if 3D run)
- `outputs/efficiency_comparison.png` (dual)
- `outputs/phi_amplitude_vs_time.png` (3D)
- `logs/run_YYYYMMDD_HHMM.txt`

## Notes
- Baseline uses rubric-correct fixed middle/tip masses per scale, arm masses ~ L², coil k_t ~ L².
- Spatial adds φ axis with torsional spring/damper and gyroscopic coupling.
- This package prioritizes numerical stability for long endurance runs.
