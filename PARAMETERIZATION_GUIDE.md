# MSSDPPG v3 - Full Parameterization Guide

## Overview

The parameterized simulator (`MSSDPPG_Parameterized.py`) implements the complete three-generator architecture with explicit models for:

- **Three Power Take-Off (PTO) points**
  - Hinge₁ generator (upper arm joint) with bidirectional clutch + gearbox
  - Hinge₂ generator (middle joint) with direct PM alternator
  - Ground alternator (mast base) with one-way clutch + flywheel

- **Advanced Control**
  - Bidirectional clutch with engagement/disengagement logic
  - One-way clutch for ground alternator (prevents back-driving)
  - Adaptive current damping (I = k|ω|)
  - Gust protection (hard stops at θ_max)

- **Complete Parametrization**
  - All geometric parameters variable
  - All mass distributions configurable
  - Full generator specifications (torque constant, resistance, efficiency)
  - Clutch and flywheel parameters

## Scenario Definition

### Basic Geometry Parameters

```python
Scenario(
    name = "Mega 15 m",           # Display name
    L1 = 12.0,                    # Upper arm length (m)
    L2 = 6.0,                     # Lower arm length (m)
    L1_L2_ratio = 2.0,            # Enforce L1 = 2×L2 (optional)

    # Masses (kg)
    m_upper_arm = 30.0,           # Upper arm structural mass
    m_middle = 100.0,             # Harvesting mass at Hinge₂
    m_lower_arm = 20.0,           # Lower arm structural mass
    m_tip = 20.0,                 # Tip mass (no hinge)

    # Wind interaction
    vane_w = 3.0,                 # Vane width (m)
    vane_h = 6.0,                 # Vane height (m)
)
```

### Mechanical Limits

```python
max_angle_rad = np.deg2rad(55)   # Hard stops: ±55°
bearing_mu = 0.020               # Bearing friction coefficient
drag_cd = 1.2                    # Aerodynamic drag coefficient
mech_loss = 0.04                 # Mechanical losses (fraction)
```

### Generator Specifications

Three independent generators, each with full specifications:

```python
GeneratorSpec(
    k_t = 1.2,                    # Torque constant (Nm/A)
    R_coil = 0.4,                 # Coil resistance (Ω)
    eff = 0.85,                   # Electrical efficiency
    Cth = 350.0,                  # Thermal capacitance (J/K)
    Rth = 1.2,                    # Thermal resistance (K/W)
    T_max = 423.15,               # Max temperature (K) = 150°C
    i_high = 7.0,                 # High current (assist on) (A)
    i_low = 1.8,                  # Low current (base damping) (A)
    rpm_min = 150,                # Min RPM for engagement
    rpm_max = 3500,               # Max RPM rating
)
```

**Typical Configuration:**

| Generator | Type | k_t | R_coil | eff | i_high | i_low | Notes |
|-----------|------|-----|--------|-----|--------|-------|-------|
| Hinge₁ | AC + gearbox | 1.2 | 0.4Ω | 85% | 7.0A | 1.8A | Via clutch + gearbox |
| Hinge₂ | Direct PM AC | 2.5 | 0.5Ω | 87% | 8.0A | 2.0A | High-torque, low-RPM |
| Ground | AC + flywheel | 1.8 | 0.35Ω | 92% | 10.0A | 2.5A | Via one-way clutch |

### Clutch Specifications

```python
ClutchSpec(
    type = 'bidirectional',       # or 'oneway'
    engagement_threshold = 0.15,  # Angular velocity to engage (rad/s)
    disengagement_threshold = 0.08, # Angular velocity to disengage
    eff = 0.98,                   # Transmission efficiency (98% = 2% loss)
)
```

**Usage:**
- **Hinge₁ clutch**: Bidirectional (transmits both directions)
  - Engages at |ω₁| > 0.15 rad/s
  - Disengages at |ω₁| < 0.08 rad/s
  - Prevents gearbox back-driving

- **Ground clutch**: One-way (only charging)
  - Transmits only positive torque (energy out)
  - Prevents flywheel back-driving motor
  - Always engaged if ω_ground > 0

### Gearbox Specification

```python
GearboxSpec(
    ratio = 15.0,                 # Gear ratio (15:1)
    eff = 0.94,                   # Efficiency (6% loss)
    max_torque = 800.0,           # Max transmitted torque (Nm)
)
```

**Purpose**: Keep Hinge₁ motor in optimal RPM range
- Low hinge speeds (0-100 RPM) → High alternator speeds (0-1500 RPM)
- Improves generator efficiency

### Flywheel Specification

```python
FlywheelSpec(
    inertia = 8.0,                # Moment of inertia (kg·m²)
    rpm_nom = 1500,               # Nominal speed
    friction_coeff = 0.0005,      # Bearing friction coefficient
)
```

**Purpose**: Smooth torque pulses from hinge motion
- Acts as mechanical energy buffer
- Reduces load on electrical system
- Inertia I = (1/2)·m·r²

## Mega 15m Configuration

The Mega 15m scenario exemplifies the full parameterization:

```python
MEGA_15M = Scenario(
    # === Geometry ===
    name="Mega 15 m",
    L1=12.0,  L2=6.0,           # 12m upper, 6m lower (L1/L2 = 2.0)

    # === Masses ===
    m_upper_arm=30.0,           # Structural
    m_middle=100.0,             # 120 kg harvesting mass (scaled)
    m_lower_arm=20.0,           # Structural
    m_tip=20.0,                 # Tip weight

    # === Wind ===
    vane_w=3.0, vane_h=6.0,     # Large wind-catching surface

    # === Limits ===
    max_angle_rad=np.deg2rad(55),  # ±55° hard stops
    bearing_mu=0.020,
    drag_cd=1.2,
    mech_loss=0.04,

    # === Three-Point PTO ===
    gen_hinge1=GeneratorSpec(
        k_t=1.2, R_coil=0.4, eff=0.85,
        i_high=7.0, i_low=1.8,
        rpm_min=150, rpm_max=3500
    ),
    gen_hinge2=GeneratorSpec(
        k_t=2.5, R_coil=0.5, eff=0.87,    # Direct PM: high k_t
        i_high=8.0, i_low=2.0,
        rpm_min=50, rpm_max=2000            # Lower RPM range
    ),
    gen_ground=GeneratorSpec(
        k_t=1.8, R_coil=0.35, eff=0.92,   # Flywheel smoothed
        i_high=10.0, i_low=2.5,
        rpm_min=300, rpm_max=4000
    ),

    # === Clutches ===
    clutch_hinge1=ClutchSpec(
        type='bidirectional',
        engagement_threshold=0.15,
        disengagement_threshold=0.08,
        eff=0.98
    ),
    clutch_ground=ClutchSpec(
        type='oneway',
        engagement_threshold=0.2,
        disengagement_threshold=0.0,
        eff=0.96
    ),

    # === Gearbox & Flywheel ===
    gearbox=GearboxSpec(ratio=15.0, eff=0.94, max_torque=800.0),
    flywheel=FlywheelSpec(inertia=8.0, rpm_nom=1500, friction_coeff=0.0005),
)
```

## Control Strategy

### Adaptive Damping Control

Current commands are computed as functions of angular velocity:

```
I₁(ω₁) = k₁·|ω₁|  (at Hinge₁)
I₂(ω₂) = k₂·|ω₂|  (at Hinge₂)
```

**Benefits:**
- Faster motion → Higher damping (more energy extraction)
- Slower motion → Lower damping (reduce stall torque)
- Self-regulating with wind speed variation

### Clutch Engagement Logic

**Hinge₁ Bidirectional Clutch:**
- Engages when |ω₁| > engagement_threshold
- Prevents gearbox back-driving during reversal
- Disengages for coast phases
- Hysteresis prevents chatter (min 0.3s engagement)

**Ground One-Way Clutch:**
- Only transmits positive torque (charging)
- Flywheel can spin up freely
- Prevents back-driving from load
- Automatic re-engagement when ω > 0

### Gust Protection

Hard stops at ±θ_max trigger rapid braking:
- At θ > ±55°: Return -25·ω acceleration (electrical braking)
- At θ > ±60°: Return -50·ω acceleration (emergency dump)

## Running Simulations

### Command Line

```bash
# Default: Mega 15m, 6h, adaptive control, assist on
python3 MSSDPPG_Parameterized.py

# Custom scenario
python3 MSSDPPG_Parameterized.py --scenario mega --duration 12 --control adaptive --assist on

# Other scenarios
python3 MSSDPPG_Parameterized.py --scenario 4x40ft --duration 6
python3 MSSDPPG_Parameterized.py --scenario 1x20ft --duration 6
python3 MSSDPPG_Parameterized.py --scenario tower --duration 6
```

### Output

```
============================================================
MSSDPPG v3 - Parameterized Multi-Generator Simulator
Scenario: mega
Duration: 6h | Control: adaptive | Assist: on
============================================================

Configuration: Mega 15 m
  Geometry: L₁=12.0m, L₂=6.0m (L₁/L₂=2.0)
  Masses: upper=30.0kg, mid=100.0kg, lower=20.0kg, tip=20.0kg
  Limits: θ_max=±55°, μ=0.02
  PTO: 3 generators (Hinge₁ + Hinge₂ + Ground)
  Flywheel: I=8.0 kg·m²

Results:
  Total Power (3-PTO): XX.XX kW avg, XX.XX kW peak
  Hinge₁ Gen: X.XX kW
  Hinge₂ Gen: X.XX kW
  Ground Gen: X.XX kW
  Energy: X.XX kWh over 6h
  Coil Tmax: XX.X°C
  Flywheel avg RPM: XXXX
```

## Key Features vs. Original

| Feature | Original v2 | Parameterized v3 |
|---------|------------|------------------|
| **Generators** | 2 (upper, lower) | 3 (upper, middle, ground) |
| **Clutches** | Implicit (lock-release) | Explicit (bidirectional + oneway) |
| **Gearbox** | Implicit | Explicit with ratio/efficiency |
| **Flywheel** | None | Explicit inertia model |
| **Control** | Lock-Release, Push-Pull | Adaptive current damping |
| **Parameterization** | Scenario-level | Per-component specs |
| **Geometric ratio** | Not enforced | Optional L1/L2 enforcement |

## Extending the Simulator

### Adding a New Scenario

```python
from MSSDPPG_Parameterized import Scenario, GeneratorSpec, ClutchSpec, FlywheelSpec, GearboxSpec

MY_SCENARIO = Scenario(
    name="My Custom System",
    L1=5.0, L2=2.5, L1_L2_ratio=2.0,
    m_upper_arm=20.0, m_middle=50.0, m_lower_arm=10.0, m_tip=15.0,
    vane_w=1.5, vane_h=3.0,
    max_angle_rad=np.deg2rad(50),
    # ... add generator, clutch, gearbox, flywheel specs
)

SCENARIOS['my_scenario'] = MY_SCENARIO
```

### Modifying Generator Specs

```python
# Scale all currents by 2x for higher power
my_scenario.gen_hinge1.i_high *= 2
my_scenario.gen_hinge1.i_low *= 2
```

### Changing Control Logic

Edit the `adaptive_current_control()` method in `Pendulum2D`:

```python
def adaptive_current_control(self, omega1, omega2, engaged_h1):
    # Custom control law
    i1 = 0.5 * abs(omega1) ** 0.8  # Non-linear damping
    i2 = 0.3 * abs(omega2) ** 0.8
    self.gen_h1.set_current(i1 if engaged_h1 else 0.0)
    self.gen_h2.set_current(i2)
```

## Notes

1. **Numerical Stability**: The large 12m arms create stiff equations. Solver tolerances are relaxed to `rtol=1e-4, atol=1e-6`.

2. **Mass Scaling**: For the Mega 15m, masses are slightly scaled for numerical stability while preserving inertial characteristics.

3. **Three-Generator Aggregation**: Total power = P_h1 + P_h2 + P_ground (scaled by n_pendulums).

4. **Flywheel Dynamics**: Ground alternator operates on flywheel inertia, smoothing torque ripple.

5. **Thermal Models**: Each generator independently tracks coil temperature with RC time constant.

## References

- **Bidirectional Clutch**: Allows free-wheeling in both directions with selective locking
- **One-Way Clutch**: Overrunning (anti-backdriving) - key for preventing motor mode
- **Flywheel Inertia**: Acts as mechanical low-pass filter on pulsatile torque
- **Adaptive Damping**: Proportional-to-speed current control without feedback sensors
