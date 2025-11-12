# MSSDPPG v3 - Scenario Parameters

Complete parameter extraction for all predefined scenarios. All parameters are fully variable and non-hardcoded.

---

## 1. **4×40ft Container System**

### Basic Geometry
```python
name = "4×40ft Container"
L1 = 2.0 m                    # Upper arm length
L2 = 2.0 m                    # Lower arm length
L1_L2_ratio = 1.0             # Ratio (1:1, not enforced)
```

### Mass Distribution
```
m_upper_arm = 5.0 kg          # Upper arm structural mass
m_middle = 30.0 kg            # Harvesting mass at Hinge₂
m_lower_arm = 3.0 kg          # Lower arm structural mass
m_tip = 5.0 kg                # Tip mass (no hinge)

Total system mass: 43.0 kg (per unit)
```

### Wind Interaction
```
vane_w = 1.0 m                # Vane width
vane_h = 2.0 m                # Vane height
Vane area: 2.0 m²
```

### Mechanical Limits & Envelope
```
max_angle_rad = ±55°          # Hard stops
container_w = 2.35 m          # Container width
container_h = 2.39 m          # Container height
bearing_mu = 0.015            # Bearing friction coefficient
drag_cd = 1.2                 # Aerodynamic drag coefficient
mech_loss = 0.03              # Mechanical loss fraction (3%)
```

### System Configuration
```
n_pendulums = 48              # Number of units (48×40ft containers)
expected_kw_at_6ms = 19.6 kW  # Design power reference
color = "#4ECDC4"             # Teal (UI color)
```

### Generator Specifications (Default)

Since not explicitly defined, uses default GeneratorSpec:

**Hinge₁ Generator** (Upper arm with gearbox):
```
k_t = 0.75 Nm/A               # Torque constant
R_coil = 0.45 Ω               # Coil resistance (25°C)
eff = 0.85                    # Electrical efficiency
Cth = 250 J/K                 # Thermal capacitance
Rth = 1.5 K/W                 # Thermal resistance
T_max = 423.15 K (150°C)      # Max temperature (derating limit)
i_high = 6.0 A                # High current (assist on)
i_low = 1.5 A                 # Low current (base damping)
rpm_min = 100 RPM             # Min RPM for engagement
rpm_max = 3000 RPM            # Max RPM rating
```

**Hinge₂ Generator** (Middle arm direct alternator):
```
k_t = 0.8 Nm/A                # Lower torque constant
R_coil = 0.6 Ω                # Higher resistance
eff = 0.87                    # Slightly higher efficiency
Cth = 200 J/K                 # Smaller thermal mass
Rth = 1.2 K/W
T_max = 423.15 K
i_high = 5.0 A
i_low = 1.2 A
rpm_min = 80 RPM
rpm_max = 2500 RPM
```

**Ground Alternator** (Mast with flywheel):
```
k_t = 1.5 Nm/A                # Higher torque constant (larger scale)
R_coil = 0.3 Ω                # Lower resistance
eff = 0.90                    # Best efficiency
Cth = 400 J/K                 # Larger thermal mass
Rth = 1.0 K/W
T_max = 423.15 K
i_high = 8.0 A
i_low = 2.0 A
rpm_min = 200 RPM
rpm_max = 4000 RPM
```

### Clutch Specifications

**Hinge₁ Clutch** (Bidirectional):
```
type = 'bidirectional'
engagement_threshold = 0.1 rad/s
disengagement_threshold = 0.05 rad/s
eff = 0.97                    # 3% loss
```

**Ground Clutch** (One-Way):
```
type = 'oneway'
engagement_threshold = 0.1 rad/s
disengagement_threshold = 0.0 rad/s
eff = 0.95                    # 5% loss
```

### Gearbox

```
ratio = 12.0                  # 12:1 speed increase
eff = 0.94                    # 6% loss
max_torque = 500.0 Nm         # Maximum transmitted torque
```

### Flywheel

```
inertia = 5.0 kg·m²           # Moment of inertia
rpm_nom = 1500 RPM            # Nominal speed
friction_coeff = 0.001        # Bearing friction coefficient
```

### Performance Summary
- **Scale**: 48 units per 4×40ft container stack
- **Total Power @ 6 m/s wind**: 19.6 kW (rated)
- **Configuration**: Multi-unit dense array
- **Control**: Adaptive damping (speed-proportional)
- **Gust Limit**: ±55° hard stops with electrical braking

---

## 2. **1×20ft Container System**

### Basic Geometry
```python
name = "1×20ft Container"
L1 = 1.4 m                    # Upper arm length
L2 = 1.4 m                    # Lower arm length
L1_L2_ratio = 1.0             # Ratio (1:1, not enforced)
```

### Mass Distribution
```
m_upper_arm = 2.45 kg         # Upper arm structural mass
m_middle = 14.7 kg            # Harvesting mass at Hinge₂
m_lower_arm = 1.47 kg         # Lower arm structural mass
m_tip = 2.45 kg               # Tip mass (no hinge)

Total system mass: 21.07 kg (per unit)
```

### Wind Interaction
```
vane_w = 0.7 m                # Vane width
vane_h = 1.4 m                # Vane height
Vane area: 0.98 m²            # ~50% of 4×40ft
```

### Mechanical Limits & Envelope
```
max_angle_rad = ±60°          # Slightly wider range (more flexibility)
container_w = 2.35 m          # Standard container width
container_h = 2.39 m          # Standard container height
bearing_mu = 0.012            # Lower bearing friction (smaller unit)
drag_cd = 1.2                 # Same aerodynamic drag
mech_loss = 0.025             # Lower mechanical loss (2.5%)
```

### System Configuration
```
n_pendulums = 24              # Number of units (24×20ft containers)
expected_kw_at_6ms = 1.79 kW  # Design power reference
color = "#95E1D3"             # Mint green (UI color)
```

### Generator Specifications

Use same defaults as 4×40ft, but scaled for smaller arm lengths:

**Hinge₁ Generator**:
```
k_t = 0.75 Nm/A
R_coil = 0.45 Ω
eff = 0.85
(... all other parameters same as 4×40ft)
```

**Hinge₂ Generator**:
```
(Same as 4×40ft defaults)
```

**Ground Alternator**:
```
(Same as 4×40ft defaults)
```

### Clutch Specifications
```
Same as 4×40ft:
- Hinge₁: bidirectional, thresholds 0.1/0.05 rad/s, eff 0.97
- Ground: oneway, threshold 0.1 rad/s, eff 0.95
```

### Gearbox
```
ratio = 12.0                  # Same ratio (keeps proportions)
eff = 0.94
max_torque = 500.0 Nm
```

### Flywheel
```
inertia = 5.0 kg·m²           # Same inertia (scaled with system)
rpm_nom = 1500 RPM
friction_coeff = 0.001
```

### Performance Summary
- **Scale**: 24 units per installation
- **Total Power @ 6 m/s wind**: 1.79 kW (rated)
- **Configuration**: Medium-density compact array
- **Arm Length**: 70% of 4×40ft (scaled system)
- **Mass**: 49% of 4×40ft unit mass
- **Control**: Adaptive damping
- **Gust Limit**: ±60° (slightly more flexible)

---

## 3. **Tower Facade System** (Cantilever)

### Basic Geometry
```python
name = "Tower Facade"
L1 = 0.75 m                   # Upper arm length
L2 = 0.75 m                   # Lower arm length
L1_L2_ratio = 1.0             # Ratio (1:1, not enforced)
```

### Mass Distribution
```
m_upper_arm = 0.28 kg         # Upper arm structural mass (light)
m_middle = 7.5 kg             # Harvesting mass at Hinge₂ (small)
m_lower_arm = 0.17 kg         # Lower arm structural mass (very light)
m_tip = 1.25 kg               # Tip mass (no hinge)

Total system mass: 9.2 kg (per unit) - Lightest system
```

### Wind Interaction
```
vane_w = 0.4 m                # Vane width (smallest)
vane_h = 0.75 m               # Vane height
Vane area: 0.3 m²             # 15% of 4×40ft vane
```

### Mechanical Limits & Envelope
```
max_angle_rad = ±65°          # Widest range (high compliance)
container_w = 1.5 m           # Smallest envelope (building mounted)
container_h = 2.5 m           # Vertical height (tower optimized)
bearing_mu = 0.010            # Lowest bearing friction (precision)
drag_cd = 1.2                 # Same drag coefficient
mech_loss = 0.02              # Lowest loss (2% - precision system)
```

### System Configuration
```
n_pendulums = 8               # Fewest units (sparse facade integration)
expected_kw_at_6ms = 0.684 kW # Design power reference (smallest)
color = "#F38181"             # Coral red (UI color)
```

### Generator Specifications

Use same defaults, optimized for small scale:

**Hinge₁ Generator**:
```
k_t = 0.75 Nm/A
R_coil = 0.45 Ω
eff = 0.85
(... same as others)
```

**Hinge₂ Generator**:
```
(Same default specs)
```

**Ground Alternator**:
```
(Same default specs)
```

### Clutch Specifications
```
Same as other scenarios:
- Hinge₁: bidirectional, 0.1/0.05 rad/s, eff 0.97
- Ground: oneway, 0.1 rad/s, eff 0.95
```

### Gearbox
```
ratio = 12.0                  # Same ratio for consistency
eff = 0.94
max_torque = 500.0 Nm
```

### Flywheel
```
inertia = 5.0 kg·m²           # Same inertia
rpm_nom = 1500 RPM
friction_coeff = 0.001
```

### Performance Summary
- **Scale**: 8 units (minimal footprint)
- **Total Power @ 6 m/s wind**: 0.684 kW (smallest)
- **Configuration**: Building facade integration
- **Arm Length**: 37.5% of 4×40ft (smallest, lightest)
- **Mass**: 21% of 4×40ft unit mass
- **Envelope**: Optimized for vertical mounting
- **Control**: Adaptive damping
- **Gust Limit**: ±65° (most forgiving - for wind gusts on buildings)

---

## **Comparative Analysis**

### Geometry Scaling

| Parameter | 4×40ft | 1×20ft | Tower |
|-----------|--------|--------|-------|
| **L1** | 2.0 m | 1.4 m | 0.75 m |
| **L2** | 2.0 m | 1.4 m | 0.75 m |
| **Ratio** | 1:1 | 1:1 | 1:1 |
| **Vane Area** | 2.0 m² | 0.98 m² | 0.3 m² |

### Mass Distribution

| Component | 4×40ft | 1×20ft | Tower |
|-----------|--------|--------|-------|
| **Upper Arm** | 5.0 kg | 2.45 kg | 0.28 kg |
| **Harvesting** | 30.0 kg | 14.7 kg | 7.5 kg |
| **Lower Arm** | 3.0 kg | 1.47 kg | 0.17 kg |
| **Tip** | 5.0 kg | 2.45 kg | 1.25 kg |
| **Total** | 43.0 kg | 21.07 kg | 9.2 kg |

### Mechanical Properties

| Property | 4×40ft | 1×20ft | Tower |
|----------|--------|--------|-------|
| **Max Angle** | ±55° | ±60° | ±65° |
| **Bearing μ** | 0.015 | 0.012 | 0.010 |
| **Mech Loss** | 3.0% | 2.5% | 2.0% |

### System Configuration

| Metric | 4×40ft | 1×20ft | Tower |
|--------|--------|--------|-------|
| **Units** | 48 | 24 | 8 |
| **Power @ 6m/s** | 19.6 kW | 1.79 kW | 0.684 kW |
| **Power per Unit** | 409 W | 75 W | 86 W |

### Characteristics by Application

**4×40ft Container System:**
- ✅ Highest density (48 units)
- ✅ Largest total power output
- ✅ Robust industrial design
- ✅ Moderate arm length (2m)
- ✅ Heavy harvesting mass (30 kg)
- **Use Case**: Shipping container stacks, industrial facilities

**1×20ft Container System:**
- ✅ Medium density (24 units)
- ✅ Balanced performance
- ✅ Moderate arm length (1.4m)
- ✅ Flexible configuration (±60°)
- ✅ Lower bearing friction
- **Use Case**: Smaller container arrangements, modular deployments

**Tower Facade System:**
- ✅ Minimal footprint (8 units sparse)
- ✅ Lightest structure (9.2 kg)
- ✅ Precision bearings (μ=0.010)
- ✅ Most forgiving limits (±65°)
- ✅ Building-integrated design
- ✅ Lowest mechanical losses (2%)
- **Use Case**: Building facades, high-rise integration, aesthetic installations

---

## **How to Customize Parameters**

### Method 1: Modify Existing Scenario

```python
from MSSDPPG_Parameterized import SCENARIOS

# Access and modify
scenario = SCENARIOS['4x40ft']
scenario.bearing_mu = 0.02      # Increase friction
scenario.m_middle = 35.0        # Heavier harvesting mass
scenario.max_angle_rad = np.deg2rad(50)  # Tighter limits
```

### Method 2: Create New Scenario Based on Existing

```python
from MSSDPPG_Parameterized import Scenario, SCENARIOS

# Clone and modify
new_scenario = Scenario(
    **vars(SCENARIOS['1x20ft'])  # Copy all parameters
)
new_scenario.name = "Modified 1×20ft"
new_scenario.n_pendulums = 30    # More units
new_scenario.bearing_mu = 0.008  # Better bearings
```

### Method 3: Add Full Custom Generator Specs

```python
from MSSDPPG_Parameterized import (
    Scenario, GeneratorSpec, ClutchSpec,
    GearboxSpec, FlywheelSpec
)

CUSTOM_TOWER = Scenario(
    name="Custom Tower System",
    L1=0.75, L2=0.75,
    m_upper_arm=0.28, m_middle=7.5, m_lower_arm=0.17, m_tip=1.25,
    vane_w=0.4, vane_h=0.75,
    max_angle_rad=np.deg2rad(65),
    bearing_mu=0.010,
    drag_cd=1.2,
    mech_loss=0.02,
    n_pendulums=8,

    # Custom generators
    gen_hinge1=GeneratorSpec(
        k_t=0.8, R_coil=0.4, eff=0.86,
        i_high=6.5, i_low=1.6,
        rpm_min=120, rpm_max=3200
    ),
    gen_hinge2=GeneratorSpec(
        k_t=0.85, R_coil=0.55, eff=0.88,
        i_high=5.5, i_low=1.3,
        rpm_min=90, rpm_max=2600
    ),
    gen_ground=GeneratorSpec(
        k_t=1.6, R_coil=0.28, eff=0.91,
        i_high=8.5, i_low=2.1,
        rpm_min=250, rpm_max=4100
    ),

    # Custom clutches
    clutch_hinge1=ClutchSpec(
        type='bidirectional',
        engagement_threshold=0.12,
        disengagement_threshold=0.07,
        eff=0.98
    ),

    # Custom gearbox & flywheel
    gearbox=GearboxSpec(ratio=13.0, eff=0.95, max_torque=450.0),
    flywheel=FlywheelSpec(inertia=4.5, rpm_nom=1500, friction_coeff=0.0008),

    expected_kw_at_6ms=0.75,
    color="#FF5733"
)
```

---

## **Running Simulations with These Parameters**

### Command Line

```bash
# 4×40ft container system
python3 MSSDPPG_Parameterized.py --scenario 4x40ft --duration 6

# 1×20ft container system
python3 MSSDPPG_Parameterized.py --scenario 1x20ft --duration 12

# Tower facade system
python3 MSSDPPG_Parameterized.py --scenario tower --duration 6
```

### Python API

```python
from MSSDPPG_Parameterized import run_simulation

# Run each scenario
for scenario_key in ['4x40ft', '1x20ft', 'tower', 'mega']:
    results, t, th1, th2, P = run_simulation(
        scenario_key=scenario_key,
        duration_h=6,
        control_mode='adaptive',
        assist=True
    )

    print(f"\n{scenario_key.upper()}")
    print(f"  Total Power: {results['P_avg_kW']:.2f} kW avg")
    print(f"  Hinge₁: {results['P_h1_avg']:.3f} kW")
    print(f"  Hinge₂: {results['P_h2_avg']:.3f} kW")
    print(f"  Ground: {results['P_gnd_avg']:.3f} kW")
    print(f"  Energy: {results['E_kWh']:.2f} kWh")
    print(f"  Coil Tmax: {results['coil_T_max']:.1f}°C")
```

---

## **Summary: All Parameters Variable**

| Aspect | Configurable | Location |
|--------|-------------|----------|
| **Geometry** | L1, L2, ratio | Scenario |
| **Masses** | m_upper, m_middle, m_lower, m_tip | Scenario |
| **Wind** | vane_w, vane_h | Scenario |
| **Limits** | max_angle_rad | Scenario |
| **Friction** | bearing_mu | Scenario |
| **Generators** | k_t, R_coil, eff, Cth, Rth, i_high, i_low, rpm_min, rpm_max | GeneratorSpec (×3) |
| **Clutches** | type, engagement threshold, eff | ClutchSpec (×2) |
| **Gearbox** | ratio, eff, max_torque | GearboxSpec |
| **Flywheel** | inertia, friction | FlywheelSpec |
| **Control** | k coefficients in adaptive_current_control() | Pendulum2D method |

**NOTHING is hardcoded!** Every parameter can be modified for your specific design.
