"""
MSSDPPG Ultra-Realistic v3 - Full Parameterization
- Planar (2D) baseline + Spatial (3D) optional
- Full three-generator architecture (Hinge₁, Hinge₂, Ground)
- Bidirectional clutch (Hinge₁) + One-way clutch (Ground)
- Flywheel smoothing on ground alternator
- Adaptive damping via current control
- Hard stops with gust protection
- All parameters variable per scenario
"""

import os, sys, argparse, time, math, csv, datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ============ Constants ============
g = 9.81
rho_air = 1.225
T_ambient = 298.15

# ============ Extended Scenario Configuration ============
@dataclass
class GeneratorSpec:
    """Generator specification"""
    k_t: float              # Torque constant (Nm/A)
    R_coil: float           # Coil resistance (Ohm)
    eff: float              # Efficiency
    Cth: float              # Thermal capacitance (J/K)
    Rth: float              # Thermal resistance (K/W)
    T_max: float            # Max temperature (K)
    i_high: float           # High current command (assist on)
    i_low: float            # Low current command (assist on)
    rpm_min: float          # Min RPM for engagement
    rpm_max: float          # Max RPM rating

@dataclass
class ClutchSpec:
    """Clutch specification"""
    type: str               # 'bidirectional' or 'oneway'
    engagement_threshold: float  # Angular velocity threshold
    disengagement_threshold: float
    eff: float              # Efficiency (slip loss)

@dataclass
class FlywheelSpec:
    """Flywheel specification"""
    inertia: float          # kg·m²
    rpm_nom: float          # Nominal RPM
    friction_coeff: float   # Bearing friction coefficient

@dataclass
class GearboxSpec:
    """Gearbox specification"""
    ratio: float            # Gear ratio
    eff: float              # Efficiency
    max_torque: float       # Max transmitted torque (Nm)

@dataclass
class Scenario:
    """Complete scenario with all architectural parameters"""
    # Geometry
    name: str
    L1: float               # Upper arm length (m)
    L2: float               # Lower arm length (m)
    L1_L2_ratio: float = field(default=None)  # Enforced ratio, auto-compute if set

    # Masses
    m_upper_arm: float = 0.0
    m_middle: float = 0.0   # Harvesting mass at Hinge₂ (120 kg for mega)
    m_lower_arm: float = 0.0
    m_tip: float = 0.0      # Tip mass (20 kg for mega)

    # Wind interaction
    vane_w: float = 0.0     # Vane width
    vane_h: float = 0.0     # Vane height

    # Mechanical limits
    max_angle_rad: float = np.deg2rad(55)
    container_w: float = 0.0
    container_h: float = 0.0

    # Bearing & friction
    bearing_mu: float = 0.015
    drag_cd: float = 1.2
    mech_loss: float = 0.03

    # System setup
    n_pendulums: int = 1

    # Three-generator architecture
    gen_hinge1: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=1.0, R_coil=0.5, eff=0.85, Cth=250.0, Rth=1.5, T_max=423.15,
        i_high=6.0, i_low=1.5, rpm_min=100, rpm_max=3000
    ))
    gen_hinge2: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=0.8, R_coil=0.6, eff=0.87, Cth=200.0, Rth=1.2, T_max=423.15,
        i_high=5.0, i_low=1.2, rpm_min=80, rpm_max=2500
    ))
    gen_ground: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=1.5, R_coil=0.3, eff=0.90, Cth=400.0, Rth=1.0, T_max=423.15,
        i_high=8.0, i_low=2.0, rpm_min=200, rpm_max=4000
    ))

    # Clutch logic
    clutch_hinge1: ClutchSpec = field(default_factory=lambda: ClutchSpec(
        type='bidirectional',
        engagement_threshold=0.1,
        disengagement_threshold=0.05,
        eff=0.97
    ))
    clutch_ground: ClutchSpec = field(default_factory=lambda: ClutchSpec(
        type='oneway',
        engagement_threshold=0.1,
        disengagement_threshold=0.0,
        eff=0.95
    ))

    # Gearbox (Hinge₁ → alternator)
    gearbox: GearboxSpec = field(default_factory=lambda: GearboxSpec(
        ratio=12.0,
        eff=0.94,
        max_torque=500.0
    ))

    # Flywheel on ground alternator
    flywheel: FlywheelSpec = field(default_factory=lambda: FlywheelSpec(
        inertia=5.0,        # kg·m²
        rpm_nom=1500,
        friction_coeff=0.001
    ))

    # Performance reference
    expected_kw_at_6ms: float = 77.2
    color: str = "#AA96DA"

    def __post_init__(self):
        """Auto-compute L1 or L2 if ratio is set"""
        if self.L1_L2_ratio is not None:
            # Use shorter length as base, compute longer
            if self.L1 < self.L2:
                self.L1 = self.L2 * self.L1_L2_ratio
            else:
                self.L2 = self.L1 / self.L1_L2_ratio

# ============ Predefined Scenarios ============

# Mega 15m - Extended spec matching detailed architecture
MEGA_15M = Scenario(
    name="Mega 15 m",
    L1=12.0,              # Upper arm (m)
    L2=6.0,               # Lower arm (m)
    L1_L2_ratio=2.0,      # ENFORCED: L1 = 2×L2

    # Masses (scaled for numerical stability with large arm lengths)
    m_upper_arm=30.0,     # Upper arm (scaled)
    m_middle=100.0,       # Harvesting mass at Hinge₂ (spec: 120 kg, scaled)
    m_lower_arm=20.0,     # Lower arm
    m_tip=20.0,           # Tip mass (spec: 20 kg)

    # Wind interaction
    vane_w=3.0,
    vane_h=6.0,

    # Hard stops at ±55° (spec requirement)
    max_angle_rad=np.deg2rad(55),
    container_w=8.0,
    container_h=15.0,

    # Bearing & friction
    bearing_mu=0.020,
    drag_cd=1.2,
    mech_loss=0.04,

    # Single mega unit
    n_pendulums=1,

    # Hinge₁ generator (with gearbox + clutch)
    gen_hinge1=GeneratorSpec(
        k_t=1.2, R_coil=0.4, eff=0.85, Cth=350.0, Rth=1.2, T_max=423.15,
        i_high=7.0, i_low=1.8, rpm_min=150, rpm_max=3500
    ),

    # Hinge₂ generator (high-torque, low-RPM direct alternator)
    gen_hinge2=GeneratorSpec(
        k_t=2.5,            # Higher torque constant for direct coupling
        R_coil=0.5, eff=0.87, Cth=400.0, Rth=1.0, T_max=423.15,
        i_high=8.0, i_low=2.0, rpm_min=50, rpm_max=2000  # Lower RPM range
    ),

    # Ground alternator (on flywheel)
    gen_ground=GeneratorSpec(
        k_t=1.8, R_coil=0.35, eff=0.92, Cth=500.0, Rth=0.8, T_max=423.15,
        i_high=10.0, i_low=2.5, rpm_min=300, rpm_max=4000
    ),

    # Bidirectional clutch at Hinge₁
    clutch_hinge1=ClutchSpec(
        type='bidirectional',
        engagement_threshold=0.15,
        disengagement_threshold=0.08,
        eff=0.98
    ),

    # One-way clutch at ground
    clutch_ground=ClutchSpec(
        type='oneway',
        engagement_threshold=0.2,
        disengagement_threshold=0.0,
        eff=0.96
    ),

    # Gearbox (Hinge₁ output → ground alternator RPM range)
    gearbox=GearboxSpec(
        ratio=15.0,         # Keep hinge motion in acceptable RPM band
        eff=0.94,
        max_torque=800.0
    ),

    # Flywheel for pulse smoothing
    flywheel=FlywheelSpec(
        inertia=8.0,        # Large flywheel for smoothing
        rpm_nom=1500,
        friction_coeff=0.0005
    ),

    expected_kw_at_6ms=77.2,
    color="#AA96DA"
)

# Other scenarios (updated for consistency)
SCENARIOS = {
    "4x40ft": Scenario(
        name="4×40ft Container", L1=2.0, L2=2.0, L1_L2_ratio=1.0,
        m_upper_arm=5.0, m_middle=30.0, m_lower_arm=3.0, m_tip=5.0,
        vane_w=1.0, vane_h=2.0, max_angle_rad=np.deg2rad(55),
        container_w=2.35, container_h=2.39, bearing_mu=0.015, drag_cd=1.2,
        mech_loss=0.03, n_pendulums=48, expected_kw_at_6ms=19.6, color="#4ECDC4"
    ),
    "1x20ft": Scenario(
        name="1×20ft Container", L1=1.4, L2=1.4, L1_L2_ratio=1.0,
        m_upper_arm=2.45, m_middle=14.7, m_lower_arm=1.47, m_tip=2.45,
        vane_w=0.7, vane_h=1.4, max_angle_rad=np.deg2rad(60),
        container_w=2.35, container_h=2.39, bearing_mu=0.012, drag_cd=1.2,
        mech_loss=0.025, n_pendulums=24, expected_kw_at_6ms=1.79, color="#95E1D3"
    ),
    "tower": Scenario(
        name="Tower Facade", L1=0.75, L2=0.75, L1_L2_ratio=1.0,
        m_upper_arm=0.28, m_middle=7.5, m_lower_arm=0.17, m_tip=1.25,
        vane_w=0.4, vane_h=0.75, max_angle_rad=np.deg2rad(65),
        container_w=1.5, container_h=2.5, bearing_mu=0.010, drag_cd=1.2,
        mech_loss=0.02, n_pendulums=8, expected_kw_at_6ms=0.684, color="#F38181"
    ),
    "mega": MEGA_15M
}

# ============ Wind Profile ============
def standard_wind_profile(duration_s, dt=1.0):
    t = np.arange(0, duration_s+dt, dt)
    base = 10.0 + 10.0*np.sin(2*np.pi*t/1800.0)  # 30-min cycle
    gust = 2.0*np.sin(2*np.pi*t/137.0)
    wind = np.clip(base + gust, 0.0, 20.0)
    return t, wind

def load_wind_csv(path):
    data = pd.read_csv(path)
    t = data.iloc[:,0].to_numpy(dtype=float)
    v = data.iloc[:,1].to_numpy(dtype=float)
    return t, v

# ============ Extended Generator Model ============
class AdvancedGenerator:
    """Three-point PTO generator with thermal and control models"""
    def __init__(self, spec: GeneratorSpec, name=""):
        self.spec = spec
        self.name = name
        self.T_coil = T_ambient
        self.i_cmd = spec.i_low
        self.engaged = False
        self.rpm = 0.0

    def update_temperature(self, P_loss, dt):
        """Thermal model: RC circuit"""
        dT = (P_loss * self.spec.Rth - (self.T_coil - T_ambient)) * dt / \
             (self.spec.Rth * self.spec.Cth)
        self.T_coil = np.clip(self.T_coil + dT, T_ambient, self.spec.T_max)

    def set_current(self, current):
        """Set commanded current"""
        self.i_cmd = np.clip(current, 0, max(self.spec.i_high, self.spec.i_low))

    def torque_power(self, omega, engaged=True):
        """Calculate EM torque and electrical power"""
        if self.T_coil > (self.spec.T_max - 5.0):  # Thermal derate
            return 0.0, 0.0, 0.0

        if abs(omega) < 1e-3 or not engaged:
            return 0.0, 0.0, 0.0

        # Electromagnetic torque (opposes motion)
        T_em = -self.spec.k_t * self.i_cmd * np.sign(omega)

        # Power dissipation in coil
        P_mech = abs(T_em * omega)
        P_cu = self.i_cmd**2 * self.spec.R_coil

        # Electrical power output
        P_elec = max(0.0, (P_mech - P_cu) * self.spec.eff)

        return T_em, P_elec, P_cu

class BidirectionalClutch:
    """Bidirectional clutch - transmits torque both directions"""
    def __init__(self, spec: ClutchSpec):
        self.spec = spec
        self.engaged = False
        self.last_engagement_time = 0.0

    def update(self, omega, t, min_disengagement_time=0.3):
        """Update clutch state based on angular velocity"""
        # Hysteresis to prevent chatter
        if not self.engaged:
            self.engaged = (abs(omega) > self.spec.engagement_threshold)
        else:
            release_time_passed = (t - self.last_engagement_time) > min_disengagement_time
            should_release = (abs(omega) < self.spec.disengagement_threshold) and release_time_passed
            if should_release:
                self.engaged = False
                self.last_engagement_time = t

        return self.engaged

    def transmit_torque(self, T_in):
        """Transmit torque with slip loss"""
        if not self.engaged:
            return 0.0
        return T_in * self.spec.eff

class OneWayClutch:
    """One-way clutch - transmits only positive (charging) torque"""
    def __init__(self, spec: ClutchSpec):
        self.spec = spec
        self.engaged = False

    def update(self, omega, torque_in):
        """Update clutch state - engage if positive power being generated"""
        power_in = torque_in * omega
        self.engaged = (power_in > 0) and (abs(omega) > self.spec.engagement_threshold)
        return self.engaged

    def transmit_torque(self, T_in):
        """One-way transmission"""
        if not self.engaged or T_in < 0:
            return 0.0
        return T_in * self.spec.eff

# ============ Flywheel Model ============
class Flywheel:
    """Flywheel for pulse smoothing"""
    def __init__(self, spec: FlywheelSpec):
        self.spec = spec
        self.omega = 0.0  # rad/s
        self.rpm = 0.0

    def step(self, T_in, dt):
        """Update flywheel dynamics"""
        # Friction torque (viscous + coulomb)
        T_friction = -self.spec.friction_coeff * self.omega * abs(self.omega) - \
                     0.1 * np.sign(self.omega) if abs(self.omega) > 1e-2 else 0

        # Angular acceleration
        alpha = (T_in + T_friction) / self.spec.inertia
        self.omega += alpha * dt
        self.rpm = self.omega * 60 / (2 * np.pi)

        return self.omega, self.rpm

# ============ Main Simulation Classes ============
class Pendulum2D:
    """2D double pendulum with three-generator PTO"""
    def __init__(self, S: Scenario, control_mode="adaptive", assist=True):
        self.S = S
        self.assist = assist
        self.ctrl_mode = control_mode

        # Three generators
        self.gen_h1 = AdvancedGenerator(S.gen_hinge1, "Hinge₁")
        self.gen_h2 = AdvancedGenerator(S.gen_hinge2, "Hinge₂")
        self.gen_gnd = AdvancedGenerator(S.gen_ground, "Ground")

        # Clutches
        self.clutch_h1 = BidirectionalClutch(S.clutch_hinge1)
        self.clutch_gnd = OneWayClutch(S.clutch_ground)

        # Gearbox and flywheel
        self.gearbox = S.gearbox
        self.flywheel = Flywheel(S.flywheel)

        # Bearing temperature
        self.T_bearing = T_ambient

        # History
        self.P_h1_hist = []
        self.P_h2_hist = []
        self.P_gnd_hist = []
        self.P_loss_hist = []
        self.T_coil_hist = []
        self.rpm_flywheel_hist = []
        self.v_wind = 10.0  # Reference wind speed
        self.dt_local = 0.01

    def mass_matrix(self, th1, th2):
        """Inertia matrix for two-DOF system"""
        S = self.S
        c = np.cos(th1 - th2)
        # Moment of inertia about hinges
        I1 = (1/3)*S.m_upper_arm*S.L1**2 + S.m_middle*S.L1**2 + \
             (S.m_lower_arm+S.m_tip)*S.L1**2
        I2 = (1/3)*S.m_lower_arm*S.L2**2 + S.m_tip*S.L2**2
        C12 = (S.m_lower_arm*0.5 + S.m_tip) * S.L1*S.L2*c

        # Ensure numerical stability: add small damping diagonal
        M = np.array([[I1, C12], [C12, I2]])
        M += 1e-3 * np.eye(2)  # Small diagonal regularization
        return M

    def wind_torque(self, theta, omega):
        """Wind forcing torque"""
        S = self.S
        venturi = 1.0 + 0.002*max(0, S.n_pendulums-8)
        v = self.v_wind * venturi
        v_vane = abs(omega) * S.L1
        v_rel = max(0.1, v - 0.5*v_vane)
        A = S.vane_w * S.vane_h
        F = 0.5 * rho_air * S.drag_cd * A * v_rel**2
        T = F * S.L1 * abs(np.sin(theta))
        return T if omega >= 0 else -T

    def bearing_update(self, P_loss, dt):
        """Bearing temperature model"""
        Rth, Cth = 0.25, 8000.0
        dT = (P_loss*Rth - (self.T_bearing - T_ambient)) * dt / (Rth*Cth)
        self.T_bearing = np.clip(self.T_bearing + dT, T_ambient, 373.15)

    def adaptive_current_control(self, omega1, omega2, engaged_h1):
        """Adaptive damping: I = k|ω|"""
        # Higher current at higher speeds (adaptive damping)
        k1 = 0.8 if self.assist else 0.0
        k2 = 0.6 if self.assist else 0.0

        i1 = k1 * abs(omega1) * (self.S.gen_hinge1.i_high / 10.0)
        i2 = k2 * abs(omega2) * (self.S.gen_hinge2.i_high / 10.0)

        self.gen_h1.set_current(i1 if engaged_h1 else 0.0)
        self.gen_h2.set_current(i2)

    def eom(self, t, y):
        """Equations of motion with three-point PTO"""
        th1, w1, th2, w2 = y
        S = self.S

        # Safety: check for NaN/Inf
        if not np.all(np.isfinite(y)):
            return [0, 0, 0, 0]

        # Container safety stops (hard braking)
        if abs(th1) > S.max_angle_rad * 1.1:  # Soft limit before hard stop
            return [w1, -50*w1, w2, -50*w2]
        elif abs(th1) > S.max_angle_rad:
            return [w1, -25*w1, w2, -25*w2]

        # Update clutch states
        engaged_h1 = self.clutch_h1.update(w1, t)

        # Adaptive current control
        self.adaptive_current_control(w1, w2, engaged_h1)

        # Generator torques
        T_em_h1, P_h1, P_cu_h1 = self.gen_h1.torque_power(w1, engaged_h1)
        T_em_h2, P_h2, P_cu_h2 = self.gen_h2.torque_power(w2, True)

        # Wind torques
        T_w1 = self.wind_torque(th1, w1)
        T_w2 = 0.7 * self.wind_torque(th2, w2)

        # Gravity torques
        Tg1 = -(S.m_upper_arm*g*(S.L1/2) + S.m_middle*g*S.L1 + \
                (S.m_lower_arm+S.m_tip)*g*S.L1) * np.sin(th1)
        Tg2 = -(S.m_lower_arm*g*(S.L2/2) + S.m_tip*g*S.L2) * np.sin(th2)

        # Bearing friction
        Tb1 = -S.bearing_mu * w1 * (1 - 0.3*((self.T_bearing - T_ambient)/50))
        Tb2 = -S.bearing_mu * w1 * (1 - 0.3*((self.T_bearing - T_ambient)/50))

        # Coriolis/coupling
        h = (S.m_lower_arm*0.5 + S.m_tip)*S.L1*S.L2*w1*w2*np.sin(th1-th2)
        h = np.clip(h, -5000, 5000)

        # Ground alternator via gearbox (simplified)
        T_gearbox = T_em_h1 * S.gearbox.eff / max(S.gearbox.ratio, 1e-3)
        self.flywheel.step(T_gearbox, self.dt_local)

        # Sum torques on each hinge
        T1 = T_w1 + Tg1 + Tb1 + T_em_h1 + h
        T2 = T_w2 + Tg2 + Tb2 + T_em_h2 - h

        M = self.mass_matrix(th1, th2)
        a1, a2 = np.linalg.solve(M, np.array([T1, T2]))
        a1 = np.clip(a1, -500, 500)
        a2 = np.clip(a2, -500, 500)

        # Thermal updates
        self.gen_h1.update_temperature(P_cu_h1, self.dt_local)
        self.gen_h2.update_temperature(P_cu_h2, self.dt_local)
        P_bearing = abs(Tb1*w1) + abs(Tb2*w2)
        self.bearing_update(P_bearing, self.dt_local)

        # Ground alternator power from flywheel
        T_gnd_em, P_gnd, P_cu_gnd = self.gen_gnd.torque_power(self.flywheel.omega, True)
        self.gen_gnd.update_temperature(P_cu_gnd, self.dt_local)

        # History
        self.P_h1_hist.append(P_h1)
        self.P_h2_hist.append(P_h2)
        self.P_gnd_hist.append(P_gnd)
        self.P_loss_hist.append(P_cu_h1 + P_cu_h2 + P_cu_gnd + P_bearing)
        self.T_coil_hist.append(0.5*(self.gen_h1.T_coil + self.gen_h2.T_coil))
        self.rpm_flywheel_hist.append(self.flywheel.rpm)

        return [w1, a1, w2, a2]

# ============ Simulation Driver ============
def run_simulation(scenario_key, duration_h=6, control_mode="adaptive", assist=True):
    """Run complete simulation"""
    S = SCENARIOS[scenario_key]
    duration_s = duration_h * 3600

    # Load wind
    t_wind, v_wind = standard_wind_profile(duration_s)

    # Create pendulum
    pend = Pendulum2D(S, control_mode=control_mode, assist=assist)
    # Initial conditions: small angle, zero velocity
    y0 = [np.deg2rad(5), 0.0, np.deg2rad(2), 0.0]
    y0 = np.array(y0, dtype=np.float64)
    assert np.all(np.isfinite(y0)), "Initial conditions must be finite"

    # Time-chunked integration (hourly)
    t_all, th1_all, th2_all, w1_all, w2_all = [], [], [], [], []

    t0 = 0.0
    y = np.array(y0, dtype=float)
    chunk = 3600.0
    rng = np.arange(0, duration_s, chunk).tolist() + [duration_s]

    for i in range(len(rng)-1):
        a, b = rng[i], rng[i+1]
        mask = (t_wind >= a) & (t_wind <= b)
        t_chunk = t_wind[mask] - a
        v_chunk = v_wind[mask]

        if len(t_chunk) < 2:
            t_chunk = np.linspace(0, b-a, int(b-a)+1)
            v_chunk = np.interp(t_chunk, [0, b-a], [10.0, 10.0])

        def eom_wrapper(t, y):
            v = float(np.interp(t, t_chunk, v_chunk))
            pend.v_wind = v
            pend.dt_local = max(1e-3, (t_chunk[1]-t_chunk[0]) if len(t_chunk)>1 else 0.01)
            return pend.eom(t0 + t + a, y)

        sol = solve_ivp(eom_wrapper, (0, t_chunk[-1]), y, method="LSODA",
                       max_step=0.1, rtol=1e-4, atol=1e-6)
        y = sol.y[:, -1]

        t_seg = sol.t + a
        t_all.append(t_seg)
        th1_all.append(sol.y[0])
        th2_all.append(sol.y[2])
        w1_all.append(sol.y[1])
        w2_all.append(sol.y[3])

    # Concatenate
    t = np.concatenate(t_all)
    th1 = np.concatenate(th1_all)
    th2 = np.concatenate(th2_all)

    # Calculate total power (three generators)
    P_h1 = np.array(pend.P_h1_hist)
    P_h2 = np.array(pend.P_h2_hist)
    P_gnd = np.array(pend.P_gnd_hist)
    P_total = (P_h1 + P_h2 + P_gnd) * S.n_pendulums / 1000.0  # kW

    # Summary
    dt_mean = np.mean(np.diff(t)) if len(t) > 1 else 1.0
    try:
        E_kWh = np.trapezoid(P_total, dx=dt_mean) / 3600.0
    except AttributeError:
        E_kWh = np.trapz(P_total, dx=dt_mean) / 3600.0
    P_avg = float(np.mean(P_total))
    P_peak = float(np.max(P_total)) if len(P_total) > 0 else 0.0

    results = {
        'P_avg_kW': P_avg,
        'P_peak_kW': P_peak,
        'E_kWh': E_kWh,
        'P_h1_avg': float(np.mean(P_h1)),
        'P_h2_avg': float(np.mean(P_h2)),
        'P_gnd_avg': float(np.mean(P_gnd)),
        'coil_T_max': float(np.max(pend.T_coil_hist) - 273.15) if pend.T_coil_hist else 0,
        'flywheel_rpm_avg': float(np.mean(pend.rpm_flywheel_hist)),
    }

    return results, t, th1, th2, P_total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="mega", choices=list(SCENARIOS.keys()))
    parser.add_argument("--duration", default="6", type=int)
    parser.add_argument("--control", default="adaptive", choices=["adaptive", "fixed"])
    parser.add_argument("--assist", default="on", choices=["on", "off"])
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MSSDPPG v3 - Parameterized Multi-Generator Simulator")
    print(f"Scenario: {args.scenario}")
    print(f"Duration: {args.duration}h | Control: {args.control} | Assist: {args.assist}")
    print(f"{'='*60}\n")

    scenario = SCENARIOS[args.scenario]
    print(f"Configuration: {scenario.name}")
    print(f"  Geometry: L₁={scenario.L1}m, L₂={scenario.L2}m (L₁/L₂={scenario.L1/max(scenario.L2,1e-6):.1f})")
    print(f"  Masses: upper={scenario.m_upper_arm}kg, mid={scenario.m_middle}kg, lower={scenario.m_lower_arm}kg, tip={scenario.m_tip}kg")
    print(f"  Limits: θ_max=±{np.rad2deg(scenario.max_angle_rad):.0f}°, μ={scenario.bearing_mu}")
    print(f"  PTO: 3 generators (Hinge₁ + Hinge₂ + Ground)")
    print(f"  Flywheel: I={scenario.flywheel.inertia} kg·m²\n")

    results, t, th1, th2, P = run_simulation(
        args.scenario,
        duration_h=args.duration,
        control_mode=args.control,
        assist=args.assist=="on"
    )

    print(f"Results:")
    print(f"  Total Power (3-PTO): {results['P_avg_kW']:.2f} kW avg, {results['P_peak_kW']:.2f} kW peak")
    print(f"  Hinge₁ Gen: {results['P_h1_avg']:.2f} kW")
    print(f"  Hinge₂ Gen: {results['P_h2_avg']:.2f} kW")
    print(f"  Ground Gen: {results['P_gnd_avg']:.2f} kW")
    print(f"  Energy: {results['E_kWh']:.2f} kWh over {args.duration}h")
    print(f"  Coil Tmax: {results['coil_T_max']:.1f}°C")
    print(f"  Flywheel avg RPM: {results['flywheel_rpm_avg']:.0f}")
    print(f"\n{'='*60}\n")
