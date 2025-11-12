"""
MSSDPPG Modular Multi-Pendulum Simulator (v4)
- Fully modular: n_pendulums variable (1, 2, 4, 8, 12, 24, 48...)
- Each pendulum: 2 generators (Hinge₁ + Hinge₂)
- Shared bidirectional ground shaft:
  * 2 flywheels (forward +ω, reverse -ω)
  * 2 alternators (forward, reverse)
  * 2 one-way clutches (direction-selective)
  * True bidirectional energy harvesting
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

# ============ Extended Modular Scenario ============
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
    engagement_threshold: float
    disengagement_threshold: float
    eff: float              # Efficiency

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
class BidirectionalShaftSpec:
    """Bidirectional ground shaft with dual flywheels and alternators"""
    n_alternators_per_dir: int = 1  # Alternators per direction
    inertia_fw: float = 5.0         # Forward flywheel (kg·m²)
    inertia_rv: float = 5.0         # Reverse flywheel (kg·m²)
    rpm_nom: float = 1500           # Nominal RPM both directions
    friction_fw: float = 0.001      # Forward friction
    friction_rv: float = 0.001      # Reverse friction

@dataclass
class Scenario:
    """Complete modular scenario"""
    name: str

    # === MODULAR: Number of Pendulums ===
    n_pendulums: int = 1    # VARIABLE: 1, 2, 4, 8, 12, 24, 48...

    # === Per-Pendulum Geometry ===
    L1: float = 2.0         # Upper arm length (m)
    L2: float = 2.0         # Lower arm length (m)
    L1_L2_ratio: float = None

    # === Per-Pendulum Masses ===
    m_upper_arm: float = 5.0
    m_middle: float = 30.0
    m_lower_arm: float = 3.0
    m_tip: float = 5.0

    # === Wind Interaction ===
    vane_w: float = 1.0
    vane_h: float = 2.0

    # === Mechanical Limits ===
    max_angle_rad: float = np.deg2rad(55)
    container_w: float = 2.35
    container_h: float = 2.39
    bearing_mu: float = 0.015
    drag_cd: float = 1.2
    mech_loss: float = 0.03

    # === Two Generators Per Pendulum ===
    gen_hinge1: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=0.75, R_coil=0.45, eff=0.85, Cth=250.0, Rth=1.5, T_max=423.15,
        i_high=6.0, i_low=1.5, rpm_min=100, rpm_max=3000
    ))
    gen_hinge2: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=0.8, R_coil=0.6, eff=0.87, Cth=200.0, Rth=1.2, T_max=423.15,
        i_high=5.0, i_low=1.2, rpm_min=80, rpm_max=2500
    ))

    # === Clutches ===
    clutch_hinge1: ClutchSpec = field(default_factory=lambda: ClutchSpec(
        type='bidirectional',
        engagement_threshold=0.1,
        disengagement_threshold=0.05,
        eff=0.97
    ))

    # === Gearbox ===
    gearbox: GearboxSpec = field(default_factory=lambda: GearboxSpec(
        ratio=12.0, eff=0.94, max_torque=500.0
    ))

    # === BIDIRECTIONAL GROUND SHAFT ===
    # With 2 flywheels (forward/reverse) and 2 alternators
    shaft_spec: BidirectionalShaftSpec = field(default_factory=lambda: BidirectionalShaftSpec(
        n_alternators_per_dir=1,
        inertia_fw=5.0,
        inertia_rv=5.0,
        rpm_nom=1500,
        friction_fw=0.001,
        friction_rv=0.001
    ))

    ground_gen_fw: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=1.5, R_coil=0.3, eff=0.90, Cth=400.0, Rth=1.0, T_max=423.15,
        i_high=8.0, i_low=2.0, rpm_min=200, rpm_max=4000
    ))
    ground_gen_rv: GeneratorSpec = field(default_factory=lambda: GeneratorSpec(
        k_t=1.5, R_coil=0.3, eff=0.90, Cth=400.0, Rth=1.0, T_max=423.15,
        i_high=8.0, i_low=2.0, rpm_min=200, rpm_max=4000
    ))

    clutch_ground_fw: ClutchSpec = field(default_factory=lambda: ClutchSpec(
        type='oneway', engagement_threshold=0.1, disengagement_threshold=0.0, eff=0.95
    ))
    clutch_ground_rv: ClutchSpec = field(default_factory=lambda: ClutchSpec(
        type='oneway', engagement_threshold=0.1, disengagement_threshold=0.0, eff=0.95
    ))

    # Performance reference
    expected_kw_at_6ms: float = 19.6
    color: str = "#4ECDC4"

    def __post_init__(self):
        if self.L1_L2_ratio is not None:
            if self.L1 < self.L2:
                self.L1 = self.L2 * self.L1_L2_ratio
            else:
                self.L2 = self.L1 / self.L1_L2_ratio

# ============ Predefined Scenarios ============

# Modular 4×40ft - NOW VARIABLE n_pendulums
SCENARIO_4X40FT_MODULAR = Scenario(
    name="4×40ft Container (Modular)",
    n_pendulums=48,         # VARIABLE: Can be 1, 2, 4, 6, 12, 24, 48
    L1=2.0, L2=2.0,
    m_upper_arm=5.0, m_middle=30.0, m_lower_arm=3.0, m_tip=5.0,
    vane_w=1.0, vane_h=2.0,
    max_angle_rad=np.deg2rad(55),
    container_w=2.35, container_h=2.39,
    bearing_mu=0.015, drag_cd=1.2, mech_loss=0.03,
    expected_kw_at_6ms=19.6, color="#4ECDC4"
)

SCENARIO_1X20FT_MODULAR = Scenario(
    name="1×20ft Container (Modular)",
    n_pendulums=24,
    L1=1.4, L2=1.4,
    m_upper_arm=2.45, m_middle=14.7, m_lower_arm=1.47, m_tip=2.45,
    vane_w=0.7, vane_h=1.4,
    max_angle_rad=np.deg2rad(60),
    container_w=2.35, container_h=2.39,
    bearing_mu=0.012, drag_cd=1.2, mech_loss=0.025,
    expected_kw_at_6ms=1.79, color="#95E1D3"
)

SCENARIO_TOWER_MODULAR = Scenario(
    name="Tower Facade (Modular)",
    n_pendulums=8,
    L1=0.75, L2=0.75,
    m_upper_arm=0.28, m_middle=7.5, m_lower_arm=0.17, m_tip=1.25,
    vane_w=0.4, vane_h=0.75,
    max_angle_rad=np.deg2rad(65),
    container_w=1.5, container_h=2.5,
    bearing_mu=0.010, drag_cd=1.2, mech_loss=0.02,
    expected_kw_at_6ms=0.684, color="#F38181"
)

SCENARIOS = {
    "4x40ft": SCENARIO_4X40FT_MODULAR,
    "1x20ft": SCENARIO_1X20FT_MODULAR,
    "tower": SCENARIO_TOWER_MODULAR,
}

# ============ Advanced Generator Model ============
class AdvancedGenerator:
    """Generator with thermal and control models"""
    def __init__(self, spec: GeneratorSpec, name=""):
        self.spec = spec
        self.name = name
        self.T_coil = T_ambient
        self.i_cmd = spec.i_low
        self.engaged = False
        self.rpm = 0.0

    def update_temperature(self, P_loss, dt):
        dT = (P_loss * self.spec.Rth - (self.T_coil - T_ambient)) * dt / \
             (self.spec.Rth * self.spec.Cth)
        self.T_coil = np.clip(self.T_coil + dT, T_ambient, self.spec.T_max)

    def set_current(self, current):
        self.i_cmd = np.clip(current, 0, max(self.spec.i_high, self.spec.i_low))

    def torque_power(self, omega, engaged=True):
        if self.T_coil > (self.spec.T_max - 5.0):
            return 0.0, 0.0, 0.0
        if abs(omega) < 1e-3 or not engaged:
            return 0.0, 0.0, 0.0

        T_em = -self.spec.k_t * self.i_cmd * np.sign(omega)
        P_mech = abs(T_em * omega)
        P_cu = self.i_cmd**2 * self.spec.R_coil
        P_elec = max(0.0, (P_mech - P_cu) * self.spec.eff)

        return T_em, P_elec, P_cu

class BidirectionalClutch:
    """Bidirectional clutch"""
    def __init__(self, spec: ClutchSpec):
        self.spec = spec
        self.engaged = False
        self.last_engagement_time = 0.0

    def update(self, omega, t, min_disengagement_time=0.3):
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
        if not self.engaged:
            return 0.0
        return T_in * self.spec.eff

class OneWayClutch:
    """One-way clutch"""
    def __init__(self, spec: ClutchSpec):
        self.spec = spec
        self.engaged = False

    def update(self, omega, torque_in):
        power_in = torque_in * omega
        self.engaged = (power_in > 0) and (abs(omega) > self.spec.engagement_threshold)
        return self.engaged

    def transmit_torque(self, T_in):
        if not self.engaged or T_in < 0:
            return 0.0
        return T_in * self.spec.eff

# ============ Bidirectional Flywheel Pair ============
class BidirectionalFlywheel:
    """Dual flywheels for forward and reverse rotation"""
    def __init__(self, spec: BidirectionalShaftSpec):
        self.spec = spec
        self.omega_fw = 0.0  # Forward angular velocity
        self.omega_rv = 0.0  # Reverse angular velocity
        self.rpm_fw = 0.0
        self.rpm_rv = 0.0

    def step(self, T_net, dt):
        """Update both flywheels based on net torque from all pendulums"""
        # Forward direction (T_net > 0)
        if T_net > 0:
            T_fw = T_net
            T_rv_friction = -self.spec.friction_rv * self.omega_rv * abs(self.omega_rv)
            alpha_fw = T_fw / self.spec.inertia_fw
            alpha_rv = T_rv_friction / self.spec.inertia_rv
        # Reverse direction (T_net < 0)
        elif T_net < 0:
            T_fw_friction = -self.spec.friction_fw * self.omega_fw * abs(self.omega_fw)
            T_rv = T_net
            alpha_fw = T_fw_friction / self.spec.inertia_fw
            alpha_rv = T_rv / self.spec.inertia_rv
        else:
            # Coast with friction
            T_fw_friction = -self.spec.friction_fw * self.omega_fw * abs(self.omega_fw)
            T_rv_friction = -self.spec.friction_rv * self.omega_rv * abs(self.omega_rv)
            alpha_fw = T_fw_friction / self.spec.inertia_fw
            alpha_rv = T_rv_friction / self.spec.inertia_rv

        self.omega_fw += alpha_fw * dt
        self.omega_rv += alpha_rv * dt

        self.rpm_fw = self.omega_fw * 60 / (2 * np.pi)
        self.rpm_rv = self.omega_rv * 60 / (2 * np.pi)

# ============ Single Pendulum (2 Generators) ============
class SinglePendulum:
    """Individual double-pendulum unit with 2 generators"""
    def __init__(self, S: Scenario, pendulum_id: int = 0, control_mode="adaptive", assist=True):
        self.S = S
        self.id = pendulum_id
        self.assist = assist
        self.ctrl_mode = control_mode

        # Two generators per pendulum
        self.gen_h1 = AdvancedGenerator(S.gen_hinge1, f"P{pendulum_id}_H1")
        self.gen_h2 = AdvancedGenerator(S.gen_hinge2, f"P{pendulum_id}_H2")

        # Clutch for Hinge₁
        self.clutch_h1 = BidirectionalClutch(S.clutch_hinge1)

        # Bearing temperature
        self.T_bearing = T_ambient

        # History
        self.P_h1_hist = []
        self.P_h2_hist = []
        self.P_loss_hist = []
        self.T_coil_hist = []

        self.v_wind = 10.0
        self.dt_local = 0.01

    def mass_matrix(self, th1, th2):
        S = self.S
        c = np.cos(th1 - th2)
        I1 = (1/3)*S.m_upper_arm*S.L1**2 + S.m_middle*S.L1**2 + \
             (S.m_lower_arm+S.m_tip)*S.L1**2
        I2 = (1/3)*S.m_lower_arm*S.L2**2 + S.m_tip*S.L2**2
        C12 = (S.m_lower_arm*0.5 + S.m_tip) * S.L1*S.L2*c
        M = np.array([[I1, C12], [C12, I2]])
        M += 1e-3 * np.eye(2)
        return M

    def wind_torque(self, theta, omega):
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
        Rth, Cth = 0.25, 8000.0
        dT = (P_loss*Rth - (self.T_bearing - T_ambient)) * dt / (Rth*Cth)
        self.T_bearing = np.clip(self.T_bearing + dT, T_ambient, 373.15)

    def adaptive_current_control(self, omega1, omega2, engaged_h1):
        k1 = 0.8 if self.assist else 0.0
        k2 = 0.6 if self.assist else 0.0
        i1 = k1 * abs(omega1) * (self.S.gen_hinge1.i_high / 10.0)
        i2 = k2 * abs(omega2) * (self.S.gen_hinge2.i_high / 10.0)
        self.gen_h1.set_current(i1 if engaged_h1 else 0.0)
        self.gen_h2.set_current(i2)

    def eom(self, t, y):
        """Equations of motion for single pendulum"""
        th1, w1, th2, w2 = y
        S = self.S

        if not np.all(np.isfinite(y)):
            return [0, 0, 0, 0]

        if abs(th1) > S.max_angle_rad * 1.1:
            return [w1, -50*w1, w2, -50*w2]
        elif abs(th1) > S.max_angle_rad:
            return [w1, -25*w1, w2, -25*w2]

        # Clutch & current control
        engaged_h1 = self.clutch_h1.update(w1, t)
        self.adaptive_current_control(w1, w2, engaged_h1)

        # Generator torques
        T_em_h1, P_h1, P_cu_h1 = self.gen_h1.torque_power(w1, engaged_h1)
        T_em_h2, P_h2, P_cu_h2 = self.gen_h2.torque_power(w2, True)

        # Wind & gravity
        T_w1 = self.wind_torque(th1, w1)
        T_w2 = 0.7 * self.wind_torque(th2, w2)
        Tg1 = -(S.m_upper_arm*g*(S.L1/2) + S.m_middle*g*S.L1 + \
                (S.m_lower_arm+S.m_tip)*g*S.L1) * np.sin(th1)
        Tg2 = -(S.m_lower_arm*g*(S.L2/2) + S.m_tip*g*S.L2) * np.sin(th2)

        # Bearing friction
        Tb1 = -S.bearing_mu * w1 * (1 - 0.3*((self.T_bearing - T_ambient)/50))
        Tb2 = -S.bearing_mu * w1 * (1 - 0.3*((self.T_bearing - T_ambient)/50))

        # Coriolis coupling
        h = (S.m_lower_arm*0.5 + S.m_tip)*S.L1*S.L2*w1*w2*np.sin(th1-th2)
        h = np.clip(h, -5000, 5000)

        # Torque sum
        T1 = T_w1 + Tg1 + Tb1 + T_em_h1 + h
        T2 = T_w2 + Tg2 + Tb2 + T_em_h2 - h

        M = self.mass_matrix(th1, th2)
        try:
            a1, a2 = np.linalg.solve(M, np.array([T1, T2]))
        except:
            a1, a2 = 0, 0

        a1 = np.clip(a1, -500, 500)
        a2 = np.clip(a2, -500, 500)

        # Thermal updates
        self.gen_h1.update_temperature(P_cu_h1, self.dt_local)
        self.gen_h2.update_temperature(P_cu_h2, self.dt_local)
        P_bearing = abs(Tb1*w1) + abs(Tb2*w2)
        self.bearing_update(P_bearing, self.dt_local)

        # History
        self.P_h1_hist.append(P_h1)
        self.P_h2_hist.append(P_h2)
        self.P_loss_hist.append(P_cu_h1 + P_cu_h2 + P_bearing)
        self.T_coil_hist.append(0.5*(self.gen_h1.T_coil + self.gen_h2.T_coil))

        return [w1, a1, w2, a2]

# ============ Multi-Pendulum System ============
class MultiPendulumSystem:
    """Multiple pendulums coupling to shared bidirectional ground shaft"""
    def __init__(self, S: Scenario, control_mode="adaptive", assist=True):
        self.S = S
        self.n_pend = S.n_pendulums

        # Create n independent pendulums
        self.pendulums = [
            SinglePendulum(S, i, control_mode, assist)
            for i in range(self.n_pend)
        ]

        # Bidirectional ground shaft
        self.shaft = BidirectionalFlywheel(S.shaft_spec)

        # Ground alternators (forward and reverse)
        self.gen_ground_fw = AdvancedGenerator(S.ground_gen_fw, "Ground_FW")
        self.gen_ground_rv = AdvancedGenerator(S.ground_gen_rv, "Ground_RV")

        # Ground clutches
        self.clutch_ground_fw = OneWayClutch(S.clutch_ground_fw)
        self.clutch_ground_rv = OneWayClutch(S.clutch_ground_rv)

        # History
        self.P_ground_fw_hist = []
        self.P_ground_rv_hist = []
        self.rpm_fw_hist = []
        self.rpm_rv_hist = []

    def step(self, t, y_all, wind_profile):
        """
        y_all: flattened state for all pendulums [th1_0, w1_0, th2_0, w2_0, ..., th1_n, w1_n, th2_n, w2_n]
        """
        # Reshape state
        y_list = []
        idx = 0
        for i in range(self.n_pend):
            y_list.append(y_all[idx:idx+4])
            idx += 4

        # Get wind speed
        v_wind = float(wind_profile)

        # Step each pendulum
        dy_list = []
        T_hinge1_all = []  # Collect all Hinge₁ torques for ground shaft

        for i, pend in enumerate(self.pendulums):
            pend.v_wind = v_wind
            dy = pend.eom(t, y_list[i])
            dy_list.append(dy)

            # Extract Hinge₁ info for ground shaft coupling
            th1, w1, th2, w2 = y_list[i]
            engaged_h1 = pend.clutch_h1.engaged
            T_em_h1, _, _ = pend.gen_h1.torque_power(w1, engaged_h1)

            # This torque contributes to ground shaft via gearbox
            T_shaft_from_h1 = T_em_h1 * self.S.gearbox.eff / max(self.S.gearbox.ratio, 1e-3)
            T_hinge1_all.append(T_shaft_from_h1)

        # Sum all Hinge₁ torques for ground shaft
        T_net_ground = sum(T_hinge1_all)

        # Step bidirectional shaft
        self.shaft.step(T_net_ground, 0.01)  # dt=0.01 (local timestep)

        # Ground alternator power (direction-selective)
        T_em_fw, P_fw, P_cu_fw = self.gen_ground_fw.torque_power(self.shaft.omega_fw, True)
        T_em_rv, P_rv, P_cu_rv = self.gen_ground_rv.torque_power(self.shaft.omega_rv, True)

        self.gen_ground_fw.update_temperature(P_cu_fw, 0.01)
        self.gen_ground_rv.update_temperature(P_cu_rv, 0.01)

        self.P_ground_fw_hist.append(P_fw)
        self.P_ground_rv_hist.append(P_rv)
        self.rpm_fw_hist.append(self.shaft.rpm_fw)
        self.rpm_rv_hist.append(self.shaft.rpm_rv)

        # Flatten derivatives
        dy_flat = []
        for dy in dy_list:
            dy_flat.extend(dy)

        return dy_flat

# ============ Simulation Driver ============
def standard_wind_profile(duration_s, dt=1.0):
    t = np.arange(0, duration_s+dt, dt)
    base = 10.0 + 10.0*np.sin(2*np.pi*t/1800.0)
    gust = 2.0*np.sin(2*np.pi*t/137.0)
    wind = np.clip(base + gust, 0.0, 20.0)
    return t, wind

def run_simulation(scenario_key, duration_h=6, n_pendulums=None, control_mode="adaptive", assist=True):
    """Run modular multi-pendulum simulation"""
    S = SCENARIOS[scenario_key]

    # Override n_pendulums if specified
    if n_pendulums is not None:
        S.n_pendulums = n_pendulums

    duration_s = duration_h * 3600
    t_wind, v_wind = standard_wind_profile(duration_s)

    # Create multi-pendulum system
    system = MultiPendulumSystem(S, control_mode, assist)

    # Initial conditions: all pendulums at small angles
    y0 = []
    for i in range(S.n_pendulums):
        y0.extend([np.deg2rad(5), 0.0, np.deg2rad(2), 0.0])
    y0 = np.array(y0, dtype=np.float64)

    # Integrate
    t_all, y_all = [], []
    t, y = 0.0, y0

    chunk = 3600.0
    rng = np.arange(0, duration_s, chunk).tolist() + [duration_s]

    for chunk_idx in range(len(rng)-1):
        t_start, t_end = rng[chunk_idx], rng[chunk_idx+1]

        # Wind in this chunk
        mask = (t_wind >= t_start) & (t_wind <= t_end)
        t_chunk = t_wind[mask] - t_start
        v_chunk = v_wind[mask]

        if len(t_chunk) < 2:
            t_chunk = np.linspace(0, t_end-t_start, int(t_end-t_start)+1)
            v_chunk = np.interp(t_chunk, [0, t_end-t_start], [10.0, 10.0])

        def eom_wrapper(t_local, y):
            v = float(np.interp(t_local, t_chunk, v_chunk))
            dy_flat = system.step(t_start + t_local, y, v)
            return dy_flat

        try:
            sol = solve_ivp(eom_wrapper, (0, t_chunk[-1]), y, method="LSODA",
                           max_step=0.1, rtol=1e-4, atol=1e-6)
            y = sol.y[:, -1]
            t_all.append(sol.t + t_start)
            y_all.append(sol.y)
        except Exception as e:
            print(f"Integration error in chunk {chunk_idx}: {e}")
            break

    # Concatenate results
    if t_all:
        t_concat = np.concatenate(t_all)
        y_concat = np.concatenate(y_all, axis=1)
    else:
        t_concat = np.array([0])
        y_concat = y0.reshape(-1, 1)

    # Calculate total power: sum all generators across all pendulums
    P_h1_total = np.sum([np.array(p.P_h1_hist) for p in system.pendulums], axis=0)
    P_h2_total = np.sum([np.array(p.P_h2_hist) for p in system.pendulums], axis=0)
    P_fw = np.array(system.P_ground_fw_hist)
    P_rv = np.array(system.P_ground_rv_hist)

    # Ensure same length (use minimum)
    min_len = min(len(P_h1_total), len(P_h2_total), len(P_fw), len(P_rv))
    P_h1_total = P_h1_total[:min_len]
    P_h2_total = P_h2_total[:min_len]
    P_fw = P_fw[:min_len]
    P_rv = P_rv[:min_len]

    P_total = (P_h1_total + P_h2_total + P_fw + P_rv) * S.n_pendulums / 1000.0  # kW

    # Summary
    dt_mean = np.mean(np.diff(t_concat)) if len(t_concat) > 1 else 1.0
    try:
        E_kWh = np.trapezoid(P_total, dx=dt_mean) / 3600.0
    except AttributeError:
        E_kWh = np.trapz(P_total, dx=dt_mean) / 3600.0

    P_avg = float(np.mean(P_total))
    P_peak = float(np.max(P_total)) if len(P_total) > 0 else 0.0

    # Per-system breakdown
    P_h1_avg = float(np.mean(P_h1_total))
    P_h2_avg = float(np.mean(P_h2_total))
    P_fw_avg = float(np.mean(P_fw))
    P_rv_avg = float(np.mean(P_rv))

    results = {
        'n_pendulums': S.n_pendulums,
        'P_avg_kW': P_avg,
        'P_peak_kW': P_peak,
        'E_kWh': E_kWh,
        'P_hinge1_avg': P_h1_avg,
        'P_hinge2_avg': P_h2_avg,
        'P_ground_fw_avg': P_fw_avg,
        'P_ground_rv_avg': P_rv_avg,
        'coil_T_max': float(max([max(p.T_coil_hist) if p.T_coil_hist else T_ambient
                                 for p in system.pendulums] + [T_ambient]) - 273.15),
        'rpm_fw_avg': float(np.mean(system.rpm_fw_hist) if system.rpm_fw_hist else 0),
        'rpm_rv_avg': float(np.mean(system.rpm_rv_hist) if system.rpm_rv_hist else 0),
    }

    return results, t_concat, y_concat, P_total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="4x40ft", choices=list(SCENARIOS.keys()))
    parser.add_argument("--duration", default="6", type=int)
    parser.add_argument("--n-pendulums", default=None, type=int, help="Override number of pendulums")
    parser.add_argument("--control", default="adaptive", choices=["adaptive"])
    parser.add_argument("--assist", default="on", choices=["on", "off"])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"MSSDPPG v4 - Modular Multi-Pendulum Simulator")
    print(f"Bidirectional Shaft with Dual Flywheels & Alternators")
    print(f"{'='*70}\n")

    scenario = SCENARIOS[args.scenario]
    n_pend = args.n_pendulums if args.n_pendulums else scenario.n_pendulums

    print(f"Scenario: {scenario.name}")
    print(f"Pendulums: {n_pend} units")
    print(f"Per-Pendulum Generators: 2 (Hinge₁ + Hinge₂)")
    print(f"Total Generators: {n_pend * 2} + 2 (ground FW/RV)")
    print(f"Ground Shaft: Bidirectional (2 flywheels, 2 alternators)\n")

    results, t, y, P = run_simulation(
        args.scenario,
        duration_h=args.duration,
        n_pendulums=args.n_pendulums,
        control_mode=args.control,
        assist=args.assist=="on"
    )

    print(f"Results ({results['n_pendulums']} pendulums, 6h):")
    print(f"  Total Power: {results['P_avg_kW']:.2f} kW avg, {results['P_peak_kW']:.2f} kW peak")
    print(f"  Hinge₁ Gens: {results['P_hinge1_avg']:.3f} kW ({results['n_pendulums']} units)")
    print(f"  Hinge₂ Gens: {results['P_hinge2_avg']:.3f} kW ({results['n_pendulums']} units)")
    print(f"  Ground FW: {results['P_ground_fw_avg']:.3f} kW")
    print(f"  Ground RV: {results['P_ground_rv_avg']:.3f} kW")
    print(f"  Energy: {results['E_kWh']:.2f} kWh")
    print(f"  Coil Tmax: {results['coil_T_max']:.1f}°C")
    print(f"  Flywheel FW: {results['rpm_fw_avg']:.0f} RPM avg")
    print(f"  Flywheel RV: {results['rpm_rv_avg']:.0f} RPM avg")
    print(f"{'='*70}\n")
