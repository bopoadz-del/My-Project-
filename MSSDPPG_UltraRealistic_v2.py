\
"""
MSSDPPG Ultra-Realistic v2
- Planar (2D) baseline + Spatial (3D) optional
- Lock–Release default control, Push–Pull optional
- Assist toggle (on/off)
- Variable wind (0–20 m/s) or CSV
- 6h / 12h endurance, adaptive solver with chunking
Outputs: PNG plots + CSV summary
"""

import os, sys, argparse, time, math, csv, datetime as dt
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------- Constants ----------------
g = 9.81
rho_air = 1.225
T_ambient = 298.15

# ---------------- Configs ----------------
@dataclass
class Scenario:
    name: str
    L1: float
    L2: float
    vane_w: float
    vane_h: float
    m_upper_arm: float
    m_middle: float  # fixed per scale
    m_lower_arm: float
    m_tip: float     # fixed per scale
    n_pendulums: int
    container_w: float
    container_h: float
    max_angle_rad: float
    bearing_mu: float
    drag_cd: float
    mech_loss: float
    expected_kw_at_6ms: float
    color: str

SCENARIOS = {
    "4x40ft": Scenario("4×40ft Container", 2.0, 2.0, 1.0, 2.0, 5.0, 30.0, 3.0, 5.0,
                       48, 2.35, 2.39, np.deg2rad(55), 0.015, 1.2, 0.03, 19.6, "#4ECDC4"),
    "1x20ft": Scenario("1×20ft Container", 1.4, 1.4, 0.7, 1.4, 2.45, 14.7, 1.47, 2.45,
                       24, 2.35, 2.39, np.deg2rad(60), 0.012, 1.2, 0.025, 1.79, "#95E1D3"),
    "tower": Scenario("Tower Facade", 0.75, 0.75, 0.4, 0.75, 0.28, 7.5, 0.17, 1.25,
                      8, 1.5, 2.5, np.deg2rad(65), 0.010, 1.2, 0.02, 0.684, "#F38181"),
    "mega": Scenario("Mega 15 m", 6.0, 6.0, 3.0, 6.0, 45.0, 120.0, 27.0, 30.0,
                     1, 8.0, 15.0, np.deg2rad(45), 0.020, 1.2, 0.04, 77.2, "#AA96DA"),
}

# ---------------- Utility: wind profile ----------------
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

# ---------------- Generator & Friction Models ----------------
class HingeGenerator:
    def __init__(self, L1):
        # scale by area ~ (L1/2)^2
        scale = (L1/2.0)**2
        self.k_t = 0.75 * scale      # Nm/A
        self.R25 = 0.45 * scale      # Ohm at 25C
        self.eff = 0.85
        self.T_coil = T_ambient
        self.Cth = 250.0 * scale     # J/K
        self.Rth = 1.5 / max(scale, 1e-3)   # K/W (smaller scale → higher Rth)
        self.Tmax = 423.15           # 150C
        self.i_high = 6.0 * max(scale**0.5, 0.2)
        self.i_low  = 1.5 * max(scale**0.5, 0.2)
        self.i_cmd = self.i_low

    def Rcoil(self):
        alpha = 0.00393
        return self.R25 * (1 + alpha*(self.T_coil - 298.15))

    def update_thermal(self, P_loss, dt):
        # simple RC cooling
        dT = (P_loss*self.Rth - (self.T_coil - T_ambient)) * dt / (self.Rth*self.Cth)
        self.T_coil = np.clip(self.T_coil + dT, T_ambient, self.Tmax)

    def torque_power(self, omega, locked, assist_on=True):
        # Coil derate near Tmax
        if self.T_coil > (self.Tmax - 10.0):
            return 0.0, 0.0, 0.0
        self.i_cmd = self.i_high if (locked and assist_on) else self.i_low if assist_on else 0.0
        if abs(omega) < 1e-3 or self.i_cmd <= 0:
            return 0.0, 0.0, 0.0
        T_em = - self.k_t * self.i_cmd * np.sign(omega)   # oppose motion
        P_mech = abs(T_em * omega)
        R = self.Rcoil()
        P_cu = self.i_cmd**2 * R
        P_elec = max(0.0, (P_mech - P_cu) * self.eff)
        return T_em, P_elec, P_cu

def bearing_torque(mu, L1, omega, T_bearing):
    # viscous + Coulomb scaled by L1^2
    visc_red = max(0.7, 1.0 - 0.3*((T_bearing - T_ambient)/50.0))
    T_visc = - mu * omega * visc_red
    T_coul = - 0.3 * (L1/2.0)**2 * np.sign(omega) if abs(omega) > 1e-3 else 0.0
    return T_visc + T_coul

# ---------------- Controller ----------------
class LockRelease:
    def __init__(self, theta_min_deg=15, theta_max_deg=55):
        self.theta_min = np.deg2rad(theta_min_deg)
        self.theta_max = np.deg2rad(theta_max_deg)
        self.last_zero_t = 0.0
        self.locked = False

    def step(self, theta1, omega1, t):
        if abs(omega1) < 0.05 and t > self.last_zero_t + 0.3:
            self.last_zero_t = t
        in_lock = (abs(omega1) > 0.1 and self.theta_min <= abs(theta1) <= self.theta_max)
        release = (t - self.last_zero_t) < 0.12
        self.locked = bool(in_lock and (not release))
        return self.locked

class PushPull:
    def __init__(self, K=5.0, cutoff=25.0):
        self.K = K; self.cutoff = cutoff
    def torque(self, theta, omega, enabled=True):
        if not enabled: return 0.0
        scale = 1.0 if abs(omega) < self.cutoff else self.cutoff/abs(omega)
        return self.K * np.sin(theta) * np.sign(omega) * scale

# ---------------- Physics Core ----------------
class Pendulum2D:
    def __init__(self, S: Scenario, control_mode="lock", assist=True):
        self.S = S
        self.assist = assist
        self.ctrl_mode = control_mode
        self.gen_upper = HingeGenerator(S.L1)
        self.gen_lower = HingeGenerator(S.L1)
        self.lock = LockRelease()
        self.push = PushPull(K=5.0*(S.L1/2.0)**2)
        self.T_bearing = T_ambient
        # Power histories
        self.P_upper_hist = []
        self.P_lower_hist = []
        self.P_shaft_hist = []
        self.P_loss_hist = []
        self.T_coil_hist = []

    def mass_matrix(self, th1, th2):
        S=self.S; c = np.cos(th1 - th2)
        I1 = (1/3)*S.m_upper_arm*S.L1**2 + S.m_middle*S.L1**2 + (S.m_lower_arm+S.m_tip)*S.L1**2
        I2 = (1/3)*S.m_lower_arm*S.L2**2 + S.m_tip*S.L2**2
        C12= (S.m_lower_arm*0.5 + S.m_tip) * S.L1*S.L2*c
        return np.array([[I1, C12],[C12, I2]])

    def wind_torque(self, theta, omega, arm='upper'):
        S=self.S
        venturi = 1.0 + 0.002*max(0, S.n_pendulums-8)
        v = self.v_wind * venturi
        v_vane = abs(omega)*S.L1
        v_rel = max(0.1, v - 0.5*v_vane)
        A = S.vane_w * S.vane_h
        F = 0.5 * rho_air * S.drag_cd * A * v_rel**2
        L = S.L1 if arm=='upper' else S.L2
        T = F * L * abs(np.sin(theta))
        return T if omega>=0 else -T

    def bearing_update(self, P_loss, dt):
        Rth=0.25; Cth=8000.0
        dT=(P_loss*Rth - (self.T_bearing - T_ambient))*dt/(Rth*Cth)
        self.T_bearing = np.clip(self.T_bearing + dT, T_ambient, 373.15)

    def clutch_torque(self, omega, P_avail):
        S=self.S
        if abs(omega) < 0.1: return 0.0, 0.0, 0.0
        T_lim = 150.0*(S.L1/2.0)
        T_avail = min(P_avail/max(abs(omega),1e-3), T_lim)
        eff=0.97
        T_c=T_avail*eff
        P_loss=T_avail*abs(omega)*(1-eff)
        T_resist = -T_c*np.sign(omega)
        P_out = T_c*abs(omega)
        return T_resist, P_out, P_loss

    def eom(self, t, y):
        th1, w1, th2, w2 = y
        S=self.S
        # container safety
        if abs(th1) > S.max_angle_rad:
            return [w1, -25*w1, w2, -25*w2]
        # lock / control
        locked = self.lock.step(th1, w1, t) if self.ctrl_mode=="lock" else False
        # EM torques
        T_em_u, P_u, P_cu_u = self.gen_upper.torque_power(w1, locked, assist_on=self.assist)
        T_em_l, P_l, P_cu_l = self.gen_lower.torque_power(w2, False, assist_on=self.assist)
        # push-pull (optional control)
        if self.ctrl_mode=="pushpull":
            T_em_u += self.push.torque(th1, w1, enabled=self.assist)
        # wind
        T_w1 = self.wind_torque(th1, w1, 'upper')
        T_w2 = 0.7*self.wind_torque(th2, w2, 'lower')
        # gravity
        Tg1 = - (S.m_upper_arm*g*(S.L1/2) + S.m_middle*g*S.L1 + (S.m_lower_arm+S.m_tip)*g*S.L1) * np.sin(th1)
        Tg2 = - (S.m_lower_arm*g*(S.L2/2) + S.m_tip*g*S.L2) * np.sin(th2)
        # bearing
        Tb1 = bearing_torque(S.bearing_mu, S.L1, w1, self.T_bearing)
        Tb2 = bearing_torque(S.bearing_mu, S.L1, w2, self.T_bearing)
        # clutch
        P_wind_upper = abs(T_w1*w1)
        P_avail = max(0.0, P_wind_upper - P_u)
        T_cl, P_shaft, P_cl_loss = self.clutch_torque(w1, P_avail)
        # coriolis/coupling
        h = (S.m_lower_arm*0.5 + S.m_tip)*S.L1*S.L2*w1*w2*np.sin(th1-th2)
        h = np.clip(h, -5000, 5000)
        # sum torques
        T1 = T_w1 + Tg1 + Tb1 + T_em_u + T_cl + h
        T2 = T_w2 + Tg2 + Tb2 + T_em_l - h
        M = self.mass_matrix(th1, th2)
        a1, a2 = np.linalg.solve(M, np.array([T1, T2]))
        a1 = np.clip(a1, -500, 500); a2 = np.clip(a2, -500, 500)
        # thermal updates (approx dt via self.dt_local)
        dt = getattr(self, "dt_local", 0.01)
        self.gen_upper.update_thermal(P_cu_u, dt)
        self.gen_lower.update_thermal(P_cu_l, dt)
        P_bearing = abs(Tb1*w1)+abs(Tb2*w2)
        self.bearing_update(P_bearing + P_cl_loss, dt)
        # store power
        self.P_upper_hist.append(P_u)
        self.P_lower_hist.append(P_l)
        self.P_shaft_hist.append(P_shaft)
        self.P_loss_hist.append(P_cu_u + P_cu_l + P_bearing + P_cl_loss)
        self.T_coil_hist.append(0.5*(self.gen_upper.T_coil + self.gen_lower.T_coil))
        return [w1, a1, w2, a2]

class Pendulum3D(Pendulum2D):
    def __init__(self, S, control_mode="lock", assist=True, offset=0.12, k_phi=10.0, c_phi=0.5, Km_phi=3.0):
        super().__init__(S, control_mode, assist)
        self.offset = offset
        self.k_phi = k_phi
        self.c_phi = c_phi
        self.Km_phi = Km_phi
        self.phi0 = 0.02; self.wphi0 = 0.0
        self.phi_hist = []

    def eom3d(self, t, y):
        th1, w1, th2, w2, phi, wphi = y
        S=self.S
        # Planar part
        self.v_wind = self.v_wind  # placeholder
        if abs(th1) > S.max_angle_rad:
            return [w1, -25*w1, w2, -25*w2, wphi, -5*wphi]
        locked = self.lock.step(th1, w1, t) if self.ctrl_mode=="lock" else False
        T_em_u, P_u, P_cu_u = self.gen_upper.torque_power(w1, locked, assist_on=self.assist)
        T_em_l, P_l, P_cu_l = self.gen_lower.torque_power(w2, False, assist_on=self.assist)
        if self.ctrl_mode=="pushpull":
            T_em_u += self.push.torque(th1, w1, enabled=self.assist)

        T_w1 = self.wind_torque(th1, w1, 'upper')
        T_w2 = 0.7*self.wind_torque(th2, w2, 'lower')
        Tg1 = - (S.m_upper_arm*g*(S.L1/2) + S.m_middle*g*S.L1 + (S.m_lower_arm+S.m_tip)*g*S.L1) * np.sin(th1)
        Tg2 = - (S.m_lower_arm*g*(S.L2/2) + S.m_tip*g*S.L2) * np.sin(th2)
        Tb1 = bearing_torque(S.bearing_mu, S.L1, w1, self.T_bearing)
        Tb2 = bearing_torque(S.bearing_mu, S.L1, w2, self.T_bearing)
        P_wind_upper = abs(T_w1*w1)
        T_cl, P_shaft, P_cl_loss = self.clutch_torque(w1, max(0.0, P_wind_upper - P_u))
        h = (S.m_lower_arm*0.5 + S.m_tip)*S.L1*S.L2*w1*w2*np.sin(th1-th2)
        h = np.clip(h, -5000, 5000)
        T1 = T_w1 + Tg1 + Tb1 + T_em_u + T_cl + h
        T2 = T_w2 + Tg2 + Tb2 + T_em_l - h
        M = self.mass_matrix(th1, th2)
        a1, a2 = np.linalg.solve(M, np.array([T1, T2]))
        a1 = np.clip(a1, -500, 500); a2 = np.clip(a2, -500, 500)

        # φ lateral axis
        T_cross = S.L1*S.L2*(S.m_tip)*w1*w2*np.sin(th1-th2)
        T_phi = - self.k_phi*phi - self.c_phi*wphi + self.Km_phi*np.sin(phi)*np.sign(wphi)
        I_phi = S.m_lower_arm * S.L2**2 + 0.5*S.m_tip*S.L2**2
        aphi = (T_phi + T_cross) / max(I_phi, 1e-6)

        dt = getattr(self, "dt_local", 0.01)
        self.gen_upper.update_thermal(P_cu_u, dt)
        self.gen_lower.update_thermal(P_cu_l, dt)
        P_bearing = abs(Tb1*w1)+abs(Tb2*w2)
        self.bearing_update(P_bearing + P_cl_loss, dt)
        self.P_upper_hist.append(P_u); self.P_lower_hist.append(P_l)
        self.P_shaft_hist.append(P_shaft); self.P_loss_hist.append(P_cu_u + P_cu_l + P_bearing + P_cl_loss)
        self.T_coil_hist.append(0.5*(self.gen_upper.T_coil + self.gen_lower.T_coil))
        self.phi_hist.append(phi)
        return [w1, a1, w2, a2, wphi, aphi]

# ---------------- Simulation Driver ----------------
def run_one(sim_mode, scenario_key, duration_s, control_mode, assist, spatial_params, wind_t, wind_v, outputs_dir):
    S = SCENARIOS[scenario_key]
    # pendulum object
    if sim_mode=="2d":
        pend = Pendulum2D(S, control_mode=control_mode, assist=assist)
        eom_func = pend.eom
        y0 = [0.15, 0.0, 0.0, 0.0]
    else:
        pend = Pendulum3D(S, control_mode=control_mode, assist=assist, **spatial_params)
        eom_func = pend.eom3d
        y0 = [0.15, 0.0, 0.0, 0.0, pend.phi0, pend.wphi0]

    # time-chunked integration (hourly chunks)
    t_all = []
    th1_all = []; th2_all = []; w1_all=[]; w2_all=[]
    phi_all=[]; P_total_all=[]; wind_all=[]

    t0 = 0.0; y = np.array(y0, dtype=float)
    chunk = 3600.0  # 1 hour per chunk
    rng = np.arange(0, duration_s, chunk).tolist() + [duration_s]
    for i in range(len(rng)-1):
        a, b = rng[i], rng[i+1]
        # slice wind
        mask = (wind_t>=a) & (wind_t<=b)
        t_chunk = wind_t[mask] - a
        v_chunk = wind_v[mask]
        if len(t_chunk) < 2:
            t_chunk = np.linspace(0, b-a, int((b-a))+1)
            v_chunk = np.interp(t_chunk, [0,b-a], [10.0,10.0])
        # set reference on pendulum
        def eom_wrapper(t, y):
            # interpolate wind at local t
            v = float(np.interp(t, t_chunk, v_chunk))
            pend.v_wind = v
            # approx dt for thermal
            pend.dt_local = max(1e-3, (t_chunk[1]-t_chunk[0]) if len(t_chunk)>1 else 0.01)
            return eom_func(t0 + t + a, y)

        sol = solve_ivp(eom_wrapper, (0, t_chunk[-1]), y, method="LSODA", max_step=0.1, rtol=1e-5, atol=1e-7)
        y = sol.y[:,-1]  # new initial
        # collect
        t_seg = sol.t + a
        t_all.append(t_seg)
        th1_all.append(sol.y[0]); w1_all.append(sol.y[1])
        th2_all.append(sol.y[2]); w2_all.append(sol.y[3])
        if sim_mode=="3d":
            phi_all.append(sol.y[4])
        wind_all.append(np.interp(sol.t, sol.t, v_chunk[:len(sol.t)]))

    t = np.concatenate(t_all)
    th1 = np.concatenate(th1_all); th2 = np.concatenate(th2_all)
    w1 = np.concatenate(w1_all);   w2 = np.concatenate(w2_all)
    wind_series = np.concatenate(wind_all)
    if sim_mode=="3d":
        phi = np.concatenate(phi_all)
    # power & energy
    P_hinge = np.array(pend.P_upper_hist) + np.array(pend.P_lower_hist)
    P_total = P_hinge + np.array(pend.P_shaft_hist) * 0.95 * 0.88  # gearbox*alt eff
    P_total_all = P_total
    # scale to system
    P_system = P_total * S.n_pendulums / 1000.0  # kW
    # averages
    dt_mean = np.mean(np.diff(t)) if len(t)>1 else 1.0
    E_kWh = np.trapz(P_system, dx=dt_mean)/3600.0
    P_avg = float(np.mean(P_system))
    P_peak = float(np.max(P_system)) if len(P_system)>0 else 0.0
    eta_total = float(np.trapz(P_total, dx=dt_mean) / max(1e-6, np.trapz(np.array(pend.P_loss_hist)+P_total, dx=dt_mean)))
    Tmax = float(np.max(pend.T_coil_hist)) if pend.T_coil_hist else float(T_ambient)
    max_angle_deg = float(np.rad2deg(np.max(np.abs(th1))))

    # plots
    if sim_mode=="2d":
        fig = plt.figure(figsize=(12,4))
        plt.plot(t/3600.0, P_system, lw=1.2, label="Power (kW)")
        plt.plot(t/3600.0, wind_series, lw=0.8, alpha=0.6, label="Wind (m/s)")
        plt.xlabel("Time (h)"); plt.title(f"{S.name} – 2D Planar – {('Assist ON' if pend.assist else 'Assist OFF')}")
        plt.legend(); plt.grid(alpha=0.3)
        outpng = os.path.join(outputs_dir, "power_vs_time_2D.png")
        plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close(fig)
    else:
        fig = plt.figure(figsize=(12,4))
        plt.plot(t/3600.0, P_system, lw=1.2, label="Power (kW)")
        plt.plot(t/3600.0, wind_series, lw=0.8, alpha=0.6, label="Wind (m/s)")
        plt.xlabel("Time (h)"); plt.title(f"{S.name} – 3D Spatial – {('Assist ON' if pend.assist else 'Assist OFF')}")
        plt.legend(); plt.grid(alpha=0.3)
        outpng = os.path.join(outputs_dir, "power_vs_time_3D.png")
        plt.tight_layout(); plt.savefig(outpng, dpi=150); plt.close(fig)

        fig2 = plt.figure(figsize=(12,3))
        plt.plot(t/3600.0, np.rad2deg(phi), lw=1.0)
        plt.xlabel("Time (h)"); plt.ylabel("φ (deg)"); plt.title("Spatial Lateral Angle")
        plt.grid(alpha=0.3)
        outpng2 = os.path.join(outputs_dir, "phi_amplitude_vs_time.png")
        plt.tight_layout(); plt.savefig(outpng2, dpi=150); plt.close(fig2)

    return {
        "P_avg_kW": P_avg, "P_peak_kW": P_peak, "E_kWh": E_kWh,
        "eta_total": eta_total, "coil_Tmax_C": (Tmax-273.15),
        "theta_max_deg": max_angle_deg
    }

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="4x40ft", choices=list(SCENARIOS.keys()))
    parser.add_argument("--duration", default="6h", choices=["6h","12h"])
    parser.add_argument("--mode", default="2d", choices=["2d","spatial","both"])
    parser.add_argument("--assist", default="on", choices=["on","off"])
    parser.add_argument("--control", default="lock", choices=["lock","pushpull"])
    parser.add_argument("--windfile", default="")
    args = parser.parse_args()

    duration_s = 6*3600 if args.duration=="6h" else 12*3600
    outputs_dir = os.path.join("outputs")
    logs_dir = os.path.join("logs")
    os.makedirs(outputs_dir, exist_ok=True); os.makedirs(logs_dir, exist_ok=True)

    # Wind profile
    if args.windfile and os.path.exists(args.windfile):
        t, v = load_wind_csv(args.windfile)
        # pad or trim to match duration
        if t[-1] < duration_s:
            # extend by repeating
            reps = int(np.ceil(duration_s / t[-1]))
            t = np.concatenate([t + i*t[-1] for i in range(reps)])
            v = np.tile(v, reps)
        mask = t <= duration_s
        t = t[mask]; v = v[mask]
    else:
        t, v = standard_wind_profile(duration_s, dt=1.0)

    # spatial params default
    spatial_params = dict(offset=0.12, k_phi=10.0, c_phi=0.6, Km_phi=3.0)

    summary_rows = []
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(logs_dir, f"run_{ts}.txt")
    with open(log_path, "w") as log:
        log.write(f"Run {ts}\nScenario={args.scenario} Duration={args.duration} Mode={args.mode} Assist={args.assist} Control={args.control}\n")

    # Run modes
    if args.mode=="both":
        res2d = run_one("2d", args.scenario, duration_s, args.control, args.assist=="on", spatial_params, t, v, outputs_dir)
        res3d = run_one("3d", args.scenario, duration_s, args.control, args.assist=="on", spatial_params, t, v, outputs_dir)
        # comparison plot
        # (already saved per-mode; here we synthesize summary only)
        summary_rows.append(["2D", *res2d.values()])
        summary_rows.append(["3D", *res3d.values()])
    elif args.mode=="spatial":
        res3d = run_one("3d", args.scenario, duration_s, args.control, args.assist=="on", spatial_params, t, v, outputs_dir)
        summary_rows.append(["3D", *res3d.values()])
    else:
        res2d = run_one("2d", args.scenario, duration_s, args.control, args.assist=="on", spatial_params, t, v, outputs_dir)
        summary_rows.append(["2D", *res2d.values()])

    # write summary CSV
    csv_path = os.path.join(outputs_dir, "performance_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Geometry","Avg_kW","Peak_kW","Energy_kWh","Eta_total","Coil_Tmax_C","Theta_max_deg"])
        for row in summary_rows:
            w.writerow(row)

    print("Done. Summary written to", csv_path)

if __name__ == "__main__":
    main()
