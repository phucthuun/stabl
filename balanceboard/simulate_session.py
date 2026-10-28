#!/usr/bin/env python3
"""Synthetic balance-board sessions for testing the pipeline.

Writes a raw log in the format of 05_balance_task.py and a truth file with the
true ground contacts, so the detection can be scored (see test_pipeline.py).

    python simulate_session.py OUT_DIR --seed 1 --rate 60 --compression 0.2

The board is simulated at 1 kHz: scripted falls, taps, long rests with
weight shifts, slow lowering, near misses, hovering close to the floor, stomps,
wall support and side switches. The floor is a hard limit (with a little
compression under load), so a contact happens whenever the scripted path
reaches it. The sensor model adds orientation noise and drift, mounting
rotation, heading drift, acceleration averaged over each sample interval (as
the DOT does), ringing at impacts, dropped packets and a wrapping tick counter.
Needs numpy.
"""
import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass

import numpy as np

import balance_core as bc

FS = 1000.0
LIFT_DEG = 3.0
LIFT_HOLD_S = 0.2


@dataclass
class SimConfig:
    seed: int = 1
    rate: float = 60.0
    trials: int = 7
    trial_s: float = 30.0
    countdown_s: float = 3.0
    break_s: float = 5.0
    theta_left: float = 11.4       # true floor angles from the true level (deg)
    theta_right: float = 12.6
    compression: float = 0.2       # extra tilt under full load (deg): hard floor ~0.2, mat ~0.8
    cal_depth: float = 1.5         # how hard the participant presses during calibration
    ori_noise: float = 0.03        # deg, white
    ori_slow: float = 0.08         # deg, slow drift of the orientation estimate
    impact_ori_err: float = 0.15   # deg transient per 50 deg/s impact
    acc_noise: float = 0.05        # m/s^2
    gyr_noise: float = 0.3         # deg/s
    sensor_r: float = 0.08         # m, sensor height above the rolling contact
    ring_gain: float = 0.06        # m/s^2 of ringing per deg/s of impact speed
    ticks_per_s: float = 1e6
    tick_start: int = (1 << 32) - 90_000_000
    drop_frac: float = 0.005
    drop_bursts: int = 3
    payload: str = "acc+gyr"       # acc+gyr | facc+gyr | orientation
    mount: str = "flat"            # flat | side
    shift_trial: int = 0           # sensor slips at the start of this trial ...
    shift_deg: float = 1.5         # ... by this much about the rocking axis


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def smooth_noise(rng, n, sd, tau_s):
    """Gaussian-smoothed white noise (kernel SD tau_s) scaled to standard deviation sd."""
    if sd == 0:
        return np.zeros(n)
    m = n + int(10 * tau_s * FS)
    w = rng.standard_normal(m)
    f = np.fft.rfftfreq(m, 1 / FS)
    y = np.fft.irfft(np.fft.rfft(w) * np.exp(-0.5 * (2 * np.pi * f * tau_s) ** 2), m)[:n]
    return y / (y.std() + 1e-12) * sd


class Script:
    """Builds the board's intended path as knots (time, tilt, velocity or None)."""

    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        self.th = {-1: cfg.theta_left, 1: cfg.theta_right}
        self.knots = [(0.0, 0.0, 0.0)]
        self.walls = []        # (t_on, t_off)
        self.phases = []       # (t, phase, trial)
        self.sway = []         # (t0, t1) balance segments that get sway noise
        self.intent = []       # scripted events for scoring
        self.stomps = []       # (t, magnitude)

    @property
    def t(self):
        return self.knots[-1][0]

    @property
    def x(self):
        return self.knots[-1][1]

    def to(self, dt, x, v=None):
        self.knots.append((self.t + max(dt, 0.002), float(x), v))

    def phase(self, name, trial=0):
        self.phases.append((self.t, name, trial))

    def u(self, a, b):
        return float(self.rng.uniform(a, b))

    # --- behaviours
    def balance(self, T, spread=1.6, sway=True):
        t0 = self.t
        end = self.t + T
        while self.t < end - 0.3:
            x = float(np.clip(self.rng.normal(0, spread), -4.5, 4.5))
            self.to(min(self.u(0.5, 1.2), end - self.t), x)
        if sway:
            self.sway.append((t0, self.t))

    def move(self, x, speed, v_end=0.0):
        self.to(1.6 * abs(x - self.x) / max(speed, 0.5), x, v_end)

    def fall(self, side, v, depth):
        th = self.th[side]
        self.move(side * th, v, side * v)
        t_c = self.t
        self.to(2 * depth / v, side * (th + depth), 0.0)
        return t_c

    def rest(self, side, T, depth=(0.4, 2.5), chatter=0.0):
        th = self.th[side]
        end = self.t + T
        while self.t < end - 0.2:
            if self.rng.random() < chatter and end - self.t > 0.8:
                lift = self.u(0.3, 2.0)
                self.to(self.u(0.12, 0.25), side * (th - lift), 0.0)
                self.to(self.u(0.12, 0.25), side * (th + self.u(*depth)), 0.0)
            else:
                d = depth[0] if depth[0] == depth[1] else self.u(*depth)
                self.to(min(self.u(0.3, 1.0), end - self.t), side * (th + d), 0.0)

    def lift(self, side, v, x_to=None):
        th = self.th[side]
        depth = abs(self.x) - th
        self.to(2 * max(depth, 0.05) / v, side * th, -side * v)
        x_to = float(self.rng.normal(0, 1.0)) if x_to is None else x_to
        self.move(x_to, v)

    def fall_rest(self, side, rest_T, chatter=0.4, v=None):
        v = v or self.u(15, 80)
        tc = self.fall(side, v, self.u(0.4, 2.5))
        self.rest(side, rest_T, chatter=chatter)
        self.lift(side, self.u(5, 30))
        self.intent.append({"kind": "fall_rest", "t": tc, "side": side, "v": v})

    def tap(self, side):
        th = self.th[side]
        v = self.u(15, 70)
        tau = self.u(0.004, 0.05)
        v_reb = v * self.u(0.4, 1.0)
        self.move(side * th, v, side * v)
        tc = self.t
        peak = v * tau / 4
        self.to(tau / 2, side * (th + peak), 0.0)
        self.to(2 * peak / v_reb, side * th, -side * v_reb)
        self.move(float(self.rng.normal(0, 1.0)), v_reb)
        self.intent.append({"kind": "tap", "t": tc, "side": side, "v": v, "tau": tau})

    def near_miss(self, side):
        th = self.th[side]
        m = self.u(0.25, 3.0)
        v = self.u(8, 50)
        a = self.u(80, 1500)
        dist = min(v * v / (2 * a), th - m - 2.0)
        a = v * v / (2 * dist)
        self.move(side * (th - m - dist), v, side * v)
        self.to(v / a, side * (th - m), 0.0)
        t_peak = self.t
        self.move(float(self.rng.normal(0, 1.0)), self.u(8, 30))
        self.intent.append({"kind": "near_miss", "t": t_peak, "side": side, "m": m, "v": v, "decel": a})

    def hover(self, side):
        th = self.th[side]
        d = self.u(0.6, 1.6)
        self.move(side * (th - d - 1.0), self.u(10, 20))
        self.to(1.0 / self.u(2, 6), side * (th - d), 0.0)
        t0 = self.t
        T = self.u(0.4, 1.5)
        self.to(T, side * (th - d), 0.0)
        self.move(float(self.rng.normal(0, 1.0)), self.u(5, 15))
        self.intent.append({"kind": "hover", "t": t0, "t1": t0 + T, "side": side, "m": d})

    def stomp(self, side):
        th = self.th[side]
        d = self.u(1.6, 2.5)
        self.move(side * (th - d), self.u(10, 25))
        self.to(0.3, side * (th - d), 0.0)
        ts = self.t
        self.stomps.append((ts, self.u(5, 12)))
        self.move(float(self.rng.normal(0, 1.0)), self.u(20, 40))
        self.intent.append({"kind": "stomp", "t": ts, "side": side, "m": d})

    def slow_rest(self, side):
        th = self.th[side]
        v = self.u(1, 4)
        self.move(side * (th - 1.5), self.u(8, 20), side * v)
        self.to(1.5 / v, side * th, side * v)
        tc = self.t
        self.to(0.4, side * (th + 0.6), 0.0)
        self.rest(side, self.u(1, 6), chatter=0.2)
        self.lift(side, self.u(4, 15))
        self.intent.append({"kind": "slow_rest", "t": tc, "side": side, "v": v})

    def switch(self, side):
        th = self.th[side]
        v = self.u(30, 80)
        tc = self.fall(side, self.u(20, 60), self.u(0.5, 2.0))
        self.rest(side, self.u(0.4, 1.5))
        self.to(2 * (abs(self.x) - th) / 20 + 0.01, side * th, -side * 20)
        other = -side
        self.move(other * self.th[other], v, other * v)
        tc2 = self.t
        self.to(0.05, other * (self.th[other] + 1.0), 0.0)
        self.rest(other, self.u(0.5, 2.0))
        self.lift(other, self.u(5, 25))
        self.intent.append({"kind": "switch", "t": tc, "t2": tc2, "side": side, "v": v})

    def support(self, touch):
        t_on = self.t
        self.balance(self.u(0.5, 1.0), spread=1.0)
        if touch:
            side = int(self.rng.choice([-1, 1]))
            self.fall(side, self.u(5, 30), self.u(0.5, 2.0))
            self.rest(side, self.u(0.5, 2.0))
            self.lift(side, self.u(5, 20))
        self.balance(self.u(0.5, 1.5), spread=1.0)
        self.walls.append((t_on, self.t))

    def trial(self, T):
        end = self.t + T
        kinds = ["fall_rest", "tap", "near_miss", "slow_rest", "support", "support_touch",
                 "switch", "hover", "stomp", "long_rest"]
        prob = np.array([0.20, 0.15, 0.24, 0.06, 0.05, 0.05, 0.05, 0.07, 0.06, 0.07])
        prob /= prob.sum()
        while self.t < end - 6.0:
            self.balance(self.u(0.8, 2.5))
            kind = self.rng.choice(kinds, p=prob)
            side = int(self.rng.choice([-1, 1]))
            if kind == "fall_rest":
                self.fall_rest(side, self.u(0.3, 3.0))
            elif kind == "long_rest":
                self.fall_rest(side, self.u(5.0, 9.0), chatter=0.5)
            elif kind == "tap":
                self.tap(side)
            elif kind == "near_miss":
                self.near_miss(side)
            elif kind == "slow_rest":
                self.slow_rest(side)
            elif kind in ("support", "support_touch"):
                self.support(kind == "support_touch")
            elif kind == "switch":
                self.switch(side)
            elif kind == "hover":
                self.hover(side)
            elif kind == "stomp":
                self.stomp(side)
        if end - self.t > 0.3:
            self.balance(end - self.t)


def build_script(cfg, rng):
    s = Script(cfg, rng)
    s.phase("setup")
    wall_t0 = s.t
    s.balance(2.0, spread=0.6)
    s.phase("cal_quiet")
    s.balance(5.0, spread=0.4)
    s.phase("setup")
    for side, name in ((-1, "cal_left"), (1, "cal_right")):
        s.fall(side, 8.0, cfg.cal_depth)
        s.rest(side, 0.8, depth=(cfg.cal_depth, cfg.cal_depth))
        s.phase(name)
        s.rest(side, 3.0, depth=(cfg.cal_depth, cfg.cal_depth))
        s.phase("setup")
        s.lift(side, 10.0, x_to=0.0)
    s.balance(1.0, spread=0.6)
    s.walls.append((wall_t0, s.t))
    for k in range(1, cfg.trials + 1):
        s.phase("countdown")
        t_on = s.t
        grounded = rng.random() < 0.2
        if grounded:
            side = int(rng.choice([-1, 1]))
            s.fall(side, 6.0, 1.2)
            s.rest(side, cfg.countdown_s)
        else:
            s.balance(cfg.countdown_s, spread=0.8)
        s.phase("trial", k)
        t_go = s.t
        if grounded:
            s.rest(side, s.u(0.3, 1.0))
            s.walls.append((t_on, t_go + s.u(0.0, 0.4)))
            s.lift(side, s.u(5, 20))
        else:
            s.walls.append((t_on, t_go + s.u(0.0, 0.3)))
        s.trial(cfg.trial_s - (s.t - t_go))
        s.phases.append((max(t_go + cfg.trial_s, s.phases[-1][0] + 1e-3), "break", 0))
        side = int(rng.choice([-1, 1]))
        s.fall(side, 15.0, 0.2)
        s.rest(side, cfg.break_s, depth=(0.2, 0.2))
        s.lift(side, 10.0, x_to=0.0)
        s.balance(1.0, spread=0.5, sway=False)
    s.phase("end")
    s.balance(1.0, spread=0.3)
    return s


def hermite(knots, tt):
    kt = np.array([k[0] for k in knots])
    kx = np.array([k[1] for k in knots])
    kv = np.zeros(len(knots))
    for i, k in enumerate(knots):
        if k[2] is not None:
            kv[i] = k[2]
        elif 0 < i < len(knots) - 1:
            kv[i] = (kx[i + 1] - kx[i - 1]) / (kt[i + 1] - kt[i - 1])
    j = np.clip(np.searchsorted(kt, tt, side="right") - 1, 0, len(kt) - 2)
    T = kt[j + 1] - kt[j]
    s = np.clip((tt - kt[j]) / T, 0, 1)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return h00 * kx[j] + h10 * T * kv[j] + h01 * kx[j + 1] + h11 * T * kv[j + 1]


def euler_zyx(R):
    roll = np.degrees(np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(R[:, 2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return roll, pitch, yaw


def quat_from_R(R):
    w = np.sqrt(np.maximum(0, 1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])) / 2
    x = np.copysign(np.sqrt(np.maximum(0, 1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2])) / 2, R[:, 2, 1] - R[:, 1, 2])
    y = np.copysign(np.sqrt(np.maximum(0, 1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2])) / 2, R[:, 0, 2] - R[:, 2, 0])
    z = np.copysign(np.sqrt(np.maximum(0, 1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2])) / 2, R[:, 1, 0] - R[:, 0, 1])
    return w, x, y, z


def rotvec_to_R(e):
    """Small rotation vectors (N,3) in rad -> (N,3,3)."""
    th = np.linalg.norm(e, axis=1)
    k = e / np.maximum(th, 1e-12)[:, None]
    K = np.zeros((len(e), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = np.eye(3)[None]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return I + s * K + (1 - c) * (K @ K)


def episodes_from_contact(tt, contact, side_arr, tilt, th):
    """Merge raw contact runs with the same lift-off rule as the detector."""
    runs = []
    d = np.diff(np.concatenate([[0], contact.astype(int), [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    for a, b in zip(starts, ends):
        runs.append((a, b, int(side_arr[a])))
    eps = []
    for a, b, side in runs:
        if eps:
            pb, pside = eps[-1]["i1"], eps[-1]["side"]
            gap = slice(pb, a)
            lifted = side != pside
            if not lifted:
                m = th[pside] - pside * tilt[gap]
                if np.any(m >= th[pside]):
                    lifted = True
                else:
                    up = m >= LIFT_DEG
                    if up.any():
                        dd = np.diff(np.concatenate([[0], up.astype(int), [0]]))
                        lens = np.where(dd == -1)[0] - np.where(dd == 1)[0]
                        lifted = lens.max() >= LIFT_HOLD_S * FS
            if not lifted:
                eps[-1]["i1"] = b
                continue
        eps.append({"i0": a, "i1": b, "side": side})
    for e in eps:
        e["onset"] = float(tt[e["i0"]])
        e["end"] = float(tt[e["i1"] - 1] + 1 / FS)
    return eps


def simulate(cfg, out_dir, name="sim"):
    rng = np.random.default_rng(cfg.seed)
    s = build_script(cfg, rng)
    T = s.t + 0.5
    n = int(T * FS)
    tt = np.arange(n) / FS

    # --- the board
    path = hermite(s.knots, tt)
    sway_w = np.zeros(n)
    for a, b in s.sway:
        i0, i1 = int(a * FS), min(int(b * FS), n)
        if i1 - i0 < 4:
            continue
        w = np.ones(i1 - i0)
        r = min(int(0.25 * FS), len(w) // 2)
        w[:r] = np.linspace(0, 1, r)
        w[-r:] = np.linspace(1, 0, r)
        sway_w[i0:i1] = np.maximum(sway_w[i0:i1], w)
    path = path + sway_w * smooth_noise(rng, n, 0.5, 0.12)
    th = {-1: cfg.theta_left, 1: cfg.theta_right}
    tilt = path.copy()
    dR = path - th[1]
    dL = -path - th[-1]
    r, l = dR > 0, dL > 0
    tilt[r] = th[1] + cfg.compression * np.minimum(1, dR[r] / 1.5)
    tilt[l] = -(th[-1] + cfg.compression * np.minimum(1, dL[l] / 1.5))
    contact = r | l
    side_arr = np.where(path >= 0, 1, -1)

    omega = np.zeros(n)
    omega[:-1] = np.diff(np.radians(tilt)) * FS
    alpha = np.zeros(n)
    alpha[1:] = np.diff(omega) * FS
    pvel = np.gradient(path) * FS

    beta = np.radians(smooth_noise(rng, n, 0.25, 0.4))
    yaw_rate = smooth_noise(rng, n, 8.0, 0.6)
    psi = np.radians(np.cumsum(yaw_rate) / FS)

    # --- truth
    eps = episodes_from_contact(tt, contact, side_arr, tilt, th)
    onsets = np.where(np.diff(np.concatenate([[0], contact.astype(int)])) == 1)[0]
    for e in eps:
        i = e["i0"]
        e["speed_dps"] = float(abs(pvel[max(i - 1, 0)]))

    # --- acceleration in the board frame (1 kHz), then sensor frame
    rs = cfg.sensor_r
    aB = np.zeros((n, 3))
    aB[:, 1] = -rs * alpha
    aB[:, 2] = -rs * omega ** 2
    for i in onsets:
        v = abs(pvel[max(i - 1, 0)])
        k = np.arange(i, min(i + 150, n))
        dt = (k - i) / FS
        aB[k, 2] += cfg.ring_gain * v * np.exp(-dt / 0.025) * np.sin(2 * np.pi * 35 * dt)
    for ts, mag in s.stomps:
        i = int(ts * FS)
        aB[i:i + 8, 2] += mag
    for ax in range(3):
        aB[:, ax] += smooth_noise(rng, n, 0.12, 0.05)
    phi = np.radians(tilt)
    # gravity direction in the board frame: Ry(beta)^T Rx(phi)^T e_z
    ux = -np.sin(beta) * np.cos(phi)
    uy = np.sin(phi)
    uz = np.cos(beta) * np.cos(phi)
    uB = np.stack([ux, uy, uz], axis=1)
    fB = aB + 9.81 * uB
    wB = np.stack([omega, np.gradient(beta) * FS, np.radians(yaw_rate)], axis=1)

    # --- sensor mounting
    if cfg.mount == "side":
        R_bs = rot_x(math.radians(90)) @ rot_z(math.radians(rng.uniform(0, 360)))
    else:
        R_bs = rot_z(math.radians(rng.uniform(0, 360))) @ rot_x(math.radians(rng.uniform(-8, 8))) \
            @ rot_y(math.radians(rng.uniform(-8, 8)))

    # --- output samples
    K = int(T * cfg.rate) - 1
    tk = (np.arange(1, K + 1)) / cfg.rate
    ik = np.minimum(np.round(tk * FS).astype(int), n - 1)
    edges = np.concatenate([[0], ik[:-1] + 1])

    phase_t = np.array([p[0] for p in s.phases])
    pj = np.searchsorted(phase_t, tk, side="right") - 1
    phase = [s.phases[j][1] for j in pj]
    trial = [s.phases[j][2] if s.phases[j][1] == "trial" else 0 for j in pj]
    wall = np.zeros(K, dtype=bool)
    for a, b in s.walls:
        wall |= (tk >= a) & (tk < b)
    shift_t = None
    if cfg.shift_trial:
        starts = [p[0] for p in s.phases if p[1] == "trial" and p[2] == cfg.shift_trial]
        shift_t = starts[0] if starts else None
    R_bs_k = np.repeat(R_bs[None], K, axis=0)
    if shift_t is not None:
        R_bs_k[tk >= shift_t] = rot_x(math.radians(cfg.shift_deg)) @ R_bs

    f_avg = np.add.reduceat(fB, edges, axis=0) / np.diff(np.concatenate([edges, [ik[-1] + 1]]))[:, None]
    w_avg = np.add.reduceat(wB, edges, axis=0) / np.diff(np.concatenate([edges, [ik[-1] + 1]]))[:, None]
    f_s = np.einsum("kji,kj->ki", R_bs_k, f_avg) + rng.standard_normal((K, 3)) * cfg.acc_noise
    w_s = np.degrees(np.einsum("kji,kj->ki", R_bs_k, w_avg)) + rng.standard_normal((K, 3)) * cfg.gyr_noise

    # board orientation at the sample times, then sensor orientation with estimation error
    R_wb = np.zeros((K, 3, 3))
    for m, i in enumerate(ik):
        R_wb[m] = rot_z(psi[i]) @ rot_x(phi[i]) @ rot_y(beta[i])
    R_ws = R_wb @ R_bs_k
    err = np.radians(rng.standard_normal((K, 3)) * cfg.ori_noise)
    slow = np.stack([smooth_noise(rng, n, cfg.ori_slow, 3.0)[ik] for _ in range(3)], axis=1)
    err += np.radians(slow)
    for i in onsets:
        v = abs(pvel[max(i - 1, 0)])
        t_c = i / FS
        sel = (tk >= t_c) & (tk < t_c + 1.0)
        direc = rng.standard_normal(3)
        direc /= np.linalg.norm(direc)
        err[sel] += np.radians(cfg.impact_ori_err * v / 50.0) * np.exp(-(tk[sel] - t_c) / 0.3)[:, None] * direc
    R_est = R_ws @ rotvec_to_R(err)
    roll, pitch, yaw = euler_zyx(R_est)
    qw, qx, qy, qz = quat_from_R(R_est)
    facc = np.einsum("kij,kj->ki", R_est, f_s) - np.array([0, 0, 9.81])

    # --- drops and ticks
    keep = rng.random(K) >= cfg.drop_frac
    for _ in range(cfg.drop_bursts):
        a = int(rng.integers(0, K - 20))
        keep[a:a + int(rng.integers(3, 15))] = False
    ticks = (cfg.tick_start + np.round(tk * cfg.ticks_per_s).astype(np.int64)) % (1 << 32)

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"{name}_beh.csv")
    clock = bc.SampleClock(cfg.ticks_per_s)
    t_first = None
    with open(log_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(bc.LOG_COLUMNS)
        for m in range(K):
            if not keep[m]:
                continue
            u = clock.unwrap(int(ticks[m]))
            ts = clock.seconds(u)
            if t_first is None:
                t_first = tk[m]
            acc = f_s[m] if cfg.payload == "acc+gyr" else None
            gyr = w_s[m] if cfg.payload in ("acc+gyr", "facc+gyr") else None
            fa = facc[m] if cfg.payload == "facc+gyr" else None
            row = [f"{ts:.6f}", u, phase[m], trial[m], int(wall[m]),
                   f"{roll[m]:.4f}", f"{pitch[m]:.4f}", f"{yaw[m]:.4f}",
                   f"{qw[m]:.6f}", f"{qx[m]:.6f}", f"{qy[m]:.6f}", f"{qz[m]:.6f}"]
            row += [f"{v:.4f}" for v in acc] if acc is not None else ["", "", ""]
            row += [f"{v:.3f}" for v in gyr] if gyr is not None else ["", "", ""]
            row += [f"{v:.4f}" for v in fa] if fa is not None else ["", "", ""]
            row += ["", "", "", "", ""]
            w.writerow(row)

    # --- truth file (times in log time: log t = sim time - t_first)
    for it in s.intent:
        if it["kind"] in ("near_miss", "hover", "stomp"):
            win = (tt >= it["t"] - 0.6) & (tt <= it.get("t1", it["t"]) + 0.6)
            sd = it["side"]
            it["true_min_margin"] = float(np.min(th[sd] - sd * tilt[win]))
            it["touched"] = bool(contact[win].any())
    off = t_first
    for e in eps:
        e["onset"] -= off
        e["end"] -= off
        del e["i0"], e["i1"]
    for it in s.intent:
        for k in ("t", "t1", "t2"):
            if k in it:
                it[k] -= off
    trial_bounds = {}
    for k_ in range(1, cfg.trials + 1):
        sel = [tk[m] for m in range(K) if trial[m] == k_]
        trial_bounds[k_] = (sel[0] - off, sel[-1] - off)
    walls = [(a - off, b - off) for a, b in s.walls]
    loaded_cal = {sd: th[sd] + cfg.compression * min(1, cfg.cal_depth / 1.5) for sd in (-1, 1)}
    truth = {"config": asdict(cfg), "episodes": eps, "intent": s.intent, "walls": walls,
             "trials": trial_bounds, "theta_true": th, "theta_cal_loaded": loaded_cal,
             "n_samples": int(keep.sum()), "n_dropped": int((~keep).sum())}
    truth_path = os.path.join(out_dir, f"{name}_truth.json")
    with open(truth_path, "w") as fh:
        json.dump(truth, fh, indent=1, default=float)
    return log_path, truth_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--name", default="sim")
    for f, v in asdict(SimConfig()).items():
        ap.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args()
    cfg = SimConfig(**{f: getattr(a, f) for f in asdict(SimConfig())})
    log, truth = simulate(cfg, a.out_dir, a.name)
    print(log)
    print(truth)


if __name__ == "__main__":
    main()
