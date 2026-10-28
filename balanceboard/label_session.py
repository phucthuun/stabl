#!/usr/bin/env python3
"""Final (offline) labels for a balance-board session.

    python label_session.py sub-XX_ses-S001_task-balanceboard_run-001_beh.csv

Recomputes tilt from the raw log, refines each side's contact angle from the
moments the board rests on the floor, runs the same contact detector as the
live script and writes, next to the raw log:

    <base>_events.tsv (+ .json)   state periods, contacts and trials (BIDS events)
    <base>_desc-labels_beh.csv    one row per sample: tilt, margin, contact, state
    <base>_desc-trials_beh.csv    one row per trial: time per state, fails, ...
    <base>_desc-contacts_beh.csv  one row per contact: onset, duration, side, cue
    <base>_qc/report.txt          calibration, which cue detected what, warnings
    <base>_qc/trial-XX.png        tilt, contacts and states per trial (needs matplotlib)
    <base>_qc/contact_angles.png  where the board rested on each side

Thresholds can be changed with options (see --help) to re-label a session
without testing anyone again. Also reads logs from 04_balance_task.py: without
calibration phases it finds the two floor contacts in the data itself.
"""
import argparse
import csv
import json
import math
import os
from collections import Counter, deque

import balance_core as bc

# ---------------------------------------------------------------------- loading


def _f(v):
    if v is None or v == "" or v == "n/a":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


class RawSession:
    """Columns of a raw log as Python lists."""

    def __init__(self, path):
        self.path = path
        self.warnings = []
        with open(path, newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            rows = [r for r in reader if len(r) == len(header)]
        self.header = header
        col = {name: k for k, name in enumerate(header)}
        self.legacy = "phase" not in col

        def get(name, conv=_f):
            k = col.get(name)
            return None if k is None else [conv(r[k]) for r in rows]

        t = get("t_s")
        if t is None:
            raise ValueError(f"{path}: no t_s column")
        keep = [i for i, v in enumerate(t) if v is not None]
        order = sorted(keep, key=lambda i: t[i])
        if order != keep:
            self.warnings.append("samples were not in time order; sorted them")
        # drop duplicate timestamps
        ordered = []
        last = None
        for i in order:
            if last is None or t[i] > last:
                ordered.append(i)
                last = t[i]
        if len(ordered) < len(keep):
            self.warnings.append(f"{len(keep) - len(ordered)} samples with repeated timestamps dropped")
        rows = [rows[i] for i in ordered]

        self.n = len(rows)
        self.t = get("t_s")
        self.roll = get("roll")
        self.pitch = get("pitch")
        if self.roll is None or self.pitch is None:
            raise ValueError(f"{path}: needs roll and pitch columns")
        self.trial = [int(v) if v is not None else 0 for v in get("trial")] if "trial" in col else [0] * self.n
        self.wall = [bool(v) for v in get("wall")] if "wall" in col else [False] * self.n
        if self.legacy:
            self.phase = ["trial" if tr > 0 else "" for tr in self.trial]
        else:
            self.phase = get("phase", conv=lambda v: v)

        def vec(prefix):
            xs = [get(prefix + a) for a in ("_x", "_y", "_z")]
            if any(x is None for x in xs):
                return None
            out = [(a, b, c) if None not in (a, b, c) else None for a, b, c in zip(*xs)]
            return out if any(v is not None for v in out) else None

        self.acc = vec("acc")
        self.gyr = vec("gyr")
        self.facc = vec("facc")
        self.acc_legacy = get("acc_dyn_ms2")
        self.state_online = get("state_online", conv=lambda v: v) if "state_online" in col else None


def fix_time_units(t, nominal_rate=None):
    """Rescale t if its unit is off by a power of ten (e.g. logs from 04 that
    divided microseconds by 10000). Returns (t, factor)."""
    dts = [b - a for a, b in zip(t[:5000], t[1:5001]) if b > a]
    if not dts:
        return t, 1.0
    med = bc.median(dts)
    target = 1.0 / nominal_rate if nominal_rate else None
    if target is None:
        if 1 / 400 <= med <= 1 / 5:
            return t, 1.0
        target = 1 / 60
    f = 10 ** round(math.log10(med / target))
    if f == 1:
        return t, 1.0
    return [v / f for v in t], f


# ------------------------------------------------------------------ calibration


def still_mask(times, values, win_s, max_range):
    """True where the tilt range within a centred window of win_s is <= max_range."""
    n = len(times)
    out = [False] * n
    half = win_s / 2.0
    maxq, minq = deque(), deque()
    hi = 0
    for i in range(n):
        ti = times[i]
        while hi < n and times[hi] <= ti + half:
            v = values[hi]
            while maxq and values[maxq[-1]] <= v:
                maxq.pop()
            maxq.append(hi)
            while minq and values[minq[-1]] >= v:
                minq.pop()
            minq.append(hi)
            hi += 1
        while maxq and times[maxq[0]] < ti - half:
            maxq.popleft()
        while minq and times[minq[0]] < ti - half:
            minq.popleft()
        if maxq and minq:
            out[i] = values[maxq[0]] - values[minq[0]] <= max_range
    return out


def _cluster(vals, start, width=0.6, rounds=3):
    c = start
    for _ in range(rounds):
        near = [v for v in vals if abs(v - c) <= width]
        if not near:
            return None, []
        c = bc.median(near)
    return c, [v for v in vals if abs(v - c) <= width]


def refine_side(vals, rate, p, reach=None):
    """Contact angle of one side from tilt values while the board was resting near
    that floor (side * tilt, within refine_search_deg of the expected angle).

    The floor is a hard limit, so the rests on it form the outermost cluster;
    holding still just above the floor would form a cluster further in. `reach`
    (side * tilt of all samples) rejects an estimate the board regularly went
    past, which a floor would not allow."""
    min_n = p.refine_min_s * rate
    if len(vals) < min_n:
        return None
    w = 0.1
    bins = Counter(int(math.floor(v / w)) for v in vals)
    lo, hi = min(bins), max(bins)
    mass = {k: sum(bins.get(j, 0) for j in range(k - 5, k + 6)) for k in range(lo, hi + 1)}
    peaks = [k for k in range(lo, hi + 1)
             if mass[k] >= min_n and mass[k] >= mass.get(k - 1, 0) and mass[k] >= mass.get(k + 1, 0)]
    if not peaks:
        return None
    c, near = _cluster(vals, (max(peaks) + 0.5) * w)
    if c is None or len(near) < min_n:
        return None
    sigma = 1.4826 * bc.median([abs(v - c) for v in near])
    # use the inner edge of the cluster: on a soft floor a light touch stops short
    # of where the board rests under full weight
    edge = min(c, bc.percentile(near, 10))
    out = {"theta": edge, "center": c, "sigma": sigma, "seconds": len(near) / rate}
    if reach is not None and sum(1 for v in reach if v > c + max(1.0, 4 * sigma)) >= 3:
        return None
    return out


def _hinge(ts, ms, descending):
    """Least-squares knee of 'slope then flat' (descending) or 'flat then slope'."""
    n = len(ts)
    sm = sum(ms)
    best = None
    for k in range(2, n - 2):
        tk = ts[k]
        xs = [max(0.0, tk - x) for x in ts] if descending else [max(0.0, x - tk) for x in ts]
        sx = sum(xs)
        sxx = sum(x * x for x in xs)
        sxm = sum(x * m for x, m in zip(xs, ms))
        det = n * sxx - sx * sx
        if det <= 1e-12:
            continue
        s = (n * sxm - sx * sm) / det
        L = (sm - s * sx) / n
        sse = sum((m - L - s * x) ** 2 for x, m in zip(xs, ms))
        if best is None or sse < best[0]:
            best = (sse, k, L, s)
    return best


def find_sensor_shift(trial_est, theta_cal, thr):
    """First trial from which the sensor sat at a different angle on the board.

    A sensor that turns on the board shifts both floors by the same amount in
    opposite directions. Standing heavier or lighter moves both floors the same
    way, and holding still just above a floor only makes one side look shorter,
    so neither can mimic it. Tries splitting the session before each trial and
    keeps the split whose before/after medians of the trials' floor rests (the
    calibration counts as one more 'before') fit the trials best, provided the
    two floors moved in opposite directions by about the same amount.
    Returns (trial, {side: shift}) or (None, None)."""
    trials = sorted(trial_est)
    best = None
    for k in trials[1:]:
        d, resid = {}, 0.0
        for s in (-1, 1):
            a = [theta_cal[s]] + [trial_est[tr][s]["center"] for tr in trials if tr < k and trial_est[tr][s]]
            b = [trial_est[tr][s]["center"] for tr in trials if tr >= k and trial_est[tr][s]]
            if len(a) < 2 or not b:
                break
            ma, mb = bc.median(a), bc.median(b)
            d[s] = mb - ma
            resid += sum(abs(v - ma) for v in a) + sum(abs(v - mb) for v in b)
        else:
            if (d[-1] * d[1] < 0 and abs(d[-1] + d[1]) <= 0.3 and max(abs(d[-1]), abs(d[1])) >= thr
                    and min(abs(d[-1]), abs(d[1])) >= thr / 2):
                if best is None or resid < best[0] - 1e-9:
                    best = (resid, k, d)
    return (best[1], best[2]) if best else (None, None)


def refine_timing(contacts, t, side_margin, band):
    """Onsets of slow contacts -> where the board arrived at the floor; ends -> where
    it left. Fits 'slope then flat' / 'flat then slope' to the margin of the contact side."""
    import bisect
    for q, c in enumerate(contacts):
        prev_end = contacts[q - 1].end if q else -math.inf
        next_on = contacts[q + 1].onset if q + 1 < len(contacts) else math.inf
        if c.cue != "impact" and (c.cue == "rest" or c.approach_dps < 15):
            t_cue = max(c.cues.values())
            a = bisect.bisect_left(t, max(prev_end, c.onset - 0.6))
            b = bisect.bisect_right(t, min(c.end, t_cue + 0.4))
            if b - a >= 8:
                ts = t[a:b]
                fit = _hinge(ts, [side_margin(i, c.side) for i in range(a, b)], True)
                if fit and fit[3] > 0.5 and fit[2] <= band:
                    new = ts[fit[1]]
                    if abs(new - c.onset) <= 0.8 and new < c.end:
                        c.onset = new
        if c.end - c.onset > 0.15:
            a = bisect.bisect_left(t, max(c.onset, c.end - 0.6))
            b = bisect.bisect_right(t, min(next_on, c.end + 0.4))
            if b - a >= 8:
                ts = t[a:b]
                fit = _hinge(ts, [side_margin(i, c.side) for i in range(a, b)], False)
                if fit and fit[3] > 0.5 and fit[2] <= band:
                    k = fit[1]
                    new = ts[k + 1] if k + 1 < len(ts) else ts[k]
                    if abs(new - c.end) <= 0.5 and new > c.onset:
                        c.end = new


def self_calibrate(times, ups, select, p, rate):
    """Find the rocking axis and both floor contacts in the data itself."""
    idx = [i for i in range(len(ups)) if select[i]]
    if len(idx) < 10 * rate:
        return None, "not enough data"
    mean = bc.unit(bc.vsum(ups[i] for i in idx))
    C = [[0.0] * 3 for _ in range(3)]
    for i in idx:
        d = bc.sub(ups[i], mean)
        for a in range(3):
            for b in range(3):
                C[a][b] += d[a] * d[b]
    v = max(((C[0][0], 0), (C[1][1], 1), (C[2][2], 2)))[1]
    v = bc.unit(tuple(C[k][v] for k in range(3)))
    for _ in range(200):
        v = bc.unit((bc.dot(C[0], v), bc.dot(C[1], v), bc.dot(C[2], v)))
    cal0 = bc.Calibration(bc.cross(mean, v), mean, 0.0, 0.0, "provisional")
    tilt0 = [cal0.tilt(u) for u in ups]
    still = still_mask(times, tilt0, p.refine_still_win_s, p.refine_still_range_deg)
    sel = [tilt0[i] for i in idx]
    ends = {}
    for side, q in ((-1, 0.5), (1, 99.5)):
        c = bc.percentile(sel, q)
        members = []
        for width in (1.5, 0.6, 0.6):
            members = [i for i in idx if still[i] and abs(tilt0[i] - c) <= width]
            if len(members) < p.refine_min_s * rate:
                return None, f"found no floor rest on the {bc.SIDE_NAME[side]} side"
            c = bc.median([tilt0[i] for i in members])
        ends[side] = bc.unit(bc.vsum(ups[i] for i in members))
    if bc.angle_deg(ends[-1], ends[1]) < 2 * p.min_contact_deg:
        return None, "the two ends are too close together to be floor contacts"
    return bc.Calibration.from_contacts(ends[-1], ends[1], source="from the data (no calibration phases)"), None


# ------------------------------------------------------------------------- main


def last_block(idx):
    """Last run of consecutive sample indices."""
    k = len(idx) - 1
    while k > 0 and idx[k - 1] >= idx[k] - 3:
        k -= 1
    return idx[k:]


def rocking_rate(t, tilt, gyr, axis):
    """Rocking rate (deg/s) from the gyroscope about the rocking axis, scaled and signed
    to match the tilt (handles deg/s or rad/s). (None, None) without a gyroscope."""
    if gyr is None:
        return None, None
    raw = [None if g is None else bc.dot(g, axis) for g in gyr]
    dtilt = [None] * len(t)
    for i in range(1, len(t) - 1):
        if t[i + 1] > t[i - 1]:
            dtilt[i] = (tilt[i + 1] - tilt[i - 1]) / (t[i + 1] - t[i - 1])
    k = bc.gyro_scale(raw, dtilt)
    if k is None:
        return None, None
    return [0.0 if r is None else k * r for r in raw], ("gyroscope (rad/s)" if abs(k) > 2 else "gyroscope (deg/s)")


def contact_index(times, contacts):
    idx = [-1] * len(times)
    import bisect
    for k, c in enumerate(contacts):
        end = c.end if c.end is not None else float("inf")
        a = bisect.bisect_left(times, c.onset)
        b = bisect.bisect_left(times, end)
        for i in range(a, b):
            idx[i] = k
    return idx


def base_name(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:-4] if stem.endswith("_beh") else stem


def label_session(log_path, params=None, calib_path=None, outdir=None, plots=True,
                  verbose=True, refine=None, board=None):
    p = params or bc.Params()
    if refine is not None:
        p.refine = refine
    raw = RawSession(log_path)
    warnings = list(raw.warnings)
    notes = []
    outdir = outdir or os.path.dirname(os.path.abspath(log_path))
    base = os.path.join(outdir, base_name(log_path))
    if calib_path is None:
        guess = os.path.join(os.path.dirname(os.path.abspath(log_path)), base_name(log_path) + "_calibration.json")
        calib_path = guess if os.path.exists(guess) else None
    meta = {}
    if calib_path:
        with open(calib_path) as fh:
            meta = json.load(fh)

    p.board = board or meta.get("board") or p.board

    # ---- time base
    t, factor = fix_time_units(raw.t, meta.get("sample_rate_hz"))
    if factor != 1.0:
        warnings.append(f"time stamps were off by a factor of {factor:g}; rescaled" +
                        (" (04 divided microseconds by 10000)" if factor == 100 else ""))
    n = raw.n
    dts = [b - a for a, b in zip(t, t[1:])]
    dt_med = bc.median(dts) if dts else 1 / 60
    rate = 1.0 / dt_med
    gaps = [(t[i], d) for i, d in enumerate(dts) if d > 1.5 * dt_med]
    missing = sum(round(d / dt_med) - 1 for _, d in gaps)
    long_gaps = [(a, d) for a, d in gaps if d > 0.1]

    # ---- orientation
    ups = [bc.up_from_euler(r, q) for r, q in zip(raw.roll, raw.pitch)]
    phases = raw.phase
    by_phase = {}
    for i, ph in enumerate(phases):
        by_phase.setdefault(ph, []).append(i)

    # ---- calibration
    cal = None
    cal_note = ""
    holds = meta.get("cal_holds") or {}

    def cal_samples(phase, key):
        """Samples of the calibration hold the task script accepted (repeats are in the log too)."""
        idx = by_phase.get(phase, [])
        if key in holds:
            a, b = holds[key]
            sel = [i for i in idx if a - 1e-6 <= t[i] <= b + 1e-6]
            if sel:
                return sel
        return last_block(idx)

    if "cal_left" in by_phase and "cal_right" in by_phase:
        left, right = cal_samples("cal_left", "left"), cal_samples("cal_right", "right")
        okl, ul, fl, _ = bc.rest_direction([ups[i] for i in left], p.cal_still_dev_deg, p.cal_still_frac)
        okr, ur, fr, _ = bc.rest_direction([ups[i] for i in right], p.cal_still_dev_deg, p.cal_still_frac)
        if bc.angle_deg(ul, ur) >= 2 * p.min_contact_deg:
            cal = bc.Calibration.from_contacts(ul, ur, source="calibration phases in the log")
            if not (okl and okr):
                warnings.append(f"calibration holds were not still (still fraction left {fl:.0%}, right {fr:.0%})")
        else:
            warnings.append("calibration contacts in the log are too close together; ignoring them")
    if cal is None and meta.get("calibration"):
        cal = bc.Calibration.from_dict(meta["calibration"])
        cal.source = "calibration file"
    if cal is None:
        # 04 logs: the hand-pressed calibration holds are in the log but not marked, so use everything
        select = [True] * n if raw.legacy else [ph in bc.LOADED_PHASES or raw.trial[i] > 0
                                                for i, ph in enumerate(phases)]
        if not any(select):
            select = [True] * n
        cal, why = self_calibrate(t, ups, select, p, rate)
        if cal is None:
            raise RuntimeError(f"cannot calibrate: {why}")
        cal_note = "side names (left/right) may be swapped"
        warnings.append("no calibration phases found; contact angles were taken from the data")

    tilt = [cal.tilt(u) for u in ups]
    off_axis = [cal.off_axis(u) for u in ups]

    # ---- dynamic acceleration
    g = meta.get("g")
    if raw.acc is not None:
        mags = [bc.norm(raw.acc[i]) for i in range(n)
                if raw.acc[i] is not None and phases[i] in ("cal_quiet", "cal_left", "cal_right")]
        if mags:
            g = bc.median(mags)
    g = g or 9.81
    if raw.acc is not None:
        adyn = [bc.dynamic_acc(ups[i], acc=raw.acc[i], g=g) if raw.acc[i] is not None else None for i in range(n)]
        acc_source = "calibrated acceleration minus gravity"
    elif raw.facc is not None:
        adyn = [bc.norm(v) if v is not None else None for v in raw.facc]
        acc_source = "free acceleration"
    elif raw.acc_legacy is not None:
        adyn = raw.acc_legacy
        acc_source = "acc_dyn_ms2 column (04 log)"
    else:
        adyn = [None] * n
        acc_source = None
        warnings.append("no acceleration in the log: the impact cue is off, contacts come from the angle only")
    impact_acc = p.impact_acc_ms2
    quiet = [adyn[i] for i in cal_samples("cal_quiet", "quiet") if adyn[i] is not None]
    if quiet:
        impact_acc = max(impact_acc, p.impact_quiet_factor * bc.percentile(quiet, 99))

    trials = bc.trial_ranges(raw.trial)
    loaded = [ph in bc.LOADED_PHASES or raw.trial[i] > 0 for i, ph in enumerate(phases)]
    rate_source = None
    if p.board == "roller":
        # ---- roller board: contacts from landing / resting, no fixed floor angle
        theta_cal = dict(cal.theta)
        theta = dict(theta_cal)
        refined, per_trial, trial_est = {-1: None, 1: None}, {}, {}
        angle_tol, rest_tol = p.angle_tol_deg, p.rest_tol_deg
        still = [False] * n
        impact_acc = max(p.roller_impact_acc_ms2, impact_acc)
        rock_rate, rate_source = rocking_rate(t, tilt, raw.gyr, cal.axis)
        if rock_rate is None:
            notes.append("no gyroscope in the log: the rocking rate comes from the tilt (noisier)")
        det = bc.RollerContactDetector(p, impact_acc=impact_acc)
        for i in range(n):
            det.update(t[i], tilt[i], adyn[i], None if rock_rate is None else rock_rate[i])
        det.finish(t[-1] + dt_med)
        contacts = det.contacts
        margins = [None] * n
        th_at = [theta] * n
    else:
        # ---- refine contact angles from rests on the floor
        theta_cal = dict(cal.theta)
        theta = dict(theta_cal)
        still = still_mask(t, tilt, p.refine_still_win_s, p.refine_still_range_deg)
        loaded = [ph in bc.LOADED_PHASES or raw.trial[i] > 0 for i, ph in enumerate(phases)]
        trials = bc.trial_ranges(raw.trial)

        def estimate(select, side, center):
            sel = list(select)
            vals = [side * tilt[i] for i in sel if still[i] and abs(side * tilt[i] - center) <= p.refine_search_deg]
            reach = [side * tilt[i] for i in sel if abs(side * tilt[i] - center) <= p.refine_search_deg + 3]
            return refine_side(vals, rate, p, reach)

        refined = {-1: None, 1: None}
        per_trial, trial_est = {}, {}
        angle_tol, rest_tol = p.angle_tol_deg, p.rest_tol_deg
        if p.refine:
            thr = p.refine_trial_shift_deg
            for tr, (i0, i1) in sorted(trials.items()):
                trial_est[tr] = {s: estimate(range(i0, i1), s, theta_cal[s]) for s in (-1, 1)}
            shift_from, shift = find_sensor_shift(trial_est, theta_cal, thr)
            t_cut = math.inf
            if shift_from is not None:
                later = [i for tr, (i0, i1) in trials.items() if tr >= shift_from for i in range(i0, i1)]
                after = {s: estimate(later, s, theta_cal[s] + shift[s]) for s in (-1, 1)}
                if after[-1] and after[1]:
                    th_after = {s: after[s]["theta"] for s in (-1, 1)}
                    for tr in trials:
                        if tr >= shift_from:
                            per_trial[tr] = th_after
                    t_cut = t[trials[shift_from][0]] if shift_from - 1 not in trials else t[trials[shift_from - 1][1] - 1]
                    warnings.append(f"from trial {shift_from} both floors moved (left {shift[-1]:+.2f}, right "
                                    f"{shift[1]:+.2f} deg): the sensor probably turned on the board. Used "
                                    f"{th_after[-1]:.2f} / {th_after[1]:.2f} deg from then on - fix the sensor firmly.")
            pool = [i for i in range(n) if loaded[i] and t[i] <= t_cut]
            for side in (-1, 1):
                r = estimate(pool, side, theta_cal[side])
                refined[side] = r
                if r is not None:
                    theta[side] = r["theta"]
                    if abs(r["center"] - theta_cal[side]) > 1.0:
                        warnings.append(f"{bc.SIDE_NAME[side]} floor: the board rested at {r['center']:.2f} deg, "
                                        f"calibration said {theta_cal[side]:.2f}")
                else:
                    notes.append(f"{bc.SIDE_NAME[side]} side: not enough resting on the floor to refine, "
                                 f"kept the calibrated angle")
            center = {s: (refined[s]["center"] if refined[s] else theta_cal[s]) for s in (-1, 1)}
            for tr, e in sorted(trial_est.items()):
                if tr in per_trial:
                    continue
                for side in (-1, 1):
                    if e[side] and abs(e[side]["center"] - center[side]) >= thr:
                        warnings.append(f"trial {tr}: the board rested near the {bc.SIDE_NAME[side]} floor at "
                                        f"{e[side]['center']:.2f} deg vs {center[side]:.2f} elsewhere - held still "
                                        f"just above the floor, or the sensor moved? kept {theta[side]:.2f}")
            if all(r is not None for r in refined.values()):
                # the floor angles are now measured where the board actually rested
                rest_tol = max(angle_tol, min(p.rest_tol_deg, angle_tol + p.rest_margin_refined_deg))
        in_trial = [i for i in range(n) if raw.trial[i] > 0]
        for side in (-1, 1):
            past = sum(1 for i in in_trial if side * tilt[i] > theta_cal[side] + 1.5) / rate
            if past > 0.2:
                warnings.append(f"the board spent {past:.1f} s more than 1.5 deg past the calibrated "
                                f"{bc.SIDE_NAME[side]} contact angle - that calibration hold was probably not on "
                                f"the floor, so near misses on that side may count as contacts")
        for side in (-1, 1):
            if theta[side] < p.min_contact_deg:
                warnings.append(f"{bc.SIDE_NAME[side]} contact angle is only {theta[side]:.1f} deg")

        # ---- detect contacts (whole recording, so contacts spanning a trial start are handled)
        det = bc.ContactDetector(theta[-1], theta[1], p, impact_acc=impact_acc,
                                 angle_tol=angle_tol, rest_tol=rest_tol)
        margins = [0.0] * n
        th_at = [None] * n
        current = None
        # samples between trials take the angles of the trial they lead into (countdown) or follow
        owner = list(raw.trial)
        nxt = 0
        for i in range(n - 1, -1, -1):
            if raw.trial[i] > 0:
                nxt = raw.trial[i]
            elif phases[i] == "countdown":
                owner[i] = nxt
        prev_tr = 0
        for i in range(n):
            if owner[i] > 0:
                prev_tr = owner[i]
            else:
                owner[i] = prev_tr
        for i in range(n):
            th = per_trial.get(owner[i], theta)
            if th is not current:
                det.set_theta(th[-1], th[1])
                current = th
            th_at[i] = th
            margins[i] = det.margin(tilt[i])[1]
            det.update(t[i], tilt[i], adyn[i])
        t_end_all = t[-1] + dt_med
        det.finish(t_end_all)
        contacts = det.contacts
        if p.refine:
            refine_timing(contacts, t, lambda i, s: th_at[i][s] - s * tilt[i], rest_tol)
    cidx_all = contact_index(t, contacts)

    # ---- label each trial
    summaries, labs, bounds = [], {}, {}
    state = [""] * n
    for tr in sorted(trials):
        i0, i1 = trials[tr]
        t0 = t[i0]
        t1 = t[i1] if i1 < n else t[i1 - 1] + dt_med
        lab = bc.label_trial(t, raw.wall, margins, contacts, i0, i1, t1, p)
        labs[tr] = lab
        bounds[tr] = (i0, i1, t0, t1)
        for j, k in enumerate(lab["cidx"]):
            state[i0 + j] = bc.state_of(raw.wall[i0 + j], k >= 0)
        s = bc.summarize_trial(tr, t0, t1, lab, contacts)
        s["theta_left_deg"] = per_trial.get(tr, theta)[-1]
        s["theta_right_deg"] = per_trial.get(tr, theta)[1]
        s["missing_samples"] = sum(round(d / dt_med) - 1 for a, d in gaps if t0 <= a < t1)
        summaries.append(s)
        close = [nm for nm in lab["near_misses"] if nm["min_margin"] <= angle_tol + 0.5]
        if close:
            when = ", ".join(f"{nm['t_min'] - t0:.1f} s" for nm in close)
            warnings.append(f"trial {tr}: {len(close)} near miss(es) within {angle_tol + 0.5:.1f} deg of the "
                            f"floor without a detected contact (at {when}) - check the plot")

    trial_of = {}
    for tr, (i0, i1, t0, t1) in bounds.items():
        for k in labs[tr]["contact_ids"]:
            if t0 - 1e-9 <= contacts[k].onset < t1:
                trial_of[k] = tr
    in_trial_contacts = sorted(trial_of)

    # ---- online vs offline agreement
    agree = None
    if raw.state_online is not None:
        pairs = [(state[i], raw.state_online[i]) for i in range(n) if state[i] and raw.state_online[i]]
        if pairs:
            agree = sum(a == b for a, b in pairs) / len(pairs)

    # ---- write outputs
    qc_dir = base + "_qc"
    os.makedirs(qc_dir, exist_ok=True)
    paths = {}
    paths["events"] = write_events(base, contacts, labs, bounds, trial_of, raw.wall, t)
    paths["labels"] = write_labels(base, t, phases, raw.trial, raw.wall, tilt, margins, adyn, cidx_all, state)
    paths["trials"] = write_rows(base + "_desc-trials_beh.csv", summaries)
    paths["contacts"] = write_rows(base + "_desc-contacts_beh.csv",
                                   contact_rows(contacts, trial_of, t, raw.wall))

    info = {
        "log": log_path, "n": n, "rate": rate, "duration": t[-1] - t[0], "missing": missing,
        "long_gaps": long_gaps, "time_factor": factor, "cal": cal, "cal_note": cal_note,
        "theta_cal": theta_cal, "theta": theta, "refined": refined, "per_trial": per_trial,
        "trial_est": trial_est,
        "angle_tol": angle_tol, "rest_tol": rest_tol, "impact_acc": impact_acc, "g": g,
        "acc_source": acc_source, "rate_source": rate_source, "trial_of": trial_of, "off_axis_p99": bc.percentile([abs(v) for v in off_axis], 99),
        "agree": agree, "params": p,
    }
    if info["off_axis_p99"] is not None and info["off_axis_p99"] > 5:
        warnings.append(f"the board also tilts front/back (99th percentile {info['off_axis_p99']:.1f} deg); "
                        f"tilt is measured about the rocking axis only")
    report = build_report(info, summaries, contacts, in_trial_contacts, trial_of, warnings, notes, paths, qc_dir)
    paths["report"] = os.path.join(qc_dir, "report.txt")
    with open(paths["report"], "w", encoding="utf-8") as fh:
        fh.write(report)
    if plots:
        try:
            import qc_plots
            paths["plots"] = qc_plots.make_all(qc_dir, t, tilt, adyn, state, contacts, labs, bounds,
                                               per_trial, theta, theta_cal, still, loaded, info, p)
        except ImportError as e:
            notes.append(f"QC plots skipped ({e}); pip install matplotlib")
    if verbose:
        print(report)
    return {"contacts": contacts, "summaries": summaries, "warnings": warnings, "paths": paths,
            "info": info, "labs": labs, "bounds": bounds, "t": t, "tilt": tilt, "state": state,
            "trial_of": trial_of, "cidx": cidx_all, "walls": raw.wall}


# ---------------------------------------------------------------------- writers


def _fmt(v, nd=4):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_rows(path, rows):
    with open(path, "w", newline="") as fh:
        if not rows:
            fh.write("")
            return path
        w = csv.writer(fh)
        keys = list(rows[0].keys())
        w.writerow(keys)
        for r in rows:
            w.writerow([_fmt(r[k]) for k in keys])
    return path


def contact_rows(contacts, trial_of, t, walls):
    import bisect
    rows = []
    for k, c in enumerate(contacts):
        i = min(bisect.bisect_left(t, c.onset), len(t) - 1)
        lead = None
        if "impact" in c.cues and "angle" in c.cues:
            lead = (c.cues["angle"] - c.cues["impact"]) * 1000.0
        rows.append({
            "contact_id": k, "trial": trial_of.get(k), "onset_s": c.onset, "end_s": c.end,
            "duration_s": c.duration, "side": c.side_name, "cue": c.cue, "cues_seen": c.cue_combo(),
            "wall_at_onset": int(walls[i]), "approach_dps": c.approach_dps,
            "jolt_peak_ms2": c.jolt_peak, "margin_min_deg": None if c.margin_min == math.inf else c.margin_min,
            "floor_deg": c.floor_deg,
            "impact_before_angle_ms": lead,
        })
    return rows


def write_labels(base, t, phases, trials, walls, tilt, margins, adyn, cidx, state):
    path = base + "_desc-labels_beh.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "phase", "trial", "wall", "tilt_deg", "margin_deg", "acc_dyn",
                    "contact", "contact_id", "state"])
        for i in range(len(t)):
            k = cidx[i]
            w.writerow([f"{t[i]:.4f}", phases[i] or "", trials[i], int(walls[i]), f"{tilt[i]:.3f}",
                        "" if margins[i] is None else f"{margins[i]:.3f}", "" if adyn[i] is None else f"{adyn[i]:.3f}",
                        int(k >= 0), k if k >= 0 else "", state[i]])
    return path


EVENTS_JSON = {
    "onset": {"Description": "Start of the event, seconds from the first sample of the raw log.",
              "Units": "s"},
    "duration": {"Description": "Duration of the event.", "Units": "s"},
    "trial_type": {"Description": "Event type.", "Levels": {
        "trial": "30-s balance trial.",
        "balance": "Wall not held and board off the floor.",
        "support": "Wall held and board off the floor.",
        "idle": "Wall held and board on the floor.",
        "fail": "Wall not held and board on the floor. Onset = moment the board touched the floor "
                "(cause 'touch').",
        "contact": "One ground contact of the board, whatever the wall state."}},
    "trial": {"Description": "Trial number."},
    "cause": {"Description": "How an idle or fail period began.", "Levels": {
        "touch": "The board touched the floor.",
        "wall_release": "The wall was released while the board was already on the floor.",
        "wall_grab": "The wall was grabbed while the board was on the floor.",
        "trial_start": "The board was already on the floor when the trial began."}},
    "cue": {"Description": "Signal that first detected the contact.", "Levels": {
        "impact": "Jolt in acceleration as the board reached the floor.",
        "angle": "Tilt reached the contact angle.",
        "rest": "Tilt stopped close to the contact angle."}},
    "side": {"Description": "Edge of the board on the floor.", "Levels": {"left": "", "right": ""}},
    "contact_id": {"Description": "Row in _desc-contacts_beh.csv."},
}


def write_events(base, contacts, labs, bounds, trial_of, walls, t):
    rows = []
    for tr, (i0, i1, t0, t1) in sorted(bounds.items()):
        rows.append((t0, 0, t1 - t0, "trial", tr, "n/a", "n/a", "n/a", "n/a"))
        for q in labs[tr]["periods"]:
            c = contacts[q["contact"]] if q["contact"] >= 0 else None
            rows.append((q["onset"], 1, q["end"] - q["onset"], q["state"], tr,
                         q["cause"] if c else "n/a",
                         c.cue if (c and q["cause"] == "touch") else "n/a",
                         c.side_name if c else "n/a", q["contact"] if c else "n/a"))
    for k, c in enumerate(contacts):
        if k in trial_of:
            rows.append((c.onset, 2, c.duration, "contact", trial_of[k], "n/a", c.cue, c.side_name, k))
    rows.sort(key=lambda r: (r[0], r[1]))
    path = base + "_events.tsv"
    with open(path, "w", newline="") as fh:
        fh.write("onset\tduration\ttrial_type\ttrial\tcause\tcue\tside\tcontact_id\n")
        for r in rows:
            fh.write("\t".join([f"{r[0]:.4f}", f"{r[2]:.4f}"] + [str(x) for x in r[3:]]) + "\n")
    with open(base + "_events.json", "w") as fh:
        json.dump(EVENTS_JSON, fh, indent=2)
    return path


# ----------------------------------------------------------------------- report


def build_report(info, summaries, contacts, in_trial, trial_of, warnings, notes, paths, qc_dir):
    p = info["params"]
    L = []
    L.append("BALANCE BOARD SESSION - FINAL LABELS")
    L.append("=" * 72)
    L.append(f"log        {os.path.basename(info['log'])}")
    L.append(f"samples    {info['n']} at {info['rate']:.1f} Hz, {info['duration']:.1f} s, "
             f"{info['missing']} missing ({100 * info['missing'] / max(1, info['n'] + info['missing']):.2f} %)")
    if info["long_gaps"]:
        worst = max(d for _, d in info["long_gaps"])
        L.append(f"           {len(info['long_gaps'])} gaps longer than 0.1 s (longest {worst:.2f} s)")
    L.append("")
    if p.board == "roller":
        L.append("BOARD      on a roller: the tilt at which an edge meets the floor depends on where the roller")
        L.append("           sits, so contacts are found from the board landing or resting, not from a set angle")
        L.append("")
        L.append("THRESHOLDS")
        L.append(f"  rest        |rocking rate| < {p.roller_still_dps:g} deg/s for {p.roller_still_dwell_s:g} s at a tilt "
                 f">= {p.roller_min_tilt_deg:g} deg   (rate from {info['rate_source'] or 'the tilt'})")
        if info["acc_source"]:
            L.append(f"  landing     jolt >= {info['impact_acc']:.1f} m/s^2 at a tilt >= {p.roller_impact_min_tilt_deg:g} deg"
                     f"   ({info['acc_source']})")
        else:
            L.append("  landing     off (no acceleration)")
        L.append(f"  lift-off    tilts back {p.roller_lift_deg:g} deg from where it rested, for {p.roller_lift_hold_s:g} s, "
                 f"or passes level")
        L.append("")
    else:
        L.append("CONTACT ANGLES (deg from level)        left     right")
        L.append(f"  calibration ({info['cal'].source})")
        L.append(f"{'':38}{info['theta_cal'][-1]:6.2f}   {info['theta_cal'][1]:6.2f}")
        if p.refine:
            def show(side):
                r = info["refined"][side]
                return f"{r['theta']:6.2f}" if r else "     -"
            def cen(side):
                r = info["refined"][side]
                return f"{r['center']:6.2f}" if r else "     -"
            L.append(f"  resting on the floor (median){'':8}{cen(-1)}   {cen(1)}")
            L.append(f"  lightest rests (10th percentile){'':5}{show(-1)}   {show(1)}")
            def spread(side):
                r = info["refined"][side]
                return f"{r['sigma']:6.2f}" if r else "     -"
            L.append(f"    spread (robust SD){'':17}{spread(-1)}   {spread(1)}")
            def secs(side):
                r = info["refined"][side]
                return f"{r['seconds']:5.1f}s" if r else "     -"
            L.append(f"    resting time used{'':18}{secs(-1)}   {secs(1)}")
        L.append(f"  used{'':33}{info['theta'][-1]:6.2f}   {info['theta'][1]:6.2f}")
        for tr, th in sorted(info["per_trial"].items()):
            L.append(f"  used in trial {tr}{'':22}{th[-1]:6.2f}   {th[1]:6.2f}")
        if info["cal_note"]:
            L.append(f"  note: {info['cal_note']}")
        L.append("")
        L.append("THRESHOLDS")
        L.append(f"  angle cue   margin <= {info['angle_tol']:.2f} deg")
        L.append(f"  rest cue    margin <= {info['rest_tol']:.2f} deg, |rate| <= {p.rest_rate_dps:g} deg/s "
                 f"for {p.rest_dwell_s:g} s")
        if info["acc_source"]:
            L.append(f"  impact cue  |acc| >= {info['impact_acc']:.2f} m/s^2 within {p.impact_zone_deg:g} deg, "
                     f"approaching >= {p.impact_approach_dps:g} deg/s   ({info['acc_source']})")
        else:
            L.append("  impact cue  off (no acceleration)")
        L.append(f"  lift-off    margin >= {p.lift_deg:g} deg for {p.lift_hold_s:g} s, or past level")
        L.append("")
    L.append("TRIALS      balance  support   idle    fail   fails  contacts  longest  near")
    L.append("                (s)      (s)    (s)     (s)                    bal (s) misses")
    for s in summaries:
        near = "  n/a" if p.board == "roller" else f"{s['near_misses']:5d}"
        L.append(f"  {s['trial']:>3}     {s['balance_s']:7.1f}  {s['support_s']:7.1f} {s['idle_s']:6.1f} "
                 f"{s['fail_s']:7.1f}   {s['n_fails']:5d}  {s['n_contacts']:8d}  {s['longest_balance_s']:7.1f}  "
                 f"{near}")
    if summaries:
        tot = {k: sum(s[k] for s in summaries) for k in ("balance_s", "support_s", "idle_s", "fail_s",
                                                         "n_fails", "n_contacts", "near_misses")}
        L.append(f"  all     {tot['balance_s']:7.1f}  {tot['support_s']:7.1f} {tot['idle_s']:6.1f} "
                 f"{tot['fail_s']:7.1f}   {tot['n_fails']:5d}  {tot['n_contacts']:8d}  {'':7}  "
                 f"{'  n/a' if p.board == 'roller' else format(tot['near_misses'], '5d')}")
    L.append("  fails = fail periods that began with a touch; time in 'fail' also includes")
    L.append("  periods where the wall was released with the board already on the floor.")
    if info["agree"] is not None:
        L.append(f"  live labels matched these on {100 * info['agree']:.1f} % of trial samples (live labels "
                 f"confirm a lift-off late)")
    L.append("")
    L.append("WHICH CUE DETECTED THE CONTACTS (contacts starting in a trial)")
    cs = [contacts[k] for k in in_trial]
    if cs:
        first = Counter(c.cue for c in cs)
        combo = Counter(c.cue_combo() for c in cs)
        cue_names = ("impact", "rest") if p.board == "roller" else bc.CUES
        L.append("  first cue      " + "   ".join(f"{cue} {first.get(cue, 0)}" for cue in cue_names))
        L.append("  cues seen      " + "   ".join(f"{k} {v}" for k, v in combo.most_common()))
        leads = [(c.cues["angle"] - c.cues["impact"]) * 1000 for c in cs if "impact" in c.cues and "angle" in c.cues]
        if leads:
            L.append(f"  impact and angle both seen in {len(leads)}: impact first in "
                     f"{sum(x > 0 for x in leads)}, same sample in {sum(x == 0 for x in leads)}, "
                     f"median lead {bc.median(leads):.0f} ms")
        only_impact = [k for k in in_trial if contacts[k].cue_combo() == "impact"]
        if only_impact:
            L.append(f"  impact only: {len(only_impact)} (quick taps or jolts near the floor - check the plots)")
        only_rest = [k for k in in_trial if contacts[k].cue_combo() == "rest"]
        if only_rest:
            L.append(f"  rest only: {len(only_rest)} " + ("(settled on the floor without a jolt)" if p.board == "roller"
                     else "(stopped close to, but not at, the contact angle)"))
        durs = sorted(c.duration for c in cs if c.duration is not None)
        if durs:
            L.append(f"  contact duration median {bc.median(durs):.2f} s, shortest {durs[0]:.3f} s, "
                     f"longest {durs[-1]:.1f} s")
    else:
        L.append("  no contacts")
    L.append("")
    if warnings:
        L.append("CHECK")
        for w in warnings:
            L.append(f"  - {w}")
        L.append("")
    if notes:
        L.append("NOTES")
        for w in notes:
            L.append(f"  - {w}")
        L.append("")
    L.append("FILES")
    for k in ("events", "trials", "contacts", "labels"):
        L.append(f"  {os.path.basename(paths[k])}")
    L.append(f"  {os.path.basename(qc_dir)}/ (report, plots)")
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------------- CLI


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="raw log (_beh.csv) written by 05_balance_task.py (or 04)")
    ap.add_argument("--calibration", help="calibration JSON (default: next to the log)")
    ap.add_argument("--outdir", help="where to write the outputs (default: next to the log)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-refine", action="store_true", help="use the calibrated contact angles as they are")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--board", choices=("roller", "rocker"), help="board type (default: roller)")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                    help="override any threshold in balance_core.Params, e.g. --set angle_tol_deg=0.7 lift_deg=2.5")
    a = ap.parse_args(argv)
    p = bc.Params()
    for kv in a.set:
        k, _, v = kv.partition("=")
        p.update(**{k: v})
    label_session(a.log, params=p, calib_path=a.calibration, outdir=a.outdir, plots=not a.no_plots,
                  verbose=not a.quiet, refine=False if a.no_refine else None, board=a.board)


if __name__ == "__main__":
    main()
