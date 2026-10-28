#!/usr/bin/env python3
"""Scores the contact detection on simulated sessions.

    python test_pipeline.py            # all scenarios, 3 seeds each
    python test_pipeline.py --seeds 5 --scenario "soft mat"

For each scenario it simulates sessions (simulate_session.py), labels them with
label_session.py, and compares the detected contacts with the true ones:
missed and extra contacts, onset and end timing, fails per trial and the
per-sample state. 'live' = calibrated contact angles only (what the task
script prints); 'final' = after refinement (the labels that are saved).
"""
import argparse
import json
import os
import tempfile
from collections import defaultdict

import balance_core as bc
import label_session as ls
import simulate_session as sim

SCENARIOS = {
    "hard floor, 60 Hz": {},
    "hard floor, 120 Hz": {"rate": 120.0},
    "soft mat (0.8 deg give)": {"compression": 0.8},
    "light press during calibration": {"compression": 0.6, "cal_depth": 0.2},
    "sensor slips 1.5 deg before trial 4": {"shift_trial": 4, "shift_deg": 1.5},
    "orientation only (no acceleration)": {"payload": "orientation"},
    "free acc, sideways sensor, 10 kHz ticks, 3% drops": {"payload": "facc+gyr", "mount": "side",
                                                          "ticks_per_s": 1e4, "drop_frac": 0.03},
}

MATCH_TOL = 0.15


def score(res, truth):
    trials = {int(k): v for k, v in truth["trials"].items()}
    windows = [(a, b + 1 / 60) for a, b in trials.values()]

    def in_trials(on, end):
        return any(on < b and end > a for a, b in windows)

    true_eps = [e for e in truth["episodes"] if in_trials(e["onset"], e["end"])]
    det = [c for c in res["contacts"] if in_trials(c.onset, c.end)]
    used = set()
    matches = []
    missed = []
    for e in true_eps:
        best, best_d = None, None
        for k, c in enumerate(det):
            if k in used:
                continue
            overlap = min(c.end, e["end"]) - max(c.onset, e["onset"])
            d = abs(c.onset - e["onset"])
            if overlap > -MATCH_TOL and d <= max(MATCH_TOL, 0.5 * (e["end"] - e["onset"])):
                if best is None or d < best_d:
                    best, best_d = k, d
        if best is None:
            missed.append(e)
        else:
            used.add(best)
            matches.append((e, det[best]))
    extra = [c for k, c in enumerate(det) if k not in used]

    # what were the extra contacts?
    intents = truth["intent"]
    extra_kind = []
    for c in extra:
        near = [it for it in intents if abs(it["t"] - c.onset) < 1.5 and it["kind"] in ("near_miss", "hover", "stomp")]
        extra_kind.append((near[0]["kind"], near[0].get("true_min_margin")) if near else ("other", None))

    # true labels through the same period logic
    t = res["t"]
    walls = res["walls"]
    true_contacts = [bc.Contact(onset=e["onset"], side=e["side"], cue="truth", end=e["end"]) for e in truth["episodes"]]
    fails_true, fails_det, fail_s_err = [], [], []
    agree = n_lab = 0
    for tr, (i0, i1, t0, t1) in res["bounds"].items():
        lab_true = bc.label_trial(t, walls, [10.0] * len(t), true_contacts, i0, i1, t1, bc.Params())
        st = bc.summarize_trial(tr, t0, t1, lab_true, true_contacts)
        sd = next(s for s in res["summaries"] if s["trial"] == tr)
        fails_true.append(st["n_fails"])
        fails_det.append(sd["n_fails"])
        fail_s_err.append(sd["fail_s"] - st["fail_s"])
        for j, k in enumerate(lab_true["cidx"]):
            truth_state = bc.state_of(walls[i0 + j], k >= 0)
            agree += truth_state == res["state"][i0 + j]
            n_lab += 1
    on_err = [(c.onset - e["onset"]) * 1000 for e, c in matches]
    end_err = [(c.end - e["end"]) * 1000 for e, c in matches]
    return {
        "true": len(true_eps), "found": len(matches), "missed": missed, "extra": extra,
        "extra_kind": extra_kind, "on_err": on_err, "end_err": end_err,
        "fails_true": sum(fails_true), "fails_det": sum(fails_det),
        "fail_trials_exact": sum(a == b for a, b in zip(fails_true, fails_det)), "n_trials": len(fails_true),
        "fail_s_err": fail_s_err, "state_agree": agree / max(1, n_lab),
    }


def pct(xs, q):
    return bc.percentile(xs, q) if xs else float("nan")


def run(scenarios, seeds, verbose=False):
    out = {}
    tmp = tempfile.mkdtemp(prefix="balance_sim_")
    for name, over in scenarios.items():
        agg = {mode: defaultdict(list) for mode in ("live", "final")}
        for seed in range(1, seeds + 1):
            cfg = sim.SimConfig(seed=seed, **over)
            d = os.path.join(tmp, f"{abs(hash(name)) % 10**6}_{seed}")
            log, truth_path = sim.simulate(cfg, d)
            with open(truth_path) as fh:
                truth = json.load(fh)
            for mode in ("live", "final"):
                od = os.path.join(d, mode)
                os.makedirs(od, exist_ok=True)
                res = ls.label_session(log, plots=False, verbose=False, refine=(mode == "final"), outdir=od,
                                       board="rocker")
                s = score(res, truth)
                a = agg[mode]
                for k, v in s.items():
                    if isinstance(v, list):
                        a[k].extend(v)
                    else:
                        a[k].append(v)
                if verbose and mode == "final":
                    for e in s["missed"]:
                        print(f"   [{name} seed {seed}] MISSED onset {e['onset']:.2f} dur {e['end'] - e['onset']:.3f} "
                              f"side {e['side']} speed {e.get('speed_dps', 0):.0f}")
                    for c, kind in zip(s["extra"], s["extra_kind"]):
                        print(f"   [{name} seed {seed}] EXTRA onset {c.onset:.2f} dur {c.duration:.3f} cue {c.cue} "
                              f"{c.cue_combo()} near {kind}")
        out[name] = agg
    return out


def report(out):
    print()
    print(f"{'scenario':52} {'mode':5} {'true':>5} {'found':>6} {'missed':>6} {'extra':>6} "
          f"{'onset ms (med, 95%)':>20} {'end ms (med)':>12} {'fails/trial':>11} {'state %':>8}")
    for name, agg in out.items():
        for mode in ("live", "final"):
            a = agg[mode]
            true = sum(a["true"])
            found = sum(a["found"])
            on = [abs(x) for x in a["on_err"]]
            print(f"{name:52} {mode:5} {true:5d} {found:6d} {len(a['missed']):6d} {len(a['extra']):6d} "
                  f"{pct(on, 50):9.0f} {pct(on, 95):9.0f} {pct(a['end_err'], 50):12.0f} "
                  f"{sum(a['fail_trials_exact'])}/{sum(a['n_trials']):<6} {100 * sum(a['state_agree']) / len(a['state_agree']):7.2f}")
        kinds = defaultdict(int)
        for k, m in agg["final"]["extra_kind"]:
            kinds[k if m is None else f"{k} (true margin {m:.2f})"] += 1
        if kinds:
            print(f"{'':52}       extra contacts during: " + "; ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scenario", default=None, help="substring of a scenario name")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    sc = {k: v for k, v in SCENARIOS.items() if a.scenario is None or a.scenario.lower() in k.lower()}
    report(run(sc, a.seeds, a.verbose))


if __name__ == "__main__":
    main()
