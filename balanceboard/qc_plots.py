"""QC figures for label_session.py (needs matplotlib)."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.lines import Line2D                  # noqa: E402
from matplotlib.patches import Patch                 # noqa: E402

import balance_core as bc                            # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
STATE_FILL = {"support": ("#2a78d6", 0.13), "idle": ("#898781", 0.22), "fail": ("#e34948", 0.20)}
CUE_STYLE = {"impact": ("v", "#6250d6"), "angle": ("o", "#eb6834"), "rest": ("s", "#1baf7a")}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8.5, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def trial_figure(path, tr, t, tilt, adyn, contacts, lab, bounds, theta, info, p):
    i0, i1, t0, t1 = bounds
    x = [v - t0 for v in t[i0:i1]]
    y = tilt[i0:i1]
    fig, (ax, axa) = plt.subplots(2, 1, figsize=(11, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.1], "hspace": 0.08})
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axa):
        _style(a)
        for q in lab["periods"]:
            if q["state"] in STATE_FILL:
                col, alpha = STATE_FILL[q["state"]]
                a.axvspan(q["onset"] - t0, q["end"] - t0, color=col, alpha=alpha, linewidth=0)

    tol = info["angle_tol"]
    roller = p.board == "roller"
    for side in ((-1, 1) if roller else ()):
        ax.axhline(side * p.roller_min_tilt_deg, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    for side in (() if roller else (-1, 1)):
        th = theta[side]
        ax.axhline(side * th, color=INK2, linewidth=1.0)
        ax.fill_between([0, t1 - t0], side * th, side * (th - tol), color=MUTED, alpha=0.18, linewidth=0)
        ax.axhline(side * (th - p.lift_deg), color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
        ax.text(t1 - t0, side * th, f" {bc.SIDE_NAME[side]} floor {th:.1f}°", va="center", ha="left",
                fontsize=8, color=INK2, clip_on=False)
    ax.plot(x, y, color=INK, linewidth=0.9)

    import bisect
    for k in lab["contact_ids"]:
        c = contacts[k]
        if c.onset < t0 - 1e-9:
            continue
        i = min(bisect.bisect_left(t, c.onset), len(t) - 1)
        marker, col = CUE_STYLE[c.cue]
        ax.plot([c.onset - t0], [tilt[i]], marker=marker, markersize=8, color=col,
                markeredgecolor=SURFACE, markeredgewidth=1.2, linestyle="none", zorder=5)
    for nm in lab["near_misses"]:
        i = min(bisect.bisect_left(t, nm["t_min"]), len(t) - 1)
        ax.plot([nm["t_min"] - t0], [tilt[i]], marker="x", markersize=7, color=MUTED,
                markeredgewidth=1.5, linestyle="none", zorder=5)

    lim = max(max(theta.values()), max(abs(v) for v in y)) + 2.0
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("tilt (°)", color=INK2, fontsize=9)

    ya = [v for v in adyn[i0:i1]]
    if any(v is not None for v in ya):
        xs = [xv for xv, v in zip(x, ya) if v is not None]
        vs = [v for v in ya if v is not None]
        axa.plot(xs, vs, color=INK2, linewidth=0.8)
        axa.axhline(info["impact_acc"], color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        axa.text(t1 - t0, info["impact_acc"], f" impact {info['impact_acc']:.1f}", va="center", ha="left",
                 fontsize=8, color=INK2, clip_on=False)
        top = max(max(vs) * 1.1, info["impact_acc"] * 1.5)
        axa.set_ylim(0, top)
    else:
        axa.text(0.5, 0.5, "no acceleration in this recording", transform=axa.transAxes,
                 ha="center", va="center", color=MUTED, fontsize=9)
        axa.set_yticks([])
    axa.set_ylabel("|acc| (m/s²)", color=INK2, fontsize=9)
    axa.set_xlabel("time in trial (s)", color=INK2, fontsize=9)
    axa.set_xlim(0, t1 - t0)

    s = [q for q in lab["periods"]]
    bal = sum(q["end"] - q["onset"] for q in s if q["state"] == "balance")
    fails = sum(1 for q in s if q["state"] == "fail" and q["cause"] == "touch")
    nm = "" if roller else f"   near misses {len(lab['near_misses'])}"
    ax.set_title(f"Trial {tr}   balance {bal:.1f} s   fails {fails}{nm}",
                 loc="left", fontsize=11, color=INK, pad=26)
    handles = [Patch(facecolor=STATE_FILL[st][0], alpha=STATE_FILL[st][1] + 0.1, label=st)
               for st in ("support", "idle", "fail")]
    handles.insert(0, Patch(facecolor=SURFACE, edgecolor=AXIS, label="balance"))
    cues = ("impact", "rest") if roller else bc.CUES
    handles += [Line2D([], [], marker=CUE_STYLE[c][0], color=CUE_STYLE[c][1], linestyle="none",
                       markersize=7, label=f"contact: {'landing' if roller and c == 'impact' else c}") for c in cues]
    if not roller:
        handles.append(Line2D([], [], marker="x", color=MUTED, linestyle="none", markersize=7, label="near miss"))
    ax.legend(handles=handles, ncol=8, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False,
              fontsize=8, handletextpad=0.4, columnspacing=1.1, labelcolor=INK2)
    fig.subplots_adjust(left=0.07, right=0.89, top=0.86, bottom=0.1)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def floor_angles_figure(path, contacts, trial_of):
    """Roller board: the tilt at which each contact rested, per side."""
    fig, ax = plt.subplots(figsize=(9, 3.2))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for side, col in ((-1, "#2a78d6"), (1, "#eb6834")):
        vals = [c.floor_deg for k, c in enumerate(contacts) if k in trial_of and c.side == side and c.floor_deg]
        if vals:
            ax.hist(vals, bins=[x * 0.5 for x in range(0, 61)], color=col, alpha=0.8, label=f"{bc.SIDE_NAME[side]} edge")
    ax.set_xlabel("tilt at which the board rested on the floor (\u00b0)", color=INK2, fontsize=8.5)
    ax.set_ylabel("contacts", color=INK2, fontsize=9)
    ax.set_title("Floor angle per contact (changes with where the roller sits)", loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def angles_figure(path, tilt, still, loaded, theta, theta_cal, info, p):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    fig.patch.set_facecolor(SURFACE)
    for ax, side in zip(axes, (-1, 1)):
        _style(ax)
        vals = [side * a for a, s, ld in zip(tilt, still, loaded)
                if s and ld and abs(side * a - theta_cal[side]) <= p.refine_search_deg + 1]
        if vals:
            ax.hist(vals, bins=60, color="#2a78d6", alpha=0.85, edgecolor=SURFACE, linewidth=0.4)
        ax.axvspan(theta[side] - info["angle_tol"], theta[side], color=MUTED, alpha=0.18, linewidth=0)
        ax.axvline(theta_cal[side], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="calibration")
        ax.axvline(theta[side], color=INK, linewidth=1.2, label="used")
        ax.set_title(f"{bc.SIDE_NAME[side]} side", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("tilt toward this side while resting (°)", color=INK2, fontsize=8.5)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    axes[0].set_ylabel("samples", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def make_all(qc_dir, t, tilt, adyn, state, contacts, labs, bounds, per_trial, theta, theta_cal,
             still, loaded, info, p):
    out = []
    for tr in sorted(bounds):
        path = os.path.join(qc_dir, f"trial-{tr:02d}.png")
        trial_figure(path, tr, t, tilt, adyn, contacts, labs[tr], bounds[tr], per_trial.get(tr, theta), info, p)
        out.append(path)
    path = os.path.join(qc_dir, "contact_angles.png")
    if p.board == "roller":
        floor_angles_figure(path, contacts, info.get("trial_of", {}))
    else:
        angles_figure(path, tilt, still, loaded, theta, theta_cal, info, p)
    out.append(path)
    return out
