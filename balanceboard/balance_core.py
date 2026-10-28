"""Shared ground-contact detection for the rocker balance board (Movella DOT).

Used live by 05_balance_task.py (console feedback) and offline by
label_session.py (final labels). Pure Python, no numpy, so it runs wherever the
task script runs.

Signals
-------
tilt    rotation of the board about its rocking axis in degrees (right = +).
        Computed from the direction of gravity in the sensor frame, so heading
        drift and the way the sensor is mounted do not matter.
theta   contact angle of each side: the tilt at which that edge rests on the
        floor, measured with the participant standing on the board.
margin  theta - |tilt| on the side the board leans to. 0 = on the floor,
        negative = pressed a little past the calibrated angle.

A contact starts at the first of three cues
    impact  dynamic acceleration >= impact_acc while the board is within
            impact_zone of the floor, was moving toward it, and is within
            rest_tol of the floor at that sample or within impact_confirm_s
    angle   margin <= angle_tol
    rest    margin <= rest_tol and the board has stopped rotating for
            rest_dwell_s (onset = the moment it entered rest_tol)
and lasts, with no stillness requirement, until lift-off: margin >= lift_deg
for lift_hold_s, or the board passes level. The end is backdated to the last
sample at the floor.

States (wall key x contact)
    balance  no wall, off the floor      support  wall, off the floor
    idle     wall, on the floor          fail     no wall, on the floor
"""
import bisect
import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field

VERSION = "11.0"

STATES = ("balance", "support", "idle", "fail")
CUES = ("impact", "angle", "rest")
SIDE_NAME = {-1: "left", 1: "right"}

# Phases written by 05_balance_task.py. The board carries the participant's
# weight in all of these, so they are used to refine the contact angles.
LOADED_PHASES = ("cal_quiet", "cal_left", "cal_right", "countdown", "trial")

LOG_COLUMNS = [
    "t_s", "ticks", "phase", "trial", "wall",
    "roll", "pitch", "yaw", "qw", "qx", "qy", "qz",
    "acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
    "facc_x", "facc_y", "facc_z",
    "tilt_deg", "margin_deg", "acc_dyn", "contact_online", "state_online",
]


@dataclass
class Params:
    """All detection thresholds. Angles in degrees, times in seconds."""
    # 'roller': board on a free roller - the floor angle depends on where the roller sits,
    #           so contacts are found from the board landing or coming to rest at a steep tilt.
    # 'rocker': curved/fixed base - each edge always meets the floor at the same angle.
    board: str = "roller"
    # --- roller board
    roller_min_tilt_deg: float = 7.0      # no contact below this tilt
    roller_still_dps: float = 3.0         # resting: |rocking rate| below this ...
    roller_still_win_s: float = 0.15      # ... over this window ...
    roller_still_dwell_s: float = 0.15    # ... for this long
    roller_impact_acc_ms2: float = 12.0   # landing: jolt at or above this ...
    roller_impact_min_tilt_deg: float = 10.0  # ... at a tilt of at least this
    roller_impact_settle_s: float = 0.25  # no lift-off decision this soon after a landing (ringing)
    roller_lift_deg: float = 2.5          # contact ends when the board tilts back this far ...
    roller_lift_hold_s: float = 0.1       # ... for this long
    roller_merge_gap_s: float = 0.25      # a new contact on the same side this soon continues the last one
    # --- rocker board: contact cues
    angle_tol_deg: float = 0.5        # 'angle': margin at or below this
    rest_tol_deg: float = 1.25        # 'rest': margin at or below this ...
    rest_rate_dps: float = 6.0        # ... with |rocking rate| below this ...
    rest_dwell_s: float = 0.12        # ... for this long
    impact_acc_ms2: float = 3.0       # 'impact': |dynamic acceleration| at or above this ...
    impact_zone_deg: float = 2.5      # ... within this margin ...
    impact_approach_dps: float = 5.0  # ... after moving toward the floor at least this fast
    impact_lookback_s: float = 0.06   #     within this window
    impact_confirm_s: float = 0.1     # ... and the board must be within rest_tol of the floor by then
    # lift-off
    lift_deg: float = 3.0             # contact ends when margin >= this ...
    lift_hold_s: float = 0.2          # ... for this long (or the board passes level)
    ground_slack_deg: float = 0.5     # 'still on the floor' = within this of the deepest point of the contact
    # bookkeeping
    rate_window_s: float = 0.04       # window for the rocking rate used by the 'rest' cue
    cue_window_s: float = 0.3         # cues seen this soon after onset are recorded with the contact
    near_miss_deg: float = 2.0        # near miss: margin came this close without a contact
    # calibration
    min_contact_deg: float = 4.0      # each contact angle must be at least this
    cal_still_dev_deg: float = 0.6    # calibration hold: samples within this of the mean direction ...
    cal_still_frac: float = 0.7       # ... must be at least this fraction
    impact_quiet_factor: float = 6.0  # impact_acc >= this x 99th percentile of the quiet stance
    # offline refinement (label_session.py)
    refine: bool = True
    refine_search_deg: float = 3.0    # look for floor rests within this of the calibrated angle
    refine_still_win_s: float = 0.3   # 'resting' = tilt range within this window ...
    refine_still_range_deg: float = 0.4  # ... at or below this
    refine_min_s: float = 0.5         # need at least this much resting data per side
    refine_trial_shift_deg: float = 0.6  # floors moving this much in opposite directions = sensor turned
    rest_margin_refined_deg: float = 0.3  # after refinement the rest band is angle_tol + this

    def update(self, **kw):
        for k, v in kw.items():
            if v is None:
                continue
            if not hasattr(self, k):
                raise KeyError(f"unknown parameter: {k}")
            cur = getattr(self, k)
            setattr(self, k, (str(v).lower() in ("1", "true", "yes")) if isinstance(cur, bool) else type(cur)(v))
        return self

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------- math helpers
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def norm(a):
    return math.sqrt(dot(a, a))


def unit(a):
    n = norm(a)
    if n == 0:
        raise ValueError("zero vector")
    return (a[0] / n, a[1] / n, a[2] / n)


def vsum(vs):
    x = y = z = 0.0
    for v in vs:
        x += v[0]
        y += v[1]
        z += v[2]
    return (x, y, z)


def angle_deg(a, b):
    return math.degrees(math.atan2(norm(cross(a, b)), dot(a, b)))


def median(xs):
    return statistics.median(xs) if xs else None


def percentile(xs, q):
    """q in 0..100, linear interpolation."""
    s = sorted(xs)
    if not s:
        return None
    k = (len(s) - 1) * q / 100.0
    f = int(math.floor(k))
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def up_from_euler(roll_deg, pitch_deg):
    """World 'up' in the sensor frame from Xsens roll/pitch (ZYX Euler). Yaw drops out."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    cp = math.cos(p)
    return (-math.sin(p), math.sin(r) * cp, math.cos(r) * cp)


def dynamic_acc(up, acc=None, facc=None, g=9.81):
    """|acceleration without gravity| in m/s^2, from calibrated or free acceleration."""
    if acc is not None:
        return norm((acc[0] - g * up[0], acc[1] - g * up[1], acc[2] - g * up[2]))
    if facc is not None:
        return norm(facc)
    return None


# ------------------------------------------------------------------ calibration
class Calibration:
    """Rocking axis, 'level' direction and contact angles, in the sensor frame."""

    def __init__(self, axis, level, theta_left, theta_right, source="calibration"):
        self.axis = unit(axis)
        self.e1 = unit(sub(level, scale(self.axis, dot(level, self.axis))))
        self.e2 = cross(self.axis, self.e1)
        self.theta = {-1: float(theta_left), 1: float(theta_right)}
        self.source = source

    @classmethod
    def from_contacts(cls, up_left, up_right, source="calibration"):
        """From the gravity direction with the board resting on each edge.
        Level = halfway between the two contacts."""
        cal = cls(cross(up_left, up_right), add(up_left, up_right), 0.0, 0.0, source)
        cal.theta = {-1: -cal.tilt(up_left), 1: cal.tilt(up_right)}
        return cal

    def tilt(self, up):
        return math.degrees(math.atan2(dot(up, self.e2), dot(up, self.e1)))

    def off_axis(self, up):
        """Tilt along the rocking axis (front/back); ~0 for a rocker."""
        return math.degrees(math.asin(max(-1.0, min(1.0, dot(up, self.axis)))))

    def to_dict(self):
        return {"axis": list(self.axis), "level": list(self.e1),
                "theta_left": self.theta[-1], "theta_right": self.theta[1], "source": self.source}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(d["axis"]), tuple(d["level"]), d["theta_left"], d["theta_right"],
                   d.get("source", "file"))


def rest_direction(ups, max_dev_deg=0.6, min_frac=0.7):
    """Gravity direction of a calibration hold.

    Returns (ok, up, still_fraction, median_deviation_deg). ok is False when the
    board was not held still enough."""
    if not ups:
        return False, None, 0.0, None
    m = unit((median([u[0] for u in ups]), median([u[1] for u in ups]), median([u[2] for u in ups])))
    devs = [angle_deg(u, m) for u in ups]
    still = [u for u, d in zip(ups, devs) if d <= max_dev_deg]
    frac = len(still) / len(ups)
    up = unit(vsum(still)) if still else m
    return frac >= min_frac, up, frac, median(devs)


# ----------------------------------------------------------------------- clock
class SampleClock:
    """Turns the sensor's 32-bit sampleTimeFine counter into seconds.

    Unwraps the counter (it wraps every ~71.6 min if it counts microseconds) and
    detects the tick unit by comparing it with the host clock."""
    WRAP = 1 << 32
    UNITS = (1e3, 1e4, 1e5, 1e6, 1e7)

    def __init__(self, ticks_per_s=None):
        self.ticks_per_s = ticks_per_s
        self.wraps = 0
        self.origin = None
        self._last = None
        self._offset = 0

    def unwrap(self, raw):
        raw = int(raw)
        if self._last is None:
            self._last = raw
        elif raw < self._last - self.WRAP // 2:       # counter wrapped
            self._offset += self.WRAP
            self.wraps += 1
            self._last = raw
        elif raw > self._last + self.WRAP // 2:       # late packet from before a wrap
            return raw + self._offset - self.WRAP
        elif raw > self._last:
            self._last = raw
        return raw + self._offset

    @classmethod
    def detect_unit(cls, pairs):
        """pairs: [(unwrapped ticks, host seconds)] spanning >= 0.5 s.
        Returns (ticks_per_s, measured_ratio) or (None, None)."""
        if len(pairs) < 2:
            return None, None
        (k0, w0), (k1, w1) = pairs[0], pairs[-1]
        if w1 - w0 < 0.5 or k1 <= k0:
            return None, None
        r = (k1 - k0) / (w1 - w0)
        best = min(cls.UNITS, key=lambda u: abs(math.log10(r / u)))
        return best, r

    def seconds(self, ticks):
        if self.origin is None:
            self.origin = ticks
        return (ticks - self.origin) / self.ticks_per_s


# --------------------------------------------------------------------- contacts
@dataclass
class Contact:
    onset: float
    side: int                       # -1 left, +1 right
    cue: str                        # cue that set the onset
    end: float = None
    cues: dict = field(default_factory=dict)   # cue -> first time seen
    margin_min: float = float("inf")
    jolt_peak: float = 0.0          # max |dynamic acc| around the onset (m/s^2)
    approach_dps: float = 0.0       # speed toward the floor just before the onset
    floor_deg: float = None         # roller board: tilt at which the board rested on the floor

    @property
    def side_name(self):
        return SIDE_NAME[self.side]

    @property
    def duration(self):
        return None if self.end is None else self.end - self.onset

    def cue_combo(self):
        return "+".join(c for c in CUES if c in self.cues)


class ContactDetector:
    """Causal ground-contact detector. Feed samples in time order with update()."""

    def __init__(self, theta_left, theta_right, params=None, impact_acc=None,
                 angle_tol=None, rest_tol=None):
        self.p = params or Params()
        self.impact_acc = self.p.impact_acc_ms2 if impact_acc is None else float(impact_acc)
        self.angle_tol = self.p.angle_tol_deg if angle_tol is None else float(angle_tol)
        self.rest_tol = self.p.rest_tol_deg if rest_tol is None else float(rest_tol)
        self.contacts = []
        self.cur = None
        self._hist = deque()        # (t, tilt) for the rocking rate
        self._recent = deque()      # (t, speed toward floor, a_dyn) for the impact cue
        self._near_since = None
        self._near_side = 0
        self._still_since = None
        self._lift_since = None
        self._last_ground = None
        self._after_ground = None
        self._pending = None        # impact waiting for the board to reach the floor
        self._prev = None           # (side, margin) of the previous sample
        self.set_theta(theta_left, theta_right)

    def set_theta(self, theta_left, theta_right):
        """Contact angles (deg, positive). Thresholds are capped for steep-sided boards."""
        self.theta = {-1: float(theta_left), 1: float(theta_right)}
        self.lim = {}
        for s, th in self.theta.items():
            at = min(self.angle_tol, 0.15 * th)
            rt = max(at, min(self.rest_tol, 0.25 * th))
            self.lim[s] = {"angle": at, "rest": rt,
                           "impact": max(rt, min(self.p.impact_zone_deg, 0.3 * th)),
                           "lift": max(rt + 0.5, min(self.p.lift_deg, 0.4 * th))}

    def margin(self, tilt):
        side = 1 if tilt >= 0 else -1
        return side, self.theta[side] - side * tilt

    @property
    def in_contact(self):
        return self.cur is not None

    def update(self, t, tilt, a_dyn=None, rate=None):
        """Process one sample. Returns a list of ('start' | 'end', Contact). `rate` is unused
        (the rocker detector takes the rate from the tilt)."""
        p = self.p
        events = []

        h = self._hist
        h.append((t, tilt))
        while len(h) > 1 and t - h[1][0] >= p.rate_window_s:
            h.popleft()
        span = t - h[0][0]
        slow = (tilt - h[0][1]) / span if p.rate_window_s <= span <= 4 * p.rate_window_s else None

        side, d = self.margin(tilt)
        lim = self.lim[side]
        rec = self._recent
        rec.append((t, side * slow if slow is not None else 0.0, a_dyn or 0.0))
        while rec and t - rec[0][0] > p.impact_lookback_s:
            rec.popleft()
        approach = max(v for _, v, _ in rec)

        impact = (a_dyn is not None and a_dyn >= self.impact_acc and d <= lim["impact"]
                  and approach >= p.impact_approach_dps)
        angle = d <= lim["angle"]
        if d <= lim["rest"]:
            if self._near_since is None or self._near_side != side:
                self._near_since, self._near_side = t, side
        else:
            self._near_since = None
        if slow is not None and abs(slow) <= p.rest_rate_dps:
            if self._still_since is None:
                self._still_since = t
        else:
            self._still_since = None
        rest = (self._near_since is not None and self._still_since is not None
                and t - max(self._near_since, self._still_since) >= p.rest_dwell_s)

        c = self.cur
        if c is None:
            prev = self._prev
            self._prev = (side, d)
            pend = self._pending
            if pend is not None and (t - pend.onset > p.impact_confirm_s or side != pend.side):
                pend = self._pending = None
            at_floor = d <= lim["rest"] or (prev is not None and prev[0] == side and prev[1] <= lim["rest"])
            if impact and not at_floor:
                # a jolt counts only if the board reaches the floor right after it
                if pend is None:
                    pend = self._pending = Contact(onset=t, side=side, cue="impact")
                    pend.cues["impact"] = t
                    pend.approach_dps = approach
                return events
            cue = "impact" if impact else "angle" if angle else "rest" if rest else None
            if pend is not None and (cue is not None or d <= lim["rest"]):
                c = pend                                   # the jolt came first
            elif cue is not None:
                c = Contact(onset=self._near_since if cue == "rest" else t, side=side, cue=cue)
                c.approach_dps = approach
            else:
                return events
            for name, hit in (("impact", impact), ("angle", angle), ("rest", rest)):
                if hit and name not in c.cues:
                    c.cues[name] = t
            c.margin_min = d
            c.jolt_peak = max(a for _, _, a in rec)
            self._pending = None
            self.cur = c
            self.contacts.append(c)
            self._last_ground = t
            self._after_ground = None
            self._lift_since = None
            events.append(("start", c))
            return events

        # in contact: follow it until lift-off, whatever the board does meanwhile
        dc = self.theta[c.side] - c.side * tilt
        limc = self.lim[c.side]
        if dc < c.margin_min:
            c.margin_min = dc
        if t - c.onset <= p.cue_window_s:
            if side == c.side:
                for name, hit in (("impact", impact), ("angle", angle), ("rest", rest)):
                    if hit and name not in c.cues:
                        c.cues[name] = t
            if a_dyn is not None and a_dyn > c.jolt_peak:
                c.jolt_peak = a_dyn
        if dc <= self._band(c):
            self._last_ground = t
            self._after_ground = None
        elif self._after_ground is None:
            self._after_ground = t
        if dc >= limc["lift"]:
            if self._lift_since is None:
                self._lift_since = t
        else:
            self._lift_since = None
        lifted = (dc >= self.theta[c.side]                                    # passed level
                  or (self._lift_since is not None and t - self._lift_since >= p.lift_hold_s))
        if lifted:
            c.end = self._after_ground if self._after_ground is not None else t
            self.cur = None
            self._prev = (side, d)
            events.append(("end", c))
        return events

    def _band(self, c):
        lim = self.lim[c.side]
        return min(lim["rest"], max(lim["angle"], c.margin_min + self.p.ground_slack_deg))

    def finish(self, t_end):
        """Close a contact still open at the end of the recording."""
        if self.cur is None:
            return []
        c = self.cur
        c.end = self._after_ground if self._after_ground is not None else t_end
        self.cur = None
        return [("end", c)]


class RollerContactDetector:
    """Ground contact for a board on a free roller (causal, feed samples in time order).

    Where the roller sits under the board changes the tilt at which an edge meets the
    floor, so no fixed angle is used. A contact starts when the board
      lands   a jolt >= roller_impact_acc at a tilt of at least roller_impact_min_tilt, or
      rests   the rocking rate stays below roller_still_dps for roller_still_dwell_s at a
              tilt of at least roller_min_tilt (the floor holds the board far stiller than
              a rider can; onset = when the board arrived at that tilt)
    and lasts until the board tilts back roller_lift_deg from where it rested (or from
    the steepest tilt since), for roller_lift_hold_s, or passes level.
    `rate` is the rocking rate in deg/s from the gyroscope; without it the rate comes
    from the tilt and the stillness threshold is doubled."""

    def __init__(self, params=None, impact_acc=None):
        self.p = params or Params()
        self.impact_acc = self.p.roller_impact_acc_ms2 if impact_acc is None else float(impact_acc)
        self.contacts = []
        self.cur = None
        self._hist = deque()          # (t, tilt, |rate|)
        self._still_since = None
        self._below_since = None
        self._last_floor = None

    def margin(self, tilt):
        return (1 if tilt >= 0 else -1), None

    @property
    def in_contact(self):
        return self.cur is not None

    def update(self, t, tilt, a_dyn=None, rate=None):
        p = self.p
        events = []
        h = self._hist
        still_dps = p.roller_still_dps
        if rate is None:
            old = next((x for x in h if t - x[0] <= 0.1), None)
            rate = (tilt - old[1]) / (t - old[0]) if old and t > old[0] else 0.0
            still_dps *= 2
        h.append((t, tilt, abs(rate), a_dyn or 0.0))
        while h and t - h[0][0] > 1.5:
            h.popleft()
        still = max(x[2] for x in h if t - x[0] <= p.roller_still_win_s) < still_dps
        if not still:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = t
        side = 1 if tilt >= 0 else -1

        c = self.cur
        if c is None:
            recent = [x for x in h if t - x[0] <= 0.15]
            peak = max(recent, key=lambda x: abs(x[1]))
            if (a_dyn is not None and a_dyn >= self.impact_acc
                    and abs(peak[1]) >= max(p.roller_impact_min_tilt_deg, p.roller_min_tilt_deg)):
                return self._open(peak[0], 1 if peak[1] >= 0 else -1, "impact", abs(peak[1]), t)
            if (self._still_since is not None and t - self._still_since >= p.roller_still_dwell_s
                    and abs(tilt) >= p.roller_min_tilt_deg):
                onset = self._still_since
                for x in reversed(h):                     # back to when it arrived at this tilt
                    if x[0] > self._still_since:
                        continue
                    if (x[1] >= 0) != (tilt >= 0) or abs(abs(x[1]) - abs(tilt)) > 1.0:
                        break
                    onset = x[0]
                return self._open(onset, side, "rest", abs(tilt), t)
            return events

        st = c.side * tilt
        if still:
            c.floor_deg = st                              # resting: this is the floor here
            if "rest" not in c.cues:
                c.cues["rest"] = t
        elif st > c.floor_deg:
            c.floor_deg = st
        if st >= c.floor_deg - 0.7:
            self._last_floor = t
        settling = c.cue == "impact" and t - c.onset < p.roller_impact_settle_s
        if st < 0 or (st < c.floor_deg - p.roller_lift_deg and not settling):
            if self._below_since is None:
                self._below_since = t
            if st < 0 or t - self._below_since >= p.roller_lift_hold_s:
                after = [x[0] for x in h if x[0] > self._last_floor]
                c.end = after[0] if after else t
                self.cur = None
                self._below_since = None
                events.append(("end", c))
        else:
            self._below_since = None
        return events

    def _open(self, onset, side, cue, floor, t):
        last = self.contacts[-1] if self.contacts else None
        self._below_since = None
        self._last_floor = t
        if last is not None and last.side == side and onset - last.end <= self.p.roller_merge_gap_s:
            last.end = None                               # same landing, ringing split it
            last.floor_deg = max(last.floor_deg, floor)
            self.cur = last
            return [("resume", last)]
        c = Contact(onset=onset, side=side, cue=cue)
        c.cues[cue] = t
        c.floor_deg = floor
        c.jolt_peak = max((x[3] for x in self._hist if x[0] >= onset - 0.15), default=0.0)
        self.cur = c
        self.contacts.append(c)
        return [("start", c)]

    def finish(self, t_end):
        if self.cur is None:
            return []
        c = self.cur
        c.end = t_end
        self.cur = None
        return [("end", c)]


def make_detector(params, theta_left=None, theta_right=None, impact_acc=None, **kw):
    """The detector for the board type in params."""
    if params.board == "roller":
        return RollerContactDetector(params, impact_acc=impact_acc)
    return ContactDetector(theta_left, theta_right, params, impact_acc=impact_acc, **kw)


def gyro_scale(rates_raw, dtilt_dt):
    """Factor turning (gyro . axis) into the rocking rate in deg/s: sign from a fit
    against the tilt's derivative, size 1 (deg/s) or 57.3 (rad/s)."""
    pairs = [(g, d) for g, d in zip(rates_raw, dtilt_dt) if g is not None and d is not None]
    sgg = sum(g * g for g, _ in pairs)
    if len(pairs) < 30 or sgg <= 0:
        return None
    k = sum(g * d for g, d in pairs) / sgg
    if abs(k) < 0.2:
        return None
    size = 1.0 if abs(math.log(abs(k))) < abs(math.log(abs(k) / 57.2958)) else 57.2958
    return math.copysign(size, k)


# ----------------------------------------------------------------------- states
def state_of(wall, contact):
    if wall:
        return "idle" if contact else "support"
    return "fail" if contact else "balance"


def label_trial(times, walls, margins, contacts, i0, i1, t_end, p):
    """Label samples i0..i1-1 (one trial).

    Returns dict with
      cidx         per-sample index into `contacts` (-1 = no contact)
      periods      runs of one state (and one contact): state, onset, end, cause, contact
      near_misses  excursions to within p.near_miss_deg of the floor without contact
      contact_ids  indices of contacts overlapping the trial
    Fail and idle periods get a cause: 'touch' (the board touched the floor),
    'wall_release' / 'wall_grab' (wall changed while already on the floor) or
    'trial_start' (already on the floor when the trial began)."""
    n = i1 - i0
    cidx = [-1] * n
    t_start = times[i0]
    ids = []
    for k, c in enumerate(contacts):
        end = c.end if c.end is not None else float("inf")
        if end <= t_start or c.onset >= t_end:
            continue
        a = bisect.bisect_left(times, c.onset, i0, i1)
        b = bisect.bisect_left(times, end, i0, i1)
        for i in range(a, b):
            cidx[i - i0] = k
        ids.append(k)

    periods = []
    key = None
    for j in range(n):
        i = i0 + j
        k = cidx[j]
        st = state_of(walls[i], k >= 0)
        if (st, k) != key:
            if periods:
                periods[-1]["end"] = times[i]
            periods.append({"state": st, "onset": times[i], "end": None, "contact": k,
                            "cause": "n/a", "first": j == 0})
            key = (st, k)
    if periods:
        periods[-1]["end"] = t_end
    for q, per in enumerate(periods):
        if per["contact"] < 0:
            continue
        c = contacts[per["contact"]]
        prev = periods[q - 1] if q > 0 else None
        if per["first"] and c.onset < per["onset"] - 1e-9:
            per["cause"] = "trial_start"
        elif prev is not None and prev["contact"] == per["contact"]:
            per["cause"] = "wall_release" if per["state"] == "fail" else "wall_grab"
        else:
            per["cause"] = "touch"

    # near misses: the board came within near_miss_deg of the floor and went back
    # (only counted after it had been clearly away, so a lift-off is not one)
    near = []
    cur = None
    armed = False
    exit_deg = p.near_miss_deg + 1.0
    for j in range(n):
        i = i0 + j
        m = margins[i]
        if m is None:
            continue
        if cidx[j] >= 0 or walls[i]:
            armed = False
            cur = None
            continue
        if cur is None:
            if m >= exit_deg:
                armed = True
            elif armed and m <= p.near_miss_deg:
                cur = {"onset": times[i], "min_margin": m, "t_min": times[i]}
                armed = False
        else:
            if m < cur["min_margin"]:
                cur["min_margin"], cur["t_min"] = m, times[i]
            if m >= exit_deg:
                cur["end"] = times[i]
                near.append(cur)
                cur = None
                armed = True
    return {"cidx": cidx, "periods": periods, "near_misses": near, "contact_ids": ids}


def summarize_trial(trial, t0, t1, lab, contacts):
    periods = lab["periods"]
    s = {"trial": trial, "onset_s": t0, "duration_s": t1 - t0}
    for st in STATES:
        s[st + "_s"] = sum((q["end"] - q["onset"] for q in periods if q["state"] == st), 0.0)
    fails = [q for q in periods if q["state"] == "fail" and q["cause"] == "touch"]
    s["n_fails"] = len(fails)
    s["n_fail_periods"] = sum(1 for q in periods if q["state"] == "fail")
    s["first_fail_s"] = (fails[0]["onset"] - t0) if fails else None
    bal = [q["end"] - q["onset"] for q in periods if q["state"] == "balance"]
    s["longest_balance_s"] = max(bal) if bal else 0.0
    new = [contacts[k] for k in lab["contact_ids"] if contacts[k].onset >= t0 - 1e-9]
    s["n_contacts"] = len(new)
    for cue in CUES:
        s["contacts_" + cue] = sum(1 for c in new if c.cue == cue)
    s["near_misses"] = len(lab["near_misses"])
    return s


def trial_ranges(trials):
    """{trial: (i0, i1)} for each trial number > 0 (first to last sample)."""
    out = {}
    for i, tr in enumerate(trials):
        if tr and tr > 0:
            if tr not in out:
                out[tr] = [i, i + 1]
            else:
                out[tr][1] = i + 1
    return {k: tuple(v) for k, v in out.items()}
