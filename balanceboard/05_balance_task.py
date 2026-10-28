"""
Balance board task v11 - rocker board, Movella DOT, BIDS names, 7 trials.

States (wall key x ground contact), shown live and labelled again offline:
    balance  wall not held, board off the floor
    support  wall held,     board off the floor
    idle     wall held,     board on the floor
    fail     wall not held, board on the floor (onset = the moment it touches)

A ground contact is found from the tilt about the board's rocking axis (at or
near the contact angle measured in calibration) or from an impact jolt next to
the floor. It lasts until the board lifts off again, however long it stays down
and however much the participant moves meanwhile. Details: balance_core.py.

Run:    python 05_balance_task.py
Needs:  movelladot_pc_sdk, pynput (pip install pynput); matplotlib for QC plots
Keys:   hold the RIGHT ARROW key while the participant holds the wall

Writes  sub-<id>_ses-S001_task-balanceboard_run-<run>...
    _beh.csv            every sensor sample with phase, trial and wall key
    _calibration.json   rocking axis, contact angles, sensor settings
and when the run ends, the final labels from label_session.py:
    _events.tsv (+.json), _desc-trials_beh.csv, _desc-contacts_beh.csv,
    _desc-labels_beh.csv, _qc/ (report and plots)
If the run is cut short, the labels can be made later with
    python label_session.py <..._beh.csv>
"""
import csv
import dataclasses
import json
import os
import sys
import threading
import time

import balance_core as bc
import label_session

try:
    import movelladot_pc_sdk.movelladot_pc_sdk_py310_64 as sdk
except ImportError:
    import movelladot_pc_sdk as sdk

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

try:
    import winsound

    def beep(freq, ms):
        winsound.Beep(freq, ms)
except ImportError:                      # not on Windows
    def beep(freq, ms):
        sys.stdout.write("\a")
        sys.stdout.flush()
        time.sleep(ms / 1000.0)

# ----------------------------- SETTINGS -----------------------------
SESSION_LABEL = "S001"
SENSOR_ADDRESS = None          # e.g. "D4:22:CD:00:12:34"; None = first sensor found
OUTPUT_RATES = (120, 60)       # Hz, tried in this order (falls back if the sensor can't keep up)
PAYLOAD_MODES = ("CustomMode4", "CustomMode5", "CustomMode1", "ExtendedEuler", "CompleteEuler")
HOLD_MODE = True               # wall = right arrow held down (False: each press toggles)

TOTAL_TRIALS = 7
SESSION_DURATION_S = 30.0      # active balancing per trial
BREAK_DURATION_S = 45.0        # rest between trials
COUNTDOWN_S = 5                # beeps before each trial
END_BREAK_S = 600              # recovery break after the run

CAL_QUIET_S = 5.0              # calibration: quiet stance holding the wall
CAL_EDGE_S = 3.0               # calibration: each edge resting on the floor
SCAN_S = 10.0                  # Bluetooth scan
PROBE_S = 2.0                  # test stream per rate / payload mode

BOARD = "roller"               # "roller": board on a free roller; "rocker": fixed curved base
PARAMS = bc.Params(board=BOARD)  # detection thresholds, see balance_core.Params
# --------------------------------------------------------------------


def instruction(text):
    print(f"\n{text}")
    answer = input(" > (Press Enter when ready, q to quit) ")
    if answer.strip().lower() == "q":
        raise KeyboardInterrupt
    return answer


def up_from_quat(q):
    """World 'up' in the sensor frame from a (w, x, y, z) sensor-to-world quaternion."""
    w, x, y, z = q
    return bc.unit((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def hold_summary(data):
    """Mean direction of a calibration hold from three independent sources."""
    out = {"euler": bc.unit(bc.vsum(d[1] for d in data))}
    qs = [d[4] for d in data if d[4] is not None]
    if qs:
        try:
            out["quaternion"] = bc.unit(bc.vsum(up_from_quat(q) for q in qs))
        except ValueError:
            pass
    accs = [d[2] for d in data if d[2] is not None]
    if accs:
        out["accelerometer"] = bc.unit(bc.vsum(accs))
    out["roll"] = bc.median([d[5] for d in data])
    out["pitch"] = bc.median([d[6] for d in data])
    return out


def explain_small_angle(left, right, apart):
    """Say why the two calibration holds came out (almost) the same."""
    a, b = hold_summary(left), hold_summary(right)
    print(f"\n  The two edges are only {apart:.1f} deg apart (need at least {2 * PARAMS.min_contact_deg:.0f}).")
    print(f"  Sensor roll/pitch: LEFT hold {a['roll']:+.1f} / {a['pitch']:+.1f} deg, "
          f"RIGHT hold {b['roll']:+.1f} / {b['pitch']:+.1f} deg")
    seen = {}
    for src in ("euler", "quaternion", "accelerometer"):
        if src in a and src in b:
            seen[src] = bc.angle_deg(a[src], b[src])
            print(f"    tilt change seen by the {src:13}: {seen[src]:5.1f} deg")
    if seen.get("accelerometer", 0) >= 2 * PARAMS.min_contact_deg and seen.get("euler", 0) < 1:
        print("  -> The accelerometer saw the board tip but the orientation output did not change. The sensor's")
        print("     orientation is not updating in this payload mode: put another mode first in PAYLOAD_MODES")
        print("     (e.g. \"CustomMode1\" or \"ExtendedEuler\") and restart.")
    elif max(seen.values(), default=0) < 1:
        print("  -> No signal changed between the two holds: the sensor did not tilt. Check that it is fixed to")
        print("     the board (not lying on the table or floor) and that the board was really tipped to each edge.")
    else:
        print("  -> The board tipped only a little. Make sure the LEFT edge is on the floor in step 2 and the")
        print("     RIGHT edge in step 3.")


def vec3(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def read_quat(packet):
    try:
        q = packet.orientationQuaternion()
    except Exception:
        return None
    for get in (lambda: (q.w(), q.x(), q.y(), q.z()), lambda: (q[0], q[1], q[2], q[3])):
        try:
            return tuple(float(v) for v in get())
        except Exception:
            pass
    return None


def fmt(v, nd):
    return "" if v is None else f"{v:.{nd}f}"


class WallKey:
    def __init__(self):
        self.active = False
        self._down = False
        self._listener = None

    def start(self):
        if keyboard is None:
            raise RuntimeError("pynput is not installed. Run:  pip install pynput")
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()

    def _on_press(self, key):
        if key != keyboard.Key.right or self._down:
            return
        self._down = True
        self.active = True if HOLD_MODE else (not self.active)

    def _on_release(self, key):
        if key != keyboard.Key.right:
            return
        self._down = False
        if HOLD_MODE:
            self.active = False


wall_key = WallKey()


class Stream(sdk.XsDotCallback):
    """Receives the sensor data (SDK thread): logs every sample, runs the live detector."""

    def __init__(self):
        sdk.XsDotCallback.__init__(self)
        self.lock = threading.Lock()
        self.found = []
        self.clock = bc.SampleClock()
        self.errors = 0
        self.probe = None
        self.file = None
        self.writer = None
        self.last_flush = 0.0
        self.phase = "setup"
        self.trial = 0
        self.collect = None
        self.cal = None
        self.det = None
        self.g = 9.81
        self.dt = None
        self.last_t = None
        self.missing = 0
        self.times, self.walls, self.trials, self.margins = [], [], [], []
        self.trial_i0 = 0
        self.trial_t0 = None
        self.live_state = None
        self.address = None
        self.beyond_warned = set()
        self.gyro_k = None

    # ---- SDK callbacks
    def onAdvertisementFound(self, port_info):
        if not port_info.isBluetooth():
            return
        with self.lock:
            if port_info.bluetoothAddress() not in [p.bluetoothAddress() for p in self.found]:
                print(f"Found: {port_info.bluetoothAddress()}")
                self.found.append(port_info)

    def onLiveDataAvailable(self, device, packet):
        try:
            self._on_packet(packet)
        except Exception as e:                 # never let an error escape into the SDK thread
            self.errors += 1
            if self.errors <= 3:
                print(f"\n  [data error] {e!r}")

    # ---- per sample
    def _on_packet(self, packet):
        host = time.perf_counter()
        has_ori = packet.containsOrientation()
        acc = vec3(packet.calibratedAcceleration()) if packet.containsCalibratedAcceleration() else None
        gyr = vec3(packet.calibratedGyroscopeData()) if packet.containsCalibratedGyroscopeData() else None
        facc = vec3(packet.freeAcceleration()) if packet.containsFreeAcceleration() else None
        raw = packet.sampleTimeFine()
        roll = pitch = yaw = None
        q = None
        if has_ori:
            e = packet.orientationEuler()
            roll, pitch, yaw = e.x(), e.y(), e.z()
            q = read_quat(packet)
        wall = wall_key.active
        with self.lock:
            ticks = self.clock.unwrap(raw)
            if self.probe is not None:
                self.probe["pairs"].append((ticks, host))
                fl = self.probe["fields"]
                if has_ori:
                    fl.add("orientation")
                if acc is not None:
                    fl.add("acc")
                if gyr is not None:
                    fl.add("gyro")
                if facc is not None:
                    fl.add("free acc")
                return
            if self.writer is None or not has_ori:
                return
            t = self.clock.seconds(ticks)
            if self.last_t is not None and self.dt and t - self.last_t > 1.5 * self.dt:
                self.missing += int(round((t - self.last_t) / self.dt)) - 1
            self.last_t = t
            up = bc.up_from_euler(roll, pitch)
            adyn = bc.dynamic_acc(up, acc, facc, self.g)
            tilt = margin = None
            contact = 0
            state = ""
            if self.det is not None:
                tilt = self.cal.tilt(up)
                side, margin = self.det.margin(tilt)
                if margin is not None and margin < -1.5 and side not in self.beyond_warned:
                    self.beyond_warned.add(side)
                    print(f"  [note] the board tipped {-margin:.1f} deg past the calibrated {bc.SIDE_NAME[side]} "
                          f"floor angle - the calibration hold was probably not on the floor")
                rate = None
                if gyr is not None and self.gyro_k is not None:
                    rate = self.gyro_k * bc.dot(gyr, self.cal.axis)
                events = self.det.update(t, tilt, adyn, rate)
                contact = int(self.det.in_contact)
                if self.trial > 0:
                    if self.trial_t0 is None:
                        self.trial_t0 = t
                    state = bc.state_of(wall, contact)
                    if state != self.live_state:
                        self._print_change(state, t, events)
                        self.live_state = state
            if self.collect is not None:
                self.collect.append((t, up, acc, facc, q, roll, pitch, gyr))
            self.times.append(t)
            self.walls.append(wall)
            self.trials.append(self.trial)
            self.margins.append(margin)
            self.writer.writerow([
                f"{t:.6f}", ticks, self.phase, self.trial, int(wall),
                f"{roll:.4f}", f"{pitch:.4f}", f"{yaw:.4f}",
                *(("", "", "", "") if q is None else (f"{v:.6f}" for v in q)),
                *(("", "", "") if acc is None else (f"{v:.4f}" for v in acc)),
                *(("", "", "") if gyr is None else (f"{v:.3f}" for v in gyr)),
                *(("", "", "") if facc is None else (f"{v:.4f}" for v in facc)),
                fmt(tilt, 3), fmt(margin, 3), fmt(adyn, 3), contact if self.det is not None else "", state,
            ])
            if host - self.last_flush > 2.0:
                self.file.flush()
                self.last_flush = host

    def _print_change(self, state, t, events):
        rel = t - self.trial_t0
        msg = ""
        for kind, c in events:
            if kind == "start":
                rel = c.onset - self.trial_t0
                how = {"impact": "landed", "rest": "resting"}.get(c.cue, c.cue) if PARAMS.board == "roller" else c.cue
                msg = f"{c.side_name} edge on the floor ({how})"
            elif kind == "resume":
                msg = f"{c.side_name} edge back on the floor"
            elif kind == "end" and state in ("balance", "support"):
                msg = f"board up after {c.duration:.2f} s on the floor"
        if not msg:
            if self.live_state is None:
                msg = "start"
            else:
                msg = "wall held" if state in ("support", "idle") else "wall released"
        print(f"  {rel:6.2f} s  {state.upper():8} {msg}")

    # ---- control (main thread)
    def begin_probe(self):
        with self.lock:
            self.probe = {"pairs": [], "fields": set()}

    def end_probe(self):
        with self.lock:
            pr, self.probe = self.probe, None
        return pr

    def start_log(self, path, ticks_per_s, dt):
        with self.lock:
            self.clock.ticks_per_s = ticks_per_s
            self.clock.origin = None
            self.dt = dt
            self.file = open(path, "w", newline="")
            self.writer = csv.writer(self.file)
            self.writer.writerow(bc.LOG_COLUMNS)

    def close_log(self):
        with self.lock:
            if self.file is not None:
                self.file.flush()
                self.file.close()
            self.writer = None
            self.file = None

    def set_phase(self, phase, trial=0):
        with self.lock:
            self.phase = phase
            self.trial = trial

    def record(self, phase, seconds):
        """Collect (t, up, acc, free acc, quaternion, roll, pitch) for `seconds`, tagging the samples with `phase`."""
        with self.lock:
            self.phase = phase
            self.collect = []
        time.sleep(seconds)
        with self.lock:
            data, self.collect = self.collect, None
            self.phase = "setup"
        if len(data) < 0.5 * seconds / (self.dt or 1 / 60):
            raise RuntimeError(f"only {len(data)} samples in {seconds:.0f} s - is the sensor still streaming?")
        return data

    def start_detection(self, cal, g, impact_acc, gyro_k):
        with self.lock:
            self.cal = cal
            self.g = g
            self.gyro_k = gyro_k
            self.det = bc.make_detector(PARAMS, cal.theta[-1], cal.theta[1], impact_acc=impact_acc)

    def start_trial(self, trial):
        with self.lock:
            self.phase = "trial"
            self.trial = trial
            self.trial_i0 = len(self.times)
            self.trial_t0 = None
            self.live_state = None

    def end_trial(self):
        """Back to 'break'; returns the live summary of the trial."""
        with self.lock:
            tr = self.trial
            self.phase = "break"
            self.trial = 0
            i0, i1 = self.trial_i0, len(self.times)
            if i1 - i0 < 2 or self.det is None:
                return None
            t_end = self.times[i1 - 1] + (self.dt or 1 / 60)
            contacts = [dataclasses.replace(c) for c in self.det.contacts]
            if contacts and contacts[-1].end is None:
                contacts[-1].end = t_end
            times, walls, margins = self.times, self.walls, self.margins
        lab = bc.label_trial(times, walls, margins, contacts, i0, i1, t_end, PARAMS)
        return bc.summarize_trial(tr, times[i0], t_end, lab, contacts)


def connect(manager, stream):
    print(f"Scanning for DOT sensors ({SCAN_S:.0f} s)...")
    manager.enableDeviceDetection()
    time.sleep(SCAN_S)
    manager.disableDeviceDetection()
    ports = stream.found
    if SENSOR_ADDRESS:
        ports = [p for p in ports if p.bluetoothAddress().upper() == SENSOR_ADDRESS.upper()]
    if not ports:
        print("No sensor found. Is it awake and not connected to the phone app?")
        return None, None
    port = ports[0]
    if len(ports) > 1:
        print("\nMore than one sensor found:")
        for k, pt in enumerate(ports, 1):
            print(f"  {k}: {pt.bluetoothAddress()}")
        while True:
            pick = input(f"Which one is on the board? [1-{len(ports)}] ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(ports):
                port = ports[int(pick) - 1]
                break
        print("  (set SENSOR_ADDRESS at the top of the script to skip this question)")
    print("Connecting to", port.bluetoothAddress())
    if not manager.openPort(port):
        print("Connection failed")
        return None, None
    return manager.device(port.deviceId()), port.bluetoothAddress()


def configure_stream(device, stream):
    """Start the fastest rate / richest payload the sensor delivers reliably.
    Returns a dict describing the running stream, or None."""
    fallback = None
    for rate in OUTPUT_RATES:
        for name in PAYLOAD_MODES:
            mode = getattr(sdk, "XsPayloadMode_" + name, None)
            if mode is None:
                continue
            if not device.setOutputRate(rate):
                print(f"  {rate} Hz not accepted")
                break
            stream.begin_probe()
            if not device.startMeasurement(mode):
                stream.end_probe()
                print(f"  {name} at {rate} Hz: could not start")
                continue
            time.sleep(PROBE_S)
            pr = stream.end_probe()
            info = evaluate_probe(pr, rate, name, mode)
            if info is None:
                print(f"  {name} at {rate} Hz: no data")
                device.stopMeasurement()
                time.sleep(0.5)
                continue
            print(f"  {name} at {rate} Hz: {info['measured']:.1f} Hz received, {100 * info['drop']:.1f} % lost, "
                  f"fields: {', '.join(sorted(info['fields']))}")
            if info["good"]:
                return info
            if "orientation" in info["fields"] and fallback is None:
                fallback = info
            device.stopMeasurement()
            time.sleep(0.5)
            if info["measured"] < 0.9 * rate:
                break                      # the sensor does not stream this fast; try the next rate
    if fallback is None:
        return None
    print(f"  Using {fallback['mode_name']} at {fallback['rate']} Hz (not all checks passed)")
    device.setOutputRate(fallback["rate"])
    stream.begin_probe()
    device.startMeasurement(fallback["mode"])
    time.sleep(PROBE_S)
    return evaluate_probe(stream.end_probe(), fallback["rate"], fallback["mode_name"], fallback["mode"])


def evaluate_probe(pr, rate, name, mode):
    pairs = pr["pairs"]
    if len(pairs) < max(10, 0.3 * rate * PROBE_S):
        return None
    tps, ratio = bc.SampleClock.detect_unit(pairs)
    if tps is None:
        return None
    dts = [(b[0] - a[0]) / tps for a, b in zip(pairs, pairs[1:]) if b[0] > a[0]]
    dt = bc.median(dts)
    span = (pairs[-1][0] - pairs[0][0]) / tps
    expected = span / dt + 1
    drop = max(0.0, 1 - len(pairs) / expected)
    fields = pr["fields"]
    good = ("orientation" in fields and ("acc" in fields or "free acc" in fields)
            and 1 / dt >= 0.9 * rate and drop <= 0.03)
    return {"rate": rate, "mode_name": name, "mode": mode, "measured": 1 / dt, "dt": dt, "drop": drop,
            "fields": fields, "ticks_per_s": tps, "tick_ratio": ratio, "good": good}


def calibrate(stream):
    p = PARAMS
    print("\n" + "=" * 60)
    print(" CALIBRATION - the participant stands on the board and keeps")
    print(" both hands on the wall throughout.")
    print("=" * 60)
    while True:
        instruction("CHECK: when you press Enter, tip the board onto the LEFT edge and then the RIGHT edge "
                    "(about 5 s).")
        data = stream.record("setup", 5.0)
        ups = [d[1] for d in data]
        moved = max(bc.angle_deg(ups[0], v) for v in ups)
        if moved >= 2 * p.min_contact_deg:
            print(f"  OK - the sensor tilted {moved:.0f} deg.")
            check_ups, check_data = ups, data
            break
        print(f"\n  The sensor tilted only {moved:.1f} deg while the board was rocked. This sensor is not moving "
              f"with the board:\n  it may be a different sensor than the one on the board (connected: "
              f"{stream.address}),\n  or it is not fixed to the board. Fix it, or press q and restart to pick "
              f"another sensor.")
    instruction("STEP 1 of 3: Board roughly level, relaxed, as still as possible.")
    quiet = stream.record("cal_quiet", CAL_QUIET_S)
    while True:
        u = {}
        for side, name, step in ((-1, "LEFT", 2), (1, "RIGHT", 3)):
            while True:
                instruction(f"STEP {step} of 3: Lower the {name} edge to the floor and let it rest there "
                            f"with full weight on the board. Hold still.")
                data = stream.record("cal_left" if side < 0 else "cal_right", CAL_EDGE_S)
                ok, up, frac, dev = bc.rest_direction([d[1] for d in data], p.cal_still_dev_deg, p.cal_still_frac)
                if ok:
                    u[side] = (up, data)
                    break
                print(f"  The board was not resting still ({frac:.0%} of samples still). Once more, please.")
        apart = bc.angle_deg(u[-1][0], u[1][0])
        if apart < 2 * p.min_contact_deg:
            explain_small_angle(u[-1][1], u[1][1], apart)
            continue
        # the floor is a hard limit: if the board went clearly further during the check,
        # that hold was not resting on the floor
        cal = bc.Calibration.from_contacts(u[-1][0], u[1][0])
        if PARAMS.board == "roller":
            break       # on a roller the floor angle moves with the roller: holds only give the axis
        reach = {s: max(s * cal.tilt(v) for v in check_ups) for s in (-1, 1)}
        short = [s for s in (-1, 1) if reach[s] > cal.theta[s] + 1.0]
        if not short:
            break
        for s in short:
            print(f"\n  In the check the board tipped {reach[s] - cal.theta[s]:.1f} deg further to the "
                  f"{bc.SIDE_NAME[s].upper()} than in the {bc.SIDE_NAME[s].upper()} hold, so that edge was not "
                  f"resting on the floor.\n  Lower it all the way until it touches the floor. Both holds again, "
                  f"please.")
    cal = bc.Calibration.from_contacts(u[-1][0], u[1][0])
    held = quiet + u[-1][1] + u[1][1]
    mags = [bc.norm(d[2]) for d in held if d[2] is not None]
    g = bc.median(mags) if mags else 9.81
    qd = [bc.dynamic_acc(d[1], d[2], d[3], g) for d in quiet]
    qd = [x for x in qd if x is not None]
    impact_acc = p.impact_acc_ms2
    if qd:
        impact_acc = max(impact_acc, p.impact_quiet_factor * bc.percentile(qd, 99))
    quiet_tilt = bc.median([cal.tilt(d[1]) for d in quiet])
    # gyroscope -> rocking rate in deg/s (sign and deg/s vs rad/s), from the rocking in the check
    gyro_k = None
    gd = [d for d in check_data if d[7] is not None]
    if len(gd) > 30:
        tilts = [cal.tilt(d[1]) for d in gd]
        dtilt = [None] + [(tilts[i + 1] - tilts[i - 1]) / (gd[i + 1][0] - gd[i - 1][0])
                          if gd[i + 1][0] > gd[i - 1][0] else None for i in range(1, len(gd) - 1)] + [None]
        gyro_k = bc.gyro_scale([bc.dot(d[7], cal.axis) for d in gd], dtilt)
    if PARAMS.board == "roller":
        impact_acc = max(impact_acc, p.roller_impact_acc_ms2)
        print(f"\nRocking axis found (edges {cal.theta[-1] + cal.theta[1]:.1f} deg apart, quiet stance at "
              f"{quiet_tilt:+.1f} deg).")
        print(f"Contact = the board landing (jolt >= {impact_acc:.0f} m/s^2) or resting still at a tilt of "
              f">= {p.roller_min_tilt_deg:g} deg; it ends when the board tilts back {p.roller_lift_deg:g} deg.")
        if gyro_k is None:
            print("  No usable gyroscope: stillness is judged from the tilt (less sensitive).")
    else:
        print(f"\nContact angles: left {cal.theta[-1]:.1f} deg, right {cal.theta[1]:.1f} deg "
              f"(quiet stance at {quiet_tilt:+.1f} deg)")
        if not (-cal.theta[-1] < quiet_tilt < cal.theta[1]):
            print("  Note: the quiet stance was not between the two edges - check that the board rested on the "
                  "floor in steps 2 and 3.")
        print(f"Contact = within {PARAMS.angle_tol_deg} deg of the floor angle, resting within "
              f"{PARAMS.rest_tol_deg} deg, or a jolt >= {impact_acc:.1f} m/s^2 next to the floor")
    if not mags and not qd:
        print("  No acceleration in this payload: no landing cue.")
    holds = {"quiet": [quiet[0][0], quiet[-1][0]],
             "left": [u[-1][1][0][0], u[-1][1][-1][0]], "right": [u[1][1][0][0], u[1][1][-1][0]]}
    return cal, g, impact_acc, holds, gyro_k


def countdown(trial, seconds):
    print("\n" + "=" * 50)
    print(("Calibration complete. Get ready to balance!" if trial == 1
           else f"Trial {trial}. Get ready to balance!").upper())
    print("=" * 50)
    print("\nSession starts in:")
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r  {i}... ")
        sys.stdout.flush()
        beep(1000, 200)
        time.sleep(0.8)
    print("\r  GO!            ")
    beep(1500, 500)


def print_summary(s):
    if s is None:
        return
    print("\n" + "=" * 50)
    print(f" TRIAL {s['trial']} (live)")
    print("=" * 50)
    print(f" Balance:   {s['balance_s']:5.1f} s    longest {s['longest_balance_s']:.1f} s")
    print(f" Support:   {s['support_s']:5.1f} s")
    print(f" Idle:      {s['idle_s']:5.1f} s")
    print(f" Fail:      {s['fail_s']:5.1f} s    fails {s['n_fails']}")
    print(f" Contacts:  {s['n_contacts']}    near misses {s['near_misses']}")
    print("=" * 50)


def main():
    print("=== BIDS Session Setup ===")
    sub_id = input("Enter Subject ID (e.g., GT6L1K): ").strip() or "GT6L1K"
    run_id = input("Enter Run Number (e.g., 001): ").strip() or "001"
    base = f"sub-{sub_id}_ses-{SESSION_LABEL}_task-balanceboard_run-{run_id}"
    log_path = base + "_beh.csv"
    cal_path = base + "_calibration.json"
    if os.path.exists(log_path):
        if input(f"{log_path} exists. Overwrite? [y/N] ").strip().lower() != "y":
            return
    print(f"\nData will be saved to: {log_path}\n")

    wall_key.start()
    stream = Stream()
    manager = sdk.XsDotConnectionManager()
    manager.addXsDotCallbackHandler(stream)
    device, address = connect(manager, stream)
    stream.address = address
    if device is None:
        wall_key.stop()
        manager.close()
        return

    print("Testing the data stream...")
    info = configure_stream(device, stream)
    if info is None:
        print("The sensor sends no orientation in any payload mode tried. Check PAYLOAD_MODES.")
        manager.close()
        wall_key.stop()
        return
    if info["tick_ratio"] and abs(info["tick_ratio"] / info["ticks_per_s"] - 1) > 0.2:
        print(f"  Note: sensor clock {info['tick_ratio']:.0f} ticks/s, using {info['ticks_per_s']:.0f}")
    print(f"Streaming {info['mode_name']} at {info['measured']:.0f} Hz "
          f"(timestamps: {info['ticks_per_s']:.0f} ticks per second)")
    stream.start_log(log_path, info["ticks_per_s"], info["dt"])

    run_completed = False
    try:
        cal, g, impact_acc, holds, gyro_k = calibrate(stream)
        meta = {
            "script": os.path.basename(__file__), "version": bc.VERSION,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sensor": address,
            "payload_mode": info["mode_name"], "fields": sorted(info["fields"]),
            "requested_rate_hz": info["rate"], "sample_rate_hz": info["measured"],
            "ticks_per_s": info["ticks_per_s"], "g": g, "impact_acc": impact_acc,
            "board": PARAMS.board, "gyro_scale": gyro_k,
            "calibration": cal.to_dict(), "cal_holds": holds, "params": PARAMS.to_dict(),
            "trials": TOTAL_TRIALS, "trial_s": SESSION_DURATION_S,
        }
        with open(cal_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        stream.start_detection(cal, g, impact_acc, gyro_k)
        instruction("Ready. The participant gets comfortable on the board, holding the wall.")
        print(f"\nProgram configured for {TOTAL_TRIALS} trials of {SESSION_DURATION_S:.0f} s.")

        for trial in range(1, TOTAL_TRIALS + 1):
            stream.set_phase("countdown")
            countdown(trial, COUNTDOWN_S)
            stream.start_trial(trial)
            print(f" [TRIAL {trial} STARTED] ".center(50, "!"))
            t_start = time.time()
            while time.time() - t_start < SESSION_DURATION_S:
                time.sleep(0.05)
            summary = stream.end_trial()
            beep(1000, 400)
            beep(800, 600)
            print_summary(summary)
            if trial < TOTAL_TRIALS:
                print(f"\n=== BREAK TIME ({BREAK_DURATION_S:.0f}s) ===")
                print("Step off the board and take a break.")
                for i in range(int(BREAK_DURATION_S), 0, -1):
                    sys.stdout.write(f"\rNext session starts in: {i}s ... ")
                    sys.stdout.flush()
                    time.sleep(1)
                print("\r" + " " * 50 + "\r", end="")
        print("\n\n=== RUN COMPLETE! ===")
        run_completed = True
    except KeyboardInterrupt:
        print("\n\nRun interrupted. Saving the data...")
    except Exception as e:
        print(f"\n\nStopped by an error: {e!r}. Saving the data...")

    # ---- stop, save, label (data is saved before anything else)
    stream.set_phase("end")
    try:
        device.stopMeasurement()
    except Exception as e:
        print(f"  stopMeasurement: {e!r}")
    manager.close()
    wall_key.stop()
    stream.close_log()
    print(f"\nRaw data saved to: {log_path}")
    if stream.missing:
        print(f"  {stream.missing} samples lost in transmission")
    if os.path.exists(cal_path):
        print("\nFinal labels (offline pass):\n")
        try:
            label_session.label_session(log_path, params=PARAMS, calib_path=cal_path)
        except Exception as e:
            print(f"Labelling failed ({e!r}). The raw data is safe; run:  python label_session.py {log_path}")

    if run_completed:
        print("\n=== STARTING 10-MINUTE RECOVERY BREAK ===")
        print("Data is already saved. The script will close automatically when the break ends.")
        print("(Press Ctrl+C to skip the break and close immediately)")
        try:
            for i in range(END_BREAK_S, 0, -1):
                mins, secs = divmod(i, 60)
                sys.stdout.write(f"\rTime remaining: {mins:02d}:{secs:02d} ... ")
                sys.stdout.flush()
                time.sleep(1)
            print("\nBreak complete. Closing program.")
        except KeyboardInterrupt:
            print("\nBreak skipped. Closing program.")


if __name__ == "__main__":
    main()
