# Balance board: ground-contact pipeline (v11)

Detects when the balance board touches the floor, for as long as it stays down,
and labels every sample of every trial with one of four states:

| state   | wall key | board          |
|---------|----------|----------------|
| balance | not held | off the floor  |
| support | held     | off the floor  |
| idle    | held     | on the floor   |
| fail    | not held | on the floor (onset = the moment the board touches) |

Tilt is measured about the board's rocking axis, found from the direction of
gravity in the two calibration holds. The sensor can be mounted in any
orientation and heading drift doesn't matter.

Set the board type at the top of `05_balance_task.py`: `BOARD = "roller"`
(default) or `"rocker"`. `label_session.py` takes `--board`. Section 6 explains
how contacts are found on each.

## 1. Requirements

| Item | Details |
|------|---------|
| PC | Windows 10/11 with Bluetooth Low Energy |
| Python | **3.10, 64-bit** (the SDK wheel is built for a specific Python version) |
| Sensor | Movella DOT, charged, firmly mounted on the board |
| Editor | Visual Studio Code (optional, any terminal works) |

## 2. Installation

### 2.1 Download the Movella DOT PC SDK

1. Download the **Movella DOT PC SDK** from the
   <a href="https://www.xsens.com/support/software-documentation?hsCtaTracking=39d661fa-2ea8-4478-955e-01d0d8885f14%7C3ad1c7d6-9c3a-42e9-b424-5b15b9d0924e">Movella software and documentation page</a>
   (software / documentation downloads >> Xsens DOT >> Latest stable software - PC >>
   Xsens DOT PC SDK (Windows)) and unzip it.
2. Inside, find the Python wheel for your Python version, e.g.
   `movelladot_pc_sdk-<version>-cp310-none-win_amd64.whl`. `cp310` means Python 3.10.

### 2.2 Create a Python environment

With conda / miniforge:

```bash
conda create -n balanceboard python=3.10
conda activate balanceboard
```

or with venv:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2.3 Install the packages

```bash
pip install "C:\path\to\movelladot_pc_sdk-<version>-cp310-none-win_amd64.whl"
pip install pynput matplotlib
```

- `movelladot_pc_sdk` communicates with the sensor.
- `pynput` reads the RIGHT ARROW key (the "holding the wall" marker).
- `matplotlib` draws the QC plots. Without it everything else still works.
- `numpy` is only needed for the simulation tests (`pip install numpy`).

### 2.4 Get the pipeline files

Put these files together in one folder. `05_balance_task.py` imports the others.

| file | what |
|------|------|
| `05_balance_task.py` | the task: sensor connection, calibration, 7 trials, live printout, raw log, final labels at the end |
| `balance_core.py` | shared detection code and all thresholds (`Params`) |
| `label_session.py` | final labels, outputs and report for a raw log (also reads 04 logs) |
| `qc_plots.py` | QC figures (needs matplotlib) |
| `simulate_session.py`, `test_pipeline.py` | simulated sessions with known contacts, and the accuracy test (need numpy) |

### 2.5 Select the environment in VS Code

1. Open the folder containing `05_balance_task.py`.
2. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → choose the `balanceboard` environment.
3. Open a new terminal (`` Ctrl+Shift+` ``) so it uses this environment.

## 3. Prepare the sensor

1. Charge the sensor and wake it up.
2. Close the Movella DOT phone app and any other program connected to the sensor.
3. Make sure Bluetooth is on in Windows.
4. Fix the sensor to the board so it cannot rotate or slide. Its orientation does
   not matter: the script finds the tilt axis itself.
5. Switch off other DOT sensors, or note the address of the one on the board.
6. Check the settings at the top of `05_balance_task.py`: `BOARD` (`"roller"` or
   `"rocker"`), `SENSOR_ADDRESS`, `TOTAL_TRIALS`, durations.

## 4. Run

```bash
python 05_balance_task.py
```

1. Enter the subject ID and run number. If several sensors are found, it asks
   which one is on the board (or set `SENSOR_ADDRESS` at the top of the script).
2. The script scans, connects and tests the stream. It tries 120 Hz, then 60 Hz,
   and several payload modes. It keeps the first one that delivers orientation
   plus acceleration without losing packets, and works out the timestamp unit
   from the PC clock. The console shows what it chose.
3. **Calibration.** The participant stands on the board with both hands on the
   wall throughout:
   - check: tip the board to each edge once (5 s). If the sensor doesn't tilt
     with the board (wrong sensor, or not fixed to it), the script says so
     before calibrating.
   - step 1: board roughly level, still (5 s)
   - step 2: LEFT edge resting on the floor, full weight on the board (3 s)
   - step 3: RIGHT edge resting on the floor (3 s)

   A step repeats automatically if the board wasn't resting still. On a roller
   board the holds only give the rocking axis, so where the roller sits doesn't
   matter. The rocking in the check also sets up the gyroscope. Type `q` at any
   prompt to stop; the data recorded so far is kept.
4. The trials run: countdown beeps, 30 s, beeps, 45 s break. Hold the
   **right arrow key** while the participant holds the wall. Each state change
   is printed:

   ```
       4.83 s  IDLE     left edge on the floor (resting)
       5.33 s  FAIL     wall released
       6.30 s  BALANCE  board up after 1.09 s on the floor
   ```

5. At the end the final labels are written and the report is printed. If the run
   is interrupted (Ctrl+C or an error), the raw data is still saved, and the
   labels can be made later with `python label_session.py <..._beh.csv>`.

## 5. Outputs

All files start with `sub-<id>_ses-S001_task-balanceboard_run-<run>`.

| file | content |
|------|---------|
| `_beh.csv` | raw log, one row per sample: time, sensor ticks, phase, trial, wall, Euler angles, quaternion, acceleration, gyro, live tilt/margin/state |
| `_calibration.json` | rocking axis, contact angles, accepted calibration holds, payload, sample rate, thresholds |
| `_events.tsv` (+ `.json`) | BIDS events: `trial`, the state periods (`balance`/`support`/`idle`/`fail`) and `contact` rows, with `cause`, `cue`, `side` |
| `_desc-trials_beh.csv` | per trial: seconds in each state, `n_fails`, contacts by cue, longest balance, time to first fail, near misses, angles used, lost samples |
| `_desc-contacts_beh.csv` | per contact: onset, end, duration, side, first cue, all cues seen, wall at onset, approach speed, peak jolt, how far past the angle |
| `_desc-labels_beh.csv` | per sample: tilt, margin, dynamic acceleration, contact id, state |
| `_qc/report.txt` | contact angles, thresholds, per-trial table, which cue found the contacts, warnings |
| `_qc/trial-NN.png` | tilt with floor lines, state shading, contact markers by cue, near misses, and acceleration with the impact threshold |
| `_qc/contact_angles.png` | roller: the tilt at which each contact rested; rocker: where the board rested near each floor, with the calibrated and used angles |

A fail period has a `cause`:

- `touch`: the board touched the floor without the wall. `n_fails` counts these.
- `wall_release`: the wall was let go while the board was already down.
- `trial_start`: the board was already down at GO.

Time in `fail` includes all three causes. Event onsets are seconds from the
first sample of the raw log.

## 6. How a contact is detected

### 6.1 Roller board (board on a free roller)

Where the roller sits under the board changes the tilt at which an edge meets the
floor: with the roller far to one side, that side's edge needs a steeper tilt and
the other edge a flatter one. In the test recordings the board rested on the
floor anywhere between about 9 and 23 deg. So no set angle is used. A contact
starts when the board

| cue | fires when |
|-----|-----------|
| landing | a jolt >= 12 m/s^2 at a tilt of at least 10 deg |
| rest | the rocking rate (gyroscope) stays below 3 deg/s for 0.15 s at a tilt of at least 7 deg. The floor holds the board far stiller than a rider can: resting on the floor the rate was 0.2-0.7 deg/s, balancing it never stayed below about 3 deg/s. The onset is backdated to when the board arrived at that tilt |

and lasts until the board tilts back 2.5 deg from where it rested (or from the
steepest tilt since), for 0.1 s, or passes level. Swinging close to the floor
without stopping is not a contact, however steep.

Thresholds are the `roller_*` fields of `balance_core.Params`.

### 6.2 Rocker board (fixed curved base)

Each side has a **contact angle**: the tilt at which that edge rests on the floor
with the participant standing on the board. The **margin** is how far the board
is from the floor on the side it leans to. A contact starts at the first of
three cues:

| cue    | fires when | catches |
|--------|-----------|---------|
| angle  | margin <= 0.5 deg | most contacts; the floor is a hard limit, so the tilt stays there as long as the board is down |
| rest   | margin <= 1.25 deg (0.8 after refinement) and the board has stopped rotating for 0.12 s | resting contacts that sit slightly short of the calibrated angle |
| impact | a jolt >= 3 m/s^2 next to the floor, after moving toward it, with the board reaching the floor within 0.1 s | the exact moment of hard landings, and quick taps between samples |

The contact lasts until the board is at least 3 deg away from the floor for
0.2 s, or passes level. The offline pass re-measures each side's contact angle
from the moments the board rested on the floor, checks whether the sensor turned
on the board mid-session, and refines onsets and ends.

### 6.3 Live vs final labels

The same detector runs twice: **live** during the task for the console printout,
and **offline** at the end of the run (`label_session.py`). The offline labels
are the final ones and can be re-made with other thresholds without testing
anyone again.

## 7. Checking and tuning

Look at the trial plots and the CHECK section of the report.

Roller board:

```
python label_session.py sub-..._beh.csv --set roller_still_dps=4 roller_min_tilt_deg=8
```

| parameter | default | raise it if | lower it if |
|-----------|---------|-------------|-------------|
| `roller_still_dps` | 3.0 | rests on the floor are missed | slow pauses in balancing count as contacts |
| `roller_min_tilt_deg` | 7 | pauses near level count as contacts | rests at a shallow angle (roller far to the side) are missed |
| `roller_impact_acc_ms2` | 12 | hard moves while balancing count as contacts | quick taps are missed |
| `roller_lift_deg` | 2.5 | one contact is split into several | separate touches merge |

Rocker board:

- **Near misses** within about 1 deg of the floor are listed. If one was really
  a touch, raise the angle tolerance.
- **Impact-only contacts** are quick taps or jolts. If they are not real,
  raise `impact_acc_ms2`.
- **Contact angles that differ from the calibration**, or a **sensor shift**,
  are warned about.

Re-label with other thresholds (any field of `balance_core.Params`):

```
python label_session.py sub-..._beh.csv --set angle_tol_deg=0.7 lift_deg=2.5
python label_session.py sub-..._beh.csv --no-refine     # calibrated angles only, like the live pass
```

| parameter | default | raise it if | lower it if |
|-----------|---------|-------------|-------------|
| `angle_tol_deg` | 0.5 | light touches are missed | near misses count as fails |
| `rest_tol_deg` | 1.25 (0.8 refined) | resting contacts are missed | holding still near the floor counts as a fail |
| `impact_acc_ms2` | 3.0 | jolts away from the floor become contacts | hard landings are timed late |
| `lift_deg`, `lift_hold_s` | 3.0, 0.2 | one fall is split into several | separate touches merge |

For thresholds to change in the live pass as well, edit `PARAMS` at the top of
`05_balance_task.py`.

## 8. Old recordings (04 script)

`python label_session.py <old _beh.csv>` also works on logs from
`04_balance_task.py`:

- It detects that the timestamps are 100 times too large (04 divided
  microseconds by 10000) and rescales them.
- Those logs have no marked calibration, so it finds both floor contacts in the
  data. The names left and right may then be swapped.
- 04 logged `| |a| - g |` rather than true dynamic acceleration, so the impact
  cue is weak there.

## 9. Results on real recordings (roller board)

Two test recordings from 28 Oct 2026:

- **Run 2 (touching on purpose, 2 trials):** all 10 touches found, including
  rests at only 9 deg (roller far to one side) and at 20 deg. Fail time 31 s.
- **Run 1 (balancing, 7 trials):** the old angle-based labels counted 25 fails,
  most of them fast swings to 15-18 deg. The roller detector finds 8, plus the
  trial starts where the board was still down. Each is a period where the board
  stopped dead at 10-23 deg for 0.6-5 s (trial 6 from 8 to 13 s, for example),
  which the floor allows but balancing doesn't.

## 10. Testing without the sensor (rocker simulation)

`python test_pipeline.py --seeds 6` simulates sessions with known contacts and
scores the final labels. A simulated session contains falls, taps of 4 to 50 ms,
rests of up to 9 s with weight shifts, slow lowering, near misses, still hovering
just above the floor, stomps, wall periods and side switches. It also models
sensor noise, lost packets and a wrapping clock.

Results from 6 simulated sessions per scenario (149 true contacts each):

| scenario | found | missed | false | onset error, median / 95 % |
|----------|-------|--------|-------|-----------------------------|
| hard floor, 60 Hz | 149 | 0 | 6 | 6 / 40 ms |
| hard floor, 120 Hz | 149 | 0 | 6 | 7 / 40 ms |
| soft mat (0.8 deg give) | 148 | 1 | 0 | 8 / 136 ms |
| participant pressed lightly in calibration | 149 | 0 | 8 | 5 / 100 ms |
| sensor turned 1.5 deg on the board before trial 4 | 147 | 2 | 9 | 6 / 39 ms |
| orientation only, no acceleration | 149 | 0 | 6 | 6 / 40 ms |
| free acceleration, sensor mounted sideways, 10 kHz ticks, 3 % packets lost | 149 | 0 | 6 | 8 / 39 ms |

Every false contact came from a near miss within about 0.5 deg of the floor, or
from the board held still within about 0.8 deg of it. The IMU cannot tell those
from a touch: 0.5 deg is about 2 mm at the edge of the board. The slow tail of
the onset error comes from boards lowered at 1-3 deg/s, where the exact moment
of touching is uncertain.

These numbers come from a model, not from real recordings. The QC plots of the
first real sessions are the actual check.

## 11. Limits

- Roller board: a touch that doesn't jolt (< 12 m/s^2) and doesn't stop the
  board (a slow graze followed by immediate recovery) is missed. A rider who
  holds the board perfectly still at a steep tilt would count as touching; in
  the recordings balancing never did.
- Roller board: thresholds come from two recordings of one person. Check the
  plots of the first real participants.

- Holding the board motionless within about 0.8 deg of the floor looks the same
  as resting on it.
- On a soft mat, a light touch stops short of where the board rests under full
  weight. The offline pass handles this by using the lightest rests; the live
  printout may miss some light taps.
- A sensor that turns on the board is caught only after a trial with rests on
  both floors. Fix it firmly.
- The wall state is only as accurate as the key presses.
- Which payload modes and rates the sensor streams depends on its firmware. The
  script tests them and prints what it got. It needs orientation, and
  acceleration for the impact cue.