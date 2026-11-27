# AEROVIGIL

### Intelligent UAV Navigation Integrity Monitoring and Adaptive GPS Spoofing Detection

## Overview

AEROVIGIL is a final-year research project on detecting GPS/GNSS spoofing attacks
against small unmanned aerial vehicles (multirotors). It compares physics-based
detectors (a Kalman filter with statistical tests, sensor-consistency checks) with
classical machine learning and LSTM models on the same flights, at the same
false-alarm budget, and then attacks them with an adaptive attacker that reshapes its
drift to avoid detection.

The project uses simulated flight logs only. It does not transmit any GPS signal, and
no result here has been validated on a real drone.

## Dataset

PX4 SITL + Gazebo UAV GPS-spoofing dataset (Anthony Finn): 360 flights (216 attacked,
144 normal) over three missions (curved, random, straight), logged at about 125–250 Hz.
Flights are split by flight (seed 42): 252 train, 54 validation, 54 test. The dataset is
not redistributed in this repository.

## What the pipeline does

| Module | Purpose |
|---|---|
| 03 | Base data: aligned sensors, measured attack phases (normal / pre / ramp / plateau / post), 36 residual features |
| 04 | Synthetic attacks: five families (abrupt jump, constant offset, linear drift, exponential ramp, low-amplitude wander), each as position-only and velocity-consistent, plus a 7-speed drift sweep (0.05–5 m/s) on the test flights |
| 05 | Classical models: logistic regression, Naive Bayes, KNN, decision tree, random forest, isolation forest, scored as event detection with a per-flight latch |
| 06 | Kalman filter with a chi-square (NIS) test and CUSUM |
| 07 | LSTM autoencoder and supervised LSTM |
| 08 | Black-box adaptive attacker (hill climber constrained on the measured mean drift speed) |
| 09 | Consistency detector (GPS velocity vs IMU, course vs heading) |
| 10 | Fused detector, cross-validated false alarms, threat-model table |
| Evaluation | One metrics module and protocol set for every detector: seen missions, unseen missions, unseen attack types, cross-validated false alarms, 5 seeds, ablations |
| Stealth | Stealth Frontier, adaptive and white-box attacks on every detector, adversarial hardening (in progress) |

Every alarm rate in the project uses the same budget: **1 false alarm per flight-hour**,
with time taken from the log timestamps, a 10 s refractory period, and a Poisson 95%
upper bound reported with the count and hours.

## Results so far

All numbers are on held-out test flights; the full tables are in `reports/tables/`.

**Real attacks (the dataset's own spoofing).** Every detector except the decision tree
and isolation forest catches 97–100% of real attacked flights. The physics detectors
react fastest: the velocity channel and the fused detector in about 0.2 s, the
chi-square test in about 0.4 s; the classical models in about 1.5 s.

**Unseen missions (train on two missions, test on the third).** This is the hardest
and most honest test. Real attacks are still all caught, but the learned models break
the false-alarm budget: logistic regression 28.8/h, random forest 22.0/h, supervised
LSTM 8.8/h, against 1.7–1.8/h for the Kalman and fused detectors. Almost all of the
excess comes from the random mission, which flies very differently from the other two.

**Window length matters.** Shortening the classical models' window from 5 s to 2 s cuts
unseen-mission false alarms sharply (random forest 22 → 0.84/h, logistic regression
29 → 6.7/h) while still catching every real attack; 10 s windows make it worse
(66–69/h). Removing the barometer features changes little.

**Cross-validated false alarms (all 144 normal flights).** Every detector stays within
0–1.33 alarms per flight-hour; the fused detector's 95% upper bound is 2.50/h.

**Smooth, velocity-consistent drift.** The physics detectors catch slow drifts
(≤ 0.5 m/s) on only about 10% of flights, and Module 08's adaptive attacker can move
the drone hundreds of metres before the chi-square test alarms. How the learned models
fare against the same drifts and against an attacker that targets them is being
measured in the Stealth Frontier stage.

## Limitations

- The synthetic attacks are added to recorded logs (open loop). A real spoofed drone
  would react in closed loop, so results on synthetic drifts may not carry over.
- Simulation only; no real-flight validation.
- Before the cross-validated stage, false-alarm rates rest on about 45 minutes of
  normal test flight, so their confidence bounds are wide.

## Running it

Windows, Python 3.11, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

The pipeline steps run in order (each reads the previous step's output), for example:

```powershell
python src\aerovigil\attacks\generate_v2.py
python src\aerovigil\baselines\build_window_tables_v2.py --sources orig,syn
python src\aerovigil\baselines\rescore_v2.py
python src\aerovigil\kalman\run_filter.py
python src\aerovigil\fusion\signals.py
python src\aerovigil\fusion\fuse.py
python src\aerovigil\evaluation\run_all.py      # rebuilds every results table
```

Checks:

```powershell
python -m black --line-length 88 src\aerovigil tests
python -m ruff check src\aerovigil tests
python -m pytest tests -q
```

## Technology

Python 3.11, NumPy, pandas, scikit-learn, PyTorch, Matplotlib, PyArrow, pytest.

## Project structure

- `configs/` configuration files
- `data/` dataset and intermediate results (large files are not committed)
- `docs/` technical notes per module
- `reports/` result tables and figures
- `src/aerovigil/` source code, one package per module
- `tests/` automated tests
