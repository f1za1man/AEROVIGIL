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
| Stealth | Stealth Frontier, adaptive (black-box) and white-box attacks on every detector, adversarial hardening |
| Explain | Three reasons and a plain-language operator message for every alarm (SHAP, reconstruction error, occlusion, filter terms), K-Means of alarms |

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

**Stealth Frontier (our own metric): how far can an attacker move the drone before
the first alarm?** Attacks are smooth, velocity-consistent drifts at 0.05–5 m/s,
from three attackers: a fixed drift, a black-box adaptive attacker that reshapes the
drift using only alarms (practises on validation flights, tested on 21 unseen test
flights), and a white-box attacker that knows the detector, its threshold and the
flight. Figures in `reports/figures/stealth/`, tables `reports/tables/D_*.csv`.

- The adaptive and white-box attackers beat every physics detector at every speed:
  the drone is moved up to several hundred metres without an alarm (the white-box
  attacker hides a ~6 m/s drift from the Kalman test, ~510 m).
- The random forest, on missions it was trained on, catches 70% of adaptive attacks
  at 0.25 m/s and 95–100% from 0.5 m/s, with the drone moved no more than about
  10 m; even the white-box attacker is caught on 81–90% of flights from 1 m/s. The
  cue is measurable: a drift shifts the GPS course away from the vehicle heading, and
  5 s window statistics pick it up where single-sample checks cannot.
- Adversarial hardening (retraining on the attacker's best flights, recalibrated at
  the same budget) leaves the random forest unchanged and halves the distance an
  adaptive attacker gets past the supervised LSTM (32–38 m to 16–22 m), at the cost of
  more false alarms on the test flights (1.1 to 2.9 per hour, on 45 minutes of flight).
- Trade-off: the learned models' advantage holds on seen missions only; on an unseen
  mission their false alarms break the budget (see above).

**Explaining alarms.** Every alarm (6,785 on the test flights, all detectors) comes
with its top three reasons and a plain-language message built from a template, for
example: *"GPS SPOOFING ALARM at 5.1 s (Kalman filter chi-square test). 1) GPS position
jumped away from where the navigation filter predicted: 2.63 m (normal up to 0.45 m,
6x normal) ... Still consistent: heading vs course, climb rate, barometer altitude."*
SHAP shows the tree models rely most on horizontal-speed consistency. Clustering the
alarms does not recover the attack families (adjusted Rand index 0.09): at the moment
of the first alarm, different attacks show the same evidence; only abrupt jumps stand
apart.

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

Python 3.11, NumPy, pandas, scikit-learn, PyTorch, SHAP, Matplotlib, PyArrow, pytest.

## Project structure

- `configs/` configuration files
- `data/` dataset and intermediate results (large files are not committed)
- `docs/` technical notes per module
- `reports/` result tables and figures
- `src/aerovigil/` source code, one package per module
- `tests/` automated tests
