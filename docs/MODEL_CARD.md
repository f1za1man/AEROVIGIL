# AEROVIGIL - Model Card

## 1. What the models are

Eight GPS-spoofing detectors for small multirotor drones, all reading the same logged
sensors (GPS position and velocity, IMU, barometer, magnetometer) and all calibrated to
the same budget of **1 false alarm per flight-hour**.

| Detector | Family | How it decides | Code |
|---|---|---|---|
| Kalman chi-square | physics | an IMU-driven, barometer-aided Kalman filter predicts the next GPS position and velocity; the alarm fires when the normalised innovation squared (NIS) passes a chi-square threshold | `src/aerovigil/kalman/` |
| CUSUM | physics | cumulative sum of the same filter's whitened innovations | `src/aerovigil/kalman/` |
| Velocity check | physics | GPS position change over 1 s against the GPS-reported velocity (2 s mean, z-score) | `src/aerovigil/fusion/` |
| Fused | physics | alarm when the Kalman or the velocity channel fires (budget split between them) | `src/aerovigil/fusion/` |
| Consistency | physics | velocity channel plus GPS course against heading | `src/aerovigil/consistency/` |
| Random forest (and 5 other classical models) | learned | statistics of 36 residual features over 5 s windows, every 2.5 s | `src/aerovigil/baselines/` |
| LSTM autoencoder | learned | reconstruction error of a 5 s window (20 steps x 36 features), trained on normal flight only | `src/aerovigil/lstm/` |
| Supervised LSTM | learned | attack logit of the same window, trained on real and synthetic attacks | `src/aerovigil/lstm/` |

Every alarm comes with three reasons and a plain-language operator message
(`src/aerovigil/explain/`).

## 2. Intended use

- **Research and teaching:** comparing physics-based and learned spoofing detectors
  under one evaluation protocol, and measuring how far an adaptive attacker can go.
- **Offline analysis of logged flights,** and as a starting point for an onboard
  monitor that a human operator would supervise.

**Not intended for:** flight-critical use, certification, or any real aircraft without
real-flight validation. No model here has seen a real drone or a real GNSS signal.

## 3. Training and calibration data

- PX4 SITL + Gazebo flights (Anthony Finn's dataset, see the
  [data card](datasets/data_card.md)): 360 flights, split by flight with seed 42 into
  252 train, 54 validation and 54 test.
- Learned models train on the training flights' real attacks, synthetic attacks made
  from training flights only (Module 04), and clean windows. The LSTM autoencoder
  trains on clean windows only.
- Thresholds are set on clean training/validation (or held-back calibration) flights,
  never on a tested flight. Learned models: 5 seeds (42-46), results averaged.
- A test checks that no window, sequence or calibration set crosses the flight split
  (`tests/evaluation/test_no_leakage.py`).

## 4. Performance (held-out test flights)

| Detector | Real attacks caught | Time to alarm | Velocity-consistent synthetic caught | False alarms / h, seen (0.75 h) | False alarms / h, unseen mission (5.25 h) | Cross-validated false alarms / h (5.25 h) |
|---|---|---|---|---|---|---|
| Kalman chi-square | 100% | 0.4 s | 60% | 3.2 | 1.8 | 1.13 |
| Velocity check | 97% | 0.2 s | 24% | 1.3 | 1.7 | 0.95 |
| Fused | 100% | 0.2 s | 58% | 1.3 | 1.7 | 1.33 |
| CUSUM | 100% | 4.1 s | 32% | 4.8 | 3.6 | 1.13 |
| Consistency | 100% | 0.3 s | 23% | 0.0 | 0.2 | 0.19 |
| Random forest | 100% | 1.6 s | 91% | 0.8 | 22.0 | 0.19 |
| LSTM autoencoder | 100% | 2.7 s | 72% | 1.3 | 3.5 | 0.38 |
| Supervised LSTM | 100% | 2.6 s | 79% | 1.1 | 8.8 | 0.19 |

Source: `reports/tables/C_all_detectors.csv`. "Seen" rates rest on 45 minutes of normal
flight and have wide 95% bounds; the cross-validated column (all 144 normal flights) is
the one to quote.

## 5. The attacker it cannot detect

Measured with the Stealth Frontier (`reports/tables/D_*.csv`, figures in
`reports/figures/stealth/`): the median distance the drone is moved before the first
alarm.

- **Physics detectors** do not stop a smooth drift that also fakes the GPS velocity
  below about 1 m/s, and they do not stop the adaptive (black-box) or white-box
  attacker at any speed tested (0.05-5 m/s): the drone is moved up to several hundred
  metres. At 0.5 m/s the adaptive attacker gets past the Kalman test on 90% of flights
  (42 m moved).
- **Random forest** catches 95-100% of adaptive attacks from 0.5 m/s with the drone
  moved about 5-10 m (70% at 0.25 m/s), and 86-95% of white-box attacks from 1 m/s,
  **on missions it was trained on only**. On an unseen mission its false alarms rise
  to 22 per hour.
- **LSTMs** sit in between (supervised LSTM: 34% at 0.5 m/s, 37 m). Adversarial
  hardening (retraining on the attacks that got through, then letting the attacker
  search again) helps only at slow speeds: at 0.5 m/s the supervised LSTM catches 90%
  instead of 34% (11 m instead of 37 m moved), but at 1-2 m/s the new attacker gets
  as far as before or further (24-63 m), and its test false alarms rise from 1.1 to
  1.6 per hour. The random forest gains a little at 0.1-0.25 m/s (47% and 90% instead
  of 21% and 70%) for false alarms 0.8 to 1.3 per hour.

Stopping a careful spoofer reliably needs a measurement the GPS cannot fake together
with the IMU: a second GNSS antenna, an airspeed sensor or a downward camera.

## 6. Deployment figures (one CPU thread, this project's laptop)

| Model | Size | Time per window | Notes |
|---|---|---|---|
| LSTM autoencoder, ONNX int8 | 33 KB | 0.11 ms | same detections as float32 |
| Supervised LSTM, ONNX int8 | 13 KB | 0.06 ms | one extra false alarm on one seed |
| Random forest, ONNX | 16 MB | 0.11 ms | int8 does not apply to trees |
| Random forest, 20 trees depth 8 | 100 KB | 0.03 ms | real attacks 100%, consistent synthetic 82% |
| Velocity check, streaming | - | 3.4 us per log row | identical alarms to the batch code |
| Kalman filter, streaming | - | 164 us per 0.1 s tick | identical alarms to the batch code |

Source: `reports/tables/F_*.csv`, CPU and versions in `data/processed/edge/run_info.json`.
Times are medians on an otherwise idle laptop; repeat runs agree within about 10% for
the models, and anything else running (a virus scan, a browser) slows them down.

## 7. After an alarm

Dead reckoning (the Kalman filter without GPS: IMU prediction, barometer height) stays
within 10 m of the truth for a median 20 s (at least 14 s in 9 of 10 runs) and within
25 m for 30 s. It buys time to start a return or landing; it cannot replace GPS for
the rest of a mission (`reports/tables/G_*.csv`).

## 8. Limitations and risks

- Simulation only; open-loop synthetic attacks (a real spoofed drone would react).
- Learned models do not transfer to a mission they were not trained on.
- A missed alarm lets the drone be steered away; a false alarm triggers an unneeded
  fallback, which itself drifts. The budget (1 false alarm per flight-hour) is a design
  choice, not a safety case.
- The adaptive and white-box attacker code measures detectors on log files only; it
  generates no radio signal (see the Ethics section of the README).
