\# Module 03 v2 addendum: rebuild from the Raw data



Module 03 (v1) built its features and windows from the Merged CSV. A review found

that Merged is a thinned copy of the Raw files and that the v1 labels could not tell

the parts of an attack apart. Module 03 v2 rebuilds the base table from Raw and adds

measured phase labels. v1 outputs are kept unchanged; v2 supersedes them for Modules

05-09. The flight split (seed 42, 252/54/54) is unchanged.



\## 1. Findings that required the rebuild



| # | Finding | Evidence |

|---|---|---|

| 1 | Merged is the exact-timestamp join of the sensor files (README script). It keeps 86.3 % of the unique Raw gps timestamps (7,893,049 of 9,149,362 rows) and cuts the end of 355 of 360 flights by 3-28 s. | check\_raw\_coverage.py; exact\_time\_match = 1.0 |

| 2 | gps.csv and raw\_gps.csv log each timestamp twice (4,705,878 repeated rows removed). raw\_gps copies are identical. gps copies differ in lat/lon in 5.3 % of pairs, only in attacked flights, spread over the whole attack (median 91 s after onset), with neither copy systematically more spoofed. This is a logging artefact. The base keeps the first copy and no feature uses the duplicates. | check\_base\_quality.py, check\_gps\_duplicates.py |

| 3 | imu, barometer, magnetometer and groundtruth drop out for 2 gaps per flight (715 runs, longest 32.4 s) while GPS continues: 695,997 / 812,243 / 700,203 / 709,434 rows (7.6-8.9 %). raw\_gps has 182 short gaps (15,303 rows, longest 2.2 s). | base\_gaps.csv |

| 4 | The original label ends about 0.7 s after the measured offset drops below 1 m (12,957 rows over 68 flights). No spoofed row is labelled normal. | build\_raw\_base.py report |

| 5 | The attack is a ramp then a constant plateau, and in the random mission it ends at a median of 67 s of a 126 s flight (snap-back). 51 % of windows are plateau; 3.8 % are ramp. | Section 4 |



\## 2. Base table (build\_raw\_base.py)



All streams are aligned to the gps timeline by nearest time (tolerance 10 ms).

Result: 360 flights, 9,149,362 rows, 53 columns. Added from Raw: eph, epv,

raw\_gps position/velocity, groundtruth position/velocity/attitude, temperature,

absolute pressure. Measured labels: offset\_m (gps vs raw\_gps), spoof\_truth,

phase (0 normal flight, 1 pre, 2 ramp, 3 plateau, 4 post), seconds\_since\_onset.

The original label is kept as attack\_label.



| Mission | Onset (s) | Ramp end (s) | Spoof end (s) | Plateau (m) | Duration (s) |

|---|---|---|---|---|---|

| curved | 5.54 | 9.41 | 265.02 | 9.78 | 265.30 |

| random | 9.73 | 13.46 | 67.17 | 9.66 | 125.94 |

| straight | 4.05 | 7.80 | 46.03 | 9.78 | 46.03 |



Label rows 4,894,379; spoofed rows 4,881,422; spoofed but not labelled 0.



\## 3. Features v2 (36 = 22 v1 + 14 IMU)



IMU features use the Module 06 convention (body\_to\_world / specific\_force / enu):

res\_accel\_horizontal, res\_accel\_vertical, res\_course\_heading (value, |value| for

signed ones, 2 s trailing mean/std/max). Median by phase:



| Feature | Normal flight | Ramp | Plateau | Post |

|---|---|---|---|---|

| res\_horizontal\_velocity | 0.050 | 2.05 | 0.050 | 0.051 |

| res\_accel\_horizontal | 0.150 | 0.152 | 0.172 | 0.189 |

| res\_course\_heading | -0.002 | 0.028 | 0.001 | -0.009 |

| res\_altitude\_abs | 0.819 | 0.818 | 0.818 | 0.819 |



The velocity residual separates the ramp. The IMU features do not see the original

attack, because the spoofer moves the position and leaves the reported velocity

unchanged. They are kept for the velocity-consistent attacks of Module 04.



\## 4. Normalization v2 and windows v2



Robust scaling (median, IQR/1.349), fitted on clean rows of the 252 training flights

only (normal flights and pre-attack rows; 515,223 rows), clipped at +/-20. No feature

has a floored scale. The clip touches 1.5-4 % of rows in the velocity-residual group

(the ramp) and at most 0.14 % elsewhere.



Windows: 5 s, hop 2.5 s, 625 points at 125 Hz, 36 features. 19,567 candidates, 2

dropped for gaps, 2,859 for NaN, 16,706 kept (11,733 / 2,488 / 2,485).



| Split | normal | pre | ramp | plateau | snapback | post |

|---|---|---|---|---|---|---|

| train | 4,487 | 115 | 440 | 5,946 | 85 | 660 |

| validation | 912 | 26 | 100 | 1,240 | 16 | 194 |

| test | 905 | 11 | 95 | 1,274 | 22 | 178 |



\## 5. Separability (best single feature, ROC AUC vs clean windows)



| Class | Windows | Best feature | AUC |

|---|---|---|---|

| ramp | 635 | res\_horizontal\_velocity | 1.000 (all splits) |

| snapback | 123 | res\_horizontal\_velocity\_roll\_std | 1.000 (all splits) |

| plateau | 8,460 | res\_accel\_vertical\_abs | 0.55 (0.62 test) |

| post | 1,032 | res\_course\_heading\_roll\_std | 0.79 |



\## 6. Consequences



\- The start and end of the original attack are detectable; the constant offset

&#x20; between them is not visible to any consistency feature. The original attack must be

&#x20; scored as event detection (ramp, snap-back) with a latch, and the plateau reported

&#x20; separately. Window-level "attack vs normal" with the plateau as positive is not a

&#x20; meaningful target for this attack.

\- Modules 05 and 07 must be re-evaluated on the v2 windows and labels.

\- AUC measures ordering, not false alarms; the false-alarm rate on normal windows is

&#x20; re-measured in the re-evaluation.



\## 7. Limitations



\- The split is by flight, but only three flight plans exist and flights are repeated

&#x20; runs; leave-one-mission-out is the unseen-trajectory test (Module 09).

\- The post-phase heading signal (AUC 0.79) is probably a flight-phase confound (end

&#x20; of long random flights) and is not used as an attack-ended detector.

\- The plateau score of 0.55-0.62 may reflect differences between flights, not the

&#x20; attack.

\- Only 11 pre-attack windows in the test split (the first second of each flight is

&#x20; NaN by design).

\- Windows touching a sensor dropout are dropped (2,859 windows).



\## 8. Tests



21 new tests (tests/module03\_v2): 8 unit tests without data (phase labelling, window

classes, IMU physics) and 13 data tests. Missing outputs fail instead of skipping

(AEROVIGIL\_ALLOW\_MISSING\_DATA=1 skips deliberately). Full suite: 140 passed.

