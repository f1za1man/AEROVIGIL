# Data Quality Report

| Check | Result |
|---|---|
| Flights | 360 |
| Rows | 7,893,049 |
| Columns | 27 |
| Missing values (raw columns) | 0 |
| Non-numeric values | 0 |
| Duplicate timestamps | 0 |
| Out-of-order rows | 140,137 (all 360 flights), corrected by sorting |
| Large gaps | 92 in 83 flights, longest 22.8 s, recorded and not repaired |
| time_usec vs timestamp | identical for all rows |
| Attack labels | 4,302,145 attack / 3,590,904 normal rows, preserved |
| Flights with no attack | 144 |
| Flights with attack | 216 (one transition: 153, two transitions: 63) |
| Sampling rate | 298 flights at 250 Hz, 62 flights at 125 Hz |
| attack_enabled vs attack_label | same distribution; attack_enabled is excluded from features |

## Engineered feature missing values

About 1% of rows have NaN in the lag-based features (the first second of each flight and the second after each gap) and about 2% in the rate features. Windows containing NaN are excluded.

## Physical validity of the substitute velocity features

On normal rows the median horizontal-velocity residual is 0.05 m/s, the median vertical-velocity residual is 0.005 m/s, and the median GPS/barometer altitude difference is 0.82 m (95th percentile 0.97 m). These are small and stable, so the units and reference frames are compatible.

## Observations

- Altitude and vertical-velocity residuals are the same for attack and normal rows to four decimals.
- The horizontal-velocity residual differs (95th percentile 0.10 for normal rows, 1.04 for attack rows).
- res_altitude and res_altitude_abs are identical because the offset is always positive.
- Validation and test normalized means stay near 0 and standard deviations near 1; the horizontal-velocity features have standard deviation about 1.06 there.
