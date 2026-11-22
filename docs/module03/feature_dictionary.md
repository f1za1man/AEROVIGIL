# Feature Dictionary

Generated from feature_metadata.csv. Features marked unavailable were not generated.

Parameters: lag L = 1 s, rolling window = 2 s (trailing, causal).

## gps_speed_from_position

- Formula: haversine(GPS_t, GPS_{t-L}) / dt
- Source columns: latitude_deg, longitude_deg, time_usec
- Units: m/s
- Assumptions: Time sorted within flight; L = 1 s; lag does not cross a gap
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## gps_vertical_rate_from_altitude

- Formula: (altitude_t - altitude_{t-L}) / dt
- Source columns: altitude, time_usec
- Units: m/s
- Assumptions: GPS altitude in metres
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## res_altitude

- Formula: altitude - pressure_altitude
- Source columns: altitude, pressure_altitude
- Units: m
- Assumptions: Both in metres with compatible reference (checked in summary)
- Missing-value behaviour: 
- Available for all flights: Yes

## res_horizontal_velocity

- Formula: \|\| [dN/dt, dE/dt] - mean(velocity_north, velocity_east) over L \|\|
- Source columns: latitude_deg, longitude_deg, velocity_north, velocity_east
- Units: m/s
- Assumptions: Reported velocity in m/s, north/east frame, same GPS solution
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## res_vertical_velocity

- Formula: (dAlt/dt) - mean(velocity_up) over L
- Source columns: altitude, velocity_up
- Units: m/s
- Assumptions: velocity_up in m/s, positive upward
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## res_altitude_abs

- Formula: \|res_altitude\|
- Source columns: res_altitude
- Units: same as base
- Assumptions: None beyond base
- Missing-value behaviour: Same as base
- Available for all flights: Yes

## res_altitude_rate

- Formula: (res_altitude_t - res_altitude_{t-L}) / dt
- Source columns: res_altitude
- Units: base unit / s
- Assumptions: Lag does not cross a gap
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## res_altitude_roll_mean

- Formula: trailing mean of res_altitude over 2 s
- Source columns: res_altitude
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_altitude_roll_std

- Formula: trailing std of res_altitude over 2 s
- Source columns: res_altitude
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_altitude_roll_max

- Formula: trailing max of res_altitude over 2 s
- Source columns: res_altitude
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_altitude_roll_min

- Formula: trailing min of res_altitude over 2 s
- Source columns: res_altitude
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_horizontal_velocity_rate

- Formula: (res_horizontal_velocity_t - res_horizontal_velocity_{t-L}) / dt
- Source columns: res_horizontal_velocity
- Units: base unit / s
- Assumptions: Lag does not cross a gap
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## res_horizontal_velocity_roll_mean

- Formula: trailing mean of res_horizontal_velocity over 2 s
- Source columns: res_horizontal_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_horizontal_velocity_roll_std

- Formula: trailing std of res_horizontal_velocity over 2 s
- Source columns: res_horizontal_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_horizontal_velocity_roll_max

- Formula: trailing max of res_horizontal_velocity over 2 s
- Source columns: res_horizontal_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_horizontal_velocity_roll_min

- Formula: trailing min of res_horizontal_velocity over 2 s
- Source columns: res_horizontal_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_vertical_velocity_abs

- Formula: \|res_vertical_velocity\|
- Source columns: res_vertical_velocity
- Units: same as base
- Assumptions: None beyond base
- Missing-value behaviour: Same as base
- Available for all flights: Yes

## res_vertical_velocity_rate

- Formula: (res_vertical_velocity_t - res_vertical_velocity_{t-L}) / dt
- Source columns: res_vertical_velocity
- Units: base unit / s
- Assumptions: Lag does not cross a gap
- Missing-value behaviour: NaN in the first 1 s of a flight and for 1 s after each gap
- Available for all flights: Yes

## res_vertical_velocity_roll_mean

- Formula: trailing mean of res_vertical_velocity over 2 s
- Source columns: res_vertical_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_vertical_velocity_roll_std

- Formula: trailing std of res_vertical_velocity over 2 s
- Source columns: res_vertical_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_vertical_velocity_roll_max

- Formula: trailing max of res_vertical_velocity over 2 s
- Source columns: res_vertical_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## res_vertical_velocity_roll_min

- Formula: trailing min of res_vertical_velocity over 2 s
- Source columns: res_vertical_velocity
- Units: same as base
- Assumptions: Trailing (causal) window; no future samples used
- Missing-value behaviour: Ignores NaN; std needs >= 2 valid samples
- Available for all flights: Yes

## GPS/IMU integrated velocity residual

- Formula: not generated
- Source columns: linear_acceleration_*, orientation_*
- Units: m/s
- Assumptions: Needs documented attitude/gravity conventions and drift handling
- Missing-value behaviour: Not generated
- Available for all flights: No - unavailable

## GPS course vs magnetometer heading residual

- Formula: not generated
- Source columns: velocity_north/east, magnetic_field_*
- Units: deg
- Assumptions: Needs documented magnetometer axes, units and tilt compensation
- Missing-value behaviour: Not generated
- Available for all flights: No - unavailable
