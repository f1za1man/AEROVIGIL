"""Module 07 v2 tests: the CUSUM that carries evidence across windows."""

import numpy as np
import pandas as pd
import pytest

import aerovigil.lstm.sequential_v2 as sq


def table(n_flights=3, n_windows=8):
    """A tiny window table: one normal flight, then attacked ones."""
    rows = []
    for f in range(n_flights):
        for w in range(n_windows):
            rows.append({
                "flight_id": f"flight_{f}", "attack_set": "sweep" if f else "orig",
                "mission_type": "curved", "role": "test",
                "attack_type": "none" if f == 0 else "linear_drift",
                "velocity_mode": "consistent", "drift_rate_mps": 0.1 * f,
                "window_start_s": 2.5 * w, "window_end_s": 2.5 * w + 5.0,
                "seconds_since_onset_end": np.nan if f == 0 else 2.5 * w - 5.0,
                "offset_end_m": 0.0 if f == 0 else 0.1 * w,
            })  # fmt: skip
    return pd.DataFrame(rows)


def test_cusum_accumulates_small_excesses_and_cannot_go_negative():
    z = np.full(20, 0.3)
    s = sq.cusum(z, 0.1)
    assert s[0] == pytest.approx(0.2)
    assert s[-1] == pytest.approx(0.2 * 20)  # 0.2 per step, accumulated
    assert (np.diff(s) > 0).all()
    quiet = sq.cusum(np.full(20, -5.0), 0.1)
    assert (quiet == 0).all()  # a one-sided CUSUM never goes below zero
    assert sq.cusum(np.r_[np.full(5, 2.0), np.full(5, -9.0)], 0.5)[-1] == 0.0


def test_a_decay_bounds_the_sum_so_it_stops_growing_with_flight_length():
    z = np.full(400, 0.3)
    plain = sq.cusum(z, 0.1, 1.0)
    decayed = sq.cusum(z, 0.1, 0.9)
    assert plain[-1] == pytest.approx(0.2 * 400)  # grows without limit
    assert decayed[-1] == pytest.approx(0.2 / (1 - 0.9), rel=1e-3)  # settles
    assert decayed[-1] == pytest.approx(decayed[200])  # same at any flight length
    assert decayed.max() < plain.max() / 10


def test_standardise_uses_the_calibration_pool_only():
    score = np.r_[np.zeros(50), np.full(10, 100.0)]
    pool = np.r_[np.ones(50, bool), np.zeros(10, bool)]
    z = sq.standardise(score, pool)
    assert abs(np.median(z[pool])) < 1e-9
    assert z[-1] > 1e3  # the attack scores sit far outside the clean spread


def test_cusum_restarts_at_every_flight_boundary():
    meta = table()
    z = np.full(len(meta), 1.0)
    s = sq.flight_cusum(z, meta, 0.0)
    first = meta.groupby("flight_id")["window_start_s"].idxmin()
    assert np.allclose(s[first.to_numpy()], 1.0)  # no carry-over between flights
    assert s.max() == pytest.approx(8.0)


def test_alarms_are_counted_once_per_refractory_period():
    t = np.arange(10) * 2.5
    series = np.full(10, 5.0)
    n, first = sq.count_alarms(series, t, 1.0)
    assert first == 0
    assert n == 3  # 10 windows over 22.5 s, one alarm every 10 s
    assert sq.count_alarms(series, t, 9.0) == (0, None)


def fake_project(root, n_calib=6, n_test=4, n_windows=30, rates=(0.05, 1.0)):
    """A scores folder and a window table, without the whole Module 03/04 chain."""
    rng = np.random.default_rng(0)
    rows = []

    def flight(fid, role, attack_type, aset, rate):
        for w in range(n_windows):
            attacked = attack_type != "none" and w >= n_windows // 3
            rows.append({
                "flight_id": fid, "source_flight_id": fid,
                "origin": "orig" if aset == "orig" else "syn", "attack_set": aset,
                "split": role if role != "calib" else "train", "role": role,
                "mission_type": "curved", "attack_type": attack_type,
                "velocity_mode": "consistent" if attacked else "na",
                "drift_rate_mps": rate, "window_start_s": 2.5 * w,
                "window_end_s": 2.5 * w + 5.0, "n_rows": 625, "missing_fraction": 0.0,
                "attack_fraction": 1.0 if attacked else 0.0,
                "spoof_fraction": 1.0 if attacked else 0.0,
                "window_class": "ramp" if attacked else "normal",
                "y_event": 1 if attacked else 0, "y_window": 1 if attacked else 0,
                "max_offset_m": 1.0 if attacked else 0.0,
                "offset_end_m": 1.0 if attacked else 0.0,
                "seconds_since_onset_end": (
                    2.5 * (w - n_windows // 3) if attack_type != "none" else np.nan
                ),
            })  # fmt: skip

    for i in range(n_calib):
        flight(f"calib_{i}", "calib", "none", "orig", np.nan)
    for i in range(n_test):
        flight(f"test_normal_{i}", "test", "none", "orig", np.nan)
        for r in rates:
            flight(f"test_sweep_{i}_{r:g}", "test", "linear_drift", "sweep", r)
    meta = pd.DataFrame(rows)
    (root / "scores").mkdir(parents=True, exist_ok=True)
    meta.to_parquet(root / "meta_orig.parquet", index=False)
    np.save(
        root / "X_orig.npy", rng.normal(0, 1, (len(meta), 20, 36)).astype("float32")
    )
    score = rng.normal(0, 1, len(meta))
    score[(meta["y_event"] == 1).to_numpy()] += (
        3.0 * meta.loc[meta["y_event"] == 1, "drift_rate_mps"].to_numpy()
    )
    np.save(root / "scores" / "clf_seed42.npy", score.astype("float32"))
    pd.DataFrame([{
        "name": "clf_seed42", "kind": "clf", "seed": 42, "protocol": "seen",
        "held_out": None, "n_train": 1, "n_val": 1, "best_epoch": 1, "best_val": 0.1,
    }]).to_csv(root / "scores" / "index.csv", index=False)  # fmt: skip
    return meta


def test_run_covers_every_k_and_decay_and_prices_each_one(tmp_path):
    fake_project(tmp_path)
    flights, frontier, cost = sq.run(output_dir=tmp_path)
    assert set(frontier["k"]) == set(sq.K_GRID)
    assert set(frontier["decay"]) == set(sq.DECAY_GRID)
    assert set(frontier["drift_rate_mps"]) == {0.05, 1.0}
    assert frontier["detection_rate"].between(0, 1).all()
    thr = pd.read_csv(tmp_path / "sequential_thresholds.csv")
    assert (thr["alarms_used"] <= thr["allowed_alarms"]).all()
    assert len(thr) == len(sq.K_GRID) * len(sq.DECAY_GRID)
    fast = frontier[frontier["drift_rate_mps"] == 1.0]["detection_rate"].mean()
    slow = frontier[frontier["drift_rate_mps"] == 0.05]["detection_rate"].mean()
    assert fast >= slow  # a stronger drift is never harder to catch
    assert len(cost) == len(sq.K_GRID) * len(sq.DECAY_GRID)
    assert {"alarms_per_hour", "within_budget", "det_0.05"} <= set(cost.columns)
    assert cost["within_budget"].eq(cost["alarms_per_hour"] <= 1.0 + 1e-9).all()
    for name in ("sequential_flights.csv", "sequential_frontier.csv",
                 "sequential_cost.csv"):  # fmt: skip
        assert (tmp_path / name).is_file()
    assert len(flights) > 0


def test_a_plain_sum_does_not_transfer_from_short_flights_to_long_ones():
    """Why the first run over-alarmed: these flights run from 46 s to 265 s."""
    rng = np.random.default_rng(1)

    def peak(n_windows, decay, reps=40):
        runs = [
            sq.cusum(rng.normal(0.7, 1.0, n_windows), 0.5, decay).max()
            for _ in range(reps)
        ]
        return float(np.median(runs))

    plain = peak(200, 1.0) / peak(20, 1.0)
    decayed = peak(200, 0.9) / peak(20, 0.9)
    assert plain > 4.0  # a threshold set on short flights is far too low for long ones
    assert decayed < 2.5
    assert decayed < plain / 2


def test_calibration_and_evaluation_respect_the_budget(monkeypatch):
    meta = table().reset_index(drop=True)
    pool = ((meta["role"] == "test") & (meta["attack_type"] == "none")).to_numpy()
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, len(meta))
    z[meta["attack_type"].to_numpy() != "none"] += 4.0
    series = sq.flight_cusum(z, meta, 0.5)
    h, allowed, used = sq.calibrate(series, meta, pool, 1.0)
    assert used <= allowed and allowed >= 1
    flights, rate = sq.evaluate(series, meta, h)
    assert len(flights) == meta["flight_id"].nunique()
    att = flights[flights["attacked"]]
    assert att["detected"].all()
    assert (att["delay_s"] >= 0).all()
    assert np.isfinite(rate)
