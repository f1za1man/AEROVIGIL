# Final figures

Every figure of the project, rebuilt from the final results by one command:

    python src\aerovigil\evaluation\make_figures.py

PNG at 200 dpi with an SVG copy of each. Names start with the plan module (M2-M13;
R09 / R10 = repo Modules 09 and 10, the consistency and fused detectors). Each
detector keeps one colour in every figure.

| Figure | What it shows | Data | Script |
|---|---|---|---|
| [M2_flight_durations.png](M2_flight_durations.png) | Length of every flight, per mission: straight flights last about 40 s, random about 110 s and curved about 230 s. | `data\interim\module03\raw_coverage.csv` | `src/aerovigil/figures/data_figs.py` |
| [M2_flights_per_mission.png](M2_flights_per_mission.png) | How many normal and attacked flights each of the three missions has. | `data\interim\module03\raw_coverage.csv` | `src/aerovigil/figures/data_figs.py` |
| [M2_logging_rate.png](M2_logging_rate.png) | Flights log at about 125 or 250 Hz; the rate differs between flights of the same mission. | `data\interim\module03\raw_coverage.csv` | `src/aerovigil/figures/data_figs.py` |
| [M3_example_flight_phases.png](M3_example_flight_phases.png) | One real attacked flight: the GPS offset with its measured phases (pre, ramp, plateau, post) and the true vs reported path. | `data\processed\module03_v2\base\Curved__Attacked__log_21_04_2023_09_58_46.parquet` | `src/aerovigil/figures/data_figs.py` |
| [M3_feature_correlation.png](M3_feature_correlation.png) | How the per-row features correlate: they group by the residual they are computed from. | `data/processed/module03_v2/features/*.parquet` | `src/aerovigil/figures/data_figs.py` |
| [M3_feature_distributions.png](M3_feature_distributions.png) | How six key consistency features differ between normal flight and the dataset's own attacks. | `data/processed/module03_v2/features/*.parquet` | `src/aerovigil/figures/data_figs.py` |
| [M3_missing_values.png](M3_missing_values.png) | Share of rows with a missing value per sensor and mission. | `data/processed/module03_v2/base/*.parquet` | `src/aerovigil/figures/data_figs.py` |
| [M3_pca_windows.png](M3_pca_windows.png) | Normal and attacked 5 s windows on the first two principal components: ramp windows stand apart, plateau windows overlap normal flight. | `data/processed/module03_v2/windows/windows_{train,test}.npz` | `src/aerovigil/figures/data_figs.py` |
| [M3_split_sizes.png](M3_split_sizes.png) | Train, validation and test sizes in flights and in 5 s windows. | `data/interim/module03/flight_splits.csv`<br>`data/processed/module03_v2/windows/window_manifest_*.csv` | `src/aerovigil/figures/data_figs.py` |
| [M4_attack_families.png](M4_attack_families.png) | Offset over time for each attack family, and how much each velocity mode changes the reported GPS velocity. | `data/interim/module04_v2/attack_manifest_v2.csv + data/processed/module04_v2/base/*.parquet` | `src/aerovigil/figures/attack_figs.py` |
| [M4_drift_sweep.png](M4_drift_sweep.png) | The drift-sweep attacks: the same flight drifted at 0.05 to 5 m/s. | `data/interim/module04_v2/attack_manifest_v2.csv + data/processed/module04_v2/base/*sweep*.parquet` | `src/aerovigil/figures/attack_figs.py` |
| [M4_onset_old_vs_new.png](M4_onset_old_vs_new.png) | The old velocity-consistent onset overshot the drift speed by 69%; the new one eases in without overshoot. | `data/interim/module04_v2/attack_manifest_v2.csv + src/aerovigil/attacks/inject_v2.py` | `src/aerovigil/figures/attack_figs.py` |
| [M4_parameter_coverage.png](M4_parameter_coverage.png) | How large, long and where in the flight the synthetic attacks are. | `data\interim\module04_v2\attack_manifest_v2.csv` | `src/aerovigil/figures/attack_figs.py` |
| [M5_confusion_matrices.png](M5_confusion_matrices.png) | Confusion matrices of the six classical models at the shared alarm budget. | `data/processed/module05_v2/all_models_window_full.csv` | `src/aerovigil/figures/classical_figs.py` |
| [M5_detection_by_attack.png](M5_detection_by_attack.png) | Share of attacked test flights each classical model catches, per attack family and velocity mode. | `data\processed\module05_v2\all_models_flight_full.csv` | `src/aerovigil/figures/classical_figs.py` |
| [M5_metrics_seen_unseen.png](M5_metrics_seen_unseen.png) | Accuracy, precision, recall and F1 of each classical model on seen and on unseen missions. | `reports\tables\C_classical_models.csv` | `src/aerovigil/figures/classical_figs.py` |
| [M5_rf_importance.png](M5_rf_importance.png) | The window features the random forest relies on most. | `data/processed/stealth/models/rf_seen_s42.joblib` | `src/aerovigil/figures/classical_figs.py` |
| [M5_roc_curves.png](M5_roc_curves.png) | ROC curves of the six classical models on real and on velocity-consistent synthetic attacks. | `data/processed/module05_v2/windows_{orig,syn}.parquet`<br>`data/processed/module05_v2/all_models_window_full.csv` | `src/aerovigil/figures/classical_figs.py` |
| [M6_cusum_trace.png](M6_cusum_trace.png) | The CUSUM statistic over time for the same three flights, with its threshold and alarms. | `data\processed\module06\innovations\orig\Curved__Normal__log_20_04_2023_18_06_31.parquet`<br>`data\processed\module06\innovations\orig\Curved__Attacked__log_21_04_2023_09_58_46.parquet`<br>`data\processed\module06\innovations\syn\Curved__Normal__log_20_04_2023_18_06_31__syn_linear_drift_00_po.parquet`<br>`data\processed\module06\detector_calibration.json` | `src/aerovigil/figures/kalman_figs.py` |
| [M6_drift_absorption.png](M6_drift_absorption.png) | The Kalman filter absorbs a smooth velocity-consistent drift into its own position estimate. | `data\processed\module06\innovations\sweep\Curved__Normal__log_20_04_2023_18_06_31__sweep_linear_drift_r0p25.parquet`<br>`data\processed\module06\innovations\sweep\Curved__Normal__log_20_04_2023_18_06_31__sweep_linear_drift_r2.parquet` | `src/aerovigil/figures/kalman_figs.py` |
| [M6_innovation_nis.png](M6_innovation_nis.png) | Filter innovations and the NIS test over time for a normal, a real attacked and a synthetic drift flight, with alarms marked. | `data\processed\module06\innovations\orig\Curved__Normal__log_20_04_2023_18_06_31.parquet`<br>`data\processed\module06\innovations\orig\Curved__Attacked__log_21_04_2023_09_58_46.parquet`<br>`data\processed\module06\innovations\syn\Curved__Normal__log_20_04_2023_18_06_31__syn_linear_drift_00_po.parquet`<br>`data\processed\module06\detector_calibration.json` | `src/aerovigil/figures/kalman_figs.py` |
| [M6_whitened_innovations.png](M6_whitened_innovations.png) | The filter's whitened innovations on normal test flights against a standard normal: velocity fits, east position is wider. | `data/processed/module06/innovations/orig/*Normal*.parquet`<br>`data\processed\module06\detector_calibration.json` | `src/aerovigil/figures/kalman_figs.py` |
| [M7_cusum_lstm.png](M7_cusum_lstm.png) | CUSUM over the supervised LSTM's score on a normal flight and the same flight with a slow drift, without and with decay. | `data/processed/module07_v2/scores/clf_seed42.npy`<br>`data\processed\module07_v2\sequential_thresholds.csv` | `src/aerovigil/figures/lstm_figs.py` |
| [M7_reconstruction_error.png](M7_reconstruction_error.png) | The autoencoder's reconstruction error on normal and attacked test windows: real attacks stand out, velocity-consistent ones mostly do not. | `data/processed/module07_v2/protocols/scores/ae_seen_s42.npy`<br>`data\processed\module07_v2\protocols\windows.csv` | `src/aerovigil/figures/lstm_figs.py` |
| [M7_roc_curves.png](M7_roc_curves.png) | ROC curves of the two LSTMs on real and on velocity-consistent synthetic attacks. | `data/processed/module07_v2/protocols/scores/*_seen_s42.npy`<br>`data\processed\module07_v2\protocols\windows.csv` | `src/aerovigil/figures/lstm_figs.py` |
| [M7_training_loss.png](M7_training_loss.png) | Training and validation loss of both LSTMs per epoch, with the epoch that was kept. | `data\processed\module07_v2\training_history.json` | `src/aerovigil/figures/lstm_figs.py` |
| [M7_window_auc.png](M7_window_auc.png) | Window-level AUC of both LSTMs for every attack family and velocity mode, including families the model never saw. | `data/processed/module07_v2/results_window.csv` | `src/aerovigil/figures/lstm_figs.py` |
| [M8_detection_by_attacker.png](M8_detection_by_attacker.png) | Share of flights each detector catches against the fixed, adaptive and white-box attackers, per speed. | `reports/tables/D_frontier_fixed.csv`<br>`reports/tables/D_frontier_adaptive.csv`<br>`reports/tables/D_whitebox.csv` | `src/aerovigil/figures/stealth_figs.py` |
| [M8_hardening.png](M8_hardening.png) | Headline: the Stealth Frontier of the random forest and the supervised LSTM before and after hardening, with 95% bands and the false-alarm cost. | `reports/tables/D_hardening.csv` | `src/aerovigil/figures/stealth_figs.py` |
| [M8_hill_climber.png](M8_hill_climber.png) | How the black-box attacker's best attack improves with each try, one curve per mean drift speed. | `data\processed\module08\adaptive_search.csv` | `src/aerovigil/figures/stealth_figs.py` |
| [M8_oracle_selftest.png](M8_oracle_selftest.png) | The attacker's oracle reproduces the old frontier, and an eased onset removes its 0.25-0.5 m/s detections: those came from the switch-on, not the drift. | `data\processed\module08\adaptive_selftest.csv` | `src/aerovigil/figures/stealth_figs.py` |
| [M8_stealth_frontier.png](M8_stealth_frontier.png) | Headline: distance the drone is moved before the first alarm vs the attacker's speed, for the fixed, adaptive and white-box attackers, per detector. | `reports/tables/D_frontier_fixed.csv`<br>`reports/tables/D_frontier_adaptive.csv`<br>`reports/tables/D_whitebox.csv` | `src/aerovigil/figures/stealth_figs.py` |
| [R09_consistency_channels.png](R09_consistency_channels.png) | The velocity channel separates attacks from clean flight; the course channel never reaches the threshold. | `data\processed\module09\alarms.csv`<br>`data\processed\module09\consistency_calibration.json`<br>`data/processed/module03_v2/base/*Normal*.parquet` | `src/aerovigil/figures/consistency_figs.py` |
| [R09_course_vs_heading.png](R09_course_vs_heading.png) | Why the course check fails: with honest GPS the heading already differs from the GPS course by 20-100 deg at low speed and in turns. | `data/processed/module03_v2/base/*Normal*.parquet (test split)` | `src/aerovigil/figures/consistency_figs.py` |
| [R09_tilt_vs_speed.png](R09_tilt_vs_speed.png) | Tilt follows speed only loosely (it mostly follows acceleration), so it cannot expose a smooth velocity-consistent drift. | `data/processed/module03_v2/base/*Normal*.parquet (test split)` | `src/aerovigil/figures/consistency_figs.py` |
| [R10_detection_delay.png](R10_detection_delay.png) | How long each detector takes to alarm after the attack starts, for real, position-only and velocity-consistent attacks. | `data\processed\evaluation\physics_flights.csv`<br>`data\processed\module05_v2\all_models_flight_full.csv.gz`<br>`data\processed\module07_v2\protocols\flights.csv.gz` | `src/aerovigil/figures/fusion_figs.py` |
| [R10_false_alarms.png](R10_false_alarms.png) | False alarms per flight-hour of every detector, seen vs unseen missions vs cross-validated, with 95% bounds. | `reports\tables\C_all_detectors.csv` | `src/aerovigil/figures/fusion_figs.py` |
| [R10_threat_model_heatmap.png](R10_threat_model_heatmap.png) | The threat-model table as a heatmap: every detector catches crude attacks; little catches smooth or adaptive drifts. | `data\processed\module10\threat_model.csv` | `src/aerovigil/figures/fusion_figs.py` |
| [M9_ablations.png](M9_ablations.png) | What happens without the barometer or magnetometer and with 2 s or 10 s windows, on seen and unseen missions. | `reports\tables\C_ablations.csv` | `src/aerovigil/figures/evaluation_figs.py` |
| [M9_detection_by_attack.png](M9_detection_by_attack.png) | Every detector against every attack family in both velocity modes. | `data/processed/evaluation/physics_flights.csv`<br>`data/processed/module05_v2/all_models_flight_full.csv.gz`<br>`data/processed/module07_v2/protocols/flights.csv.gz` | `src/aerovigil/figures/evaluation_figs.py` |
| [M9_model_size.png](M9_model_size.png) | Larger LSTMs do not detect velocity-consistent attacks better. | `reports\tables\C_lstm.csv` | `src/aerovigil/figures/evaluation_figs.py` |
| [M9_protocols.png](M9_protocols.png) | Detection per detector on seen missions, an unseen mission and an unseen attack type. | `reports\tables\C_all_detectors.csv` | `src/aerovigil/figures/evaluation_figs.py` |
| [M9_seeds.png](M9_seeds.png) | How much the learned detectors' results move between training seeds. | `data\processed\module05_v2\all_models_per_seed_full.csv`<br>`data\processed\module07_v2\protocols\per_seed.csv` | `src/aerovigil/figures/evaluation_figs.py` |
| [M10_alarm_attribution.png](M10_alarm_attribution.png) | How one real alarm is explained: each sensor check against its normal limit, and the message the operator sees. | `data\processed\explain\alarms.csv.gz`<br>`data/processed/explain/normals.json` | `src/aerovigil/figures/explain_figs.py` |
| [M10_alarm_clusters.png](M10_alarm_clusters.png) | Unsupervised clusters of alarms do not recover the attack families (ARI 0.09-0.16). | `data\processed\explain\alarm_clusters.csv`<br>`reports\tables\E_kmeans.csv`<br>`reports\tables\E_kmeans_silhouette.csv` | `src/aerovigil/figures/explain_figs.py` |
| [M10_shap_summary.png](M10_shap_summary.png) | SHAP summary: the tree models rely first on horizontal-speed consistency, then on course and GPS speed. | `reports\tables\E_shap_global.csv` | `src/aerovigil/figures/explain_figs.py` |
| [M11_compact_forests.png](M11_compact_forests.png) | How much detection smaller random forests give up for a smaller file. | `reports\tables\F_compact_forests.csv` | `src/aerovigil/figures/edge_figs.py` |
| [M11_size_latency.png](M11_size_latency.png) | Model size, time per window and detection before and after ONNX export and int8 quantization: the LSTMs shrink 2.5x with the same detections. | `reports\tables\F_edge_models.csv` | `src/aerovigil/figures/edge_figs.py` |
| [M12_fallback_error.png](M12_fallback_error.png) | Position error with and without the dead-reckoning fallback after an alarm: the fallback helps for about 15-20 s, then drifts. | `data/processed/mitigation/curves.npz`<br>`data/processed/mitigation/choice.txt` | `src/aerovigil/figures/fallback_figs.py` |
| [M12_within_limits.png](M12_within_limits.png) | Share of dead-reckoning runs still within 10 m and 25 m of the truth over time (median 20 s and 30 s). | `data/processed/mitigation/coast_runs.csv` | `src/aerovigil/figures/fallback_figs.py` |
| [M13_dashboard.png](M13_dashboard.png) | The offline demo dashboard (default view): flight, attack and detector controls, the path and offset, the detector comparison, the out-drift grid, the signal charts and the alarm explanation. | `data/processed/demo/*.parquet`<br>`app/app.py` | `src/aerovigil/figures/demo_figs.py` |

## Earlier figures and what replaces them

The module figures made during development (reports/figures/module03-module07) came
from the first version of the data and attacks (before the Module 03 v2 rebuild and the
removal of the velocity overshoot), so they were removed from the repository; Git
history keeps them. Nothing they showed is lost:

| Earlier figure | Replaced by |
|---|---|
| module03 feature_correlation, missing_values | M3_feature_correlation, M3_missing_values |
| module03 pca_train / pca_validation / pca_test | M3_pca_windows |
| module03 sampling_intervals | M2_logging_rate |
| module04 M4_fig1_offset_profiles | M4_attack_families |
| module04 M4_fig2_true_vs_spoofed_path | M3_example_flight_phases (path panel) |
| module04 M4_fig3_parameter_coverage | M4_parameter_coverage |
| module04 M4_fig4_sweep | M4_drift_sweep |
| module05 M5_fig1_f1_seen_vs_unseen | M5_metrics_seen_unseen, M9_protocols |
| module05 M5_fig2_roc_seen | M5_roc_curves |
| module05 M5_fig3_detection_by_attack | M5_detection_by_attack, M9_detection_by_attack |
| module05 M5_fig4_rf_importance | M5_rf_importance |
| module06 M6_fig1_innovation_plot | M6_innovation_nis |
| module06 M6_fig2_drift_absorption | M6_drift_absorption |
| module06 M6_fig3_detection_by_attack | M9_detection_by_attack |
| module06 M6_fig4_stealth_frontier (old overshoot artefact) | M8_oracle_selftest, M8_stealth_frontier |
| module07 M7_fig1_window_auc | M7_window_auc |
| module07 M7_fig2_ae_timeline | M7_reconstruction_error |
| module07 M7_fig3_detection_by_attack | M9_detection_by_attack |
| module07 M7_fig4_stealth_preview | M8_stealth_frontier |

The step figures in reports/figures/stealth (D1-D3), explain (E1-E2) and mitigation
(G1-G2) are still drawn by their own steps; M8_stealth_frontier, M8_detection_by_attacker,
M8_hardening, M10_* and M12_* are their final versions here.
