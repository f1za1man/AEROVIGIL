# AEROVIGIL one-command targets. On Windows without make, use make.ps1 (same targets):
#   .\make.ps1 <target>
# Order for a full rebuild: data -> train -> eval (each reads the previous one's output).
PY ?= python

.PHONY: help download data train eval tables figures test smoke demo demo-data

help:
	@echo "make download    fetch and check the dataset (2.6 GB, not redistributed)"
	@echo "make data        Modules 02-04: base tables, features, splits, attacks, windows"
	@echo "make train       Modules 05-10: every detector, trained and calibrated"
	@echo "make eval        Steps B-G: protocols, Stealth Frontier, explanations, edge,"
	@echo "                 fallback, then every table (~15 h)"
	@echo "make tables      rebuild every results table from saved results (minutes)"
	@echo "make figures     redraw the Stealth Frontier and fallback figures"
	@echo "make test        black, ruff and pytest"
	@echo "make smoke       tests plus tables and headline figure from a fresh clone"
	@echo "make demo        open the offline demo dashboard (builds its data the first time)"

download:
	$(PY) src/aerovigil/data/download.py

data: download
	$(PY) src/aerovigil/data/inspect_raw_files.py
	$(PY) -m aerovigil.data.inspect_dataset
	$(PY) src/aerovigil/data/build_raw_base.py
	$(PY) -m aerovigil.features.build_features_v2
	$(PY) -m aerovigil.preprocessing.normalize_features_v2
	$(PY) -m aerovigil.windowing.make_windows_v2
	$(PY) src/aerovigil/attacks/generate_v2.py
	$(PY) src/aerovigil/attacks/features_v2_m04.py
	$(PY) src/aerovigil/attacks/attack_separability.py
	$(PY) src/aerovigil/baselines/build_window_tables_v2.py --sources orig,syn
	$(PY) src/aerovigil/lstm/build_sequences_v2.py

train:
	$(PY) src/aerovigil/baselines/rescore_v2.py
	$(PY) src/aerovigil/kalman/run_filter.py
	$(PY) src/aerovigil/kalman/evaluate_detectors.py
	$(PY) src/aerovigil/lstm/train_v2.py
	$(PY) src/aerovigil/lstm/evaluate_v2.py
	$(PY) src/aerovigil/lstm/sequential_v2.py
	$(PY) src/aerovigil/consistency/evaluate.py
	$(PY) src/aerovigil/fusion/signals.py
	$(PY) src/aerovigil/fusion/fuse.py
	$(PY) src/aerovigil/fusion/adaptive.py
	$(PY) src/aerovigil/fusion/threat_model.py

DETECTORS = kalman velocity fused cusum consistency random_forest lstm_ae lstm_supervised

eval:
	$(PY) src/aerovigil/baselines/all_models_v2.py
	$(PY) src/aerovigil/baselines/all_models_v2.py --full
	$(PY) src/aerovigil/evaluation/physics_protocols.py
	$(PY) src/aerovigil/evaluation/lstm_protocols.py
	$(PY) src/aerovigil/evaluation/ablations.py
	$(PY) src/aerovigil/stealth/frontier.py
	for d in $(DETECTORS); do $(PY) src/aerovigil/stealth/adaptive.py --detector $$d || exit 1; done
	$(PY) src/aerovigil/stealth/adaptive.py --summarise
	$(PY) src/aerovigil/stealth/whitebox.py
	$(PY) src/aerovigil/stealth/harden.py --model random_forest
	$(PY) src/aerovigil/stealth/harden.py --model lstm_supervised
	$(PY) src/aerovigil/explain/run.py
	$(PY) src/aerovigil/edge/benchmark.py
	$(PY) src/aerovigil/mitigation/run.py
	$(PY) src/aerovigil/evaluation/run_all.py

tables:
	$(PY) src/aerovigil/evaluation/run_all.py

figures:
	$(PY) src/aerovigil/stealth/figures.py
	$(PY) src/aerovigil/mitigation/run.py --figures

test:
	$(PY) -m black --check --line-length 88 src/aerovigil tests app
	$(PY) -m ruff check src/aerovigil tests app
	$(PY) -m pytest tests -q

# a fresh clone has no dataset: data tests skip instead of failing
smoke: export AEROVIGIL_ALLOW_MISSING_DATA = 1
smoke: test tables
	$(PY) src/aerovigil/stealth/figures.py

demo-data:
	$(PY) src/aerovigil/demo/build.py

demo:
	@test -f data/processed/demo/scenarios.parquet || $(PY) src/aerovigil/demo/build.py
	@echo "AEROVIGIL demo: open http://localhost:8501 (Ctrl+C to stop)"
	$(PY) -m streamlit run app/app.py --server.headless true --server.port 8501 \
		--browser.gatherUsageStats false
