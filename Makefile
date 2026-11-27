# AEROVIGIL one-command targets. On Windows without make, use make.ps1 (same targets):
#   .\make.ps1 demo
PY ?= python

.PHONY: help demo demo-data

help:
	@echo "make demo        open the offline demo dashboard (builds its data the first time)"
	@echo "make demo-data   (re)build the demo's precomputed scenarios (~10 min)"

demo-data:
	$(PY) src/aerovigil/demo/build.py

demo:
	@test -f data/processed/demo/scenarios.parquet || $(PY) src/aerovigil/demo/build.py
	@echo "AEROVIGIL demo: open http://localhost:8501 (Ctrl+C to stop)"
	$(PY) -m streamlit run app/app.py --server.headless true --server.port 8501 \
		--browser.gatherUsageStats false
