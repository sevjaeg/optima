# Tested with Python 3.9
PYTHON=python3

.PHONY: fit
fit:
	$(PYTHON) fit.py

.PHONY: plot-raw
plot-raw:
	$(PYTHON) plot-raw-discharge.py
	$(PYTHON) plot-raw-power.py

.PHONY: clean
clean:
	rm -rf out/ __pycache__/
