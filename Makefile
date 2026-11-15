PY ?= python3
.PHONY: reproduce test test-all manifest dsv41 dsv41-check rtl clean

## Regenerate every number, figure and doc from the committed inputs (no network; a few minutes).
reproduce:
	$(PY) -m experiments.run_all

## Fast invariants (seconds). `make test-all` also re-runs the DeepSeek-V4.1 experiments.
test:
	$(PY) -m pytest -q tests -m "not slow"

test-all:
	$(PY) -m pytest -q tests

## Rebuild the checkpoint manifest from the Hub (headers only; needs network).
manifest:
	$(PY) tools/fetch_checkpoint_manifest.py --repo thinkingmachines/Inkling-2-Small \
	    --revision 8361e7a31ae7e9966a49a91d3994fde618936c2b --out-dir data/inkling2_small

## Re-run the DeepSeek-V4.1 study's experiments on the shared engine (results/dsv41/).
dsv41:
	$(PY) -m experiments.dsv41.run_all

## Re-run the DeepSeek-V4.1 study from its own clone and record whether it reproduces itself.
dsv41-check:
	$(PY) tools/check_dsv41_reproduction.py --repo $(DSV41) --python $(PY)

clean:
	rm -f results/phase0_workload.json results/calibration_*.json results/inkling.json results/fig_*.png
