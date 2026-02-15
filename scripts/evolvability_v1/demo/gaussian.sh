#!/bin/bash
# Demo: Run simulation in Gaussian random field environment
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/evolvability_v1/scripts/run.py configs/evolvability_v1/run/demo/gaussian.yaml "$@"
