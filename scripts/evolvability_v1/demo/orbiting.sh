#!/bin/bash
# Demo: Orbiting Gaussian environment - resource blob orbits center
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/evolvability_v1/scripts/run.py configs/evolvability_v1/run/demo/orbiting.yaml "$@"
