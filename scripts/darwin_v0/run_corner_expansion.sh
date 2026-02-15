#!/bin/bash
# Corner expansion - agents spread from corner with reproduction, no regen
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/corner_expansion.yaml "$@"
