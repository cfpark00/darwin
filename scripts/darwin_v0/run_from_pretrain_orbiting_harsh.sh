#!/bin/bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/from_pretrain_orbiting_harsh.yaml "$@"
