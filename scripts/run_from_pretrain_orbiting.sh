#!/bin/bash
# Transfer simple agents from pretrain to orbiting gaussian environment
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/scripts/transfer_simple.py configs/run/from_pretrain_orbiting.yaml "$@"
