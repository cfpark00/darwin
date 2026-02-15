#!/bin/bash
# Transfer simple agents from pretrain to temporal gaussian environment
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/from_pretrain_temporal.yaml "$@"
