#!/bin/bash
# Transfer simple agents from pretrain to default-like environment (no toxin)
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/from_pretrain_simple.yaml "$@"
