#!/bin/bash
# Transfer simple agents from pretrain to default-like environment (no toxin)
uv run python src/scripts/transfer_simple.py configs/run/from_pretrain_simple.yaml "$@"
