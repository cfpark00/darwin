#!/bin/bash
# Transfer simple agents from pretrain to full environment (with toxin/attack)
uv run python src/darwin_v0/scripts/transfer_simple_to_full.py configs/darwin_v0/run/from_pretrain.yaml "$@"
