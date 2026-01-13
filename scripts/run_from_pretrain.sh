#!/bin/bash
# Transfer simple agents from pretrain to full environment (with toxin/attack)
uv run python src/scripts/transfer_simple_to_full.py configs/run/from_pretrain.yaml "$@"
