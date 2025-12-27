#!/bin/bash
# Uses transfer_simple.py to load pretrained agents from run_pretrain
uv run python src/scripts/transfer_simple.py configs/run/thermotaxis.yaml "$@"
