#!/bin/bash
# Uses transfer_simple.py to load pretrained agents from run_pretrain
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/thermotaxis.yaml "$@"
