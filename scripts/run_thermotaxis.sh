#!/bin/bash
# Uses transfer.py to load pretrained agents from run_pretrain
uv run python src/scripts/transfer.py configs/run/thermotaxis.yaml "$@"
