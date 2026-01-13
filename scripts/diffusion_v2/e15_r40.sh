#!/bin/bash
# Diffusion v2 experiment - E=15, resource level 40 (with resource clamp)
uv run python src/scripts/transfer_simple.py configs/run/diffusion_v2_e15/r40.yaml "$@"
