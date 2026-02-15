#!/bin/bash
# Diffusion v2 experiment - E=15, resource level 20 (with resource clamp)
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/diffusion_v2_e15/r20.yaml "$@"
