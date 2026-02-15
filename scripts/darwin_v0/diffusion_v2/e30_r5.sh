#!/bin/bash
# Diffusion v2 experiment - E=30, resource level 5 (with resource clamp)
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/diffusion_v2_e30/r5.yaml "$@"
