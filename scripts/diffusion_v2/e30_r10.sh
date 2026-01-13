#!/bin/bash
# Diffusion v2 experiment - E=30, resource level 10 (with resource clamp)
uv run python src/scripts/transfer_simple.py configs/run/diffusion_v2_e30/r10.yaml "$@"
