#!/bin/bash
# Diffusion v2 experiment - resource level 10 (with resource clamp)
uv run python src/scripts/transfer_simple.py configs/run/diffusion_v2/r10.yaml "$@"
