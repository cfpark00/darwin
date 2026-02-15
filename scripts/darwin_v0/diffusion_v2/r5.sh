#!/bin/bash
# Diffusion v2 experiment - resource level 5 (with resource clamp)
uv run python src/darwin_v0/scripts/transfer_simple.py configs/darwin_v0/run/diffusion_v2/r5.yaml "$@"
