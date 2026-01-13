#!/bin/bash
# Diffusion experiment - resource level 30 (energy=30)
uv run python src/scripts/transfer_simple.py configs/run/diffusion_e30/r30.yaml "$@"
