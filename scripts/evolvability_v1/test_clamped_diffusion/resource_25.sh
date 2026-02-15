#!/bin/bash
# Test: Clamped diffusion with high resources (25)
# Energy clamped to 30, reproduction disabled, resources clamped to 25
# Loads agents from pretrain checkpoint
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

PRETRAIN_CKPT="data/evolvability_v1/pretrain/default/checkpoints/latest.pkl"

if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: Pretrain checkpoint not found at $PRETRAIN_CKPT"
    echo "Run pretrain first: bash scripts/evolvability_v1/pretrain/default.sh"
    exit 1
fi

uv run python src/evolvability_v1/scripts/run.py \
    configs/evolvability_v1/run/test_clamped_diffusion/resource_25.yaml \
    --from-checkpoint "$PRETRAIN_CKPT" \
    "$@"
