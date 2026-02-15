#!/bin/bash
# Pretrain: Simple environment for basic survival training
# Food decays from 30 to 12 over 10k steps (curriculum learning)
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/evolvability_v1/scripts/run.py configs/evolvability_v1/run/pretrain/default.yaml "$@"
