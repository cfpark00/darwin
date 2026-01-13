#!/bin/bash
# Single agent observation in uniform resource arena
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
uv run python src/scripts/transfer_simple.py configs/run/single_agent.yaml "$@"
