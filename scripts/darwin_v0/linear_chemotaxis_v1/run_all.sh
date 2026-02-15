#!/bin/bash
set -e

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

echo "=== Linear chemotaxis v1 ==="

echo "--- from_pretrain_simple ---"
bash scripts/linear_chemotaxis_v1/from_pretrain_simple.sh "$@"

echo "--- from_temporal ---"
bash scripts/linear_chemotaxis_v1/from_temporal.sh "$@"

echo "--- from_orbiting ---"
bash scripts/linear_chemotaxis_v1/from_orbiting.sh "$@"

echo "=== All linear chemotaxis v1 experiments complete ==="
