#!/bin/bash
# Run all divergent_genotype_v3 experiments
# Action profiles at high and low resource levels for clusters 5 and 17

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Running action_high_c5 ==="
bash "$SCRIPT_DIR/action_high_c5.sh" "$@"

echo "=== Running action_high_c17 ==="
bash "$SCRIPT_DIR/action_high_c17.sh" "$@"

echo "=== Running action_low_c5 ==="
bash "$SCRIPT_DIR/action_low_c5.sh" "$@"

echo "=== Running action_low_c17 ==="
bash "$SCRIPT_DIR/action_low_c17.sh" "$@"

echo "=== All v3 experiments complete ==="
