#!/bin/bash
# Run all demo experiments
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Running demo/quick..."
bash "$SCRIPT_DIR/quick.sh" "$@"

echo "Running demo/gaussian..."
bash "$SCRIPT_DIR/gaussian.sh" "$@"

echo "Running demo/uniform..."
bash "$SCRIPT_DIR/uniform.sh" "$@"

echo "Running demo/large..."
bash "$SCRIPT_DIR/large.sh" "$@"

echo "Running demo/bridge..."
bash "$SCRIPT_DIR/bridge.sh" "$@"

echo "Running demo/temporal..."
bash "$SCRIPT_DIR/temporal.sh" "$@"

echo "Running demo/orbiting..."
bash "$SCRIPT_DIR/orbiting.sh" "$@"

echo "Running demo/toxin_ring..."
bash "$SCRIPT_DIR/toxin_ring.sh" "$@"

echo "Running demo/toxin_maze..."
bash "$SCRIPT_DIR/toxin_maze.sh" "$@"

echo "Running demo/full_with_toxin..."
bash "$SCRIPT_DIR/full_with_toxin.sh" "$@"

echo "All demos complete!"
