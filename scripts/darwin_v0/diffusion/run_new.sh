#!/bin/bash
# Run new diffusion experiments only (r5, r35, r40)

set -e  # Exit on error

echo "=== Diffusion Experiment: Resource Level 5 ==="
bash scripts/diffusion/r5.sh "$@"

echo "=== Diffusion Experiment: Resource Level 35 ==="
bash scripts/diffusion/r35.sh "$@"

echo "=== Diffusion Experiment: Resource Level 40 ==="
bash scripts/diffusion/r40.sh "$@"

echo "=== New diffusion experiments complete ==="
