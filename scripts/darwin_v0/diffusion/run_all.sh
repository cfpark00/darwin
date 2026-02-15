#!/bin/bash
# Run all diffusion experiments (r5, r10, r15, r20, r25, r30, r35, r40)

set -e  # Exit on error

echo "=== Diffusion Experiment: Resource Level 5 ==="
bash scripts/diffusion/r5.sh "$@"

echo "=== Diffusion Experiment: Resource Level 10 ==="
bash scripts/diffusion/r10.sh "$@"

echo "=== Diffusion Experiment: Resource Level 15 ==="
bash scripts/diffusion/r15.sh "$@"

echo "=== Diffusion Experiment: Resource Level 20 ==="
bash scripts/diffusion/r20.sh "$@"

echo "=== Diffusion Experiment: Resource Level 25 ==="
bash scripts/diffusion/r25.sh "$@"

echo "=== Diffusion Experiment: Resource Level 30 ==="
bash scripts/diffusion/r30.sh "$@"

echo "=== Diffusion Experiment: Resource Level 35 ==="
bash scripts/diffusion/r35.sh "$@"

echo "=== Diffusion Experiment: Resource Level 40 ==="
bash scripts/diffusion/r40.sh "$@"

echo "=== All diffusion experiments complete ==="
