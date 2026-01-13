# Vision System Exploration

## Summary
Explored options for adding vision capabilities to agents. Discussed implementation approaches, computational costs, and alternatives. No code changes made - this was a design discussion.

## Current Perception System
Agents currently have minimal perception:
- Food/temperature at current cell (with noise)
- Toxin detection (binary, within radius via convolution)
- 4 contact sensors (adjacent cells only)

## Vision Options Discussed

### Option 1: Patch-Based Vision
Extract NxN grid around agent.
- Simple to implement
- But omnidirectional (not realistic), no occlusion
- Large input dimension (5×5×4 = 100 inputs)

### Option 2: Ray Tracing (Recommended)
Cast N rays in forward cone, march through cells.
- Directional and realistic
- Natural occlusion handling
- Smaller input (5 rays × 4 values = 20 inputs)
- More complex implementation (ray marching in JAX)

### Option 3: Cone Vision
Check cells in triangular region ahead.
- Middle ground between patch and ray tracing
- Simpler than ray tracing but no occlusion

## Computational Cost Analysis

Compared ray tracing vs pairwise operations (like gravity):

| Operation | Complexity | N=512 agents |
|-----------|------------|--------------|
| Pairwise | O(N²) | 262,144 ops |
| Ray tracing | O(N × R × D) | 25,600 ops |

Ray tracing is ~10× cheaper and scales better with agent count.

## MuJoCo / MJX Discussion
Discussed MuJoCo MJX (JAX-accelerated physics) as alternative:
- Built-in camera sensors and ray casting
- GPU-batched simulation
- Fully open source (Apache 2.0)
- Would require replacing grid-based simulation entirely

## Files Created
- `docs/maybe_vision.md` - Detailed discussion document for future reference

## Decision
Not implementing vision now. Discussion captured in docs/maybe_vision.md for future reference. If pursued, ray tracing is the recommended approach.

## Next Steps (if vision is pursued)
1. Create `agent_vision.py` with CNN+LSTM or larger LSTM
2. Implement ray casting in JAX with `jax.lax.scan`
3. Add vision parameters to config (FOV, num_rays, max_distance)
4. Test if small networks can learn to use visual input
