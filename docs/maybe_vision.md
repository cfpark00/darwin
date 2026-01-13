# Maybe Vision: Discussion Notes

## Context
Discussion about adding vision capabilities to agents (2026-01-01).

## Current Agent Perception

**Full version (7 inputs):**
- Food level at current cell (with noise)
- Temperature at current cell (with noise)
- Toxin detected (binary, within radius)
- 4 contact sensors (front, back, left, right)

**Simple version (6 inputs):**
- Same but no toxin detection

All perception is essentially **local** - agents only sense their immediate cell and adjacent cells for contact.

## Vision Options Discussed

### Option 1: Patch-Based Vision
Extract NxN grid around agent (e.g., 5x5).

**Pros:**
- Simple to implement (`jax.lax.dynamic_slice`)
- Easy to vectorize

**Cons:**
- Omnidirectional (not realistic vision)
- No occlusion
- Large input dimension (5×5×4 channels = 100 inputs)

### Option 2: Ray Tracing (Preferred)
Cast N rays in forward cone, march through cells until hit.

**Example:** 5 rays, 90° FOV, max distance 10
```
        \  |  /
         \ | /
          \|/
           A  ← agent facing up
```

Each ray returns: `(distance, food, is_agent, is_obstacle)`
→ 5 rays × 4 values = 20 inputs

**Pros:**
- Directional (realistic)
- Occlusion handled naturally
- Smaller input than patch
- Computationally cheaper than O(N²) alternatives

**Cons:**
- More complex to implement
- Need ray marching loop in JAX (`jax.lax.scan`)
- Orientation handling for ray directions

### Option 3: Cone Vision (Middle Ground)
Check cells in triangular region ahead, no per-ray occlusion.

**Pros:**
- Simpler than ray tracing
- Still directional

**Cons:**
- No occlusion (unrealistic)

## Implementation Complexity

| Component | Difficulty |
|-----------|------------|
| Ray direction calculation | Low |
| Grid stepping along ray | Medium |
| First-hit detection (occlusion) | Medium |
| Vectorizing over all agents | Medium |
| New agent architecture (CNN+LSTM or larger LSTM) | Medium |
| Integration with evolution | Unknown |

**Estimate:** ~200-300 lines of new code

## Computational Cost Comparison

| Operation | Complexity |
|-----------|------------|
| Pairwise (e.g., gravity) | O(N²) |
| Ray tracing | O(N × R × D) |

For N=512 agents, R=5 rays, D=10 distance:
- Pairwise: 262,144 operations
- Ray tracing: 25,600 operations

**Ray tracing is ~10× cheaper** and scales better with agent count.

## Architecture Considerations

Current network: 2-layer LSTM, hidden_dim=8, ~1,055 parameters

With vision, options:
1. **Flatten visual input → larger LSTM input** (simple but param explosion)
2. **CNN encoder → LSTM** (appropriate but more complex)
3. **Attention over visual field** (overkill for now)

## Open Questions

1. **What should agents see?**
   - Just food levels?
   - Other agents?
   - Toxin/obstacles?
   - All of the above?

2. **Can small LSTM learn to use vision?**
   - May need larger hidden_dim
   - May need much longer evolution

3. **Is vision worth the complexity?**
   - Current contact sensors already provide some spatial info
   - Vision might enable more interesting emergent behaviors (hunting, flocking, avoidance)

## Alternative: MuJoCo / MJX

MuJoCo MJX (JAX-accelerated physics) offers:
- Built-in camera sensors and ray casting
- GPU-batched simulation
- Real 3D physics

Trade-offs:
- Would replace grid-based simulation entirely
- Different evolutionary dynamics
- More complex setup
- Potentially richer behaviors

MJX is fully open source (Apache 2.0), part of main MuJoCo repo.

## Decision

**Not implementing now.** This doc captures the discussion for future reference. If we pursue vision, ray tracing is the recommended approach.
