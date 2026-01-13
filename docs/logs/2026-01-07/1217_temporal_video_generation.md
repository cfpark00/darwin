# Temporal Gaussian Video Generation

## Summary
Created video generation tooling for the temporal gaussian run (`data/run_from_pretrain_temporal/`).

## Videos Created

### 1. Checkpoint-based Video
- **Script**: `scratch/temporal_video/make_video.py`
- **Output**: `scratch/temporal_video/temporal_gaussian.mp4` (8.1 MB)
- **Method**: Loads all 82 checkpoints (step 0 to 200001), generates frames, encodes to video
- **Framerate**: 15 fps (~5.5 seconds total)
- Shows resource field evolution and agent positions over full 200k step run

### 2. Step-by-Step Resume Video
- **Script**: `scratch/temporal_video/resume_video.py`
- **Output**: `scratch/temporal_video/temporal_resume.mp4` (14 MB)
- **Method**: Loads last checkpoint, resumes simulation, saves 1 frame per step
- **Parameters**: `--steps N` (default 500), `--fps N` (default 30)
- Generated 500 frames at 30 fps = 16.7 seconds of video

## Technical Details

### Resume Script
The resume script demonstrates how to continue simulation from checkpoint:
1. Load checkpoint (contains full state including Fourier coefficients for temporal evolution)
2. Load world and run configs
3. Create `SimulationSimple` instance with same configs
4. Call `sim.step(state, key)` repeatedly
5. Save matplotlib frame each iteration

### Key Learnings
- Checkpoint structure: `{'step': int, 'state': {...}}`
- State contains: world, positions, orientations, params, states, energies, ages, alive, max_agents, actions
- World for temporal_gaussian includes `_fourier_coefficients` and `_fourier_omega` for time evolution
- No explicit "resume" method needed - just pass loaded state to `step()`

## Files Created
- `scratch/temporal_video/make_video.py`
- `scratch/temporal_video/resume_video.py`
- `scratch/temporal_video/frames/` (checkpoint video frames)
- `scratch/temporal_video/resume_frames/` (step-by-step frames)
- `scratch/temporal_video/temporal_gaussian.mp4`
- `scratch/temporal_video/temporal_resume.mp4`
