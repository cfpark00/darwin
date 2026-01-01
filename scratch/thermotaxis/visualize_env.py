"""
Thermotaxis Environment Visualization

Environment specs:
- Grid: 512 x 512
- Food: Gradient (0 at bottom, 20 at top) + Gaussian noise field
- Temperature:
  - Top (y=512): sinusoidal oscillation between 0.0 and 1.0
  - Amplitude and phase decay with depth (soil heat diffusion model)
- Reproduction success: 1.0 if temp <= 0.5, linear decay to 0.0 at temp = 1.0
- Metabolic penalty: proportional to |temp - 0.5| (optimal at 0.5)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Grid dimensions (matches simulation: square grid)
HEIGHT = 512  # y
WIDTH = 512   # x

# Create coordinate grids
y = np.arange(HEIGHT)
x = np.arange(WIDTH)
X, Y = np.meshgrid(x, y)

# --- Food Distribution (gradient + noise) ---
def generate_gaussian_field(shape, length_scale, seed=42):
    """Generate smooth Gaussian random field."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    # Smooth with Gaussian filter (sigma ~ length_scale)
    smooth = gaussian_filter(noise, sigma=length_scale)
    # Normalize to [0, 1]
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth

food_gradient = 20 * Y / (HEIGHT - 1)  # 0 at bottom, 20 at top
food_noise = generate_gaussian_field((HEIGHT, WIDTH), length_scale=30, seed=42)
food_noise_amplitude = 8.0  # +/- variation around gradient
food = np.clip(food_gradient + food_noise_amplitude * (food_noise - 0.5), 0, None)

# --- Temperature Model ---
def get_temperature(y_coords, time, period=1000, damping_depth=128.0):
    """
    Soil temperature model with phase lag (inspired by Ramot et al. 2008).

    Modified to have vertical gradient in mean temperature:
    - Top (y=max): mean=0.5, oscillates between 0.0 and 1.0
    - Bottom (y=0): mean=0.25, no oscillation (amplitude decayed)

    T(y,t) = mean(y) + 0.5 * exp(-z/zd) * sin(2πt/p - z/zd)
    where mean(y) = 0.25 + 0.25 * (y / max_y)
    """
    max_y = HEIGHT - 1
    depth = max_y - y_coords  # depth from surface (0 at top, max at bottom)

    # Mean temperature: 0.5 at surface, 0.25 at bottom
    mean_temp = 0.25 + 0.25 * (y_coords / max_y)

    # Amplitude decay and phase lag both scale with depth/zd
    decay = np.exp(-depth / damping_depth)
    phase_lag = depth / damping_depth

    # Oscillation around the mean
    temp = mean_temp + 0.5 * decay * np.sin(2 * np.pi * time / period - phase_lag)

    return temp

# --- Reproduction Success ---
def reproduction_success(temp):
    return np.clip(1.0 - 2 * np.maximum(0, temp - 0.5), 0, 1)

# --- Metabolic Penalty (optimal at temp = 0.5) ---
def metabolic_penalty(temp, coeff=0.75):
    """Penalty proportional to distance from optimal temperature (0.5)."""
    return coeff * np.abs(temp - 0.5)

# --- Create visualizations ---
output_dir = Path(__file__).parent

# Figure 1: Combined overview
# Row 1: Food, Temp peak, Temp trough
# Row 2: (empty), Repro peak, Repro trough
# Row 3: (empty), Metab peak, Metab trough
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
period = 1000

temp_peak = get_temperature(Y, period/4, period=period)  # t=T/4 is actual peak for sin
temp_trough = get_temperature(Y, 3*period/4, period=period)  # t=3T/4 is actual trough

# Row 0: Food, Temp peak, Temp trough
ax = axes[0, 0]
im = ax.imshow(food, origin='lower', aspect='auto', cmap='Greens', extent=[0, WIDTH, 0, HEIGHT])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Food Distribution')
plt.colorbar(im, ax=ax, label='Food (0-20)')

ax = axes[0, 1]
im = ax.imshow(temp_peak, origin='lower', aspect='auto', cmap='coolwarm',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Temperature (PEAK, top=1.0)')
plt.colorbar(im, ax=ax, label='Temperature')

ax = axes[0, 2]
im = ax.imshow(temp_trough, origin='lower', aspect='auto', cmap='coolwarm',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Temperature (TROUGH, top=0.0)')
plt.colorbar(im, ax=ax, label='Temperature')

# Row 1: (empty), Repro peak, Repro trough
axes[1, 0].axis('off')

ax = axes[1, 1]
repro_peak = reproduction_success(temp_peak)
im = ax.imshow(repro_peak, origin='lower', aspect='auto', cmap='RdYlGn',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Repro Success (PEAK)')
plt.colorbar(im, ax=ax, label='Success Prob')

ax = axes[1, 2]
repro_trough = reproduction_success(temp_trough)
im = ax.imshow(repro_trough, origin='lower', aspect='auto', cmap='RdYlGn',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Repro Success (TROUGH)')
plt.colorbar(im, ax=ax, label='Success Prob')

# Row 2: (empty), Metab peak, Metab trough
axes[2, 0].axis('off')

ax = axes[2, 1]
metab_peak = metabolic_penalty(temp_peak)
im = ax.imshow(metab_peak, origin='lower', aspect='auto', cmap='Reds',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=0.375)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Metabolic Penalty (PEAK)')
plt.colorbar(im, ax=ax, label='Penalty')

ax = axes[2, 2]
metab_trough = metabolic_penalty(temp_trough)
im = ax.imshow(metab_trough, origin='lower', aspect='auto', cmap='Reds',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=0.375)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Metabolic Penalty (TROUGH)')
plt.colorbar(im, ax=ax, label='Penalty')

plt.suptitle('Thermotaxis Environment Overview (512 x 512)', fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / 'environment_overview.png', dpi=150)
plt.close()

# Figure 2: Temperature heatmap (y vs time)
n_times = 200
times = np.linspace(0, 2 * period, n_times)  # 2 full cycles
y_vals = np.arange(HEIGHT)

# Build heatmap: rows=y, cols=time
temp_heatmap = np.zeros((HEIGHT, n_times))
for i, t in enumerate(times):
    temp_heatmap[:, i] = get_temperature(y_vals, t, period=period)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(temp_heatmap, origin='lower', aspect='auto', cmap='coolwarm',
               extent=[0, 2*period, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('Time (steps)')
ax.set_ylabel('y position')
ax.set_title('Temperature over Time (2 cycles)')
plt.colorbar(im, ax=ax, label='Temperature')
plt.tight_layout()
plt.savefig(output_dir / 'temperature_yt_heatmap.png', dpi=150)
plt.close()

# Figure 3: Metabolic penalty heatmap (y vs time) - shows optimal zone at temp=0.5
metab_heatmap = metabolic_penalty(temp_heatmap)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(metab_heatmap, origin='lower', aspect='auto', cmap='Greens_r',
               extent=[0, 2*period, 0, HEIGHT], vmin=0, vmax=0.375)
ax.set_xlabel('Time (steps)')
ax.set_ylabel('y position')
ax.set_title('Metabolic Penalty over Time (dark = optimal, temp=0.5)')
plt.colorbar(im, ax=ax, label='Penalty')

# Add contour line for temp=0.5 (optimal zone)
ax.contour(temp_heatmap, levels=[0.5], colors='white', linewidths=2,
           extent=[0, 2*period, 0, HEIGHT])

plt.tight_layout()
plt.savefig(output_dir / 'metabolic_penalty_yt_heatmap.png', dpi=150)
plt.close()

print(f"Saved: environment_overview.png, temperature_yt_heatmap.png, metabolic_penalty_yt_heatmap.png")
