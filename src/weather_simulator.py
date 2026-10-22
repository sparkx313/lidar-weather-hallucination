"""
Weather Simulator — Synthesizes rain, fog, and snow corruptions on LiDAR point clouds.
"""

import numpy as np
from typing import Tuple, Literal


def simulate_rain(
    points: np.ndarray,
    intensity: float = 0.5,
    rain_density: float = 2000,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Add rain artifacts to a LiDAR point cloud.

    Rain droplets appear as bright, small, randomly-placed returns
    scattered throughout the scan volume.

    Args:
        points: (N, 3) xyz point cloud (clean)
        intensity: 0.0-1.0 rain severity
        rain_density: number of rain points to add
        rng: numpy random generator
    Returns:
        Corrupted point cloud (N + rain_points, 3)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_rain = int(rain_density * intensity)

    # Rain points: random xyz within sensor volume (0-50m range)
    rain_pts = np.zeros((n_rain, 3), dtype=np.float32)
    rain_pts[:, 0] = rng.uniform(-25, 25, n_rain)   # x
    rain_pts[:, 1] = rng.uniform(-25, 25, n_rain)   # y
    rain_pts[:, 2] = rng.uniform(0.5, 2.5, n_rain)  # z (ground-adjacent)

    # Simulate streaking: offset points along gravity axis
    streak_len = 0.05 + 0.15 * intensity
    rain_pts[:, 2] -= streak_len

    return np.vstack([points, rain_pts]).astype(np.float32)


def simulate_fog(
    points: np.ndarray,
    intensity: float = 0.5,
    max_range: float = 80.0,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate fog: attenuates distant returns and adds backscatter noise.

    Args:
        points: (N, 3) clean point cloud
        intensity: 0.0-1.0 fog density
    Returns:
        Fog-affected point cloud
    """
    if rng is None:
        rng = np.random.default_rng()

    ranges = np.linalg.norm(points[:, :2], axis=1)  # 2D range

    # Fog drop-off: exponential visibility decay
    visibility_range = max_range * np.exp(-2.5 * intensity)
    fog_mask = ranges > visibility_range

    # Remove (or heavily thin) points beyond visibility
    keep = ~fog_mask
    # Randomly thin remaining far points
    thin_prob = (ranges[keep] - visibility_range * 0.5) / (visibility_range * 0.5)
    thin_prob = np.clip(thin_prob, 0, intensity * 0.8)
    thin_mask = rng.random(thin_prob.shape) < thin_prob
    keep_indices = np.where(keep)[0]
    drop_indices = keep_indices[thin_mask]

    surviving = np.delete(keep_indices, np.where(keep_indices[:, None] == drop_indices).any(axis=1))
    result = points[surviving]

    # Add backscatter noise points (dense fog droplets in range)
    n_backscatter = int(len(points) * 0.15 * intensity)
    bs_pts = np.zeros((n_backscatter, 3), dtype=np.float32)
    bs_pts[:, 0] = rng.uniform(-30, 30, n_backscatter)
    bs_pts[:, 1] = rng.uniform(-30, 30, n_backscatter)
    bs_pts[:, 2] = rng.uniform(0.1, 3.0, n_backscatter)

    return np.vstack([result, bs_pts]).astype(np.float32)


def simulate_snow(
    points: np.ndarray,
    intensity: float = 0.5,
    snow_density: float = 1500,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Simulate snow: floating white particles + partial ground occlusion.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_snow = int(snow_density * intensity)

    # Snowflake points: slow fall, more clustered than rain
    snow_pts = np.zeros((n_snow, 3), dtype=np.float32)
    snow_pts[:, 0] = rng.normal(0, 15, n_snow)
    snow_pts[:, 1] = rng.normal(0, 15, n_snow)
    snow_pts[:, 2] = rng.uniform(0.0, 4.0, n_snow)  # floating at various heights

    # Add small drift
    drift_x = rng.normal(0, 0.1 * intensity, n_snow)
    drift_y = rng.normal(0, 0.1 * intensity, n_snow)
    snow_pts[:, 0] += drift_x
    snow_pts[:, 1] += drift_y

    # Remove low ground points (snow covers road surface)
    ground_mask = points[:, 2] < 0.2
    drop_frac = rng.random(ground_mask.sum()) < intensity * 0.6
    drop_idx = np.where(ground_mask)[0][drop_frac]
    result = np.delete(points, drop_idx, axis=0)

    return np.vstack([result, snow_pts]).astype(np.float32)


def corrupt_point_cloud(
    points: np.ndarray,
    weather: Literal["rain", "fog", "snow"],
    intensity: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Apply weather corruption to a clean point cloud."""
    rng = np.random.default_rng(seed)
    if weather == "rain":
        return simulate_rain(points, intensity, rng=rng)
    elif weather == "fog":
        return simulate_fog(points, intensity, rng=rng)
    elif weather == "snow":
        return simulate_snow(points, intensity, rng=rng)
    else:
        raise ValueError(f"Unknown weather: {weather}")
