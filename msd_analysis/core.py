"""Core simulation and regression functions."""

import numpy as np

def generate_3d_random_walk_ensemble(n_steps: int, n_particles: int) -> np.ndarray:
    """Generate ensemble of 3D random walks on cubic lattice."""
    step_choices = np.array([
        [1, 0, 0], [-1, 0, 0],  # ±x
        [0, 1, 0], [0, -1, 0],  # ±y
        [0, 0, 1], [0, 0, -1]   # ±z
    ]) * np.sqrt(6)
    
    # Generate all random steps at once - no loop needed!
    step_indices = np.random.randint(0, 6, size=(n_particles, n_steps))
    steps = step_choices[step_indices]
    
    positions = np.zeros((n_particles, n_steps + 1, 3))
    positions[:, 1:, :] = np.cumsum(steps, axis=1)
    
    return positions

def calculate_msd_with_time_avg(positions: np.ndarray, max_lag: int) -> tuple:
    """Calculate MSD with ensemble and time-origin averaging."""
    n_particles, n_steps_plus_one, _ = positions.shape
    n_steps = n_steps_plus_one - 1
    max_lag = min(max_lag, n_steps)
    
    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(max_lag)
    
    for i, lag in enumerate(lags):
        n_origins = n_steps - lag + 1
        diff = positions[:, lag:, :] - positions[:, :n_origins, :]
        msd[i] = np.einsum('ijk,ijk->', diff, diff) / (n_particles * n_origins)
    
    return lags, msd

def calculate_msd_no_time_avg(positions: np.ndarray, max_lag: int) -> tuple:
    """Calculate MSD with ensemble averaging only (t=0 as only origin)."""
    n_particles, n_steps_plus_one, _ = positions.shape
    n_steps = n_steps_plus_one - 1
    max_lag = min(max_lag, n_steps)
    
    lags = np.arange(1, max_lag + 1)
    
    displacements = positions[:, lags, :] - positions[:, 0:1, :]
    squared_displacements = np.sum(displacements**2, axis=2).T
    msd = np.mean(squared_displacements, axis=1)
    
    return lags, msd

def fit_generalized_vectorized(
    lags: np.ndarray,
    msd_matrix: np.ndarray,
    W: np.ndarray) -> np.ndarray:
    """
    Universal fitting function for all methods.
    
    Calculates inv(X.T @ W @ X) @ X.T @ W @ Y) / 6
    
    Args:
        lags: Time lags, shape (n_lags,)
        msd_matrix: Multiple MSDs, shape (n_lags, n_trajectories)
        W: Weight matrix, shape (n_lags, n_lags)
           - Identity matrix → OLS
           - pinv(diag(var(MSD))) → WLS
           - pinv(covar(MSD)) → GLS
    
    Returns:
        Diffusion coefficients, shape (n_trajectories,)
    """
    X = np.column_stack([np.ones_like(lags), lags])
    
    # Compute regression matrix
    XtW = X.T @ W
    regression_matrix = np.linalg.pinv(XtW @ X) @ XtW
    
    # Apply to all MSDs at once
    coefficients = regression_matrix @ msd_matrix
    
    return coefficients[1, :] / 6
