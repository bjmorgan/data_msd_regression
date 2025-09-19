"""Core simulation and regression functions."""

import numpy as np
from scipy import stats
from statsmodels.stats.correlation_tools import cov_nearest

def generate_3d_random_walk_ensemble(n_steps: int, n_particles: int) -> np.ndarray:
    """Generate ensemble of 3D random walks on cubic lattice."""
    seed = np.random.RandomState()
    possible_moves = np.zeros((6, 3))
    j = 0
    for i in range(0, 6, 2):
        possible_moves[i, j] = 2.4494897428
        possible_moves[i + 1, j] = -2.4494897428
        j += 1
    choices = seed.choice(len(range(len(possible_moves))), size=(n_particles, n_steps))
    steps = np.zeros((n_particles, n_steps, 3))
    for i in range(steps.shape[0]):
        for j in range(steps.shape[1]):
            steps[i, j] = possible_moves[choices[i, j]]
    cum_steps = np.cumsum(steps, axis=1) 
    return cum_steps

def calculate_msd_with_time_avg(positions: np.ndarray, max_lag: int) -> tuple:
    """Calculate MSD with ensemble and time-origin averaging."""
    n_particles, n_steps_plus_one, _ = positions.shape
    n_steps = n_steps_plus_one - 1
    max_lag = min(max_lag, n_steps)
    
    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(max_lag)
    
    for i, lag in enumerate(lags):
        n_origins = n_steps - lag + 1
        displacements = np.concatenate([positions[:, np.newaxis, i],
                                        np.subtract(positions[:, lag:], positions[:, :n_origins])],
                                        axis=1)
        squared_displacements = np.sum(displacements**2, axis=-1)
        msd[i] = np.mean(squared_displacements)
    
    return lags, msd

def calculate_msd_no_time_avg(positions: np.ndarray, max_lag: int) -> tuple:
    """Calculate MSD with ensemble averaging only (t=0 as only origin)."""
    n_particles, n_steps_plus_one, _ = positions.shape
    n_steps = n_steps_plus_one - 1
    max_lag = min(max_lag, n_steps)
    
    lags = np.arange(1, max_lag + 1)
    msd = np.zeros(max_lag)
    
    for i, lag in enumerate(lags):
        displacements = positions[:, lag, :] - positions[:, 0, :]
        squared_displacements = np.sum(displacements**2, axis=-1)
        msd[i] = np.mean(squared_displacements)
    
    return lags, msd

def fit_ols(lags: np.ndarray, msd: np.ndarray) -> float:
    """Ordinary least squares regression."""
    X = np.array([np.ones(lags.size), lags]).T
    Y = msd.T
    return (np.linalg.pinv(X.T @ X) @ X.T @ Y)[1] / 6

def fit_wls(lags: np.ndarray, msd: np.ndarray, variance: np.ndarray) -> float:
    """Weighted least squares regression with provided weights."""
    X = np.column_stack([np.ones_like(lags), lags])
    Y = msd.T
    W = np.linalg.pinv(np.diag(variance))
    return (np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ Y)[1] / 6

def fit_wls_sqrtlag(lags: np.ndarray, msd: np.ndarray) -> float:
    """WLS using 1/sqrt(lag) weights (incorrect but commonly used)."""
    weights = np.sqrt(lags)
    return fit_wls(lags, msd, weights)

def fit_gls(lags: np.ndarray, msd: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Generalized least squares regression."""
    X = np.column_stack([np.ones_like(lags), lags])
    Y = msd.T
    W = np.linalg.pinv(cov_matrix)
    return (np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ Y)[1] / 6
