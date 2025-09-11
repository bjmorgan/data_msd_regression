"""Core simulation and regression functions."""

import numpy as np
from scipy import stats
from statsmodels.stats.correlation_tools import cov_nearest

def generate_3d_random_walk_ensemble(n_steps: int, n_particles: int) -> np.ndarray:
    """Generate ensemble of 3D random walks on cubic lattice."""
    step_choices = np.array([
        [1, 0, 0], [-1, 0, 0],  # ±x
        [0, 1, 0], [0, -1, 0],  # ±y
        [0, 0, 1], [0, 0, -1]   # ±z
    ]) * np.sqrt(6)
    
    positions = np.zeros((n_particles, n_steps + 1, 3))
    for p in range(n_particles):
        step_indices = np.random.randint(0, 6, size=n_steps)
        steps = step_choices[step_indices]
        positions[p, 1:] = np.cumsum(steps, axis=0)
    
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
        displacements = positions[:, lag:, :] - positions[:, :n_origins, :]
        squared_displacements = np.sum(displacements**2, axis=2)
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
        squared_displacements = np.sum(displacements**2, axis=1)
        msd[i] = np.mean(squared_displacements)
    
    return lags, msd

def fit_ols(lags: np.ndarray, msd: np.ndarray) -> float:
    """Ordinary least squares regression."""
    slope, _, _, _, _ = stats.linregress(lags, msd)
    return slope / 6

def fit_wls(lags: np.ndarray, msd: np.ndarray, weights: np.ndarray) -> float:
    """Weighted least squares regression with provided weights."""
    w_sqrt = np.sqrt(weights)
    X = np.column_stack([np.ones_like(lags), lags])
    X_weighted = X * w_sqrt[:, None]
    msd_weighted = msd * w_sqrt
    
    coeffs = np.linalg.lstsq(X_weighted, msd_weighted, rcond=None)[0]
    return coeffs[1] / 6

def fit_wls_sqrtlag(lags: np.ndarray, msd: np.ndarray) -> float:
    """WLS using 1/sqrt(lag) weights (incorrect but commonly used)."""
    weights = 1.0 / np.sqrt(lags)
    return fit_wls(lags, msd, weights)

def fit_gls(lags: np.ndarray, msd: np.ndarray, cov_matrix: np.ndarray) -> float:
    """Generalized least squares regression."""
    try:
        cov_psd = cov_nearest(cov_matrix + np.eye(len(cov_matrix)) * 1e-10)
        L = np.linalg.cholesky(cov_psd)
        X = np.column_stack([np.ones_like(lags), lags])
        
        X_white = np.linalg.solve(L, X)
        msd_white = np.linalg.solve(L, msd)
        
        coeffs = np.linalg.lstsq(X_white, msd_white, rcond=None)[0]
        return coeffs[1] / 6
    except np.linalg.LinAlgError:
        return np.nan
