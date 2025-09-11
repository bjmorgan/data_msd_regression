"""Data structures and utility functions."""

from dataclasses import dataclass
import numpy as np

@dataclass
class RegressionResult:
    """Results from regression analysis."""
    method: str
    values: np.ndarray
    n_steps: int
    max_lag: int
    time_averaged: bool
    n_particles: int
    n_simulations: int
    
    @property
    def mean(self) -> float:
        return np.nanmean(self.values)
    
    @property
    def std(self) -> float:
        return np.nanstd(self.values, ddof=1)
    
    @property
    def var(self) -> float:
        return np.nanvar(self.values, ddof=1)
    
    @property
    def n_valid(self) -> int:
        return np.sum(~np.isnan(self.values))


def calculate_usler_variance(n_particles: int, 
                             n_dim: int=3,
                             D_true: float = 1.0) -> float:
    """
    Calculate the Usler estimate for variance in the plateau limit.
    
    From Usler et al. doi:10.1002/jcc.27090 Eq. 14, 
    in plateau limit where [1 - exp(-<r²>/s²)] ≈ 1:
    u_rel = sqrt(2 / (d * N_k))
    """
    u_rel = np.sqrt(2 / (n_dim * n_particles))
    variance = (u_rel * D_true)**2
    return variance
