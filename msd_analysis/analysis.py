"""Analysis workflow functions for running MSD regression comparisons."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from .core import (
    generate_3d_random_walk_ensemble,
    calculate_msd_with_time_avg,
    calculate_msd_no_time_avg,
    # fit_ols, fit_wls, fit_wls_sqrtlag, fit_gls,
    fit_generalized, fit_generalized_vectorized
)
from .utils import RegressionResult


def analyze_condition(n_steps: int, max_lag: int, n_simulations: int, 
                     n_particles: int, base_seed: int, time_average: bool = True) -> dict:
    """Analyze a single (n_steps, max_lag) condition."""
    
    msd_func = calculate_msd_with_time_avg if time_average else calculate_msd_no_time_avg
    
    def generate_single_msd(seed):
        np.random.seed(seed)
        positions = generate_3d_random_walk_ensemble(n_steps, n_particles)
        lags, msd = msd_func(positions, max_lag)
        return lags, msd
    
    seeds = [base_seed + i for i in range(n_simulations)]
    msd_results = Parallel(n_jobs=-1, batch_size='auto')(
        delayed(generate_single_msd)(seed) for seed in seeds
    )
    
    lags = msd_results[0][0]
    msd_array = np.array([msd for _, msd in msd_results])
    msd_matrix = np.array(msd_array).T
    
    # Calculate weight matrices for each regression method
    cov_matrix = np.cov(msd_array, rowvar=False, ddof=1)
    W_ols = np.eye(len(lags))
    W_wls = np.linalg.pinv(np.diag(np.diag(cov_matrix)))
    W_wls_sqrt = np.diag(1.0 / np.sqrt(lags))
    W_gls = np.linalg.pinv(cov_matrix)
    
    d_ols = fit_generalized_vectorized(lags, msd_matrix, W_ols)
    d_wls = fit_generalized_vectorized(lags, msd_matrix, W_wls)
    d_wls_sqrtlag = fit_generalized_vectorized(lags, msd_matrix, W_wls_sqrt)
    d_gls = fit_generalized_vectorized(lags, msd_matrix, W_gls)
    
    return {
        'ols': RegressionResult('OLS', d_ols, n_steps, max_lag, 
                               time_average, n_particles, n_simulations),
        'wls': RegressionResult('WLS', d_wls, n_steps, max_lag, 
                               time_average, n_particles, n_simulations),
        'wls_sqrtlag': RegressionResult('WLS-SqrtLag', d_wls_sqrtlag, n_steps, max_lag, 
                                        time_average, n_particles, n_simulations),
        'gls': RegressionResult('GLS', d_gls, n_steps, max_lag, 
                               time_average, n_particles, n_simulations)
    }


def run_analysis(n_steps_values, max_lag_values, n_simulations, n_particles,
                random_seed, time_average=True):
    """Run complete analysis for all conditions."""
    
    if time_average:
        conditions = [(i, j, n, m) for i, n in enumerate(n_steps_values) 
                     for j, m in enumerate(max_lag_values) if m <= n]
    else:
        conditions = [(i, i, n, n) for i, n in enumerate(n_steps_values)]
    
    print(f"Analyzing {len(conditions)} conditions {'with' if time_average else 'without'} time averaging")
    
    results = {}
    for i, j, n_steps, max_lag in tqdm(conditions, desc="Conditions"):
        base_seed = random_seed + i * 1000 + j * 10000
        results[(n_steps, max_lag)] = analyze_condition(
            n_steps, max_lag, n_simulations, n_particles, base_seed, time_average
        )
    
    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Convert results to DataFrame for analysis."""
    records = []
    for (n_steps, max_lag), method_results in results.items():
        for method_name, result in method_results.items():
            records.append({
                'n_steps': n_steps,
                'max_lag': max_lag,
                'method': method_name.upper(),
                'mean': result.mean,
                'std': result.std,
                'var': result.var,
                'n_valid': result.n_valid,
                'time_averaged': result.time_averaged,
                'n_particles': result.n_particles,
                'n_simulations': result.n_simulations,
                'fitting_fraction': max_lag / n_steps,
                'is_diagonal': n_steps == max_lag,
            })
    
    return pd.DataFrame(records)
