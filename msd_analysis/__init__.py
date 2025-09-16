"""MSD regression analysis package for comparing OLS, WLS, and GLS methods."""

from .core import (
    generate_3d_random_walk_ensemble,
    calculate_msd_with_time_avg,
    calculate_msd_no_time_avg,
    fit_ols,
    fit_wls,
    fit_wls_sqrtlag,
    fit_gls
)

from .analysis import (
    analyze_condition,
    run_analysis,
    results_to_dataframe
)

from .plotting import (
    plot_all_methods,
    plot_comparison_time_averaging,
    plot_ols_vs_wls_sqrtlag,
    plot_wls_comparison,
    plot_comprehensive_wls_comparison,
    plot_methods_single_panel
)

from .utils import (
    RegressionResult,
    calculate_usler_variance
)

__version__ = "0.1.0"