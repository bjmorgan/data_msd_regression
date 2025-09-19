"""MSD regression analysis package for comparing OLS, WLS, and GLS methods."""

from .core import (
    generate_3d_random_walk_ensemble,
    calculate_msd_with_time_avg,
    calculate_msd_no_time_avg
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
    plot_methods_single_panel,
    plot_ols_comparison
)

from .utils import (
    RegressionResult,
    calculate_usler_variance
)

__version__ = "0.2.0"
