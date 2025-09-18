"""Plotting functions for MSD regression analysis."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd

# Apply figure formatting (required dependency)
from figure_formatting import figure_formatting as ff
ff.set_formatting()

from .utils import calculate_usler_variance


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_method_colors(n_colors: int, color_range: tuple[float, float] = (0.2, 0.8)) -> np.ndarray:
    """Get consistent colors from viridis colormap.
    
    Args:
        n_colors: Number of colors needed.
        color_range: Range within colormap to use (0-1).
    
    Returns:
        Array of RGBA color values.
    """
    return plt.cm.viridis(np.linspace(color_range[0], color_range[1], n_colors))


def format_log_axis(
    ax: plt.Axes, 
    df: pd.DataFrame | None = None,
    n_steps_values: np.ndarray | None = None,
    xlabel: str = '$t_\\mathrm{sim}$',
    ylabel: str = '$\\sigma^2[\\widehat{D}^*]$',
    show_ylabel: bool = True
) -> None:
    """Apply common formatting for log-log plot axes.
    
    Args:
        ax: Matplotlib axis to format.
        df: DataFrame to extract n_steps values from (if n_steps_values not provided).
        n_steps_values: Values for x-axis ticks (extracted from df if not provided).
        xlabel: X-axis label.
        ylabel: Y-axis label.
        show_ylabel: Whether to show y-axis label.
    """
    ax.set_box_aspect(1)
    
    # Auto-extract n_steps if needed
    if n_steps_values is None and df is not None:
        if 'is_diagonal' in df.columns:
            n_steps_values = sorted(df[df['is_diagonal']]['n_steps'].unique())
        else:
            n_steps_values = sorted(df['n_steps'].unique())
    
    if n_steps_values is not None:
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xticks(n_steps_values[::2])
        ax.set_xticklabels(n_steps_values[::2])
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if show_ylabel else '')


def add_scaling_guide(
    ax: plt.Axes,
    x_range: list[float],
    y_scale: float,
    exponent: float = -1.0,
    label: str | None = None,
    label_x_shift: float = 0.0,
    label_y_shift: float = -0.05
) -> None:
    """Add a scaling guide line to show theoretical variance scaling.
    
    Args:
        ax: Matplotlib axis to add guide to.
        x_range: [start, end] x-coordinates for the guide line.
        y_scale: Y-scale factor for positioning the guide line.
        exponent: Scaling exponent (e.g., -1.0 for N^-1 scaling).
        label: Optional label for the scaling guide.
        label_x_shift: Multiplicative shift for label x-position (0 = center).
        label_y_shift: Multiplicative shift for label y-position from line.
    """
    x_theory = np.array(x_range)
    y_theory = y_scale * (256 / x_theory**(-exponent))
    ax.loglog(x_theory, y_theory, 'k-', alpha=0.5, base=2)
    
    if label:
        # Center in log space
        x_center = np.sqrt(x_range[0] * x_range[1])
        x_label = x_center * (10 ** label_x_shift)  # Log shift
        y_on_line = y_scale * (256 / x_label**(-exponent))
        y_label = y_on_line * (10 ** label_y_shift)  # Log shift
        ax.text(x_label, y_label, label, fontsize=8, ha='center')


def add_legend(
    ax: plt.Axes,
    reverse_order: bool = False,
    outside: bool = False,
    fontsize: int = 8,
    **kwargs
) -> None:
    """Add legend with common configurations.
    
    Args:
        ax: Axis to add legend to.
        reverse_order: Whether to reverse the order of legend entries.
        outside: If True, place legend outside plot area (right side).
        fontsize: Font size for legend text.
        **kwargs: Additional arguments passed to ax.legend().
    """
    handles, labels = ax.get_legend_handles_labels()
    
    if reverse_order:
        handles = handles[::-1]
        labels = labels[::-1]
    
    # Set defaults based on position
    if outside:
        kwargs.setdefault('bbox_to_anchor', (1.1, 1.05))
    else:
        kwargs.setdefault('loc', 'best')
    
    ax.legend(handles, labels, fontsize=fontsize, **kwargs)


def add_usler_line(ax: plt.Axes, n_particles: int = 128) -> None:
    """Add Usler plateau reference line to axis.
    
    Args:
        ax: Matplotlib axis to add line to.
        n_particles: Number of particles for Usler calculation.
    """
    usler_var = calculate_usler_variance(n_particles)
    ax.axhline(y=usler_var, color='#D55E00', linestyle='--', 
               linewidth=1.5, alpha=0.8, label='Usler plateau')


def get_diagonal_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract diagonal cases from DataFrame.
    
    Args:
        df: Full results DataFrame.
        
    Returns:
        Filtered DataFrame with only diagonal cases.
    """
    return df.query('is_diagonal').copy()


def filter_method_data(
    df: pd.DataFrame,
    method: str,
    time_averaged: bool | None = None
) -> pd.DataFrame:
    """Filter DataFrame for specific method and optionally time averaging.
    
    Args:
        df: Results DataFrame.
        method: Method name (e.g., 'OLS', 'WLS', 'GLS').
        time_averaged: If specified, filter by time averaging status.
        
    Returns:
        Filtered and sorted DataFrame.
    """
    # Ensure method name is uppercase
    method = method.upper()
    
    # Build query string
    query_parts = [f'method == "{method}"']
    if time_averaged is not None:
        query_parts.append(f'time_averaged == {time_averaged}')
    
    query_str = ' and '.join(query_parts)
    return df.query(query_str).sort_values('n_steps')


def plot_data_if_exists(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_col: str = 'n_steps',
    y_col: str = 'var',
    marker: str = 'o-',
    **kwargs
) -> None:
    """Plot data on log-log scale if DataFrame is not empty.
    
    Args:
        ax: Axis to plot on.
        data: DataFrame to plot (may be empty).
        x_col: Column name for x-axis.
        y_col: Column name for y-axis.
        marker: Marker style.
        **kwargs: Additional arguments for loglog.
    """
    if not data.empty:
        ax.loglog(data[x_col], data[y_col], marker, 
                 linewidth=kwargs.pop('linewidth', 1),
                 markersize=kwargs.pop('markersize', 2),
                 alpha=kwargs.pop('alpha', 0.8),
                 base=2,
                 **kwargs)


# -----------------------------------------------------------------------------
# Main Plotting Functions
# -----------------------------------------------------------------------------

def plot_all_methods(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (7, 2)
) -> matplotlib.figure.Figure:
    """Create 3-panel plot comparing OLS, WLS, and GLS methods."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.15)
    
    # Get unique values for setup
    max_lag_values = sorted(df['max_lag'].unique())
    colors = get_method_colors(len(max_lag_values), color_range=(0, 1))
    
    # Method configurations: (method, title, scaling_guides)
    methods = [
        ('OLS', '(a) OLS', [([48, 362], 3e-4, -1.0, r'$N^{-1}$', 0.0, -0.3)]),
        ('WLS', '(b) WLS', [([48, 362], 0.6e-4, -1.0, r'$N^{-1}$', 0.0, -0.3),
                            ([90, 362], 0.3e-4, -0.5, r'$N^{-0.5}$', 0.2, 0.1)]),
        ('GLS', '(c) GLS', [([48, 362], 2.2e-5, -1.0, r'$N^{-1}$', 0.0, -0.3)])
    ]
    
    for idx, ((method, title, scaling_guides), ax) in enumerate(zip(methods, axes)):
        # Plot data for each max_lag value
        df_method = filter_method_data(df, method)
        for j, max_lag in enumerate(max_lag_values):
            data = df_method.query(f'max_lag == {max_lag}')
            plot_data_if_exists(ax, data, color=colors[j], 
                              label=f'$\\Delta t_{{\\max}}$ = {max_lag}')
        
        # Add scaling guides
        for guide_params in scaling_guides:
            add_scaling_guide(ax, *guide_params)
        
        format_log_axis(ax, df, show_ylabel=(idx == 0))
        ax.set_title(title)
    
    # Add legend to last subplot
    add_legend(axes[-1], reverse_order=True, outside=True)
    
    plt.tight_layout(rect=[0, 0, 1, 1], w_pad=2.0)
    return fig


def plot_comparison_time_averaging(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (7, 2)
) -> matplotlib.figure.Figure:
    """Create comparison plot for time averaging effect."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.subplots_adjust(wspace=0.15)
  
    # Get diagonal cases
    df_diagonal = get_diagonal_data(df)
    
    # Colors for with/without time averaging
    colors = get_method_colors(2)
    
    # Method configurations: (method, title, show_usler_line)
    methods = [
        ('OLS', '(a) OLS', True),
        ('WLS', '(b) WLS', False),
        ('GLS', '(c) GLS', False)
    ]
    
    for idx, ((method, title, show_usler), ax) in enumerate(zip(methods, axes)):
        # Get data for this method
        df_method = df_diagonal.query(f'method == "{method}"')
        
        # Plot configurations with colors: (time_avg, marker, color, label)
        plot_configs = [
            (False, 's-', colors[1], 'No time avg'),
            (True, 'o-', colors[0], 'With time avg')
        ]
        
        for time_avg, marker, color, label in plot_configs:
            data = df_method[df_method['time_averaged'] == time_avg].sort_values('n_steps')
            plot_data_if_exists(ax, data, marker=marker, color=color, label=label)
        
        # Add Usler line for OLS
        if show_usler:
            add_usler_line(ax)
        
        format_log_axis(ax, df_diagonal, show_ylabel=(idx == 0))
        ax.set_title(title)
    
    # Add legend to last subplot
    add_legend(axes[-1], outside=True)
    
    plt.tight_layout(rect=[0, 0, 1, 1], w_pad=2.0)
    return fig


def plot_ols_vs_wls_sqrtlag(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 2)
) -> matplotlib.figure.Figure:
    """Plot comparing OLS and WLS with sqrt(lag) weights."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get diagonal cases only
    df_diagonal = get_diagonal_data(df)
    
    # Colors
    colors = get_method_colors(4, color_range=(0.1, 0.9))
    
    # Plot configurations: (method, time_avg, marker, color, label)
    plot_configs = [
        ('OLS', False, 's-', colors[0], 'OLS (no time avg)'),
        ('OLS', True, 'o-', colors[1], 'OLS (with time avg)'),
        ('WLS_SQRTLAG', False, '^-', colors[2], 'WLS-SqrtLag (no time avg)'),
        ('WLS_SQRTLAG', True, 'v-', colors[3], 'WLS-SqrtLag (with time avg)')
    ]
    
    for method, time_avg, marker, color, label in plot_configs:
        data = filter_method_data(df_diagonal, method, time_averaged=time_avg)
        plot_data_if_exists(ax, data, marker=marker, color=color, label=label)
    
    # Add Usler reference line
    add_usler_line(ax)
    
    format_log_axis(ax, df_diagonal)
    
    ax.set_title('OLS vs WLS-SqrtLag ($w \\propto 1/\\sqrt{\\Delta t}$)')
    add_legend(ax, outside=True)
    
    plt.tight_layout(pad=0.5)
    return fig


def plot_wls_comparison(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 2)
) -> matplotlib.figure.Figure:
    """Plot comparing WLS with different weight schemes."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter for diagonal, time-averaged cases
    df_filtered = df.query('is_diagonal and time_averaged').copy()
    
    # Colors
    colors = get_method_colors(2)
    
    # Plot configurations: (method, marker, color, label)
    plot_configs = [
        ('WLS_SQRTLAG', 'v-', colors[0], 'WLS-SqrtLag ($w \\propto 1/\\sqrt{\\Delta t}$)'),
        ('WLS', 'o-', colors[1], 'WLS-proper ($w \\propto 1/\\sigma^2$)')
    ]
    
    for method, marker, color, label in plot_configs:
        data = filter_method_data(df_filtered, method)
        plot_data_if_exists(ax, data, marker=marker, color=color, label=label)
    
    # Add Usler reference line
    add_usler_line(ax)
    
    format_log_axis(ax, df_filtered)
    
    ax.set_title('WLS: Effect of Weight Choice (with time averaging)')
    add_legend(ax, outside=True)
    
    plt.tight_layout(pad=0.5)
    return fig


def plot_comprehensive_wls_comparison(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 2)
) -> matplotlib.figure.Figure:
    """Plot comprehensive comparison of OLS, WLS variants, and Usler prediction."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get diagonal cases only
    df_diagonal = get_diagonal_data(df)
    
    # Colors
    colors = get_method_colors(4, color_range=(0.1, 0.9))
    
    # Plot configurations: (method, time_avg, marker, color, label)
    plot_configs = [
        ('OLS', False, 's-', colors[0], 'OLS (no time avg)'),
        ('WLS_SQRTLAG', True, 'v-', colors[1], 'WLS-1/√Δt (with time avg)'),
        ('WLS', False, '^-', colors[2], 'WLS-proper (no time avg)'),
        ('WLS', True, 'o-', colors[3], 'WLS-proper (with time avg)')
    ]
    
    for method, time_avg, marker, color, label in plot_configs:
        data = filter_method_data(df_diagonal, method, time_averaged=time_avg)
        plot_data_if_exists(ax, data, marker=marker, color=color, label=label)
    
    # Add Usler reference line
    add_usler_line(ax)
    
    format_log_axis(ax, df_diagonal)
    
    ax.set_title('WLS Comparison')
    add_legend(ax, outside=True)
    
    plt.tight_layout(pad=0.5)
    return fig
    

def plot_methods_single_panel(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 2)
) -> matplotlib.figure.Figure:
    """Plot OLS, WLS, and GLS on single panel (time-averaged, diagonal only).
    
    Args:
        df: Results DataFrame.
        figsize: Figure size as (width, height).
        
    Returns:
        The created matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter for diagonal, time-averaged cases
    df_filtered = df.query('is_diagonal and time_averaged').copy()
    
    # Colors and markers for each method
    colors = get_method_colors(3, color_range=(0.2, 0.8))
    
    # Plot configurations: (method, marker, color, label)
    plot_configs = [
        ('OLS', 'o-', colors[0], 'OLS', ([48, 362], 3.8e-5, 0.0, r'$N^{0}$', 0.0, +0.1)),
        ('WLS', 's-', colors[1], 'WLS', ([48, 362], 3.2e-5, -0.5, r'$N^{-0.5}$', +0.1, +0.15)),
        ('GLS', '^-', colors[2], 'GLS', ([48, 362], 2e-5, -1.0, r'$N^{-1}$', -0.1, -0.5))
    ]
    
    for method, marker, color, label, scaling_guide_params in plot_configs:
        data = filter_method_data(df_filtered, method)
        plot_data_if_exists(ax, data, marker=marker, color=color, label=label)
        add_scaling_guide(ax, *scaling_guide_params)
    
    # Add Usler reference line
    add_usler_line(ax)
    
    # Format axis
    format_log_axis(ax, df_filtered)
    
    ax.set_title('Method Comparison (time-averaged, diagonal)')
    add_legend(ax, outside=True)
    
    plt.tight_layout()
    return fig