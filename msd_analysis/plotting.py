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

def format_log_axis(
    ax: plt.Axes, 
    n_steps_values: np.ndarray,
    xlabel: str = '$t_\\mathrm{sim}$',
    ylabel: str = '$\\sigma^2[\\widehat{D}^*]$',
    show_ylabel: bool = True
) -> None:
    """Apply common formatting for log-log plot axes.
    
    Args:
        ax: Matplotlib axis to format.
        n_steps_values: Values for x-axis ticks.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        show_ylabel: Whether to show y-axis label.
    """
    ax.set_box_aspect(1)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks(n_steps_values[::2])
    ax.set_xticklabels(n_steps_values[::2])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel if show_ylabel else '')


def add_scaling_guide(
    ax: plt.Axes,
    x_range: list[float],
    y_position: float,
    exponent: float = -1.0,
    label: str | None = None
) -> None:
    """Add a scaling guide line to show theoretical variance scaling.
    
    Args:
        ax: Matplotlib axis to add guide to.
        x_range: [start, end] x-coordinates for the guide line.
        y_position: Y-position for the guide line.
        exponent: Scaling exponent (e.g., -1.0 for N^-1 scaling).
        label: Optional label for the scaling guide.
    """
    x_theory = np.array(x_range)
    y_theory = y_position * (256 / x_theory**(-exponent))
    ax.loglog(x_theory, y_theory, 'k-', alpha=0.5)
    
    if label:
        # Position label appropriately based on exponent
        if exponent == -1.0:
            x_label = x_range[0] * 2
            y_label = y_position * 0.9
        else:
            x_label = 160
            y_label = y_position * 23
        ax.text(x_label, y_label, label, fontsize=8)


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


# -----------------------------------------------------------------------------
# Main Plotting Functions
# -----------------------------------------------------------------------------

def plot_all_methods(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (7, 2.5)
) -> matplotlib.figure.Figure:
    """Create 3-panel plot comparing OLS, WLS, and GLS methods.
    
    Args:
        df: Results DataFrame with columns: method, n_steps, max_lag, var.
        figsize: Figure size as (width, height).
        
    Returns:
        The created matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    
    # Get unique values for setup
    n_steps_values = sorted(df['n_steps'].unique())
    max_lag_values = sorted(df['max_lag'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(max_lag_values)))
    
    # Method configurations: (method, title, scaling_guides)
    # Each scaling guide is: (x_range, y_position, exponent, label)
    methods = [
        ('OLS', '(a) OLS', [([48, 362], 3e-4, -1.0, r'$N^{-1}$')]),
        ('WLS', '(b) WLS', [([48, 362], 0.6e-4, -1.0, r'$N^{-1}$'),
                           ([90, 362], 0.3e-4, -0.5, r'$N^{-0.5}$')]),
        ('GLS', '(c) GLS', [([48, 362], 2.2e-5, -1.0, r'$N^{-1}$')])
    ]
    
    for idx, (method, title, scaling_guides) in enumerate(methods, 1):
        ax = plt.subplot(1, 3, idx)
        
        # Plot data for each max_lag value
        df_method = filter_method_data(df, method)
        for j, max_lag in enumerate(max_lag_values):
            data = df_method.query(f'max_lag == {max_lag}')
            if not data.empty:
                ax.loglog(data['n_steps'], data['var'], 'o-',
                         color=colors[j],
                         label=f'$\\Delta t_\\mathrm{{max}}$ = {max_lag}',
                         linewidth=1, markersize=2, alpha=0.8)
        
        # Add scaling guides
        for x_range, y_pos, exp, guide_label in scaling_guides:
            add_scaling_guide(ax, x_range, y_pos, exp, guide_label)
        
        # Format axis
        format_log_axis(ax, n_steps_values, show_ylabel=(idx == 1))
        ax.set_title(title)
    
    # Add legend to last subplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], 
              bbox_to_anchor=(1.1, 1.05), fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_comparison_time_averaging(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (7, 2.5)
) -> matplotlib.figure.Figure:
    """Create comparison plot for time averaging effect.
    
    Args:
        df: Combined results DataFrame with time_averaged column.
        figsize: Figure size as (width, height).
        
    Returns:
        The created matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    
    # Get diagonal cases and n_steps values
    df_diagonal = get_diagonal_data(df)
    n_steps_values = sorted(df_diagonal['n_steps'].unique())
    
    # Colors for with/without time averaging
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 2))
    
    # Method configurations: (method, title, show_usler_line)
    methods = [
        ('OLS', '(a) OLS', True),   # Include Usler line
        ('WLS', '(b) WLS', False),
        ('GLS', '(c) GLS', False)
    ]
    
    for idx, (method, title, show_usler) in enumerate(methods, 1):
        ax = plt.subplot(1, 3, idx)
        
        # Get data for this method
        df_method = df_diagonal.query(f'method == "{method}"')
        
        # Plot without time averaging
        no_avg = df_method.query('not time_averaged').sort_values('n_steps')
        if not no_avg.empty:
            ax.loglog(no_avg['n_steps'], no_avg['var'], 's-',
                     color=colors[1], label='No time avg',
                     linewidth=1, markersize=2, alpha=0.8)
        
        # Plot with time averaging
        with_avg = df_method.query('time_averaged').sort_values('n_steps')
        if not with_avg.empty:
            ax.loglog(with_avg['n_steps'], with_avg['var'], 'o-',
                     color=colors[0], label='With time avg',
                     linewidth=1, markersize=2, alpha=0.8)
        
        # Add Usler line for OLS
        if show_usler:
            add_usler_line(ax)
        
        # Format axis
        xlabel = '$t_\\mathrm{sim} = \\Delta t_\\mathrm{max}$'
        format_log_axis(ax, n_steps_values, xlabel=xlabel, show_ylabel=(idx == 1))
        ax.set_title(title)
        
        # Add legend to last subplot
        if idx == 3:
            ax.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    return fig


def plot_ols_vs_wls_sqrtlag(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 2)
) -> matplotlib.figure.Figure:
    """Plot comparing OLS and WLS with sqrt(lag) weights.
    
    Args:
        df: Combined results DataFrame.
        figsize: Figure size as (width, height).
        
    Returns:
        The created matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get diagonal cases only
    df_diagonal = get_diagonal_data(df)
    
    # Define plot configurations: (method, time_avg, marker, color_idx, label)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, 4))
    plot_configs = [
        ('OLS', False, 's-', 0, 'OLS (no time avg)'),
        ('OLS', True, 'o-', 1, 'OLS (with time avg)'),
        ('WLS_SQRTLAG', False, '^-', 2, 'WLS-SqrtLag (no time avg)'),
        ('WLS_SQRTLAG', True, 'v-', 3, 'WLS-SqrtLag (with time avg)')
    ]
    
    for method, time_avg, marker, color_idx, label in plot_configs:
        data = filter_method_data(df_diagonal, method, time_averaged=time_avg)
        if not data.empty:
            ax.loglog(data['n_steps'], data['var'], marker,
                     color=colors[color_idx], label=label,
                     linewidth=1, markersize=2, alpha=0.8)
    
    # Add Usler reference line
    add_usler_line(ax)
    
    # Format axis
    n_steps_values = sorted(df_diagonal['n_steps'].unique())
    xlabel = '$t_\\mathrm{sim} = \\Delta t_\\mathrm{max}$'
    format_log_axis(ax, n_steps_values, xlabel=xlabel)
    
    ax.set_title('OLS vs WLS-SqrtLag ($w \\propto 1/\\sqrt{\\Delta t}$)')
    ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize=8)
    
    plt.tight_layout()
    return fig


def plot_wls_comparison(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (4, 2)
) -> matplotlib.figure.Figure:
    """Plot comparing WLS with different weight schemes.
    
    Args:
        df: Results DataFrame.
        figsize: Figure size as (width, height).
        
    Returns:
        The created matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter for diagonal, time-averaged cases
    df_filtered = df.query('is_diagonal and time_averaged').copy()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 2))
    
    # Plot configurations: (method, marker, color, label)
    plot_configs = [
        ('WLS_SQRTLAG', 'v-', colors[0], 'WLS-SqrtLag ($w \\propto 1/\\sqrt{\\Delta t}$)'),
        ('WLS', 'o-', colors[1], 'WLS-proper ($w \\propto 1/\\sigma^2$)')
    ]
    
    for method, marker, color, label in plot_configs:
        data = filter_method_data(df_filtered, method)
        if not data.empty:
            ax.loglog(data['n_steps'], data['var'], marker,
                     color=color, label=label,
                     linewidth=1, markersize=3, alpha=0.8)
    
    # Add Usler reference line
    add_usler_line(ax)
    
    # Format axis
    n_steps_values = sorted(df_filtered['n_steps'].unique())
    xlabel = '$t_\\mathrm{sim} = \\Delta t_\\mathrm{max}$'
    format_log_axis(ax, n_steps_values, xlabel=xlabel)
    
    ax.set_title('WLS: Effect of Weight Choice (with time averaging)')
    ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize=8)
    
    plt.tight_layout()
    return fig
    
def plot_comprehensive_wls_comparison(
        df: pd.DataFrame,
        figsize: tuple[float, float] = (5, 2)
    ) -> matplotlib.figure.Figure:
        """Plot comprehensive comparison of OLS, WLS variants, and Usler prediction.
        
        Shows:
        - OLS (no time avg)
        - WLS with 1/sqrt(lag) weights (with time avg) 
        - Usler formula prediction
        - Proper WLS (no time avg)
        - Proper WLS (with time avg)
        
        Args:
            df: Combined results DataFrame with all methods and conditions.
            figsize: Figure size as (width, height).
            
        Returns:
            The created matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get diagonal cases only
        df_diagonal = get_diagonal_data(df)
        
        # Use viridis colormap for consistency with other plots
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
        
        # Plot configurations: (method, time_avg, marker, color_idx, label)
        plot_configs = [
            ('OLS', False, 's-', 0, 'OLS (no time avg)'),
            ('WLS_SQRTLAG', True, 'v-', 1, 'WLS-1/√Δt (with time avg)'),
            ('WLS', False, '^-', 2, 'WLS-proper (no time avg)'),
            ('WLS', True, 'o-', 3, 'WLS-proper (with time avg)')
        ]
        
        for method, time_avg, marker, color_idx, label in plot_configs:
            data = filter_method_data(df_diagonal, method, time_averaged=time_avg)
            if not data.empty:
                ax.loglog(data['n_steps'], data['var'], marker,
                         color=colors[color_idx], label=label,
                         linewidth=1, markersize=2, alpha=0.8)
        
        # Add Usler reference line using consistent colour
        add_usler_line(ax)
        
        # Format axis using the standard formatting function
        n_steps_values = sorted(df_diagonal['n_steps'].unique())
        xlabel = '$t_\\mathrm{sim} = \\Delta t_\\mathrm{max}$'
        format_log_axis(ax, n_steps_values, xlabel=xlabel)
        
        ax.set_title('Comprehensive WLS Comparison')
        ax.legend(bbox_to_anchor=(1.1, 1.05), fontsize=8)
        
        plt.tight_layout()
        return fig
