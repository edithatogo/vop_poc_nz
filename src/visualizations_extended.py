"""
Additional Visualization Functions

New visualization capabilities for enhanced analysis reporting:
1. Acceptability Frontier
2. Population EVPI Timeline
3. Threshold Waterfall
4. Multi-Intervention Radar Plot
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def plot_acceptability_frontier(
    ceac_data: pd.DataFrame,
    wtp_range: List[float],
    output_dir: str = "output/figures/",
    intervention_names: Optional[List[str]] = None
):
    """
    Plot the cost-effectiveness acceptability frontier.
    
    Shows which intervention is optimal (highest probability of cost-effectiveness)
    at each WTP threshold.
    
    Args:
        ceac_data: DataFrame with WTP thresholds and probabilities per intervention
        wtp_range: List of WTP thresholds
        output_dir: Output directory for the plot
        intervention_names: List of intervention names
    """
    from .visualizations import apply_default_style, save_figure
    
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    # Assuming ceac_data has columns: wtp, intervention, probability
    # Find optimal intervention at each WTP
    optimal_interventions = []
    optimal_probs = []
    
    for wtp in wtp_range:
        wtp_data = ceac_data[ceac_data['wtp'] == wtp]
        if not wtp_data.empty:
            max_idx = wtp_data['probability'].idxmax()
            optimal_interventions.append(wtp_data.loc[max_idx, 'intervention'])
            optimal_probs.append(wtp_data.loc[max_idx, 'probability'])
        else:
            optimal_interventions.append(None)
            optimal_probs.append(0)
    
    # Plot frontier
    ax.plot(wtp_range, optimal_probs, 'o-', linewidth=2, markersize=8, label='Frontier')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50% Threshold')
    
    # Add intervention labels
    prev_intervention = None
    for i, (wtp, intervention) in enumerate(zip(wtp_range, optimal_interventions)):
        if intervention != prev_intervention and intervention:
            ax.annotate(intervention, (wtp, optimal_probs[i]),
                       textcoords="offset points", xytext=(0,10), ha='center')
            prev_intervention = intervention
    
    ax.set_xlabel('Willingness-to-Pay Threshold ($)')
    ax.set_ylabel('Probability of Cost-Effectiveness')
    ax.set_title('Cost-Effectiveness Acceptability Frontier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "acceptability_frontier")
    plt.close(fig)


def plot_population_evpi_timeline(
    evpi_per_person: float,
    population_per_year: List[int],
    time_horizon: int = 10,
    discount_rate: float = 0.03,
    output_dir: str = "output/figures/"
):
    """
    Plot population EVPI over time with discounting.
    
    Shows the decline in research value as decisions are made and
    population changes over time.
    
    Args:
        evpi_per_person: Per-person EVPI value
        population_per_year: List of population sizes per year
        time_horizon: Number of years to project
        discount_rate: Annual discount rate
        output_dir: Output directory
    """
    from .visualizations import apply_default_style, save_figure
    
    apply_default_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    years = list(range(1, time_horizon + 1))
   
    # Ensure we have population for all years
    if len(population_per_year) < time_horizon:
        population_per_year = population_per_year + [population_per_year[-1]] * (time_horizon - len(population_per_year))
    
    # Calculate annual and cumulative EVPI
    annual_evpi = []
    cumulative_evpi = 0
    cumulative_evpi_list = []
    
    for year in years:
        pop = population_per_year[year - 1]
        discount_factor = (1 + discount_rate) ** (year - 1)
        annual_value = (evpi_per_person * pop) / discount_factor
        annual_evpi.append(annual_value)
        cumulative_evpi += annual_value
        cumulative_evpi_list.append(cumulative_evpi)
    
    # Plot 1: Annual EVPI
    ax1.bar(years, annual_evpi, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Annual Population EVPI ($)')
    ax1.set_title('Annual Value of Research (Discounted)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative EVPI
    ax2.plot(years, cumulative_evpi_list, 'o-', linewidth=2, color='darkgreen', markersize=6)
    ax2.fill_between(years, 0, cumulative_evpi_list, alpha=0.3, color='green')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Population EVPI ($)')
    ax2.set_title(f'Total Research Value (Discounted @ {discount_rate*100:.1f}%)')
    ax2.grid(True, alpha=0.3)
    
    # Add final value annotation
    ax2.annotate(f'Total: ${cumulative_evpi:,.0f}',
                xy=(time_horizon, cumulative_evpi),
                xytext=(time_horizon-2, cumulative_evpi*0.8),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, output_dir, "population_evpi_timeline")
    plt.close(fig)


def plot_threshold_waterfall(
    threshold_results: Dict[str, Dict],
    output_dir: str = "output/figures/",
    base_decision: str = "Cost-Effective"
):
    """
    Plot threshold waterfall showing parameter ranges where decision changes.
    
    Tornado-style visualization highlighting decision-critical parameters.
    
    Args:
        threshold_results: Dict with parameter names and threshold ranges
        output_dir: Output directory
        base_decision: Base case decision
    """
    from .visualizations import apply_default_style, save_figure
    
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Extract parameter names and threshold ranges
    params = []
    lower_bounds = []
    upper_bounds = []
    ranges = []
    
    for param_name, param_data in threshold_results.items():
        params.append(param_name)
        lower = param_data.get('lower_threshold', 0)
        upper = param_data.get('upper_threshold', 0)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        ranges.append(upper - lower)
    
    # Sort by range (largest impact first)
    sorted_indices = np.argsort(ranges)[::-1]
    params = [params[i] for i in sorted_indices]
    lower_bounds = [lower_bounds[i] for i in sorted_indices]
    upper_bounds = [upper_bounds[i] for i in sorted_indices]
    
    y_pos = np.arange(len(params))
    
    # Create horizontal bars
    for i, (param, lower, upper) in enumerate(zip(params, lower_bounds, upper_bounds)):
        ax.barh(i, upper - lower, left=lower, height=0.6,
               color='steelblue' if i % 2 == 0 else 'lightblue',
               edgecolor='black', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params)
    ax.set_xlabel('Parameter Value')
    ax.set_title(f'Decision Threshold Analysis\n(Base Case: {base_decision})')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "threshold_waterfall")
    plt.close(fig)


def plot_intervention_radar(
    intervention_results: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    output_dir: str = "output/figures/",
    normalize: bool = True
):
    """
    Plot multi-intervention comparison using radar/spider plot.
    
    Visualizes trade-offs across multiple dimensions:
    - ICER (inverted so lower is better)
    - NMB (normalized)
    - Equity impact
    - Budget impact
    
    Args:
        intervention_results: Dict of intervention names to results
        metrics: List of metrics to plot
        output_dir: Output directory
        normalize: Whether to normalize metrics to [0, 1]
    """
    from .visualizations import apply_default_style, save_figure
    
    apply_default_style()
    
    if metrics is None:
        metrics = ['NMB', 'Equity', 'Affordability', 'Certainty']
    
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300, subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(intervention_results)))
    
    for idx, (intervention_name, results) in enumerate(intervention_results.items()):
        # Extract values for each metric
        values = []
        for metric in metrics:
            if metric == 'NMB':
                val = results.get('incremental_nmb', 0)
            elif metric == 'Equity':
                equity_data = results.get('dcea_equity_analysis', {})
                val = 1 - equity_data.get('atkinson_index', 0)  # Higher is better
            elif metric == 'Affordability':
                # Inverse of net cost (placeholder)
                val = 100000 / (abs(results.get('incremental_cost', 1)) + 1)
            elif metric == 'Certainty':
                # Probability of cost-effectiveness (placeholder)
                val = 0.75  # Would come from CEAC
            else:
                val = 0
            
            values.append(val)
        
        # Normalize if requested
        if normalize and values:
            max_val = max(max(values), 1)
            values = [v / max_val for v in values]
        
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=intervention_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1 if normalize else None)
    ax.set_title('Multi-Dimensional Intervention Comparison', size=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    save_figure(fig, output_dir, "intervention_radar")
    plt.close(fig)
