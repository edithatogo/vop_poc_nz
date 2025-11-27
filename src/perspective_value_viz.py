"""
Perspective Value Analysis Visualizations and Reporting

Functions for visualizing and reporting the comprehensive value of perspective analysis:
- Value of Perspective dashboard
- Perspective comparison tables
- Decision discordance visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def plot_perspective_value_dashboard(
    vop_results: Dict,
    output_dir: str = "output/figures/",
    intervention_name: str = "Intervention"
):
    """
    Create comprehensive 4-panel dashboard for Value of Perspective analysis.
    
    Shows all four perspective value metrics and their components.
    
    Args:
        vop_results: Results from calculate_value_of_perspective()
        output_dir: Output directory
        intervention_name: Name of intervention
    """
    from src.visualizations import apply_default_style, save_figure
    
    apply_default_style()
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    wtp = vop_results['wtp_threshold']
    
    # Panel 1: Main metrics comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = [
        'Expected Value of\nPerspective (EVP)',
        'Perspective\nPremium',
        'Decision\nDiscordance Cost',
        'Information\nValue'
    ]
    values = [
        vop_results['expected_value_of_perspective'],
        vop_results['perspective_premium'],
        vop_results['decision_discordance_cost'],
        vop_results['information_value']
    ]
    
    colors = ['steelblue', 'forestgreen', 'coral', 'purple']
    bars = ax1.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        width = bar.get_width()
        label_x = width if width > 0 else 0
        ax1.text(label_x, bar.get_y() + bar.get_height()/2,
                f'  ${val:,.0f}' if abs(val) >= 1 else f'  ${val:.2f}',
                ha='left' if width > 0 else 'right', va='center',
                fontweight='bold', fontsize=10)
    
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Value ($)', fontsize=12)
    ax1.set_title('Perspective Value Metrics Summary', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Panel 2: NMB comparison
    ax2 = fig.add_subplot(gs[0, 1])
    perspectives = ['Health System', 'Societal', 'Optimal\n(Adaptive)']
    nmb_values = [
        vop_results['expected_nmb_health_system'],
        vop_results['expected_nmb_societal'],
        vop_results['expected_value_optimal']
    ]
    
    colors_nmb = ['steelblue', 'forestgreen', 'gold']
    bars2 = ax2.bar(perspectives, nmb_values, color=colors_nmb, alpha=0.7, 
                    edgecolor='black', linewidth=2, width=0.6)
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height,
                f'${height/1e6:.2f}M' if abs(height) >= 1e6 else f'${height/1e3:.0f}K',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=10)
    
    # Highlight chosen perspective
    chosen_idx = 0 if vop_results['chosen_perspective'] == 'health_system' else 1
    bars2[chosen_idx].set_edgecolor('red')
    bars2[chosen_idx].set_linewidth(4)
    
    ax2.set_ylabel(f'Expected NMB @ ${wtp:,}/QALY', fontsize=11)
    ax2.set_title('Expected Value by Perspective', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Panel 3: Decision discordance
    ax3 = fig.add_subplot(gs[1, 0])
    
    discordance_data = [
        vop_results['prob_health_system_optimal'],
        vop_results['prob_societal_optimal'],
        vop_results['proportion_discordant']
    ]
    labels = ['HS Optimal', 'Societal Optimal', 'Discordant\nDecisions']
    
    # Pie chart
    colors_pie = ['steelblue', 'forestgreen', 'coral']
    wedges, texts, autotexts = ax3.pie(
        [discordance_data[0], discordance_data[1], discordance_data[2]],
        labels=labels,
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0, 0.1),
        textprops={'fontsize': 10, 'fontweight': 'bold'}
    )
    
    ax3.set_title(f'Decision Concordance\n({vop_results["proportion_discordant"]*100:.1f}% Discordant)', 
                  fontsize=13, fontweight='bold')
    
    # Panel 4: Summary table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    table_data = [
        ['EVP (Opportunity Loss)', f'${vop_results["expected_value_of_perspective"]:,.0f}'],
        ['Perspective Premium', f'${vop_results["perspective_premium"]:,.0f}'],
        ['Discordance Cost', f'${vop_results["decision_discordance_cost"]:,.0f}'],
        ['Information Value', f'${vop_results["information_value"]:,.0f}'],
        ['', ''],
        ['Correlation (HS ↔ Soc)', f'{vop_results["correlation_hs_soc"]:.3f}'],
        ['Variance (HS)', f'${vop_results["variance_health_system"]:,.0f}'],
        ['Variance (Societal)', f'${vop_results["variance_societal"]:,.0f}'],
        ['', ''],
        ['Chosen Perspective', vop_results['chosen_perspective'].replace('_', ' ').title()],
        ['WTP Threshold', f'${vop_results["wtp_threshold"]:,}/QALY'],
    ]
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight empty rows (separators)
    for i in [6, 10]:
        for j in range(2):
            table[(i, j)].set_facecolor('#F0F0F0')
    
    ax4.set_title('Detailed Metrics', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle(f'Value of Perspective Analysis: {intervention_name}\nWTP = ${wtp:,}/QALY', 
                fontsize=16, fontweight='bold', y=0.98)
    
    save_figure(fig, output_dir, f"perspective_value_dashboard_{intervention_name}")
    plt.close(fig)


def generate_perspective_value_table(
    vop_results: Dict,
    intervention_name: str = "Intervention"
) -> pd.DataFrame:
    """
    Generate formatted table of perspective value metrics for manuscript.
    
    Args:
        vop_results: Results from calculate_value_of_perspective()
        intervention_name: Name of intervention
        
    Returns:
        DataFrame with formatted perspective value metrics
    """
    data = {
        'Intervention': [intervention_name] * 4,
        'Metric': [
            'Expected Value of Perspective (EVP)',
            'Perspective Premium (Societal - HS)',
            'Decision Discordance Cost',
            'Information Value of Perspective Choice'
        ],
        'Value ($)': [
            vop_results['expected_value_of_perspective'],
            vop_results['perspective_premium'],
            vop_results['decision_discordance_cost'],
            vop_results['information_value']
        ],
        'Interpretation': [
            'Opportunity loss from using chosen perspective',
            'Incremental value of societal perspective',
            'Cost when perspectives give conflicting recommendations',
            'Value of knowing which perspective is correct'
        ],
        'Proportion Discordant': [vop_results['proportion_discordant']] * 4,
        'Correlation (HS-Soc)': [vop_results['correlation_hs_soc']] * 4,
    }
    
    df = pd.DataFrame(data)
    
    # Format currency columns
    df['Value ($)'] = df['Value ($)'].apply(lambda x: f'${x:,.0f}')
    df['Proportion Discordant'] = df['Proportion Discordant'].apply(lambda x: f'{x:.1%}')
    df['Correlation (HS-Soc)'] = df['Correlation (HS-Soc)'].apply(lambda x: f'{x:.3f}')
    
    return df


def plot_perspective_comparison_scatter(
    psa_results_hs: pd.DataFrame,
    psa_results_soc: pd.DataFrame,
    wtp_threshold: float = 50000,
    output_dir: str = "output/figures/",
    intervention_name: str = "Intervention"
):
    """
    Scatter plot of NMB: Health System vs Societal perspectives.
    
    Shows correlation and identifies discordant decisions.
    
    Args:
        psa_results_hs: PSA results from health system perspective
        psa_results_soc: PSA results from societal perspective
        wtp_threshold: WTP threshold
        output_dir: Output directory
        intervention_name: Intervention name
    """
    from src.visualizations import apply_default_style, save_figure
    
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
    
    # Calculate NMB
    def _nmb(df):
        inc_qaly = df["qaly_nt"] - df["qaly_sc"]
        inc_cost = df["cost_nt"] - df["cost_sc"]
        return (inc_qaly * wtp_threshold) - inc_cost
    
    nmb_hs = _nmb(psa_results_hs)
    nmb_soc = _nmb(psa_results_soc)
    
    # Identify decision types
    concordant_adopt = (nmb_hs > 0) & (nmb_soc > 0)
    concordant_reject = (nmb_hs <= 0) & (nmb_soc <= 0)
    discordant = ~(concordant_adopt | concordant_reject)
    
    # Plot points
    ax.scatter(nmb_hs[concordant_adopt], nmb_soc[concordant_adopt],
              c='green', alpha=0.5, s=20, label='Concordant Adopt', edgecolors='none')
    ax.scatter(nmb_hs[concordant_reject], nmb_soc[concordant_reject],
              c='red', alpha=0.5, s=20, label='Concordant Reject', edgecolors='none')
    ax.scatter(nmb_hs[discordant], nmb_soc[discordant],
              c='orange', alpha=0.7, s=30, label='Discordant', marker='x', linewidths=2)
    
    # Reference lines
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # 45-degree line (perfect agreement)
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k-', alpha=0.5, linewidth=2, label='Perfect Agreement')
    
    # Correlation
    corr = np.corrcoef(nmb_hs, nmb_soc)[0, 1]
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}\nDiscordant: {discordant.mean():.1%}',
           transform=ax.transAxes, fontsize=12, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(f'NMB: Health System Perspective ($)', fontsize=12)
    ax.set_ylabel(f'NMB: Societal Perspective ($)', fontsize=12)
    ax.set_title(f'Perspective Agreement Analysis: {intervention_name}\nWTP = ${wtp_threshold:,}/QALY',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    save_figure(fig, output_dir, f"perspective_scatter_{intervention_name}")
    plt.close(fig)
