Plot Feature Matrix
===================

Matrix of plots by analysis type, visualization style, perspective, and implementation status. Scope indicates whether the plot is built per intervention or comparative across interventions.

| Plot | Analysis Domain | Visualization Type | Perspective | Scope | Function/Location | Status | Notes/Next Steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | CEA Structure | Graphviz tree | N/A | Single | `plot_decision_tree` (`src/visualizations.py`) | Implemented | Outputs PNG/PDF for each intervention. |
| Cost-Effectiveness Plane | CEA | Scatter + WTP line | Societal (current) | Comparative | `plot_cost_effectiveness_plane` (`src/visualizations.py`) | Implemented | Needs HS/Both perspective handling in unified version. |
| CEAC | VOI | Line | Societal | Comparative | `plot_ceac` (`src/visualizations.py`) | Implemented | Uses PSA CEAC; assumes societal data. |
| CEAF | VOI | Line | Societal | Comparative | `plot_ceaf` (`src/visualizations.py`) | Implemented | Frontier overlay included. |
| EVPI | VOI | Line | Societal | Comparative | `plot_evpi` (`src/visualizations.py`) | Implemented | Per-person EVPI per WTP. |
| Population EVPI | VOI | Line | Societal | Comparative | `plot_pop_evpi` (`src/visualizations.py`) | Implemented | Uses assumed population sizes. |
| EVPPI | VOI | Line | Societal | Comparative | `plot_evppi` (`src/visualizations.py`) | Implemented | By parameter group. |
| Net Benefit Curves | CEA | Line + ribbon | Societal | Comparative | `plot_net_benefit_curves` (`src/visualizations.py`) | Implemented | Mean and 95% CI across WTPs. |
| Value of Perspective | CEA | Bar | Societal | Comparative | `plot_value_of_perspective` (`src/visualizations.py`) | Implemented | Probability CE at $50k WTP. |
| Comparative 2-Way DSA | DSA | Histograms | HS/Soc | Comparative | `plot_comparative_two_way_dsa` (`src/visualizations.py`) | Implemented | Uses comparative grids. |
| Comparative 3-Way DSA | DSA | Histograms | HS/Soc | Comparative | `plot_comparative_three_way_dsa` (`src/visualizations.py`) | Implemented | Pairwise comps across 3-way DSA. |
| One-Way DSA Tornado | DSA | Tornado bars | HS/Soc | Single | `plot_one_way_dsa_tornado` (`src/dsa_analysis.py`) | Implemented | Saves to `data/data_outputs/figures/`. |
| Two-Way DSA Heatmaps | DSA | Heatmaps | HS/Soc | Single | `plot_two_way_dsa_heatmaps` (`src/dsa_analysis.py`) | Implemented | Per-parameter pair. |
| Three-Way DSA 3D | DSA | 3D surface | HS/Soc | Single | `plot_three_way_dsa_3d` (`src/dsa_analysis.py`) | Implemented | Plots slices for parameter 3. |
| Equity Impact Plane | DCEA | Scatter | Societal equity | Single | `plot_equity_impact_plane` (`src/dcea_equity_analysis.py`) | Implemented | Efficiency vs equity trade-off. |
| Lorenz Curve | DCEA | Lorenz curve | Societal equity | Single | `plot_lorenz_curve` (`src/dcea_equity_analysis.py`) | Implemented | Health gains distribution. |
| Cluster Analysis | Clustering | Multi-panel | N/A | Single | `plot_cluster_analysis` (`src/visualizations.py`) | Implemented | PCA scatter, bars, CE plane by cluster. |
| Comparative Clusters | Clustering | Multi-panel | N/A | Comparative | `plot_comparative_clusters` (`src/visualizations.py`) | Implemented | Sizes, silhouette, archetypes. |
| Decision Discordance Matrix | Concordance | Matrix (todo) | HS/Soc | Comparative | `plot_decision_reversal_matrix` (`src/visualizations.py`) | Implemented | Uses perspective-specific NMB. |
| Markov Trace (Cohort) | CEA Dynamics | Line/stacked area | HS/Soc | Single | `plot_markov_trace` (`src/visualizations.py`) | Implemented | Plot state occupancy over cycles. |
| Disaggregated Cost & QALY Breakdown | CEA | Stacked bar | HS/Soc | Single | `plot_cost_qaly_breakdown` (`src/visualizations.py`) | Implemented | Cost/QALY components by perspective. |
| Hexbin or KDE CE Plane | CEA | Hexbin/KDE | HS/Soc | Comparative | `plot_density_ce_plane` (`src/visualizations.py`) | Implemented | Density view of PSA on CE plane. |
| Forest Plot (Subgroup) | Heterogeneity | Forest | HS/Soc | Comparative | `plot_subgroup_forest` (`src/visualizations.py`) | Implemented | Effect/ICER by subgroup. |
| Annual Cash Flow (BIA) | BIA | Line/bar | HS/Soc/BIA | Single | `plot_annual_cash_flow` (`src/visualizations.py`) | Implemented | Budget impact over years. |
| Expansion Path (Efficiency Frontier) | CEA Frontier | Frontier line | HS/Soc | Comparative | `plot_efficiency_frontier` (`src/visualizations.py`) | Implemented | Trace efficient set across WTP. |
| Time-to-ROI (Cumulative NMB) | CEA/BIA | Cumulative line | HS/Soc | Single | `plot_cumulative_nmb` (`src/visualizations.py`) | Implemented | Cycles to break-even. |
| Scenario Waterfall | Scenario | Waterfall | HS/Soc | Comparative | `plot_scenario_waterfall` (`src/visualizations.py`) | Implemented | Incremental drivers by scenario. |
| Survival GOF (Parametric vs KM) | Survival | Overlay | HS/Soc | Single | `plot_survival_gof` (`src/visualizations.py`) | Implemented | KM overlaid with parametric fits. |
| PSA Convergence | VOI QA | Line/scatter | HS/Soc | Single | `plot_psa_convergence` (`src/visualizations.py`) | Implemented | ESS/stability vs iterations. |
| Validation/Calibration Plot | Model QA | Scatter/line | HS/Soc | Single | `plot_model_calibration` (`src/visualizations.py`) | Implemented | Observed vs predicted targets. |
| Rank Probability (Rankogram) | VOI | Rank bars | HS/Soc | Comparative | `plot_rankogram` (`src/visualizations.py`) | Implemented | Probability of rank by WTP. |
| Health Concentration Curve | Equity | Lorenz-style | Societal equity | Single | Partial | Existing Lorenz plot; may need income ranking version. |
| Price Acceptability Curve | Pricing | Line | HS/Soc | Single | `plot_price_acceptability_curve` (`src/visualizations.py`) | Implemented | Max acceptable price vs target CE. |
| Resource Capacity Constraint | Operational | Line/area | HS/Soc | Single | `plot_resource_constraint` (`src/visualizations.py`) | Implemented | Bottleneck utilization over time. |
| Expected Loss (Regret) Curve | VOI | Line | HS/Soc | Comparative | `plot_expected_loss_curve` (`src/visualizations.py`) | Implemented | Expected loss vs WTP. |
| Life-Years vs QALYs Decomposition | Outcomes | Scatter/bar | HS/Soc | Single | `plot_ly_vs_qaly` (`src/visualizations.py`) | Implemented | Split LY and QALY gains. |
| Equity-Efficiency Impact Plane | DCEA | Scatter/quadrants | Societal | Comparative | `plot_equity_efficiency_plane` (`src/visualizations.py`) | Implemented | Similar to equity impact plane but pairwise. |
| Staircase of Inequality | DCEA | Step/stack | Societal | Single | `plot_inequality_staircase` (`src/visualizations.py`) | Implemented | Decompose inequality contributions. |
| Financial Risk Protection (ECEA) | ECEA | Bar/line | Societal | Single | `plot_financial_risk_protection` (`src/visualizations.py`) | Implemented | Catastrophic/impov results. |
| Inequality Aversion (Atkinson) | DCEA | Curve | Societal | Single | `plot_inequality_aversion_curve` (`src/visualizations.py`) | Implemented | Welfare vs epsilon. |
| Net Cash Flow Waterfall | BIA | Waterfall | HS/Soc | Single | `plot_net_cash_flow_waterfall` (`src/visualizations.py`) | Implemented | Cash inflow/outflow components. |
| Affordability Ribbon (BIA Sensitivity) | BIA | Ribbon | HS/Soc | Single | `plot_affordability_ribbon` (`src/visualizations.py`) | Implemented | Budget impact under sensitivity ranges. |
| Threshold Crossing Plot | Threshold | Line/scatter | HS/Soc | Comparative | `plot_threshold_crossing` (`src/visualizations.py`) | Implemented | Where ICER/NMB crosses WTP. |
| Structural Waterfall (Scenario) | Structural SA | Waterfall | HS/Soc | Comparative | `plot_structural_waterfall` (`src/visualizations.py`) | Implemented | Structural assumptions impact. |
| Decision Reversal Matrix | Concordance | Heatmap | HS/Soc | Comparative | `plot_decision_reversal_matrix` (`src/visualizations.py`) | Implemented | Reversals across assumptions/WTP. |
| Societal Driver Decomposition | CEA | Tornado/bar | Societal | Single | `plot_societal_drivers` (`src/visualizations.py`) | Implemented | Contribution of societal components. |
| Expected Value of Perspective (EVP) Curve | VOI | Line | HS vs Soc | Comparative | `plot_evp_curve` (`src/visualizations.py`) | Implemented | EVP across WTP. |
| Structural Sensitivity Tornado | Structural SA | Tornado | HS/Soc | Single | `plot_structural_tornado` (`src/visualizations.py`) | Implemented | Treat perspective/structure as parameters. |

Suggested additional matrix columns to consider:
- Data Inputs Required (e.g., PSA draws, subgroup metadata, survival curves).
- Output Filename/Path (to standardize naming).
- Automation Trigger (function call site or CLI flag).
- Test Coverage (unit/visual regression planned).
- Status Owner/Priority (for scheduling delivery).
