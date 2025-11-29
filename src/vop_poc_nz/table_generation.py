"""
Table Generation Module.

This module handles the generation of Markdown tables for the NZMJ manuscript and supplements.
It covers Core Tables, DCEA Tables, BIA Tables, and VOI Tables.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def dataframe_to_markdown(df):
    """Convert a pandas DataFrame to a Markdown table string."""
    return df.to_markdown(index=False)


def generate_core_tables(params, output_dir):
    """
    Generate Core Tables (1-9) based on parameters and static content.
    """
    tables = {}

    # Table 1: Study Setting
    # Extract common parameters
    first_intervention = next(iter(params.values())) if params else {}
    discount_rate = first_intervention.get("discount_rate", 0.03)

    table_1_data = [
        {"Item": "Jurisdiction", "Description": "Aotearoa New Zealand"},
        {
            "Item": "Perspectives",
            "Description": "Health System (Base Case), Societal (Scenario)",
        },
        {
            "Item": "Target Population",
            "Description": "Varies by intervention (see Table 22)",
        },
        {"Item": "Time Horizon", "Description": "Lifetime (CEA), 5 Years (BIA)"},
        {
            "Item": "Discount Rate",
            "Description": f"{discount_rate:.1%} per annum (Costs and QALYs)",
        },
        {"Item": "Currency / Price Year", "Description": "NZD 2025"},
        {"Item": "WTP Threshold", "Description": "NZ$50,000 / QALY (Base Case)"},
    ]
    tables["Table_1_Study_Setting"] = dataframe_to_markdown(pd.DataFrame(table_1_data))

    # Table 2: Interventions
    rows_2 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        rows_2.append(
            {
                "Intervention": name.replace("_", " ").title(),
                "Description": "See Methods Supplement for full description",
                "Comparator": "Standard Care / Status Quo",
            }
        )
    tables["Table_2_Interventions"] = dataframe_to_markdown(pd.DataFrame(rows_2))

    # Table 3: Baseline Epidemiology (extracted from initial_population/states)
    rows_3 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        states = p.get("states", [])
        init_pop = p.get("initial_population", [])
        total_pop = sum(init_pop)
        if total_pop > 0:
            dist = [
                f"{s}: {n / total_pop:.1%}" for s, n in zip(states, init_pop) if n > 0
            ]
            rows_3.append(
                {
                    "Intervention": name.replace("_", " ").title(),
                    "Baseline State Distribution": ", ".join(dist),
                }
            )
    tables["Table_3_Baseline_Epidemiology"] = dataframe_to_markdown(
        pd.DataFrame(rows_3)
    )

    # Table 6: Unit Costs (Sample)
    rows_6 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        states = p.get("states", [])
        costs_hs = p.get("costs", {}).get("health_system", {}).get("new_treatment", [])
        costs_soc = p.get("costs", {}).get("societal", {}).get("new_treatment", [])

        for i, state in enumerate(states):
            if i < len(costs_hs) and costs_hs[i] > 0:
                rows_6.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Category": "Health System",
                        "State/Item": state,
                        "Unit Cost (Annual)": f"${costs_hs[i]:,.0f}",
                    }
                )
            if i < len(costs_soc) and costs_soc[i] > 0:
                rows_6.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Category": "Societal (Productivity/Other)",
                        "State/Item": state,
                        "Unit Cost (Annual)": f"${costs_soc[i]:,.0f}",
                    }
                )
    tables["Table_6_Unit_Costs"] = dataframe_to_markdown(pd.DataFrame(rows_6))

    # Table 7: Utilities
    rows_7 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        states = p.get("states", [])
        qalys = p.get("qalys", {}).get("new_treatment", [])
        for i, state in enumerate(states):
            if i < len(qalys):
                rows_7.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Health State": state,
                        "Utility Weight": f"{qalys[i]:.2f}",
                    }
                )
    tables["Table_7_Health_State_Utilities"] = dataframe_to_markdown(
        pd.DataFrame(rows_7)
    )

    # Table 11: Parameter Distributions (from dsa_parameter_ranges)
    # Assuming dsa_parameter_ranges is in the first intervention or global
    dsa_params = first_intervention.get("dsa_parameter_ranges", {})
    if dsa_params:
        rows_11 = []
        for param, details in dsa_params.items():
            rows_11.append(
                {
                    "Parameter": param.replace("_", " ").title(),
                    "Base Value": details.get("base", "-"),
                    "Range/Distribution": f"{details.get('range', '-')}",
                    "Source": details.get("source", "-"),
                }
            )
        tables["Table_11_Parameter_Distributions"] = dataframe_to_markdown(
            pd.DataFrame(rows_11)
        )
        tables["Table_12_DSA_Specifications"] = dataframe_to_markdown(
            pd.DataFrame(rows_11)
        )  # Similar content

    return tables


def generate_qualitative_tables(params, output_dir):
    """Generate Qualitative Tables (4, 5, 8, 9)."""
    tables = {}

    # Table 4: Treatment effects (Placeholder/Extraction)
    rows_4 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        effect = p.get("treatment_effect", {})
        if effect:
            desc = f"RR/OR: {effect.get('value', '-')} (Source: {effect.get('source', 'Literature')})"
        else:
            desc = "See Methods Supplement"

        rows_4.append(
            {
                "Intervention": name.replace("_", " ").title(),
                "Effect Measure": "Relative Risk / Hazard Ratio",
                "Estimate": desc,
                "Source": "Clinical Trials / Meta-analysis",
            }
        )
    tables["Table_4_Treatment_Effects"] = dataframe_to_markdown(pd.DataFrame(rows_4))

    # Table 5: Resource Use (Placeholder)
    rows_5 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        rows_5.append(
            {
                "Intervention": name.replace("_", " ").title(),
                "Resource Item": "Intervention Delivery",
                "Quantity": "1 per person (course)",
                "Source": "Program Guidelines",
            }
        )
    tables["Table_5_Resource_Use"] = dataframe_to_markdown(pd.DataFrame(rows_5))

    # Table 8: Model Structure
    rows_8 = [
        {"Feature": "Model Type", "Description": "Markov Cohort Model"},
        {"Feature": "Cycle Length", "Description": "1 Year"},
        {"Feature": "Time Horizon", "Description": "Lifetime (100 Years)"},
        {"Feature": "Perspective", "Description": "Health System & Societal"},
        {"Feature": "Discounting", "Description": "3% per annum (Costs & Benefits)"},
        {"Feature": "Half-cycle Correction", "Description": "Applied"},
    ]
    tables["Table_8_Model_Structure"] = dataframe_to_markdown(pd.DataFrame(rows_8))

    # Table 9: Key Assumptions
    rows_9 = [
        {
            "Assumption": "Adherence",
            "Rationale": "100% adherence assumed for initial treatment phase (simplification).",
        },
        {
            "Assumption": "Waning Effect",
            "Rationale": "Treatment effect constant over time unless specified otherwise.",
        },
        {
            "Assumption": "General Mortality",
            "Rationale": "Based on NZ Life Tables 2020-2022.",
        },
        {"Assumption": "Costs", "Rationale": "All costs adjusted to 2025 NZD."},
    ]
    tables["Table_9_Key_Assumptions"] = dataframe_to_markdown(pd.DataFrame(rows_9))

    return tables


def generate_cea_tables(results, output_dir):
    """Generate CEA Results Tables (10-15)."""
    tables = {}

    # Table 10: Base-case costs, effects, and ICERs
    rows = []
    for name, res in results["intervention_results"].items():
        for perspective in ["health_system", "societal"]:
            if perspective in res:
                data = (
                    res[perspective].get("human_capital", res[perspective])
                    if perspective == "societal"
                    else res[perspective]
                )

                rows.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Perspective": perspective.replace("_", " ").title(),
                        "Total Costs (NT)": f"${data['cost_new_treatment']:,.0f}",
                        "Total QALYs (NT)": f"{data['qalys_new_treatment']:.2f}",
                        "Inc. Costs": f"${data['incremental_cost']:,.0f}",
                        "Inc. QALYs": f"{data['incremental_qalys']:.3f}",
                        "ICER": "Dominant"
                        if data["incremental_cost"] < 0
                        and data["incremental_qalys"] > 0
                        else f"${data['icer']:,.0f}"
                        if data["incremental_qalys"] > 0
                        else "Dominated",
                        "NMB (50k)": f"${data['incremental_nmb']:,.0f}",
                    }
                )

    df_10 = pd.DataFrame(rows)
    tables["Table_10_Base_Case_CEA"] = dataframe_to_markdown(df_10)

    # Table 13: Key DSA Results
    dsa_rows = []
    if "dsa_analysis" in results and "1_way" in results["dsa_analysis"]:
        for name, dsa_res in results["dsa_analysis"]["1_way"].items():
            for perspective in ["health_system", "societal"]:
                if perspective in dsa_res:
                    res_p = dsa_res[perspective]
                    sorted_params = sorted(
                        res_p.items(), key=lambda x: x[1]["range"], reverse=True
                    )[:5]
                    for param, vals in sorted_params:
                        dsa_rows.append(
                            {
                                "Intervention": name.replace("_", " ").title(),
                                "Perspective": perspective.replace("_", " ").title(),
                                "Parameter": param,
                                "Low Input": f"{vals['low_input']:.2f}",
                                "High Input": f"{vals['high_input']:.2f}",
                                "NMB Range": f"${vals['range']:,.0f}",
                            }
                        )
    if dsa_rows:
        tables["Table_13_Key_DSA_Results"] = dataframe_to_markdown(
            pd.DataFrame(dsa_rows)
        )

    # Table 14: PSA Summary
    psa_rows = []
    if "probabilistic_results" in results:
        for name, psa_res in results["probabilistic_results"].items():
            for perspective in ["health_system", "societal"]:
                if perspective in psa_res:
                    data = psa_res[perspective]
                    costs = np.array(data["incremental_costs"])
                    qalys = np.array(data["incremental_qalys"])
                    nmb = (qalys * 50000) - costs
                    prob_ce = np.mean(nmb > 0)

                    psa_rows.append(
                        {
                            "Intervention": name.replace("_", " ").title(),
                            "Perspective": perspective.replace("_", " ").title(),
                            "Mean Inc. Cost (95% CI)": f"${np.mean(costs):,.0f} (${np.percentile(costs, 2.5):,.0f} - ${np.percentile(costs, 97.5):,.0f})",
                            "Mean Inc. QALYs (95% CI)": f"{np.mean(qalys):,.3f} ({np.percentile(qalys, 2.5):,.3f} - {np.percentile(qalys, 97.5):,.3f})",
                            "Prob. CE @ $50k": f"{prob_ce:.1%}",
                        }
                    )
    if psa_rows:
        tables["Table_14_PSA_Summary"] = dataframe_to_markdown(pd.DataFrame(psa_rows))

    return tables


def generate_subgroup_tables(results, output_dir):
    """Generate Subgroup Tables (15)."""
    tables = {}

    # Table 15: Subgroup Results
    # Attempting to extract from DCEA results if explicit subgroup results are missing
    rows_15 = []
    if "dcea_equity_analysis" in results:
        for name, dcea_res_dict in results["dcea_equity_analysis"].items():
            if not dcea_res_dict:
                continue
            # Use Societal perspective for subgroup breakdown
            res = dcea_res_dict.get("societal", {})
            if "distribution_of_net_health_benefits" in res:
                dist = res["distribution_of_net_health_benefits"]
                # dist is likely {group: value}
                for group, nhb in dist.items():
                    rows_15.append(
                        {
                            "Intervention": name.replace("_", " ").title(),
                            "Subgroup": group,
                            "Net Health Benefit": f"{nhb:.2f}",
                            "Cost-Effective?": "Likely (Positive NHB)"
                            if nhb > 0
                            else "Unlikely",
                        }
                    )

    if rows_15:
        tables["Table_15_Subgroup_Results"] = dataframe_to_markdown(
            pd.DataFrame(rows_15)
        )

    return tables


def generate_dcea_tables(results, output_dir):
    """Generate DCEA Tables (16-21)."""
    tables = {}

    if "dcea_equity_analysis" not in results:
        return tables

    # Table 16: Equity-relevant population stratification (from params)
    # We need to access params, but this function only gets results.
    # Ideally params should be passed or stored in results.
    # Assuming we can infer from results keys if they contain group names

    # Table 19: Post-intervention outcomes by group
    dcea_rows = []
    for name, dcea_res_dict in results["dcea_equity_analysis"].items():
        if not dcea_res_dict:
            continue
        for perspective, dcea_res in dcea_res_dict.items():
            if not dcea_res:
                continue
            if "outcomes_by_group" in dcea_res:
                for group, outcomes in dcea_res["outcomes_by_group"].items():
                    dcea_rows.append(
                        {
                            "Intervention": name.replace("_", " ").title(),
                            "Perspective": perspective.replace("_", " ").title(),
                            "Group": group,
                            "Inc. QALYs": f"{outcomes.get('incremental_qalys', 0):.4f}",
                            "Inc. Costs": f"${outcomes.get('incremental_costs', 0):,.0f}",
                            "Net Health Benefit": f"{outcomes.get('net_health_benefit', 0):.4f}",
                        }
                    )
    if dcea_rows:
        tables["Table_19_DCEA_Outcomes_by_Group"] = dataframe_to_markdown(
            pd.DataFrame(dcea_rows)
        )

    # Table 20: Equity Impact Metrics
    metrics_rows = []
    for name, dcea_res_dict in results["dcea_equity_analysis"].items():
        if not dcea_res_dict:
            continue
        for perspective, dcea_res in dcea_res_dict.items():
            if not dcea_res:
                continue
            if "equity_metrics" in dcea_res:
                metrics = dcea_res["equity_metrics"]
                metrics_rows.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Perspective": perspective.replace("_", " ").title(),
                        "Atkinson Index (Health)": f"{metrics.get('atkinson_index', 0):.4f}",
                        "Health Achievement Index": f"{metrics.get('health_achievement_index', 0):.4f}",
                        "Equity-Weighted NMB": f"${metrics.get('equity_weighted_nmb', 0):,.0f}",
                    }
                )
    if metrics_rows:
        tables["Table_20_Equity_Impact_Metrics"] = dataframe_to_markdown(
            pd.DataFrame(metrics_rows)
        )

    return tables


def generate_extended_dcea_tables(results, params, output_dir):
    """Generate Extended DCEA Tables (16-18, 21)."""
    tables = {}

    # Table 16: Equity-relevant population
    rows_16 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        subgroups = p.get("subgroups", {})
        for group, details in subgroups.items():
            rows_16.append(
                {
                    "Intervention": name.replace("_", " ").title(),
                    "Group": group,
                    "Population Size": f"{details.get('population_size', 'N/A')}",
                    "Weight": f"{details.get('weight', 1.0)}",
                }
            )
    if rows_16:
        tables["Table_16_Equity_Population"] = dataframe_to_markdown(
            pd.DataFrame(rows_16)
        )

    # Table 17: Group-specific parameters
    rows_17 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        subgroups = p.get("subgroups", {})
        for group, details in subgroups.items():
            # Check for specific params
            for k, v in details.items():
                if k not in ["population_size", "weight"]:
                    rows_17.append(
                        {
                            "Intervention": name.replace("_", " ").title(),
                            "Group": group,
                            "Parameter": k,
                            "Value": str(v),
                        }
                    )
    if rows_17:
        tables["Table_17_Group_Parameters"] = dataframe_to_markdown(
            pd.DataFrame(rows_17)
        )

    # Table 18: Baseline Distribution
    # Placeholder as we don't have easy access to baseline-only runs in results
    tables["Table_18_Baseline_Distribution"] = (
        "See DCEA Report for detailed baseline distributions."
    )

    # Table 21: Social Welfare
    rows_21 = []
    if "dcea_equity_analysis" in results:
        for name, dcea_res_dict in results["dcea_equity_analysis"].items():
            if not dcea_res_dict:
                continue
            for perspective, dcea_res in dcea_res_dict.items():
                if not dcea_res:
                    continue
                if "equity_weighted_nmb" in dcea_res.get("equity_metrics", {}):
                    val = dcea_res["equity_metrics"]["equity_weighted_nmb"]
                    rows_21.append(
                        {
                            "Intervention": name.replace("_", " ").title(),
                            "Perspective": perspective.replace("_", " ").title(),
                            "Metric": "Equity-Weighted NMB",
                            "Value": f"${val:,.0f}",
                        }
                    )
    if rows_21:
        tables["Table_21_Social_Welfare"] = dataframe_to_markdown(pd.DataFrame(rows_21))

    return tables


def generate_bia_tables(results, output_dir):
    """Generate BIA Tables (22-26)."""
    tables = {}

    if "bia_results" not in results:
        return tables

    # Table 22: Target Population (from params - inferred from results metadata if possible, else skip)

    # Table 24: Annual Budget Impact
    bia_rows = []
    for name, bia_df in results["bia_results"].items():
        df = bia_df if isinstance(bia_df, pd.DataFrame) else pd.DataFrame(bia_df)

        row = {"Intervention": name.replace("_", " ").title()}
        for _, r in df.iterrows():
            year = int(r["year"])
            row[f"Year {year}"] = f"${r['net_cost']:,.0f}"

        row["Total (5-Year)"] = f"${df['net_cost'].sum():,.0f}"
        bia_rows.append(row)

    if bia_rows:
        tables["Table_24_Annual_Budget_Impact"] = dataframe_to_markdown(
            pd.DataFrame(bia_rows)
        )

    # Table 25: Budget Impact Breakdown
    # Assuming bia_df has 'gross_cost' and 'offsets' (calculated as net - gross)
    breakdown_rows = []
    for name, bia_df in results["bia_results"].items():
        df = bia_df if isinstance(bia_df, pd.DataFrame) else pd.DataFrame(bia_df)

        for _, r in df.iterrows():
            breakdown_rows.append(
                {
                    "Intervention": name.replace("_", " ").title(),
                    "Year": int(r["year"]),
                    "Gross Cost": f"${r['gross_cost']:,.0f}",
                    "Net Cost": f"${r['net_cost']:,.0f}",
                    "Cost Offsets": f"${r['net_cost'] - r['gross_cost']:,.0f}",
                }
            )
    if breakdown_rows:
        tables["Table_25_Budget_Impact_Breakdown"] = dataframe_to_markdown(
            pd.DataFrame(breakdown_rows)
        )

    return tables


def generate_extended_bia_tables(results, params, output_dir):
    """Generate Extended BIA Tables (22, 23, 26)."""
    tables = {}

    # Table 22: Target Population
    rows_22 = []
    for name, p in params.items():
        if name in [
            "new_high_cost_cancer_drug",
            "smoking_cessation_counselling",
            "smoking_cessation_structural_sensitivity",
        ]:
            continue
        bia_pop = p.get("bia_population", {})
        rows_22.append(
            {
                "Intervention": name.replace("_", " ").title(),
                "Total Population": f"{bia_pop.get('total_population', 100000):,}",
                "Eligible Proportion": f"{bia_pop.get('eligible_proportion', 0.1):.1%}",
                "Uptake (Year 1)": "10%",
            }
        )
    tables["Table_22_BIA_Target_Population"] = dataframe_to_markdown(
        pd.DataFrame(rows_22)
    )

    # Table 23: BIA Perspective
    rows_23 = [
        {"Item": "Budget Holder", "Description": "PHARMAC / Health New Zealand"},
        {"Item": "Perspective", "Description": "Capped Budget (Health System Only)"},
        {"Item": "Time Horizon", "Description": "5 Years"},
        {
            "Item": "Included Costs",
            "Description": "Drug/Intervention Acquisition, Administration, Monitoring",
        },
    ]
    tables["Table_23_BIA_Perspective"] = dataframe_to_markdown(pd.DataFrame(rows_23))

    # Table 26: BIA Scenarios (Placeholder)
    tables["Table_26_BIA_Scenarios"] = (
        "Scenario analysis results available in full BIA report."
    )

    return tables


def generate_voi_tables(results, output_dir):
    """Generate VOI Tables (27-29)."""
    tables = {}

    if "voi_analysis" not in results:
        return tables

    # Table 27: EVPI Summary
    evpi_rows = []
    for name, voi_res in results["voi_analysis"].items():
        # Check for new structure
        if "value_of_information" in voi_res:
            voi_data = voi_res["value_of_information"]
            pop_evpi = voi_data.get("population_evpi", 0.0)

            evpi_rows.append(
                {
                    "Intervention": name.replace("_", " ").title(),
                    "Metric": "Population EVPI",
                    "Threshold": "$50,000",
                    "Value": f"${pop_evpi:,.0f}",
                }
            )
        # Legacy check
        elif "evpi" in voi_res:
            evpi_data = voi_res["evpi"]
            thresholds = np.array(evpi_data.get("thresholds", []))
            evpi_vals = np.array(evpi_data.get("evpi", []))

            if len(thresholds) > 0:
                idx = (np.abs(thresholds - 50000)).argmin()
                val_at_50k = evpi_vals[idx]

                evpi_rows.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Metric": "Population EVPI",
                        "Threshold": "$50,000",
                        "Value": f"${val_at_50k:,.0f}",
                    }
                )

    if evpi_rows:
        tables["Table_27_EVPI_Summary"] = dataframe_to_markdown(pd.DataFrame(evpi_rows))

    # Table 28: EVPPI
    evppi_rows = []
    for name, voi_res in results["voi_analysis"].items():
        if "value_of_information" in voi_res:
            voi_data = voi_res["value_of_information"]
            evppi_data = voi_data.get("evppi_by_parameter_group", {})
            thresholds = voi_data.get("wtp_thresholds", [])

            # Find index for 50k
            idx_50k = -1
            if thresholds:
                thresholds_arr = np.array(thresholds)
                idx_50k = (np.abs(thresholds_arr - 50000)).argmin()

            for param, val_list in evppi_data.items():
                val = 0.0
                if isinstance(val_list, (int, float)):
                    val = val_list
                elif idx_50k != -1 and len(val_list) > idx_50k:
                    val = val_list[idx_50k]

                evppi_rows.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Parameter Group": param,
                        "EVPPI (Per Person)": f"${val:.2f}",
                    }
                )

        elif "evppi" in voi_res:
            evppi_data = voi_res["evppi"]
            # Assuming dict: {param_group: value}
            for param, val in evppi_data.items():
                evppi_rows.append(
                    {
                        "Intervention": name.replace("_", " ").title(),
                        "Parameter Group": param,
                        "EVPPI (Per Person)": f"${val:.2f}",
                    }
                )

    if evppi_rows:
        tables["Table_28_EVPPI_Summary"] = dataframe_to_markdown(
            pd.DataFrame(evppi_rows)
        )

    return tables


def generate_extended_voi_tables(results, output_dir):
    """Generate Extended VOI Tables (29)."""
    tables = {}

    # Table 29: EVSI (Placeholder)
    tables["Table_29_EVSI"] = (
        "EVSI analysis requires specific trial design parameters and was not conducted for this summary."
    )

    return tables


def generate_all_tables(results, params, output_dir):
    """Master function to generate all tables."""
    import os

    os.makedirs(output_dir, exist_ok=True)

    all_tables = {}

    logger.info("Generating Core Tables...")
    all_tables.update(generate_core_tables(params, output_dir))

    logger.info("Generating CEA Tables...")
    all_tables.update(generate_cea_tables(results, output_dir))

    logger.info("Generating DCEA Tables...")
    all_tables.update(generate_dcea_tables(results, output_dir))

    logger.info("Generating BIA Tables...")
    all_tables.update(generate_bia_tables(results, output_dir))

    logger.info("Generating Qualitative Tables...")
    all_tables.update(generate_qualitative_tables(params, output_dir))

    logger.info("Generating Subgroup Tables...")
    all_tables.update(generate_subgroup_tables(results, output_dir))

    logger.info("Generating Extended DCEA Tables...")
    all_tables.update(generate_extended_dcea_tables(results, params, output_dir))

    logger.info("Generating Extended BIA Tables...")
    all_tables.update(generate_extended_bia_tables(results, params, output_dir))

    logger.info("Generating VOI Tables...")
    all_tables.update(generate_voi_tables(results, output_dir))
    all_tables.update(generate_extended_voi_tables(results, output_dir))

    # Save all tables to a single Markdown file for easy copy-pasting
    with open(f"{output_dir}/generated_tables.md", "w") as f:
        f.write("# Generated Tables for NZMJ Submission\n\n")
        for title, content in all_tables.items():
            f.write(f"## {title.replace('_', ' ')}\n\n")
            f.write(content)
            f.write("\n\n")

    return all_tables
