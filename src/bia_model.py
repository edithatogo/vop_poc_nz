from typing import List, Union

import pandas as pd


def project_bia(
    population_size: int,
    eligible_prop: float,
    uptake_by_year: List[float],
    cost_per_patient: float,
    offset_cost_per_patient: float,
    implementation_cost_year1: float = 0.0,
    horizon_years: int = 5,
    adherence: float = 1.0,
    discount_rate: float = 0.03,
) -> pd.DataFrame:
    """
    Project budget impact over multiple years with optional discounting.

    Args:
        population_size: Total eligible population
        eligible_prop: Proportion of population that is eligible
        uptake_by_year: List of uptake rates by year (e.g., [0.1, 0.2, 0.3])
        cost_per_patient: Cost per treated patient
        offset_cost_per_patient: Avoided costs per patient (from standard care)
        implementation_cost_year1: One-time implementation cost in year 1
        horizon_years: Number of years to project
        adherence: Proportion of patients who adhere to treatment
        discount_rate: Annual discount rate (default 3%)

    Returns:
        DataFrame with year-by-year budget impact
    """
    years = list(range(1, horizon_years + 1))
    out: List[dict[str, float]] = []
    cumulative_discounted_net = 0.0

    for i, y in enumerate(years):
        uptake = uptake_by_year[min(i, len(uptake_by_year) - 1)]
        treated = population_size * eligible_prop * uptake * adherence
        gross = treated * cost_per_patient + (
            implementation_cost_year1 if y == 1 else 0.0
        )
        offsets = treated * offset_cost_per_patient
        net = gross - offsets

        # Apply discounting (start of year)
        discount_factor = (1 + discount_rate) ** (y - 1)
        discounted_gross = gross / discount_factor
        discounted_offsets = offsets / discount_factor
        discounted_net = net / discount_factor
        cumulative_discounted_net += discounted_net

        out.append(
            {
                "year": y,
                "treated": round(treated),
                "gross_cost": gross,
                "offsets": offsets,
                "net_cost": net,
                "discounted_gross_cost": discounted_gross,
                "discounted_offsets": discounted_offsets,
                "discounted_net_cost": discounted_net,
                "cumulative_discounted_net": cumulative_discounted_net,
            }
        )
    return pd.DataFrame(out)


def bia_to_markdown_table(
    df: pd.DataFrame, currency: str = "NZD", base_year: str = "2023"
) -> str:
    def to_int_safe(x: Union[int, float]) -> int:
        try:
            return round(float(x))
        except Exception:
            return 0

    header = (
        f"| Year | Treated | Gross cost ({currency} {base_year}) | Offsets ({currency} {base_year}) | Net cost ({currency} {base_year}) |\n"
        "|---:|---:|---:|---:|---:|"
    )
    rows: List[str] = []
    for r in df.itertuples(index=False):
        year = to_int_safe(getattr(r, "year", 0))
        treated = to_int_safe(getattr(r, "treated", 0))
        gross = float(getattr(r, "gross_cost", 0.0))
        offsets = float(getattr(r, "offsets", 0.0))
        net = float(getattr(r, "net_cost", 0.0))
        rows.append(
            f"| {year} | {treated:,} | {gross:,.0f} | {offsets:,.0f} | {net:,.0f} |"
        )
    return header + "\n" + "\n".join(rows)
