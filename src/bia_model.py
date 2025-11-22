from typing import Union
import pandas as pd


def project_bia(population_size:int,
                eligible_prop:float,
                uptake_by_year:list,
                cost_per_patient:float,
                offset_cost_per_patient:float,
                implementation_cost_year1:float=0.0,
                horizon_years:int=5,
                adherence:float=1.0):
    years = list(range(1, horizon_years+1))
    out = []
    for i, y in enumerate(years):
        uptake = uptake_by_year[min(i, len(uptake_by_year)-1)]
        treated = population_size * eligible_prop * uptake * adherence
        gross = treated * cost_per_patient + (implementation_cost_year1 if y == 1 else 0.0)
        offsets = treated * offset_cost_per_patient
        net = gross - offsets
        out.append({'year': y, 'treated': int(round(treated)), 'gross_cost': gross, 'offsets': offsets, 'net_cost': net})
    return pd.DataFrame(out)


def bia_to_markdown_table(df: pd.DataFrame, currency: str = 'NZD', base_year: str = '2023') -> str:
    def to_int_safe(x: Union[int, float]) -> int:
        try:
            return int(round(float(x)))
        except Exception:
            return 0
    header = (
        f"| Year | Treated | Gross cost ({currency} {base_year}) | Offsets ({currency} {base_year}) | Net cost ({currency} {base_year}) |\n"
        "|---:|---:|---:|---:|---:|"
    )
    rows = []
    for r in df.itertuples(index=False):
        year = to_int_safe(getattr(r, 'year', 0))
        treated = to_int_safe(getattr(r, 'treated', 0))
        gross = float(getattr(r, 'gross_cost', 0.0))
        offsets = float(getattr(r, 'offsets', 0.0))
        net = float(getattr(r, 'net_cost', 0.0))
        rows.append(f"| {year} | {treated:,} | {gross:,.0f} | {offsets:,.0f} | {net:,.0f} |")
    return header + "\n" + "\n".join(rows)
