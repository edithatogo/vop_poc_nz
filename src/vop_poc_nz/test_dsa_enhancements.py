"""Manual smoke harness for profiling and perspective-value DSA.

Importing this module is deliberately side-effect free. Execute it as a module to
generate the diagnostic plots, tables, and profiling report.
"""

from pathlib import Path

from .dsa_analysis import perform_one_way_dsa, plot_one_way_dsa_tornado
from .perspective_value_dsa import (
    generate_perspective_value_dsa_table,
    perform_perspective_value_dsa,
    plot_perspective_value_dsa,
)
from .pipeline.analysis import load_parameters
from .profiling import (
    print_profiling_report,
    profile_section,
    reset_profiler,
    save_profiling_report,
)


def main(output_dir: str | Path = "output/test_dsa") -> None:
    """Run the bounded manual DSA/profiling smoke harness."""
    destination = Path(output_dir)
    print("=" * 80)
    print("TESTING PROFILING MODULE AND PERSPECTIVE VALUE DSA")
    print("=" * 80)
    reset_profiler()

    with profile_section("Load Parameters"):
        params = load_parameters()
        hpv_params = params["hpv_vaccination"]

    with profile_section("Perspective Value DSA"):
        dsa_results = perform_perspective_value_dsa(
            hpv_params,
            intervention_name="HPV Vaccination",
            wtp_range=(25_000, 75_000),
            n_wtp_points=10,
            n_psa_samples=100,
        )

    with profile_section("General One-Way DSA"):
        general_dsa_results = perform_one_way_dsa(
            {"HPV Vaccination": hpv_params},
            wtp_threshold=50_000,
            n_points=5,
        )

    destination.mkdir(parents=True, exist_ok=True)
    with profile_section("Plot Generation"):
        plot_perspective_value_dsa(dsa_results, output_dir=f"{destination}/")
        plot_one_way_dsa_tornado(general_dsa_results, output_dir=f"{destination}/")

    table = generate_perspective_value_dsa_table(dsa_results)
    table.to_csv(destination / "perspective_value_dsa_table.csv", index=False)
    print_profiling_report()
    save_profiling_report(str(destination / "profiling_report.txt"))
    print(f"Outputs saved to: {destination}")


if __name__ == "__main__":
    main()
