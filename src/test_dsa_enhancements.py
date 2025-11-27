"""
Test script for profiling module and perspective value DSA.

Runs on HPV vaccination intervention to verify:
1. Profiling decorators work
2. Performance metrics are captured
3. Perspective value DSA executes correctly
4. Outputs are generated
"""



from .profiling import profile_function, profile_section, print_profiling_report, save_profiling_report, reset_profiler
from .pipeline.analysis import load_parameters
from .perspective_value_dsa import perform_perspective_value_dsa, plot_perspective_value_dsa, generate_perspective_value_dsa_table
import os

print("=" * 80)
print("TESTING PROFILING MODULE AND PERSPECTIVE VALUE DSA")
print("=" * 80)

# Reset profiler
reset_profiler()

# Load parameters
with profile_section("Load Parameters"):
    params = load_parameters('src/parameters.yaml')
    hpv_params = params['hpv_vaccination']

print(f"\n✓ Loaded HPV vaccination parameters")

# Run perspective value DSA (small sample for testing)
print(f"\n{'=' * 80}")
print("Running Perspective Value DSA (Test Mode)")
print(f"{'=' * 80}")

with profile_section("Perspective Value DSA"):
    dsa_results = perform_perspective_value_dsa(
        hpv_params,
        intervention_name="HPV Vaccination",
        wtp_range=(25000, 75000),
        n_wtp_points=10,  # Reduced for testing
        n_psa_samples=100,  # Reduced for testing
    )

print(f"\n✓ Perspective Value DSA completed")

# Run General One-Way DSA (Test Mode)
print(f"\n{'=' * 80}")
print("Running General One-Way DSA (Test Mode)")
print(f"{'=' * 80}")

from .dsa_analysis import perform_one_way_dsa, plot_one_way_dsa_tornado

with profile_section("General One-Way DSA"):
    # Create a wrapper dict as expected by perform_one_way_dsa
    models = {"HPV Vaccination": hpv_params}
    
    # Run DSA with reduced points for testing
    general_dsa_results = perform_one_way_dsa(
        models, 
        wtp_threshold=50000, 
        n_points=5  # Reduced for speed
    )

print(f"\n✓ General One-Way DSA completed")
print(f"  Parameters analyzed: {list(general_dsa_results['HPV Vaccination']['dsa_results'].keys())}")

# Generate outputs
os.makedirs('output/test_dsa', exist_ok=True)

print(f"\nGenerating visualization...")
with profile_section("Plot Generation"):
    plot_perspective_value_dsa(dsa_results, output_dir='output/test_dsa/')
    plot_one_way_dsa_tornado(general_dsa_results, output_dir='output/test_dsa/')

print(f"✓ Plots saved to: output/test_dsa/")

print(f"\nGenerating summary table...")
table = generate_perspective_value_dsa_table(dsa_results)
table_path = 'output/test_dsa/perspective_value_dsa_table.csv'
table.to_csv(table_path, index=False)
print(f"✓ Table saved to: {table_path}")

# Print sample results
print(f"\n{'=' * 80}")
print("SAMPLE RESULTS")
print(f"{'=' * 80}")
print(f"\nPerspective Value Metrics at WTP=$50,000:")
mid_idx = len(dsa_results['wtp_thresholds']) // 2
print(f"  EVP: ${dsa_results['evp'][mid_idx]:,.0f}")
print(f"  Perspective Premium: ${dsa_results['perspective_premium'][mid_idx]:,.0f}")
print(f"  Discordance Cost: ${dsa_results['discordance_cost'][mid_idx]:,.0f}")
print(f"  Information Value: ${dsa_results['information_value'][mid_idx]:,.0f}")
print(f"  Proportion Discordant: {dsa_results['proportion_discordant'][mid_idx]:.1%}")

# Print profiling report
print(f"\n{'=' * 80}")
print("PROFILING REPORT")
print(f"{'=' * 80}")
print_profiling_report()

# Save profiling report
prof_path = 'output/test_dsa/profiling_report.txt'
save_profiling_report(prof_path)

print(f"\n{'=' * 80}")
print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
print(f"{'=' * 80}")
print(f"\nOutputs saved to: output/test_dsa/")
print(f"  - perspective_value_dsa_HPV Vaccination.png")
print(f"  - perspective_value_dsa_table.csv")
print(f"  - profiling_report.txt")
