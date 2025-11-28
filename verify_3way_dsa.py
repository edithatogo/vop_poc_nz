
"""
Systematic verification of 3-way DSA calculations
Checks 1, 2, and 3 for correctness
"""
import json
import numpy as np
from scipy.interpolate import griddata

# Load results
results_path = "../nzmj_feedback/complete_analysis_results.json"
with open(results_path, "r") as f:
    results = json.load(f)

dsa_3way = results["dsa_analysis"]["3_way"]

print("="*100)
print("SYSTEMATIC VERIFICATION OF 3-WAY DSA")
print("="*100)

# CHECK 1: Verify input parameters and ranges
print("\n" + "="*100)
print("CHECK 1: INPUT PARAMETERS AND RANGES")
print("="*100)

for intervention_name, data in dsa_3way.items():
    print(f"\n{intervention_name}:")
    print(f"  Param 1: {data['param1_name']}")
    print(f"    Range: {data['p1_range'][:3]} ... {data['p1_range'][-3:]}")
    print(f"    Min/Max: {min(data['p1_range']):.3f} / {max(data['p1_range']):.3f}")
    
    print(f"  Param 2: {data['param2_name']}")
    print(f"    Range: {data['p2_range'][:3]} ... {data['p2_range'][-3:]}")
    print(f"    Min/Max: {min(data['p2_range']):.3f} / {max(data['p2_range']):.3f}")
    
    print(f"  Param 3: {data['param3_name']}")
    print(f"    Range: {data['p3_range'][:3]} ... {data['p3_range'][-3:]}")
    print(f"    Min/Max: {min(data['p3_range']):.3f} / {max(data['p3_range']):.3f}")

# CHECK 2: Verify NMB variation within each slice
print("\n" + "="*100)
print("CHECK 2: NMB VARIATION WITHIN SLICES")
print("="*100)

for intervention_name, data in dsa_3way.items():
    print(f"\n{intervention_name}:")
    
    points = np.array([[pt["p1"], pt["p2"], pt["p3"]] for pt in data["dsa_grid"]])
    values = np.array([pt["nmb"] for pt in data["dsa_grid"]])
    
    unique_p3 = np.unique(points[:, 2])
    n_p3 = len(unique_p3)
    p3_slices = [unique_p3[0], unique_p3[n_p3 // 2], unique_p3[-1]]
    
    # Calculate tolerance
    if len(unique_p3) > 1:
        p3_spacing = unique_p3[1] - unique_p3[0]
        tolerance = p3_spacing / 2
    else:
        tolerance = 0.05
    
    for i, p3_slice in enumerate(p3_slices):
        mask = (points[:, 2] >= p3_slice - tolerance) & (points[:, 2] <= p3_slice + tolerance)
        slice_values = values[mask]
        
        if len(slice_values) > 0:
            print(f"  Slice {i+1} (p3={p3_slice:.3f}):")
            print(f"    N points: {len(slice_values)}")
            print(f"    NMB Min:   ${slice_values.min():,.0f}")
            print(f"    NMB Max:   ${slice_values.max():,.0f}")
            print(f"    NMB Range: ${slice_values.max() - slice_values.min():,.0f}")
            print(f"    NMB Std:   ${slice_values.std():,.0f}")
            print(f"    Variation: {100 * slice_values.std() / abs(slice_values.mean()):.2f}%")

# CHECK 3: Verify actual grid calculations
print("\n" + "="*100)
print("CHECK 3: SAMPLE GRID CALCULATIONS (Manual verification)")
print("="*100)

# For each intervention, show a few sample calculations
for intervention_name, data in dsa_3way.items():
    print(f"\n{intervention_name}:")
    
    # Get first, middle, and last grid points
    grid = data["dsa_grid"]
    sample_indices = [0, len(grid) // 2, -1]
    
    for idx in sample_indices:
        pt = grid[idx]
        print(f"  Point {idx}:")
        print(f"    p1={pt['p1']:.3f}, p2={pt['p2']:.3f}, p3={pt['p3']:.3f}")
        print(f"    NMB: ${pt['nmb']:,.0f}")
        print(f"    ICER: ${pt['icer']:,.0f}/QALY")
    
    # Check if all NMB values are identical
    all_nmb = [pt["nmb"] for pt in grid]
    unique_nmb = len(set(all_nmb))
    print(f"  Total grid points: {len(grid)}")
    print(f"  Unique NMB values: {unique_nmb}")
    if unique_nmb == 1:
        print(f"  WARNING: All NMB values are identical! This would create a flat surface.")
    elif unique_nmb < len(grid) * 0.1:
        print(f"  WARNING: Very few unique values ({unique_nmb}/{len(grid)}). Surface may appear flat.")

print("\n" + "="*100)
print("VERIFICATION COMPLETE")
print("="*100)
