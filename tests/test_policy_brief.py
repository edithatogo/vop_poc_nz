from vop_poc_nz.policy_brief_generator import generate_policy_brief


def test_policy_brief_generation(tmp_path):
    """
    Test that the policy brief generator produces a file with expected content.
    """
    # Dummy results
    results = {
        "Intervention A": {
            "health_system": {"icer": 60000, "is_cost_effective": False},
            "societal": {"human_capital": {"icer": 40000, "is_cost_effective": True}},
            "dcea_equity_analysis": {
                "weighted_total_health_gain": 120,
                "total_health_gain": 100,
            },
        },
        "Intervention B": {
            "health_system": {"icer": 20000, "is_cost_effective": True},
            "societal": {"human_capital": {"icer": 10000, "is_cost_effective": True}},
            # No equity analysis
        },
    }

    output_dir = tmp_path / "reports"
    generate_policy_brief(results, output_dir=str(output_dir))

    brief_path = output_dir / "policy_brief.md"
    assert brief_path.exists()

    content = brief_path.read_text()

    # Check for key sections
    assert "# Executive Summary" in content
    assert "## 1. The Policy Matrix" in content
    assert "## 2. The Value of Perspective" in content

    # Check for table content
    assert "Intervention A" in content
    assert "REJECT" in content  # HS decision for A
    assert "FUND" in content  # Soc decision for A
    assert "Positive (Pro-Poor)" in content  # Equity for A

    assert "Intervention B" in content
    assert "Neutral" in content  # Default equity
