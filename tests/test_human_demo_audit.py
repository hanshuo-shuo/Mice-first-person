from analysis.human_demo_audit import audit_human_demos


def test_no_human_data_is_clear_non_error(tmp_path, capsys):
    data_root = tmp_path / "missing_data"
    output_dir = tmp_path / "audit"
    summary = audit_human_demos(data_root, output_dir)
    assert summary["status"] == "no_data"
    assert summary["episodes"] == 0
    assert (output_dir / "audit_summary.json").exists()
    assert "No human demonstration sessions found" in capsys.readouterr().out
