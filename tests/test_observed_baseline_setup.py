"""Observation-initialized baseline dates and dispatch, without full inference."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_baseline_config_preserves_matched_diagnostics():
    baseline = yaml.safe_load((ROOT/"examples/diagnostics-era5-oras5-sst-support-observed.yaml").read_text())
    existing = yaml.safe_load((ROOT/"examples/diagnostics-era5-oras5-sst-support.yaml").read_text())
    for key in ("checkpoint", "training_source", "references", "reference_kind", "anomaly_method",
                "co2_netcdf", "co2_sha256", "training", "index_period", "variables"):
        assert baseline[key] == existing[key]
    assert baseline["forecast"] != existing["forecast"]
    assert baseline["evaluation"] == baseline["reference_evaluation"] == [1959, 2025]
    generated = np.arange(np.datetime64("1958-03", "M"), np.datetime64("2026-01", "M"))
    assert len(generated) == 814
    assert len(generated[generated >= np.datetime64("1959-01")]) == 67*12


@pytest.mark.parametrize("diagnostics_only", [False, True])
def test_baseline_launcher_resumes_mps_forecast_before_diagnostics(tmp_path, diagnostics_only):
    stub = tmp_path/"python_stub"
    record = tmp_path/"calls.jsonl"
    stub.write_text(f"#!{sys.executable}\nimport json, os, sys\n"
                    "with open(os.environ['MD_LAUNCH_LOG'], 'a') as stream:\n"
                    "    stream.write(json.dumps(sys.argv[1:])+'\\n')\n")
    stub.chmod(0o755)
    command = ["bash", str(ROOT/"scripts/recipes/run_era5_oras5_sst_support_observed_rollout.sh")]
    if diagnostics_only:
        command.append("--diagnostics-only")
    subprocess.run(command, check=True, capture_output=True, text=True,
                   env=dict(os.environ, MD_PYTHON=str(stub), MD_LAUNCH_LOG=str(record)))
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(calls) == (1 if diagnostics_only else 3)
    assert "scripts/research/diagnose_long_rollout.py" in calls[-1]
    assert "--skip-decoder" in calls[-1]
    if not diagnostics_only:
        assert calls[0][0] == "-"  # Read-only storage preflight.
        forecast = calls[1]
        assert "--resume" in forecast and "--require-trained" in forecast
        for option, expected in (("--device", "mps"), ("--months", "814"),
                                 ("--members", "5"), ("--initialization", "1958-02"),
                                 ("--seed", "20260716")):
            assert forecast[forecast.index(option)+1] == expected
        assert "--gaussian-start" not in forecast and "--spinup-months" not in forecast
