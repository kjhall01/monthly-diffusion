"""Check shell stage ordering and failure handling without launching training."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("fail_ocean", [False, True])
@pytest.mark.parametrize(
    "script",
    ["train_no_co2.sh", "train_no_transforms.sh", "train_masked_autoencoders.sh"],
)
def test_launcher_runs_sequentially_and_stops_on_failure(tmp_path, fail_ocean, script):
    source = Path(__file__).resolve().parents[1] / "scripts" / script
    launcher = tmp_path / script
    shutil.copyfile(source, launcher)
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'echo "$2" >> calls.txt\n'
        'echo "training $2"\n'
        'if [[ "${FAIL_OCEAN}" == 1 && "$2" == train_ocean_autoencoder.py ]]; then\n'
        "    exit 7\n"
        "fi\n"
    )
    fake_python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(launcher)],
        cwd="/",
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHON": str(fake_python),
            "FAIL_OCEAN": str(int(fail_ocean)),
        },
    )
    expected = ["train_atmosphere_autoencoder.py", "train_ocean_autoencoder.py"]
    if not fail_ocean and script == "train_no_co2.sh":
        expected.append("train_latent_diffusion.py")
    assert (tmp_path / "calls.txt").read_text().splitlines() == expected
    assert result.returncode == (7 if fail_ocean else 0)
