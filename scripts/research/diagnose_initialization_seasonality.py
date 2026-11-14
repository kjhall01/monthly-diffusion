"""Separate initialization offsets from internal variability in saved readouts.

No new inference. The fixed-training-baseline comparison is preserved; these
alternative anomalies must NOT be used as evidence of matching mean states.
The monthly/trend fits use the stated simulation window, not physical years
1966–2006, so they reproduce an anomaly operator, not the original experiment.
"""

# Make the repository workflow helpers available when running this file directly.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


sys.path.insert(0, str(Path(__file__).resolve().parent))
from workflows.experiments.diagnose_isotropic_cycling_sweep import OUTPUT


from workflows.experiments.diagnose_initialization_seasonality import run

run(OUTPUT / "confirm_january")
