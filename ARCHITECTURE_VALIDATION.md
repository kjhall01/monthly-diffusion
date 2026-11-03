# Validation performed

CPU validation in a clean extracted-package directory, without the old production
package on the import path:

- `python -m pytest tests/architecture -q`: **7 passed**.
- `python -m examples.architecture_walkthrough`: successful reconstruction forward,
  backward, and denoiser forward.
- Python source compilation: passed.

Environment: Python 3.12.14 runtime; PyTorch 2.15.0+cu132 (CPU execution), NumPy 2.3.5,
SciPy 1.17.0, pytest 9.2.0. Dependency minimums in requirements are intended API
baselines, not a tested version matrix.

Coverage: reference vs degree-bucketed ECC values/input gradients/parameter
gradients for mean and sum; directed hierarchy index bounds and coverage; AE
shape/finite-gradient checks; shared and separate conditioning; state_dict round
trip; frozen encoder isolation during a decoder update; fixed ocean support;
paired denoiser forward/backward and explicit diffusion time requirement.

Not performed: GPU/MPS benchmarks, production-resolution training, climate skill,
full production checkpoint migration, installation through the old setup.py, or
end-to-end production training/forecast integration. No old results are claimed to
be reproduced. Runtime code, guide and tests are included for local review.
