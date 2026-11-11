"""Encoded field-noise initialization, calendar boundary and exact restart."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from monthly_diffusion.forecast import Forecast, rollout
from monthly_diffusion.data.dataloader import month_features
from monthly_diffusion.data.preprocessing import LatentNormalizer
from monthly_diffusion.data.sources import hold_first_co2_back_to
from test_gaussian_forecast import ToyModel

ROOT = Path(__file__).resolve().parents[1]


class ToyEncoder:
    mask_aware = True

    def __init__(self, nodes):
        self.nodes, self.calls = nodes, []

    def encode(self, x, statics, months, forcing, node_mask):
        self.calls.append((x.clone(), months.clone(), forcing.clone(), node_mask.clone()))
        # Nontrivial, deterministic encoding; don't sample any Torch noise.
        z = x.nanmean((1, 2, 3)) + months[:, 0] + forcing[:, 0]
        return z[:, None, None].expand(-1, self.nodes, 4).clone()


class NoiseModel(ToyModel):
    def __init__(self, fail_at=None):
        super().__init__(fail_at=fail_at)
        self.names = dict(atmosphere=['a0', 'a1'], ocean=['o0', 'o1'])
        support = xr.DataArray(np.ones((4, 8), dtype=bool), dims=('lat', 'lon'))
        support[0, :2] = False
        stats = {name: dict(mask=xr.ones_like(support, dtype=bool))
                 for names in self.names.values() for name in names}
        stats['a1']['mask'][3, 0] = False  # Per-variable missingness also matters.
        stats['o1']['mask'][2, 0] = False
        self.preprocessor = SimpleNamespace(stats=stats, ocean_support_mask=support,
            co2_mean=300., co2_std=20., static_tensor=lambda ds: torch.zeros(2, 4, 8))
        self.statics = None
        self.autoencoder = SimpleNamespace(atmosphere=ToyEncoder(32), ocean=ToyEncoder(32), eval=lambda: None)
        self.latent_normalizers = {}
        for component in self.names:
            norm = LatentNormalizer()
            norm.mean, norm.std = torch.full((4,), .2), torch.full((4,), 2.)
            self.latent_normalizers[component] = norm


def test_encoded_histories_masks_conditioning_and_single_normalization():
    model = NoiseModel()
    dates = xr.date_range('1799-11', periods=3, freq='MS')
    co2 = xr.DataArray([280., 282., 283.], dims='time', coords=dict(time=dates))
    before = torch.get_rng_state().clone()
    forecast = Forecast(model, None, None, co2=co2, members=3, seed=51, encoded_noise_start='1800-01')
    assert torch.equal(before, torch.get_rng_state())
    assert [str(t)[:7] for t in forecast.times] == ['1799-12', '1799-11']
    expected_months = month_features(xr.DataArray(dates[-2::-1], dims='time', coords=dict(time=dates[-2::-1])))
    for i, component in enumerate(model.names):
        calls = getattr(model.autoencoder, component).calls
        assert len(calls) == 3
        valid = np.stack([model.preprocessor.stats[n]['mask'].values for n in model.names[component]])
        if component == 'ocean':
            valid &= model.preprocessor.ocean_support_mask.values[None]
        rng = np.random.default_rng(51+1000+i)
        for j, (x, months, forcing, mask) in enumerate(calls):
            expected = rng.standard_normal((2, *valid.shape)).astype('float32')
            expected[:, ~valid] = np.nan
            np.testing.assert_array_equal(x, expected)
            torch.testing.assert_close(months, expected_months)
            torch.testing.assert_close(forcing[:, 0], torch.tensor([-.9, -1.]))
            np.testing.assert_array_equal(mask, np.broadcast_to(valid[0], (2, 4, 8)))
            raw = x.nanmean((1, 2, 3)) + months[:, 0] + forcing[:, 0]
            torch.testing.assert_close(forecast.history[i][j, :, 0, 0], (raw-.2)/2.)
        assert not torch.equal(calls[0][0], calls[1][0])
    same = Forecast(NoiseModel(), None, None, co2=co2, members=3, seed=51, encoded_noise_start='1800-01')
    for a, b in zip(forecast.history, same.history):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize('fail_at', [15, 603])
def test_fifty_year_spinup_retains_all_latents_and_resumes_exactly(tmp_path, fail_at):
    original = xr.DataArray([284., 285., 287.], dims='time',
                           coords=dict(time=xr.date_range('1850-01', periods=3, freq='MS')))
    co2 = hold_first_co2_back_to(original, '1799-11')
    kwargs = dict(co2=co2, months=3, members=2, seed=72,
                  encoded_noise_start='1800-01', spinup_months=600, save_spinup_latents=True)
    broken = NoiseModel(fail_at=fail_at)
    with pytest.raises(RuntimeError, match='simulated'):
        rollout(broken, None, None, path=tmp_path/'resumed', resume=False, **kwargs)
    resumed = NoiseModel()
    actual = rollout(resumed, None, None, path=tmp_path/'resumed', resume=True, **kwargs).compute()
    uninterrupted = NoiseModel()
    expected = rollout(uninterrupted, None, None, path=tmp_path/'complete', resume=False, **kwargs).compute()
    xr.testing.assert_equal(actual, expected)
    assert uninterrupted.decoded == 3
    assert resumed.decoded == (3 if fail_at == 15 else 1)
    assert resumed.transition.calls == (591 if fail_at == 15 else 1)
    assert str(actual.time.values[0])[:7] == '1850-01'
    np.testing.assert_array_equal(actual.co2, original)
    blocks = sorted((tmp_path/'resumed/spinup_latents').glob('year-*.pt'))
    assert len(blocks) == 50
    for block in blocks:
        a = torch.load(block, weights_only=False)
        b = torch.load(tmp_path/'complete/spinup_latents'/block.name, weights_only=False)
        np.testing.assert_array_equal(a['time'], b['time'])
        for component in ['atmosphere', 'ocean']:
            assert a['latents'][component].shape == (12, 2, 32, 4)
            np.testing.assert_array_equal(a['latents'][component], b['latents'][component])
    first, last = [torch.load(p, weights_only=False) for p in (blocks[0], blocks[-1])]
    assert str(first['time'][0])[:7] == '1800-01'
    assert str(last['time'][-1])[:7] == '1849-12'
    assert last['state']['step'] == 600
    metadata = json.loads((tmp_path/'resumed/forecast.json').read_text())
    assert metadata['initialization_kind'] == 'encoded_standard_normal_fields'
    assert metadata['initialization_co2_ppm'] == [284., 284.]
    assert metadata['spinup_co2_ppm'] == [284.]*600
    assert metadata['co2_ppm'] == [284., 285., 287.]
    with pytest.raises(ValueError, match='differs'):
        rollout(NoiseModel(), None, None, path=tmp_path/'resumed', resume=True,
                **dict(kwargs, save_spinup_latents=False))


def test_noise_modes_and_missing_history_forcing_fail():
    with pytest.raises(ValueError, match='not both'):
        Forecast(NoiseModel(), None, None, co2=280., members=1, seed=7,
                 gaussian_start='1800-01', encoded_noise_start='1800-01')
    with pytest.raises(ValueError):
        rollout(NoiseModel(), None, None, co2=280., months=1, members=1, seed=7,
                path=None, resume=False, encoded_noise_start='1800-01',
                spinup_months=600, save_spinup_latents=True)
    co2 = xr.DataArray([280.], dims='time', coords=dict(time=[np.datetime64('1800-01-01')]))
    with pytest.raises((ValueError, KeyError)):
        Forecast(NoiseModel(), None, None, co2=co2, members=1, seed=7, encoded_noise_start='1800-01')


@pytest.mark.parametrize('diagnostics_only', [False, True])
def test_launcher_and_diagnostics_config(tmp_path, diagnostics_only):
    config = yaml.safe_load((ROOT/'examples/diagnostics-era5-oras5-sst-support-encoded-noise.yaml').read_text())
    baseline = yaml.safe_load((ROOT/'examples/diagnostics-era5-oras5-sst-support-observed.yaml').read_text())
    for key in ['checkpoint', 'training_source', 'references', 'anomaly_method', 'training',
                'index_period', 'co2_netcdf', 'co2_sha256', 'variables']:
        assert config[key] == baseline[key]
    assert config['forecast'] != baseline['forecast']
    assert config['evaluation'] == [1850, 2100]
    stub = tmp_path/'python_stub'
    record = tmp_path/'calls.jsonl'
    stub.write_text(f'#!{sys.executable}\nimport os,sys,json\n'
                    "with open(os.environ['MD_LAUNCH_LOG'],'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')\n")
    stub.chmod(0o755)
    command = ['bash', str(ROOT/'scripts/run_era5_oras5_sst_support_encoded_noise_rollout.sh')]
    if diagnostics_only:
        command.append('--diagnostics-only')
    subprocess.run(command, check=True, capture_output=True, text=True,
                   env=dict(os.environ, MD_PYTHON=str(stub), MD_LAUNCH_LOG=str(record)))
    calls = [json.loads(s) for s in record.read_text().splitlines()]
    assert len(calls) == (1 if diagnostics_only else 3)
    assert 'scripts/diagnose_long_rollout.py' in calls[-1]
    if not diagnostics_only:
        forecast = calls[1]
        for option, expected in [('--device', 'mps'), ('--months', '3012'), ('--members', '5'),
                                 ('--encoded-noise-start', '1800-01'), ('--spinup-months', '600'),
                                 ('--hold-first-co2-back-to', '1799-11')]:
            assert forecast[forecast.index(option)+1] == expected
        assert '--resume' in forecast and '--save-spinup-latents' in forecast
        assert '--gaussian-start' not in forecast
