"""Keep spin-up drift visible and avoid member-splicing in temporal metrics."""
import numpy as np
import torch
import pytest

from monthly_diffusion.experiments import diagnose_spinup_latents as analysis


def test_fixed_climatology_preserves_offset_and_drift():
    months = np.tile(np.arange(1, 13), 20)
    climate = np.arange(12)[:, None, None]*2.
    reference = np.broadcast_to(climate[months-1], (240, 3, 2))
    drift = np.linspace(0, 5, 240)[:, None, None, None]
    values = reference[:, None]+3.+drift
    residual, ref = analysis.fixed_monthly_anomalies(values, months, reference, months)
    np.testing.assert_allclose(residual[:, 0, 0, 0], 3.+drift[:, 0, 0, 0])
    np.testing.assert_allclose(ref, 0.)


def test_window_statistics_keep_members_separate():
    t = np.arange(240)
    x = np.stack([np.sin(2*np.pi*t/48), 8+2*np.cos(2*np.pi*t/48)], axis=1)
    psd, acf = analysis.window_statistics(x)
    assert psd.sizes['member'] == acf.sizes['member'] == 2
    np.testing.assert_allclose(acf.sel(lag_months=0), 1.)
    assert (acf.sel(lag_months=24) < -.8).all()


def test_extend_from_current_restart_slot_and_validate_boundary(tmp_path):
    dates = np.arange('1830-01', '1851-01', dtype='datetime64[M]').astype('datetime64[ns]')
    metadata = dict(spinup_months=240, spinup_valid_times=[str(t) for t in dates[:240]],
                    valid_times=[str(t) for t in dates[240:]])
    directory = tmp_path/'spinup_latents'
    directory.mkdir()
    for year in range(20):
        end = 12*(year+1)
        values = np.arange(end-11, end+1, dtype='float32')[:, None, None, None]*np.ones((1, 2, 3, 4))
        history = torch.from_numpy(np.stack([values[-1], values[-2]], axis=1))
        torch.save(dict(time=dates[end-12:end], latents={c:values for c in analysis.COMPONENTS},
                        state=dict(step=end, history=[history, history])), directory/f'year-{year+1:04d}.pt')
    for i in range(12):
        directory = tmp_path/f'month-{i+1:06d}.zarr'
        directory.mkdir()
        history = torch.stack([torch.full((2, 3, 4), 241.+i), torch.full((2, 3, 4), 240.+i)], dim=1)
        torch.save(dict(step=241+i, times=[dates[240+i], dates[239+i]], history=[history, history]),
                   directory/'forecast_state.pt')
    raw, actual_dates = analysis.load_latent_trajectory(tmp_path, metadata, through_year=1850)
    np.testing.assert_array_equal(actual_dates, dates)
    for c in analysis.COMPONENTS:
        np.testing.assert_array_equal(raw[c][:, 0, 0, 0], np.arange(1, 253))
    raw, actual_dates = analysis.load_latent_trajectory(tmp_path, metadata)
    assert len(actual_dates) == 240  # Original spin-up-only mode unaffected.
    path = tmp_path/'month-000001.zarr/forecast_state.pt'
    broken = torch.load(path, weights_only=False)
    broken['history'][0][:, 1] += 1
    torch.save(broken, path)
    with pytest.raises(AssertionError):
        analysis.load_latent_trajectory(tmp_path, metadata, through_year=1850)
