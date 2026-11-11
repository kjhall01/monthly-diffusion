"""Calendar, train/test leakage, probability/RPSS and legacy scoring parity."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from monthly_diffusion.analysis.seasonal_hindcast import (
    winter_dates, reference_winters, co2_increment_climatology, forecast_co2,
    djf_mean, terciles, categorize, probabilities, rps, aggregate_rpss, reliability)


def test_dates_and_strict_test_seasons():
    assert reference_winters() == list(range(1967, 2007))
    history, future = winter_dates(2008)
    assert list(history.month) == [9, 10]
    assert list(future.month) == [11, 12, 1, 2]
    assert future[1] == pd.Timestamp("2007-12-01")
    assert winter_dates(2007)[1][1] == pd.Timestamp("2006-12-01")
    assert len([y for y in range(2007, 2015) if y > 2007]) == 7


def test_native_calendar_day_weights_and_missing_month():
    x = np.array([0., 0., 1.])
    assert djf_mean(x, year=2008, calendar="noleap") == 28/90
    assert djf_mean(x, year=2008, calendar="proleptic_gregorian") == 29/91
    assert djf_mean(x, year=2009, calendar="proleptic_gregorian") == 28/90
    assert np.isnan(djf_mean([0., np.nan, 1.]))


def test_future_co2_does_not_leak_into_increment_climatology():
    dates = pd.date_range("1965-01-01", "2015-12-01", freq="MS")
    co2 = xr.DataArray(300+np.arange(len(dates))*.1, dims="time", coords=dict(time=dates))
    increments, count = co2_increment_climatology(co2)
    co2.loc[dict(time=slice("2007", None))] += 10000.
    np.testing.assert_array_equal(increments, co2_increment_climatology(co2)[0])
    np.testing.assert_array_equal(count, [40]+[41]*11)
    np.testing.assert_allclose(forecast_co2(380., winter_dates(2008)[1], increments), [380.1, 380.2, 380.3, 380.4])


def test_tercile_ties_missing_and_uniform_probability_score():
    q, valid, degenerate, tied, counts = terciles(np.arange(12.)[:, None, None])
    np.testing.assert_allclose(q[:, 0, 0], [11/3, 22/3])
    np.testing.assert_array_equal(categorize(q[:, 0, 0, None, None], q)[:, 0, 0], [0, 1])
    assert valid.all()
    assert not terciles(np.zeros((12, 1, 1)))[1].any()
    targets = np.arange(3)[:, None, None]
    np.testing.assert_allclose(rps(np.full((3, 3, 1, 1), 1/3), targets).ravel(), [5/9, 2/9, 5/9])
    perfect = np.eye(3)[:, :, None, None]
    assert (rps(perfect, targets) == 0).all()
    assert aggregate_rpss(rps(perfect, targets), rps(np.full_like(perfect, 1/3), targets))[0] == 1.


def test_rpss_ratio_of_matched_sums_not_mean_ratios():
    s = np.array([0.1, 0.2, np.nan])[:, None, None]
    c = np.array([5/9, 2/9, 5/9])[:, None, None]
    result, count, num, den = aggregate_rpss(s, c)
    np.testing.assert_allclose(result, 1-.3/(7/9))
    assert count == 2
    assert not np.isclose(result, np.nanmean(1-s/c))


def test_reliability_area_weights_and_member_bins():
    p = np.zeros((2, 3, 2, 1))
    p[:, 0] = .2
    p[:, 1] = .3
    p[:, 2] = .5
    y = np.array([[[0], [2]], [[0], [2]]])
    rows = reliability(p, y, np.array([[1.], [.5]]), np.ones((2, 1), bool))
    low = next(r for r in rows if r['category'] == 0 and r['pair_count'])
    np.testing.assert_allclose(low['observed_frequency'], 2/3)
    np.testing.assert_allclose(low['case_equivalent_weight'], 2.)


def test_existing_scoring_parity():
    old = Path(__file__).resolve().parents[2]/"umd/analysis/djf_hindcast.py"
    if not old.exists():
        return  # Package tests also work in the standalone frozen repository.
    spec = importlib.util.spec_from_file_location("legacy_djf_stats", old)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    rng = np.random.default_rng(123)
    calibration = rng.normal(size=(40, 3, 4))
    q = terciles(calibration)[0]
    np.testing.assert_array_equal(q, legacy.terciles(calibration)[0])
    f = rng.normal(size=(8, 20, 3, 4))
    t = rng.normal(size=(8, 3, 4))
    p, y = probabilities(f, q), categorize(t, q)
    np.testing.assert_array_equal(p, legacy.probabilities(f, q))
    np.testing.assert_array_equal(rps(p, y), legacy.rps(p, y))


def test_scoring_pipeline_same_member_case_preservation(tmp_path):
    """Tiny end-to-end archive->DJF->training thresholds->test scores."""
    import json
    from monthly_diffusion.experiments import test_set_seasonal_hindcasts as runner
    out = tmp_path/"e3sm"
    (out/"provenance").mkdir(parents=True)
    rows = []
    rng = np.random.default_rng(12)
    for year in [1967, 1968, 1969, 2007, 2008, 2009]:
        for member in ["r2i1p1f1", "r3i1p1f1"]:
            rows.append(dict(year=year, initialization_member=member, calibration=year < 2007,
                             training_overlap=year == 2007))
            path = out/"forecasts"/f"djf{year}_{member}"
            path.mkdir(parents=True)
            _, future = winter_dates(year)
            target = xr.Dataset({v: (("time", "lat", "lon"), rng.normal(size=(3, 2, 4)))
                                 for v in runner.VARIABLES}, coords=dict(time=future[1:], lat=[-45., 45.], lon=[0., 90., 180., 270.]))
            runner.save_zarr(target, path/"target.zarr")
            for start in range(0, 20, 4):
                monthly = xr.Dataset({v: (("time", "member", "lat", "lon"), rng.normal(size=(4, 4, 2, 4)))
                                      for v in runner.VARIABLES}, coords=dict(time=future, member=np.arange(start, start+4),
                                                                            lat=target.lat, lon=target.lon))
                runner.save_zarr(monthly, path/f"members_{start:03d}.zarr")
            (path/"completion.json").write_text(json.dumps(dict(members=20)))
    pd.DataFrame(rows).to_csv(out/"cases.csv", index=False)
    config = dict(output=str(tmp_path), forecast_members=20, inference_batch_size=4,
                  training_years=[1966, 2006], experiments=dict(e3sm=dict(calendar="noleap",
                  sources={"r2i1p1f1": "synthetic", "r3i1p1f1": "synthetic"})))
    runner.score(config, "e3sm")
    result = xr.open_zarr(out/"scores/t2m.zarr", consolidated=True).load()
    assert result.sizes["case"] == 6
    assert set(result.initialization_member.values) == {"r2i1p1f1", "r3i1p1f1"}
    assert (result.case_count_strict_test == 4).all()
    np.testing.assert_allclose(result.forecast_probability.sum("category"), 1.)
    expected = 1-result.rps_md.sel(case=result.year > 2007).sum("case") / result.rps_climatology.sel(case=result.year > 2007).sum("case")
    np.testing.assert_allclose(result.rpss_strict_test, expected)


def test_e3sm_calibration_and_verification_members_are_disjoint():
    import yaml
    path = Path(__file__).resolve().parents[1]/"examples/test-set-djf-hindcasts.yaml"
    c = yaml.safe_load(path.read_text())["experiments"]["e3sm"]
    assert c["calibration_members"] == ["r1i1p1f1"]
    assert c["evaluation_members"] == [f"r{i}i1p1f1" for i in range(2, 6)]
    assert not set(c["calibration_members"]) & set(c["evaluation_members"])
