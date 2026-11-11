"""Current frozen checkpoints: same-trajectory, test-only DJF hindcasts.

Reuses MonthlyDiffusionV1.initialize/decode unchanged. History is always
[current, previous] inside Forecast; callers supply chronological physical ICs.
Only training winters fit reference/model thresholds. All months of the three
scored fields and both full latent states are archived, allowing other fields
to be decoded later without rerunning stochastic transitions.
"""

from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from numcodecs import Blosc

ROOT = Path(__file__).resolve().parents[2]
from monthly_diffusion import MonthlyDiffusionV1
from monthly_diffusion.data.sources import open_canonical_state, open_co2
from monthly_diffusion.analysis.seasonal_hindcast import (
    winter_dates,
    reference_winters,
    co2_increment_climatology,
    forecast_co2,
    djf_mean,
    terciles,
    categorize,
    probabilities,
    rps,
    aggregate_rpss,
    reliability,
)

VARIABLES = ("mtpr", "t2m", "sst")
UNITS = dict(mtpr="mm day-1", t2m="K", sst="degC")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    temp = path.with_suffix(".partial.json")
    temp.write_text(json.dumps(value, indent=2, default=str))
    temp.replace(path)


def save_zarr(ds, path):
    """One new atomic group per member batch; never overwrite completed groups."""
    partial = path.with_suffix(".partial.zarr")
    ds = ds.copy()
    for v in ds.variables:
        ds[v].encoding = {}
    encoding = {
        v: dict(
            compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
            chunks=tuple(
                1
                if d in ("member", "time", "case")
                else min(ds.sizes[d], 15)
                if d == "lat"
                else ds.sizes[d]
                for d in ds[v].dims
            ),
        )
        for v in ds.data_vars
    }
    ds.to_zarr(partial, mode="w", consolidated=True, encoding=encoding)
    if path.exists():
        raise FileExistsError(path)
    partial.rename(path)


def prepare(config, key, device):
    c = config["experiments"][key]
    out = ROOT / config["output"] / key
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / c["checkpoint"]
    sha = digest(checkpoint)
    # Isolate experiment identity: changing E3SM's calibration membership
    # must not invalidate the numerically unchanged reanalysis forecasts.
    scientific_config = {
        k: v for k, v in config.items() if k not in ("output", "experiments")
    }
    scientific_config["experiment"] = c
    identity = dict(
        configuration=scientific_config,
        experiment=key,
        checkpoint_sha256=sha,
        initialization_source_order=list(c["sources"]),
        protocol_version=1,
        history_order="current, previous at EVERY step",
        forcing="actual history; October CO2 plus training-only destination-month increments",
        interpretation="fixed historical climatology; skill may include forced change",
        stored_fields=list(VARIABLES),
        stored_latents="all atmosphere and ocean channels/nodes",
    )
    identity_path = out / "identity.json"
    if identity_path.exists():
        previous = json.loads(identity_path.read_text())
        if "experiments" in previous["configuration"]:
            old_config = previous["configuration"]
            previous["configuration"] = {
                k: v
                for k, v in old_config.items()
                if k not in ("output", "experiments")
            }
            previous["configuration"]["experiment"] = old_config["experiments"][key]
        previous.setdefault(
            "initialization_source_order",
            list(previous["configuration"]["experiment"]["sources"]),
        )
        if previous != identity:
            raise ValueError(f"Experiment identity changed: {out}")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; refusing silent CPU inference")
    model = MonthlyDiffusionV1.load(checkpoint, device=device)
    train = config["training_years"]
    expected = pd.date_range(f"{train[0]}-01-01", f"{train[1]}-12-01", freq="MS").values
    np.testing.assert_array_equal(model.preprocessor.training_times, expected)
    co2 = open_co2(
        config["co2_netcdf"], expected_sha256=config["co2_sha256"]
    ).reset_coords(drop=True)
    np.testing.assert_allclose(
        co2.sel(time=expected), model.preprocessor.training_co2, rtol=0, atol=1e-5
    )
    increments, counts = co2_increment_climatology(
        co2, f"{train[0]}-01", f"{train[1]}-12"
    )
    sources = {}
    calibration_members = c.get("calibration_members", list(c["sources"]))
    evaluation_members = c.get("evaluation_members", list(c["sources"]))
    if not set(calibration_members + evaluation_members) <= set(c["sources"]):
        raise ValueError("Calibration/evaluation member lacks a source archive")
    training_member = model.preprocessor.source_attrs["atmosphere"]["member_id"]
    if key == "e3sm" and (
        calibration_members != [training_member]
        or training_member in evaluation_members
    ):
        raise ValueError(
            "E3SM calibration must be training-member-only and evaluation held-out-member-only"
        )
    for member, path in c["sources"].items():
        a, o, _ = open_canonical_state(path)
        if a.attrs["member_id"] != member:
            raise ValueError("Source identity mismatch")
        if c["calendar"] == "noleap" and "noleap" not in a.attrs["source_calendars"]:
            raise ValueError("Calendar mismatch")
        trained_attrs = model.preprocessor.source_attrs["atmosphere"]
        for attribute in ("source_lock_sha256", "product_release_id", "target_grid"):
            if a.attrs.get(attribute) != trained_attrs.get(attribute):
                raise ValueError(
                    f"Source provenance differs from trained product: {member}/{attribute}"
                )
        for coordinate in ("lat", "lon"):
            np.testing.assert_array_equal(a[coordinate], model.statics[coordinate])
        sources[member] = (a, o)
    rows = []
    for calibration, years in (
        (True, reference_winters(*train)),
        (False, range(config["test_years"][0], config["test_years"][1] + 1)),
    ):
        for year in years:
            for member in calibration_members if calibration else evaluation_members:
                history, future = winter_dates(year)
                # Exact selection fails on unavailable months, never fills/substitutes.
                for fields in sources[member]:
                    fields.sel(time=history.append(future))
                rows.append(
                    dict(
                        year=year,
                        initialization_member=member,
                        calibration=calibration,
                        training_overlap=year == train[1] + 1,
                        initialization_month=str(history[-1].date()),
                        first_valid=str(future[0].date()),
                        last_valid=str(future[-1].date()),
                    )
                )
    cases = pd.DataFrame(rows)
    save_json(identity_path, identity)
    cases.to_csv(out / "cases.csv", index=False)
    save_json(
        out / "model_and_data.json",
        dict(
            stages=model.stages,
            model_config=model.config.to_dict(),
            sources={k: dict(a.attrs) for k, (a, o) in sources.items()},
            co2_monthly_increments=increments.tolist(),
            increment_sample_counts=counts.tolist(),
            calibration_cases=int(cases.calibration.sum()),
            evaluation_cases=int((~cases.calibration).sum()),
            strictly_test_cases=int(
                (~cases.calibration & ~cases.training_overlap).sum()
            ),
            forecast_members=config["forecast_members"],
            torch_version=torch.__version__,
        ),
    )
    provenance = out / "provenance"
    provenance.mkdir(exist_ok=True)
    snapshot = provenance / "model.pt"
    if not snapshot.exists():
        shutil.copy2(checkpoint, snapshot)
    elif digest(snapshot) != sha:
        raise ValueError("Checkpoint snapshot disagrees")
    # Snapshot readable executable package, not imported legacy prototype modules.
    if not (provenance / "monthly_diffusion").exists():
        shutil.copytree(
            ROOT / "monthly_diffusion",
            provenance / "monthly_diffusion",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(__file__, provenance / Path(__file__).name)
    # Analysis/plotting may evolve without changing inference. The actual
    # model/sampler/preprocessing implementation must match its snapshot.
    hashes = {}
    for module in (ROOT / "monthly_diffusion").rglob("*.py"):
        relative = module.relative_to(ROOT / "monthly_diffusion")
        if relative.parts[0] == "analysis":
            continue
        hashes[str(relative)] = digest(module)
        if hashes[str(relative)] != digest(provenance / "monthly_diffusion" / relative):
            raise ValueError(
                f"Inference code changed after forecasts began: {relative}"
            )
    save_json(out / "inference_code_sha256.json", hashes)
    return model, sources, co2, increments, cases, out


def infer(config, key, device, *, pilot=False):
    model, sources, co2, increments, cases, out = prepare(config, key, device)
    c = config["experiments"][key]
    total = config["forecast_members"]
    batch = config["inference_batch_size"]
    if pilot:
        # One real test initialization; subsequent run reuses the committed group.
        cases = cases[(~cases.calibration) & (~cases.training_overlap)].iloc[:1]
    started = time.monotonic()
    completed = 0
    for index, row in cases.iterrows():
        member, year = row.initialization_member, int(row.year)
        path = out / "forecasts" / f"djf{year}_{member}"
        path.mkdir(parents=True, exist_ok=True)
        history_dates, future = winter_dates(year)
        history_co2 = co2.sel(time=history_dates).values
        future_co2 = forecast_co2(history_co2[-1], future, increments)
        forcing = xr.DataArray(
            np.r_[history_co2, future_co2],
            dims="time",
            coords={"time": history_dates.append(future)},
            attrs=dict(units="ppm"),
        )
        atmosphere, ocean = sources[member]
        target_path = path / "target.zarr"
        if not target_path.exists():
            target = (
                xr.merge(
                    [
                        atmosphere[list(VARIABLES[:2])].sel(time=future[1:]),
                        ocean[["sst"]].sel(time=future[1:]),
                    ]
                )
                .reset_coords(drop=True)
                .load()
            )
            target["sst"] = target.sst.where(model.preprocessor.stats["sst"]["mask"])
            target["djf_days"] = (
                "time",
                [31.0, 31.0, 28.0]
                if c["calendar"] == "noleap"
                else future[1:].days_in_month.to_numpy(float),
            )
            target.attrs = dict(
                initialization_member=member, year=year, calendar=c["calendar"]
            )
            save_zarr(target, target_path)
        ic = None
        for start in range(0, total, batch):
            if pilot and start:
                break
            group_path = path / f"members_{start:03d}.zarr"
            if group_path.exists():
                continue
            if shutil.disk_usage(out).free < 5 * 1024**3:
                raise RuntimeError(
                    "Less than 5 GiB free; safely resume after freeing space"
                )
            tick = time.monotonic()
            if ic is None:
                ic = (
                    atmosphere.sel(time=history_dates).load(),
                    ocean.sel(time=history_dates).load(),
                )
            seed = int(
                np.random.SeedSequence(
                    [
                        config["seed"],
                        0 if key == "e3sm" else 1,
                        year,
                        list(sources).index(member),
                        start,
                    ]
                ).generate_state(1)[0]
            )
            n = min(batch, total - start)
            forecast = model.initialize(*ic, co2=forcing, members=n, seed=seed)
            np.testing.assert_array_equal(forecast.times, history_dates.values[::-1])
            latents, monthly = [], []
            for lead, date in enumerate(future):
                old = [h[:, 0].clone() for h in forecast.history]
                a, o = next(forecast)
                np.testing.assert_array_equal(a.time, [date.to_datetime64()])
                for i, z in enumerate((a, o)):
                    torch.testing.assert_close(
                        forecast.history[i][:, 1], old[i], rtol=0, atol=0
                    )
                    np.testing.assert_array_equal(
                        forecast.history[i][:, 0].cpu().numpy(), z.values[0]
                    )
                decoded_a, decoded_o = model.decode(a, o, co2=forcing)
                fields = xr.merge([decoded_a[list(VARIABLES[:2])], decoded_o[["sst"]]])
                for variable in VARIABLES:
                    mask = np.asarray(model.preprocessor.stats[variable]["mask"])
                    if not np.isfinite(fields[variable].values[..., mask]).all():
                        raise FloatingPointError(
                            f"Nonfinite forecast in trained support: {variable}"
                        )
                monthly.append(fields)
                latents.append(xr.Dataset(dict(latent_atmosphere=a, latent_ocean=o)))
            result = xr.merge([xr.concat(monthly, "time"), xr.concat(latents, "time")])
            result = result.assign_coords(
                member=np.arange(start, start + n), lead=("time", np.arange(1, 5))
            )
            result["co2"] = ("time", future_co2, dict(units="ppm"))
            result.attrs.update(
                initialization_member=member,
                year=year,
                seed=seed,
                initialization_month=str(history_dates[-1].date()),
                calendar=c["calendar"],
                history_order="current, previous at every step",
                checkpoint_sha256=digest(out / "provenance/model.pt"),
            )
            save_zarr(result, group_path)
            completed += n
            print(
                f"{key} DJF {year} {member}: {start + n}/{total} members, group {time.monotonic() - tick:.1f}s",
                flush=True,
            )
            if pilot:
                size = sum(
                    p.stat().st_size for p in group_path.rglob("*") if p.is_file()
                )
                report = dict(
                    passed=True,
                    elapsed_seconds=time.monotonic() - started,
                    group_seconds=time.monotonic() - tick,
                    group_members=n,
                    estimated_all_experiment_hours=(time.monotonic() - tick)
                    * len(pd.read_csv(out / "cases.csv"))
                    * total
                    / n
                    / 3600,
                    estimated_monthly_archive_GiB=size
                    * len(pd.read_csv(out / "cases.csv"))
                    * total
                    / n
                    / 1024**3,
                    history_order_verified_all_steps=True,
                    valid_dates=[str(d.date()) for d in future],
                    forecast_co2=future_co2.tolist(),
                    device=str(model.device),
                )
                save_json(out / "pilot.json", report)
                print(json.dumps(report, indent=2), flush=True)
        if not pilot:
            save_json(
                path / "completion.json",
                dict(
                    members=total,
                    groups=list(range(0, total, batch)),
                    initialization_member=member,
                    year=year,
                ),
            )
    print(
        f"{key} inference complete, newly generated trajectories={completed}",
        flush=True,
    )


def score(config, key):
    """Process small latitude strips; no eager 40-winter forecast field stack."""
    c = config["experiments"][key]
    out = ROOT / config["output"] / key
    cases = pd.read_csv(out / "cases.csv")
    seasonal = []
    for index, row in cases.iterrows():
        path = out / "forecasts" / f"djf{row.year}_{row.initialization_member}"
        if not (path / "completion.json").exists():
            raise RuntimeError(f"Forecast not complete: {path}")
        means_path = path / "seasonal_means.zarr"
        if not means_path.exists():
            target = xr.open_zarr(path / "target.zarr", consolidated=True)
            groups = [
                xr.open_zarr(path / f"members_{start:03d}.zarr", consolidated=True)
                for start in range(
                    0, config["forecast_members"], config["inference_batch_size"]
                )
            ]
            monthly = xr.concat(
                groups, "member", data_vars="minimal", coords="minimal", compat="equals"
            )
            for group in groups:
                if (
                    group.attrs.get("initialization_member", row.initialization_member)
                    != row.initialization_member
                ):
                    raise ValueError("Forecast/target initialization-member mismatch")
                if int(group.attrs.get("year", row.year)) != int(row.year):
                    raise ValueError("Forecast/target winter mismatch")
            np.testing.assert_array_equal(
                monthly.member, np.arange(config["forecast_members"])
            )
            _, valid = winter_dates(int(row.year))
            np.testing.assert_array_equal(monthly.time, valid.values)
            np.testing.assert_array_equal(target.time, valid[1:].values)
            fields = {}
            for variable in VARIABLES:
                factor = 86400.0 if variable == "mtpr" else 1.0
                t = (
                    djf_mean(
                        target[variable].transpose("time", "lat", "lon").values,
                        calendar=c["calendar"],
                        year=int(row.year),
                    )
                    * factor
                )
                f = (
                    djf_mean(
                        monthly[variable]
                        .isel(time=slice(1, None))
                        .transpose("time", "member", "lat", "lon")
                        .values,
                        calendar=c["calendar"],
                        year=int(row.year),
                    )
                    * factor
                )
                fields["target_" + variable] = (
                    ("lat", "lon"),
                    t,
                    dict(units=UNITS[variable]),
                )
                fields["forecast_" + variable] = (
                    ("member", "lat", "lon"),
                    f,
                    dict(units=UNITS[variable]),
                )
            means = xr.Dataset(
                fields,
                coords=dict(lat=target.lat, lon=target.lon, member=monthly.member),
                attrs=dict(row),
            )
            save_zarr(means, means_path)
            for dataset in groups + [target]:
                dataset.close()
        seasonal.append(
            xr.open_zarr(means_path, consolidated=True).expand_dims(case=[index])
        )
    means = xr.concat(seasonal, "case", combine_attrs="drop")
    calibration = cases.calibration.to_numpy(bool)
    evaluation = ~calibration
    evaluated = cases[evaluation]
    years = evaluated.year.to_numpy(int)
    n = len(evaluated)
    lat, lon = means.lat.values, means.lon.values
    area = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon)))
    summaries, bins = [], []
    scores_path = out / "scores"
    scores_path.mkdir(exist_ok=True)
    for variable in VARIABLES:
        destination = scores_path / f"{variable}.zarr"
        if destination.exists():
            ds = xr.open_zarr(destination, consolidated=True).load()
        else:
            shape = (len(lat), len(lon))
            p_all = np.full((n, 3, *shape), np.nan, dtype=np.float32)
            y_all = np.full((n, *shape), -1, dtype=np.int8)
            s_all = np.full((n, *shape), np.nan)
            c_all = np.full_like(s_all, np.nan)
            qref, qmd = np.full((2, *shape), np.nan), np.full((2, *shape), np.nan)
            masks = {
                name: np.zeros(shape, bool)
                for name in (
                    "valid",
                    "reference_degenerate",
                    "forecast_degenerate",
                    "reference_ties",
                    "forecast_ties",
                )
            }
            cr_all, cm_all = (
                np.zeros((3, *shape), np.int32),
                np.zeros((3, *shape), np.int32),
            )
            target_eval = np.full((n, *shape), np.nan)
            for start in range(0, len(lat), 10):
                sl = slice(start, start + 10)
                t = means["target_" + variable].isel(lat=sl).values
                f = means["forecast_" + variable].isel(lat=sl).values
                qr, vr, dr, tr, cr = terciles(t[calibration])
                qm, vm, dm, tm, cm = terciles(f[calibration].reshape(-1, *t.shape[1:]))
                valid = (
                    vr
                    & vm
                    & np.isfinite(t[evaluation]).all(0)
                    & np.isfinite(f[evaluation]).all((0, 1))
                )
                p = np.where(
                    valid[None, None], probabilities(f[evaluation], qm), np.nan
                )
                y = np.where(valid[None], categorize(t[evaluation], qr), -1)
                np.testing.assert_allclose(p.sum(1)[:, valid], 1.0)
                s, baseline = rps(p, y), rps(np.full_like(p, 1 / 3), y)
                np.testing.assert_allclose(
                    baseline[:, valid], np.where(y[:, valid] == 1, 2 / 9, 5 / 9)
                )
                p_all[:, :, sl], y_all[:, sl] = p, y
                s_all[:, sl], c_all[:, sl] = s, baseline
                qref[:, sl], qmd[:, sl] = qr, qm
                cr_all[:, sl], cm_all[:, sl] = cr, cm
                target_eval[:, sl] = t[evaluation]
                for name, values in zip(masks, (valid, dr, dm, tr, tm)):
                    masks[name][sl] = values
            fields = dict(
                reference_threshold=(("quantile", "lat", "lon"), qref),
                forecast_threshold=(("quantile", "lat", "lon"), qmd),
                reference_calibration_category_count=(
                    ("category", "lat", "lon"),
                    cr_all,
                ),
                forecast_calibration_category_count=(
                    ("category", "lat", "lon"),
                    cm_all,
                ),
                target_mean=(("case", "lat", "lon"), target_eval),
                forecast_probability=(("case", "category", "lat", "lon"), p_all),
                target_category=(("case", "lat", "lon"), y_all),
                rps_md=(("case", "lat", "lon"), s_all),
                rps_climatology=(("case", "lat", "lon"), c_all),
            )
            fields.update(
                {name: (("lat", "lon"), value) for name, value in masks.items()}
            )
            for subset, select in (
                ("strict_test", years > config["training_years"][1] + 1),
                ("including_2007", np.ones(n, bool)),
            ):
                skill, count, numerator, denominator = aggregate_rpss(
                    s_all[select], c_all[select]
                )
                fields["rpss_" + subset] = (("lat", "lon"), skill)
                fields["case_count_" + subset] = (("lat", "lon"), count)
                fields["target_frequency_" + subset] = (
                    ("category", "lat", "lon"),
                    np.where(
                        masks["valid"][None],
                        np.stack([(y_all[select] == j).mean(0) for j in range(3)]),
                        np.nan,
                    ),
                )
            ds = xr.Dataset(
                fields,
                coords=dict(
                    lat=lat,
                    lon=lon,
                    case=evaluated.index.to_numpy(),
                    year=("case", years),
                    initialization_member=(
                        "case",
                        evaluated.initialization_member.to_numpy(str),
                    ),
                    category=[0, 1, 2],
                    quantile=[1 / 3, 2 / 3],
                ),
                attrs=dict(
                    variable=variable,
                    physical_units=UNITS[variable],
                    calendar=c["calendar"],
                    calibration_winters="1967-2006",
                    definition="1 - sum_case(RPS forecast) / sum_case(RPS equal terciles); same-member target",
                    interpretation="Fixed historical thresholds: skill includes forced change, not isolated IC skill",
                    ties="lower <= q1; middle q1 < x <= q2; upper > q2; degenerate/material ties excluded",
                ),
            )
            save_zarr(ds, destination)
        for subset, select in (
            ("strict_test", years > config["training_years"][1] + 1),
            ("including_2007", np.ones(n, bool)),
        ):
            skill, count, numerator, denominator = aggregate_rpss(
                ds.rps_md.values[select], ds.rps_climatology.values[select]
            )
            valid = ds.valid.values
            weights = np.where(valid, area, 0.0)
            if weights.sum() == 0:
                raise ValueError(
                    f"No valid tercile-verification domain for {key}/{variable}"
                )
            summary = dict(
                experiment=key,
                variable=variable,
                subset=subset,
                cases=int(select.sum()),
                winters=int(len(np.unique(years[select]))),
                valid_cells=int(valid.sum()),
                area_mean_rpss=float(np.nansum(skill * weights) / weights.sum()),
                pooled_area_weighted_rpss=float(
                    1 - np.sum(numerator * weights) / np.sum(denominator * weights)
                ),
                positive_skill_area_fraction=float(
                    weights[skill > 0].sum() / weights.sum()
                ),
            )
            for j, name in enumerate(("lower", "middle", "upper")):
                summary[name + "_target_frequency"] = float(
                    np.sum(((ds.target_category.values[select] == j).mean(0)) * weights)
                    / weights.sum()
                )
            summaries.append(summary)
            for domain, mask in (
                ("full", valid),
                ("positive_rpss_postselected", valid & (skill > 0)),
            ):
                rows = reliability(
                    ds.forecast_probability.values[select],
                    ds.target_category.values[select],
                    area,
                    mask,
                    n_members=config["forecast_members"],
                )
                bins.extend(
                    dict(variable=variable, subset=subset, domain=domain, **row)
                    for row in rows
                )
            print(f"SCORED {key}: {summary}", flush=True)
        ds.close()
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(bins).to_csv(out / "reliability.csv", index=False)
    save_json(
        out / "verification_metadata.json",
        dict(
            main_period="Complete DJFs 2008-2014; January-year labels; all IC and target months in test interval",
            sensitivity_period="DJF 2007-2014; December 2006 target and Sep/Oct 2006 ICs overlap training",
            calibration="Separate model and reference terciles fit on identical DJF 1967-2006 winters; training member only",
            calibration_members=c.get("calibration_members", list(c["sources"])),
            evaluation_members=c.get("evaluation_members", list(c["sources"])),
            initialization="Real September and October history from the same trajectory as verification target",
            lead="History ends October; generate November-February; verify December-February",
            target="Same held-out E3SM member, never ensemble-mean truth"
            if key == "e3sm"
            else "ERA5/ORAS5 trajectory",
            calendar=c["calendar"],
            forecast_members=config["forecast_members"],
            strict_test_cases=int((~cases.calibration & ~cases.training_overlap).sum()),
            calibration_cases=int(cases.calibration.sum()),
            all_evaluation_cases=int((~cases.calibration).sum()),
            years_per_source_member=7,
            uncertainty="No spatial-independent error bars; only seven evaluation winters",
            interpretation="Fixed historical climatology: skill includes forced change; not isolated initialization skill",
            display="Yellow-to-red positive RPSS; gray nonpositive skill; white unavailable/degenerate support. All negative scores retained.",
            reliability="Cosine-area weighting within cases, equal weighting of winters and initialization members; empty bins omitted",
            secondary_reliability="Positive-RPSS cells selected on SAME evaluation outcomes; descriptive, selection-biased",
            archive="Three scored physical monthly fields and both full latent states; other prognostic variables can be re-decoded",
        ),
    )
    for dataset in seasonal:
        dataset.close()
    shutil.copy2(__file__, out / "provenance" / "completed_pipeline.py")


def comparison_map(config):
    """Both strict-test RPSS maps, one shared pooled display scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    out = ROOT / config["output"]
    paths = {
        (key, v): out / key / "scores" / f"{v}.zarr"
        for key in config["experiments"]
        for v in VARIABLES
    }
    if not all(p.exists() for p in paths.values()):
        return
    fields = {
        kv: xr.open_zarr(path, consolidated=True).rpss_strict_test.load()
        for kv, path in paths.items()
    }
    positive = np.concatenate([f.values[f.values > 0] for f in fields.values()])
    limit = (
        min(1.0, max(0.1, np.ceil(np.percentile(positive, 99) * 10) / 10))
        if positive.size
        else 1.0
    )
    clipped = bool(positive.size and positive.max() > limit)
    with plt.rc_context({"font.size": 11, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(
            2, 3, figsize=(13.8, 6.9), subplot_kw=dict(projection=ccrs.Robinson())
        )
        fig.subplots_adjust(
            left=0.08, right=0.99, bottom=0.18, top=0.86, wspace=0.06, hspace=0.08
        )
        for row, key in enumerate(config["experiments"]):
            for col, variable in enumerate(VARIABLES):
                ax = axes[row, col]
                field = fields[key, variable]
                value, lon = add_cyclic_point(field.values, coord=field.lon)
                ax.set_global()
                ax.pcolormesh(
                    lon,
                    field.lat,
                    np.where(np.isfinite(value) & (value <= 0), 1.0, np.nan),
                    transform=ccrs.PlateCarree(),
                    cmap=ListedColormap(["#D5D5D5"]),
                    rasterized=True,
                )
                ax.pcolormesh(
                    lon,
                    field.lat,
                    np.ma.masked_where(~np.isfinite(value) | (value <= 0), value),
                    transform=ccrs.PlateCarree(),
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=limit,
                    rasterized=True,
                )
                ax.coastlines(resolution="110m", linewidth=0.4)
                ax.text(
                    0,
                    1.01,
                    chr(97 + 3 * row + col) + ")",
                    transform=ax.transAxes,
                    weight="bold",
                )
                if row == 0:
                    ax.set_title(
                        dict(
                            mtpr="Precipitation",
                            t2m="2-meter Temperature",
                            sst="Sea Surface Temperature",
                        )[variable],
                        pad=12,
                    )
                if col == 0:
                    ax.text(
                        -0.065,
                        0.5,
                        config["experiments"][key]["label"],
                        transform=ax.transAxes,
                        rotation=90,
                        ha="center",
                        va="center",
                    )
        cbax = fig.add_axes([0.33, 0.115, 0.4, 0.02])
        fig.colorbar(
            ScalarMappable(norm=Normalize(0, limit), cmap="YlOrRd"),
            cax=cbax,
            orientation="horizontal",
            extend="max" if clipped else "neither",
            label="RPSS",
        )
        fig.suptitle(
            "Test-set DJF 2008–2014 · October initialization · 20 members",
            y=0.965,
            fontsize=14,
        )
        fig.text(
            0.535,
            0.905,
            "1967–2006 calibration: E3SM r1 / reanalysis · test verification: E3SM r2–r5 / reanalysis",
            ha="center",
            fontsize=10,
        )
        fig.text(
            0.535,
            0.015,
            "Gray: nonpositive skill; white: unavailable/excluded. Fixed historical terciles; skill may include forced change.",
            ha="center",
            fontsize=9,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"djf_rpss_strict_test_comparison.{suffix}", dpi=350)
        plt.close(fig)
    save_json(
        out / "comparison_scale.json",
        dict(vmin=0, vmax=limit, saturated=clipped, positive_percentile=99),
    )
    summaries = [
        pd.read_csv(out / key / "summary.csv") for key in config["experiments"]
    ]
    pd.concat(summaries, ignore_index=True).to_csv(out / "summary.csv", index=False)


def plot(config, key):
    """RPSS display masks negatives only; saved skill retains every value."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from cartopy.util import add_cyclic_point
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    out = ROOT / config["output"] / key
    datasets = {
        v: xr.open_zarr(out / "scores" / f"{v}.zarr", consolidated=True).load()
        for v in VARIABLES
    }
    bins = pd.read_csv(out / "reliability.csv")
    colors = ["#277DA1", "#8764A2", "#D65B32"]
    limits = {}
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    ):
        for subset in ("strict_test", "including_2007"):
            positives = np.concatenate(
                [
                    d["rpss_" + subset].values[d["rpss_" + subset].values > 0]
                    for d in datasets.values()
                ]
            )
            limit = (
                min(1.0, max(0.1, np.ceil(np.percentile(positives, 99) * 10) / 10))
                if positives.size
                else 1.0
            )
            clipped = bool(positives.size and positives.max() > limit)
            limits[subset] = dict(vmin=0, vmax=limit, percentile=99, saturated=clipped)
            for domain in ("full", "positive_rpss_postselected"):
                fig = plt.figure(figsize=(13.8, 8.4))
                grid = fig.add_gridspec(
                    2,
                    3,
                    left=0.065,
                    right=0.97,
                    bottom=0.25,
                    top=0.90,
                    wspace=0.23,
                    hspace=0.22,
                    height_ratios=[1, 1.30],
                )
                for col, (variable, title) in enumerate(
                    zip(
                        VARIABLES,
                        (
                            "Precipitation",
                            "2-meter Temperature",
                            "Sea Surface Temperature",
                        ),
                    )
                ):
                    ax = fig.add_subplot(grid[0, col], projection=ccrs.Robinson())
                    field = datasets[variable]["rpss_" + subset]
                    values, lon = add_cyclic_point(field.values, coord=field.lon)
                    ax.set_global()
                    ax.pcolormesh(
                        lon,
                        field.lat,
                        np.where(np.isfinite(values) & (values <= 0), 1.0, np.nan),
                        transform=ccrs.PlateCarree(),
                        cmap=ListedColormap(["#D5D5D5"]),
                        rasterized=True,
                    )
                    ax.pcolormesh(
                        lon,
                        field.lat,
                        np.ma.masked_where(
                            ~np.isfinite(values) | (values <= 0), values
                        ),
                        transform=ccrs.PlateCarree(),
                        cmap="YlOrRd",
                        vmin=0,
                        vmax=limit,
                        rasterized=True,
                    )
                    ax.coastlines(resolution="110m", linewidth=0.45, color=".3")
                    ax.set_title(title, pad=9)
                    ax.text(
                        0,
                        1.01,
                        chr(97 + col) + ")",
                        transform=ax.transAxes,
                        weight="bold",
                    )
                    lower = grid[1, col].subgridspec(
                        2, 1, height_ratios=[5, 0.85], hspace=0.16
                    )
                    ax = fig.add_subplot(lower[0])
                    histogram = fig.add_subplot(lower[1], sharex=ax)
                    ax.plot([0, 1], [0, 1], "--", lw=1, color=".5")
                    for j, label in enumerate(
                        ("Lower tercile", "Middle tercile", "Upper tercile")
                    ):
                        table = bins[
                            (bins.variable == variable)
                            & (bins.subset == subset)
                            & (bins.domain == domain)
                            & (bins.category == j)
                        ]
                        nonempty = table[table.case_equivalent_weight > 0]
                        ax.plot(
                            nonempty.mean_probability,
                            nonempty.observed_frequency,
                            "o-",
                            ms=3,
                            lw=1.5,
                            color=colors[j],
                            label=label,
                        )
                        total = table.case_equivalent_weight.sum()
                        if total:
                            histogram.plot(
                                table.bin / config["forecast_members"],
                                table.case_equivalent_weight / total,
                                color=colors[j],
                                lw=1.1,
                            )
                    ax.set(xlim=(-0.025, 1.025), ylim=(-0.025, 1.025))
                    ax.grid(alpha=0.16)
                    ax.tick_params(labelbottom=False)
                    ax.text(
                        0,
                        1.015,
                        chr(100 + col) + ")",
                        transform=ax.transAxes,
                        weight="bold",
                    )
                    histogram.set_xlabel("Forecast probability")
                    histogram.set_ylim(bottom=0)
                    if col == 0:
                        ax.set_ylabel("Observed frequency")
                        histogram.set_ylabel("Freq.", fontsize=9)
                    for axis in (ax, histogram):
                        axis.spines[["top", "right"]].set_visible(False)
                cbax = fig.add_axes([0.32, 0.145, 0.36, 0.018])
                fig.colorbar(
                    ScalarMappable(norm=Normalize(0, limit), cmap="YlOrRd"),
                    cax=cbax,
                    orientation="horizontal",
                    extend="max" if clipped else "neither",
                    label="RPSS",
                )
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.030),
                    ncol=3,
                    frameon=False,
                )
                period = (
                    "2008–2014"
                    if subset == "strict_test"
                    else "2007–2014 (DJF 2007 overlaps training)"
                )
                selection = (
                    " · positive-skill cells (post-selected)"
                    if domain != "full"
                    else ""
                )
                fig.suptitle(
                    f"{config['experiments'][key]['label']} · DJF {period}" + selection,
                    y=0.975,
                    fontsize=13,
                )
                calibration_label = (
                    "r1-only calibration" if key == "e3sm" else "reanalysis calibration"
                )
                fig.text(
                    0.5,
                    0.938,
                    f"October initialization · 20 members · {calibration_label}: 1967–2006",
                    ha="center",
                    fontsize=10,
                )
                destination = out / f"djf_rpss_{subset}_{domain}"
                fig.savefig(destination.with_suffix(".png"), dpi=350)
                fig.savefig(destination.with_suffix(".pdf"))
                plt.close(fig)
    save_json(out / "figure_scales.json", limits)
    for dataset in datasets.values():
        dataset.close()
