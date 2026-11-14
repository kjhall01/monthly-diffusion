"""Fit separate small atmosphere/ocean MLPs to raw global means, using CO2 only.

First run prepare_co2_global_means.py. No calendar, time index, other forcing,
seasonal adjustment, autoencoder output, or latent representation is used here.
"""

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
import xarray as xr

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-monthly-comparison")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "runs/test_gridpoint_no_co2/analysis/co2_global_means"
)
EPOCHS = 4000
WIDTH = 32
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
SEED = 20261112
torch.set_num_threads(2)

data = xr.load_dataset(OUTPUT / "reference_global_means.nc")
names = [name for name in data.data_vars if name != "co2"]
components = [data[name].attrs["component"] for name in names]
units = [data[name].attrs["units"] for name in names]
years = data.time.dt.year.values
co2 = data.co2.values.astype("float64")[:, None]
observed = np.column_stack([data[name].values for name in names])
assert np.isfinite(co2).all() and np.isfinite(observed).all()

# Ten five-year holdouts (1860–64, 1885–89, ..., 2085–89) sample the CO2 range.
# This evaluates interpolation within this simulation, not a future-only forecast.
block_holdout = ((years - 1850) // 5) % 5 == 2
experiments = ["full_range", "historical_only"]
training_masks = [~block_holdout, (years >= 1966) & (years <= 2006)]
predictions = np.full((2, *observed.shape), np.nan)
linear_predictions = np.full_like(predictions, np.nan)
metrics, training_history, model_records = [], [], []

for experiment_index, (experiment, train) in enumerate(
    zip(experiments, training_masks)
):
    if experiment == "full_range":
        evaluation_periods = {"training": train, "held_out_blocks": block_holdout}
    else:
        evaluation_periods = {
            "training": train,
            "early_1850_1957": years <= 1957,
            "ae_validation_1958_1965": (years >= 1958) & (years <= 1965),
            "future_2007_2100": years >= 2007,
            "late_2070_2100": years >= 2070,
        }
    for component in ("atmosphere", "ocean"):
        indices = [i for i, group in enumerate(components) if group == component]
        targets = observed[:, indices]
        # Both input/output scaling use TRAINING months only. Targets remain raw
        # physical global means; this affine scaling simply balances their units.
        x_mean, x_std = co2[train].mean(0), co2[train].std(0)
        y_mean, y_std = targets[train].mean(0), targets[train].std(0)
        assert np.all(x_std > 0) and np.all(y_std > 0)
        x = torch.tensor((co2 - x_mean) / x_std, dtype=torch.float32)
        y = torch.tensor((targets - y_mean) / y_std, dtype=torch.float32)
        x_train, y_train = x[train], y[train]

        torch.manual_seed(SEED)
        model = nn.Sequential(
            nn.Linear(1, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, WIDTH),
            nn.SiLU(),
            nn.Linear(WIDTH, len(indices)),
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, EPOCHS, eta_min=3e-5
        )
        started = perf_counter()
        # Fixed training budget: held-out targets never select epochs or settings.
        for epoch in range(1, EPOCHS + 1):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x_train) - y_train).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite MLP loss")
            loss.backward()
            optimizer.step()
            schedule.step()
            if epoch == 1 or epoch % 500 == 0:
                training_history.append(
                    dict(
                        experiment=experiment,
                        component=component,
                        epoch=epoch,
                        standardized_training_mse=float(loss.detach()),
                    )
                )
        model.eval()
        with torch.no_grad():
            fitted = model(x).numpy().astype("float64") * y_std + y_mean
        predictions[experiment_index][:, indices] = fitted

        # A linear CO2 regression checks how much the MLP's nonlinearity adds.
        design = np.column_stack([np.ones(len(co2)), (co2 - x_mean) / x_std])
        coefficients = np.linalg.lstsq(
            design[train], (targets[train] - y_mean) / y_std, rcond=None
        )[0]
        linear = (design @ coefficients) * y_std + y_mean
        linear_predictions[experiment_index][:, indices] = linear
        for split, selected in evaluation_periods.items():
            truth = targets[selected]
            variance = truth.var(axis=0)
            assert np.all(variance > 0)
            for estimator, prediction in (("mlp", fitted), ("linear", linear)):
                residual = prediction[selected] - truth
                mse = (residual**2).mean(0)
                r2 = 1 - mse / variance
                explained_variance = 1 - residual.var(0) / variance
                for j, index in enumerate(indices):
                    metrics.append(
                        dict(
                            experiment=experiment,
                            component=component,
                            estimator=estimator,
                            split=split,
                            variable=names[index],
                            units=units[index],
                            months=int(selected.sum()),
                            r2=float(r2[j]),
                            explained_variance=float(explained_variance[j]),
                            rmse=float(np.sqrt(mse[j])),
                            bias=float(residual[:, j].mean()),
                            target_std=float(np.sqrt(variance[j])),
                        )
                    )
        record = dict(
            experiment=experiment,
            component=component,
            variables=[names[i] for i in indices],
            architecture=[1, WIDTH, WIDTH, len(indices)],
            activation="SiLU",
            parameters=sum(p.numel() for p in model.parameters()),
            seed=SEED,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            training_months=int(train.sum()),
            training_co2_min=float(co2[train].min()),
            training_co2_max=float(co2[train].max()),
        )
        torch.save(
            dict(
                **record,
                model=model.state_dict(),
                input_mean=x_mean,
                input_std=x_std,
                output_mean=y_mean,
                output_std=y_std,
                training_time=data.time.values[train],
            ),
            OUTPUT / f"{experiment}_{component}_mlp.pt",
        )
        model_records.append(record)
        print(
            f"{experiment} / {component}: {perf_counter()-started:.1f} s; training MSE {float(loss.detach()):.5f}",
            flush=True,
        )

assert np.isfinite(predictions).all()
scores = pd.DataFrame(metrics)
scores.to_csv(OUTPUT / "scores.csv", index=False)
pd.DataFrame(training_history).to_csv(OUTPUT / "training_history.csv", index=False)
xr.Dataset(
    data_vars=dict(
        observed=(("time", "variable"), observed),
        mlp_prediction=(("experiment", "time", "variable"), predictions),
        linear_prediction=(("experiment", "time", "variable"), linear_predictions),
        used_for_training=(
            ("experiment", "time"),
            np.array(training_masks).astype("int8"),
        ),
        co2=("time", co2[:, 0]),
    ),
    coords=dict(
        time=data.time,
        variable=names,
        experiment=experiments,
        component=("variable", components),
        units=("variable", units),
    ),
    attrs=dict(
        data.attrs,
        input="CO2 ppm only",
        targets="raw monthly global means; seasonality retained",
    ),
).to_netcdf(OUTPUT / "predictions.nc")
(OUTPUT / "experiment.json").write_text(
    json.dumps(
        dict(
            models=model_records,
            full_range_holdout_years=np.unique(years[block_holdout]).tolist(),
            historical_training_period=[1966, 2006],
            score="R2 = 1 - mean squared prediction error / target variance within the evaluation split",
            explained_variance="1 - residual variance / target variance; unlike R2 this ignores constant bias",
            caveat="Single member and scenario: CO2 also tracks time and covarying forcings. Predictive association is not causal attribution.",
        ),
        indent=2,
    )
    + "\n"
)

primary = scores.query("experiment == 'full_range' and split == 'held_out_blocks'")
summary = primary.pivot(
    index=["component", "variable", "units"], columns="estimator", values="r2"
).reset_index()
future = scores.query(
    "experiment == 'historical_only' and split == 'future_2007_2100' and estimator == 'mlp'"
)
summary = summary.merge(
    future[["variable", "r2"]].rename(columns={"r2": "historical_mlp_future_r2"}),
    on="variable",
)
summary.to_csv(OUTPUT / "summary.csv", index=False)
print(summary.to_string(index=False), flush=True)

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
for ax, component in zip(axes, ("atmosphere", "ocean")):
    group = (
        summary[summary.component == component]
        .set_index("variable")
        .loc[[name for name, c in zip(names, components) if c == component]]
    )
    positions = np.arange(len(group))
    ax.barh(
        positions - 0.18, group.mlp * 100, height=0.34, label="MLP", color="#2066a8"
    )
    ax.barh(
        positions + 0.18,
        group.linear * 100,
        height=0.34,
        label="Linear CO₂",
        color="#a8b4c0",
    )
    ax.set_yticks(
        positions, group.index.str.replace("ohc700_minus_ohc300", "ohc300–700")
    )
    ax.invert_yaxis()
    ax.set_xlim(min(-10, 100 * group[["mlp", "linear"]].min().min() - 5), 110)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Held-out R² (%)")
    ax.set_title(f"{component.capitalize()} — separate MLP", fontweight="bold")
    ax.grid(axis="x", alpha=0.15)
    for y_position, value in zip(positions, group.mlp):
        ax.text(
            max(0, value * 100) + 1,
            y_position - 0.18,
            f"{value:.1%}",
            va="center",
            fontsize=9,
        )
axes[0].legend(loc="lower right", frameon=False)
fig.suptitle(
    "How much of the raw monthly global means can CO₂ predict?",
    fontsize=16,
    fontweight="bold",
)
fig.text(
    0.5,
    0.025,
    "1850–2100; test on ten held-out five-year blocks. Seasonal cycles retained.\n"
    "CO₂ is the only input. R² includes bias; negative values are worse than the test-period mean. Association does not establish causation.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.085, 1, 0.94), w_pad=3)
for extension in ("png", "pdf"):
    fig.savefig(OUTPUT / f"variance_explained.{extension}", dpi=180)
plt.close(fig)

factors = {
    "mtpr": 86400,
    "mslp": 0.01,
    "ssh": 100,
    "sic": 100,
    "ohc300": 1e-9,
    "ohc700_minus_ohc300": 1e-9,
}
display_units = {
    "mtpr": "mm/day",
    "mslp": "hPa",
    "ssh": "cm",
    "sic": "%",
    "ohc300": "GJ/m²",
    "ohc700_minus_ohc300": "GJ/m²",
}
for component, shape in (("atmosphere", (4, 4)), ("ocean", (3, 3))):
    indices = [i for i, c in enumerate(components) if c == component]
    for experiment_index, experiment in enumerate(experiments):
        fig, axes = plt.subplots(
            *shape, figsize=(16, 12 if component == "atmosphere" else 10), sharex=True
        )
        split = "held_out_blocks" if experiment == "full_range" else "future_2007_2100"
        for ax, index in zip(axes.flat, indices):
            name = names[index]
            factor = factors.get(name, 1)
            row = scores.query(
                "experiment == @experiment and split == @split and estimator == 'mlp' and variable == @name"
            ).iloc[0]
            ax.plot(
                data.time.values,
                observed[:, index] * factor,
                color="#777777",
                linewidth=0.6,
                alpha=0.75,
            )
            ax.plot(
                data.time.values,
                predictions[experiment_index, :, index] * factor,
                color="#2066a8",
                linewidth=1.5,
            )
            if experiment == "historical_only":
                ax.axvspan(
                    np.datetime64("1966-01"),
                    np.datetime64("2007-01"),
                    color="#d45c24",
                    alpha=0.13,
                )
            ax.set_title(
                f"{name.replace('ohc700_minus_ohc300', 'ohc300–700')}  |  R²={row.r2:.1%}",
                fontsize=10,
                fontweight="bold",
            )
            ax.set_ylabel(display_units.get(name, units[index]))
            ax.grid(alpha=0.15)
            ax.tick_params(axis="x", labelrotation=30)
            ax.set_xlim(data.time.values[0], data.time.values[-1])
        for ax in axes[-1]:
            ax.set_xlabel("Year (display only; not an input)")
        label = (
            "held-out five-year blocks"
            if experiment == "full_range"
            else "2007–2100, trained only on 1966–2006"
        )
        fig.suptitle(
            f"{component.capitalize()} global means from CO₂ only",
            fontsize=17,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.945,
            f"Gray: raw reference monthly means. Blue: MLP prediction. Panel scores: {label}.",
            ha="center",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.02, 1, 0.92), h_pad=1.6, w_pad=1.6)
        for extension in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{component}_{experiment}.{extension}", dpi=170)
        plt.close(fig)
print("Saved results:", OUTPUT, flush=True)
