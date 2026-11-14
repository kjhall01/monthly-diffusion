#!/usr/bin/env python
"""Raw annual global means: E3SM/reanalysis and their real-state-start emulators.

Streams only the selected physical variables; never runs inference. Spatial means use
cos(latitude) over valid cells, with fixed training-SST support for SST. Annual
means give equal weight to 12 monthly means, following long-rollout diagnostics.
No climatology subtraction, trend removal, rebaselining, or bias correction.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np
import pandas as pd
import xarray as xr
import yaml

from workflows.analysis.archives import ClimateArchive
from workflows.analysis.climate import annual_means
from workflows.analysis.enso_dynamics import fixed_sst_support
from workflows.analysis.long_rollout import save_dataset
from workflows.analysis.spatial import weighted_mean
from monthly_diffusion.preprocessing import Variable
from workflows.data.dataloader import validate_units

STYLE = {
    "t2m": ("2-meter Temperature", "°C", 1.0, -273.15, "K", "T2M (°C)"),
    "sst": ("Sea Surface Temperature", "°C", 1.0, 0.0, "degC", "SST (°C)"),
    "mtpr": (
        "Precipitation",
        "mm day⁻¹",
        86400.0,
        0.0,
        "kg m-2 s-1",
        "Precipitation (mm day⁻¹)",
    ),
}


def calculate(config, root):
    """Cache monthly native-unit means and complete-year display-unit means."""
    supports, provenance, tables = {}, {}, []
    for name, paths in config["supports"].items():
        path = root / f"{name}_sst_support.nc"
        if path.exists():
            support = xr.load_dataset(path).sst_support.astype(bool)
        else:
            logging.info("Deriving fixed training SST support: %s", name)
            support = fixed_sst_support(
                ClimateArchive(paths), tuple(config["training"])
            )
            save_dataset(support.rename("sst_support").to_dataset(), path)
        supports[name] = support
    for name, spec in config["sources"].items():
        support = supports[spec["support"]]
        archive = ClimateArchive(
            spec["paths"],
            forecast=spec.get("forecast", False),
            ocean_support=support,
            ocean_variables=("sst",),
        )
        if not np.array_equal(archive.ocean_support.values, support.values):
            raise ValueError(
                "Forecast's archived SST support differs from requested training support"
            )
        validate_units(
            xr.Dataset(
                {v: xr.DataArray(attrs={"units": archive.units[v]}) for v in STYLE}
            ),
            [Variable(v, "", STYLE[v][4]) for v in STYLE],
        )
        identity = dict(
            paths=list(map(str, archive.paths)),
            members=list(map(str, archive.members)),
            checkpoint_sha256=archive.record.get("checkpoint_sha256"),
            support_sha256=hashlib.sha256(
                np.asarray(support, dtype=np.uint8).tobytes()
            ).hexdigest(),
            source_metadata_sha256={
                str(p): hashlib.sha256(
                    (
                        p / ("forecast.json" if archive.forecast else ".zmetadata")
                    ).read_bytes()
                ).hexdigest()
                for p in archive.paths
            },
        )
        marker = root / f"{name}_identity.json"
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError("Source identity changed: use a new output directory")
        marker.write_text(json.dumps(identity, indent=2) + "\n")
        monthly_path = root / f"{name}_monthly.nc"
        if monthly_path.exists():
            monthly = xr.load_dataset(monthly_path)
        else:
            selected = np.flatnonzero(
                (archive.time.dt.year >= config["years"][0])
                & (archive.time.dt.year <= config["years"][1])
            )
            pieces = []
            for i, block in enumerate(archive.blocks(selected, 12)):
                part = {}
                for variable in config["variables"]:
                    field = archive.read(variable, block)
                    part[variable] = weighted_mean(field)
                    part[variable].attrs["units"] = archive.units[variable]
                pieces.append(xr.Dataset(part))
                if i % 25 == 0:
                    logging.info(
                        "%s: global means through %s",
                        name,
                        str(field.time.values[-1])[:7],
                    )
            monthly = xr.concat(pieces, "time")
            monthly.attrs.update(
                spatial_mean="cos(latitude), finite cells; SST uses training-fixed ocean support",
                processing="raw physical fields; no anomaly subtraction or bias adjustment",
            )
            save_dataset(monthly, monthly_path)
        annual = annual_means(monthly)
        for variable in config["variables"]:
            _, units, scale, offset, _, _ = STYLE[variable]
            annual[variable] = annual[variable] * scale + offset
            annual[variable].attrs["units"] = units
        annual.attrs.update(
            temporal_mean="equal-month average of complete 12-month calendar years only",
            note="Temperature converted K to degC; precipitation kg m-2 s-1 to mm/day. These are not anomalies.",
        )
        if any(not np.isfinite(annual[v]).all() for v in annual):
            raise ValueError("Nonfinite annual global mean")
        save_dataset(annual, root / f"{name}_annual.nc")
        for variable in config["variables"]:
            table = annual[variable].to_dataframe(name="value").reset_index()
            table["member"] = table.member.astype(str)
            table["statistic"] = "individual_member"
            mean = (
                annual[variable]
                .mean("member", skipna=False)
                .to_dataframe(name="value")
                .reset_index()
            )
            mean["member"], mean["statistic"] = "ensemble_mean", "ensemble_mean"
            tables.append(
                pd.concat([table, mean], ignore_index=True).assign(
                    source=name, variable=variable, units=STYLE[variable][1]
                )
            )
        provenance[name] = dict(
            **identity,
            first_month=str(monthly.time.values[0])[:10],
            last_month=str(monthly.time.values[-1])[:10],
            first_complete_year=int(annual.year[0]),
            last_complete_year=int(annual.year[-1]),
            months=monthly.sizes["time"],
            sst_support_cells=int(support.sum()),
            members_count=len(archive.members),
        )
    pd.concat(tables, ignore_index=True).to_csv(
        root / "annual_global_means.csv", index=False
    )
    (root / "data_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def draw_lines(ax, annual, sources, *, inset=False, style=None):
    """All members first, heavier means last, so member traces cannot hide means."""
    style = {} if style is None else style
    for name, values in annual.items():
        if values.sizes["member"] <= 1:
            continue
        spec = sources[name]
        ax.plot(
            values.year,
            values.transpose("year", "member"),
            color=spec["color"],
            ls=spec["linestyle"],
            lw=style.get("inset_member_linewidth", 0.65)
            if inset
            else style.get("member_linewidth", 0.8),
            alpha=style.get("member_alpha", 0.27),
            zorder=2,
        )
    for name, values in annual.items():
        spec = sources[name]
        ax.plot(
            values.year,
            values.mean("member", skipna=False),
            color=spec["color"],
            ls=spec["linestyle"],
            lw=style.get("inset_mean_linewidth", 1.5) if inset else 2.25,
            zorder=5 if name == "era5_reference" else 4,
        )


def plot(config, root):
    variables = config["variables"]
    if (
        not variables
        or len(set(variables)) != len(variables)
        or any(v not in STYLE for v in variables)
    ):
        raise ValueError("Select unique variables from t2m, sst, mtpr")
    data_root = Path(config.get("data_directory", root))
    data = {
        name: xr.load_dataset(data_root / f"{name}_annual.nc")
        for name in config["sources"]
    }
    main_years = config.get("main_years", config["years"])
    main_ticks = config.get("main_xticks", [1850, 1900, 1950, 2000, 2050, 2100])
    inset_ticks = config.get("inset_xticks", [1960, 1980, 2000, 2020])
    inset_axis_labels = config.get("show_inset_axis_labels", True)
    inset_tick_labels = config.get("show_inset_tick_labels", True)
    inset_title = config.get("show_inset_title", True)
    style_legend = config.get("show_style_legend", True)
    display = {}
    inset_checks = {}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    ):
        fig, axes = plt.subplots(
            1,
            len(variables),
            squeeze=False,
            figsize=config.get("figsize", (18.8 * len(variables) / 3, 6.4)),
        )
        fig.subplots_adjust(
            left=0.06,
            right=0.985,
            top=0.94,
            bottom=config.get("subplot_bottom", 0.245 if style_legend else 0.18),
            wspace=0.255,
        )
        for letter, variable, ax in zip("abc", variables, axes[0]):
            values = {name: ds[variable] for name, ds in data.items()}
            main_values = {
                name: v.sel(year=slice(*main_years)) for name, v in values.items()
            }
            ax.axvspan(
                config["training"][0], config["training"][1] + 1, color=".91", zorder=0
            )
            draw_lines(
                ax, main_values, config["sources"], style=config.get("line_style")
            )
            main_traces = list(ax.lines)
            ax.set(
                xlim=main_years,
                xlabel="Year" if config.get("show_main_axis_label", True) else "",
                ylabel=f"{STYLE[variable][0]} ({STYLE[variable][1]})",
                xticks=main_ticks,
            )
            ax.text(
                -0.12,
                1.025,
                f"{letter})",
                transform=ax.transAxes,
                weight="bold",
                fontsize=13,
            )
            ax.grid(axis="y", color=".88", linewidth=0.55, zorder=0)
            ax.spines[["top", "right"]].set_visible(config.get("full_frame", False))
            ax.tick_params(labelsize=9)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            if "main_box_aspect" in config:
                ax.set_box_aspect(config["main_box_aspect"])  # axes height / width

            # Reserve whitespace over the inset's actual horizontal footprint,
            # above or below the data. Insets must not conceal member traces.
            rect = config.get("inset_bounds", {}).get(
                variable, (0.105, 0.605, 0.555, 0.305)
            )
            clearance = config.get("inset_clearance", 0.17)
            x0, x1 = main_years
            inset_span = (
                x0 + (rect[0] - 0.05) * (x1 - x0),
                x0 + (rect[0] + rect[2] + 0.04) * (x1 - x0),
            )
            all_values = np.concatenate(
                [v.values.ravel() for v in main_values.values()]
            )
            lo, hi = all_values.min(), all_values.max()
            pad = max((hi - lo) * 0.065, 0.015)
            bottom, top = lo - pad, hi + pad
            footprint = [
                v.sel(year=slice(*inset_span))
                for v in main_values.values()
                if v.sel(year=slice(*inset_span)).sizes["year"]
            ]
            if rect[1] + rect[3] / 2 < 0.5:
                boundary = rect[1] + rect[3] + clearance
                local_min = min(float(v.min()) for v in footprint)
                bottom = min(
                    bottom, (local_min - pad - boundary * top) / (1 - boundary)
                )
            else:
                boundary = rect[1] - clearance
                local_max = max(float(v.max()) for v in footprint)
                top = max(top, bottom + (local_max + pad - bottom) / boundary)
            ax.set_ylim(bottom, top)
            zoom = ax.inset_axes(rect)
            inset_checks[variable] = (zoom, main_traces)
            zoom.set_facecolor("white")
            zoom.axvspan(
                config["training"][0], config["training"][1] + 1, color=".91", zorder=0
            )
            restricted = {
                name: v.sel(year=slice(*config["inset_years"]))
                for name, v in values.items()
            }
            draw_lines(
                zoom,
                restricted,
                config["sources"],
                inset=True,
                style=config.get("line_style"),
            )
            zoom.set(
                xlim=config["inset_years"],
                xlabel="Year" if inset_axis_labels else "",
                ylabel=STYLE[variable][5] if inset_axis_labels else "",
                xticks=inset_ticks,
            )
            if inset_title:
                zoom.set_title(
                    f"{config['inset_years'][0]}–{config['inset_years'][1]}",
                    fontsize=9,
                    pad=4,
                )
            zoom.xaxis.label.set_size(8)
            zoom.yaxis.label.set_size(8)
            zoom.xaxis.labelpad = 1.5
            zoom.yaxis.labelpad = 2
            zoom.tick_params(axis="both", labelsize=7.5, length=2.5, pad=2)
            zoom.tick_params(labelbottom=inset_tick_labels, labelleft=inset_tick_labels)
            zoom.yaxis.set_major_locator(MaxNLocator(3))
            zoom.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            zoom.grid(axis="y", color=".9", linewidth=0.45)
            for spine in zoom.spines.values():
                spine.set_edgecolor(".55")
                spine.set_linewidth(0.6)
            window_box = None
            if config.get("outline_main_window", main_years != config["years"]):
                # Outline the EXACT main-panel x/y extent on the longer timeline.
                # Ensure all four edges are visible even if the main panel has
                # extra vertical headroom reserved for its inset.
                zlo, zhi = zoom.get_ylim()
                margin = 0.025 * (zhi - zlo)
                zoom.set_ylim(min(zlo, bottom - margin), max(zhi, top + margin))
                window_box = [
                    main_years[0],
                    bottom,
                    main_years[1] - main_years[0],
                    top - bottom,
                ]
                zoom.add_patch(
                    Rectangle(
                        window_box[:2],
                        window_box[2],
                        window_box[3],
                        fill=False,
                        edgecolor=".32",
                        linewidth=0.9,
                        zorder=7,
                    )
                )
            inset_window_box = None
            connectors = []
            if config.get("outline_inset_window", False):
                zlo, zhi = zoom.get_ylim()
                inset_window_box = [
                    config["inset_years"][0],
                    zlo,
                    config["inset_years"][1] - config["inset_years"][0],
                    zhi - zlo,
                ]
                ax.add_patch(
                    Rectangle(
                        inset_window_box[:2],
                        inset_window_box[2],
                        inset_window_box[3],
                        fill=False,
                        edgecolor=".32",
                        linewidth=0.9,
                        zorder=7,
                    )
                )
                if config.get("connect_inset_corners", False):
                    # Draw below the member traces and inset. End exactly at
                    # the selected corners, with no arrowheads or endpoint gaps.
                    connection_style = config.get("inset_connection_styles", {}).get(
                        variable, config.get("inset_connection_style")
                    )
                    if connection_style == "bottom_to_top":
                        pairs = (
                            ((0, 0), (config["inset_years"][0], zhi)),
                            ((1, 0), (config["inset_years"][1], zhi)),
                        )
                    elif connection_style == "top_to_bottom":
                        pairs = (
                            ((0, 1), (config["inset_years"][0], zlo)),
                            ((1, 1), (config["inset_years"][1], zlo)),
                        )
                    else:
                        pairs = (
                            ((0, 0), (config["inset_years"][0], zlo)),
                            ((1, 1), (config["inset_years"][1], zhi)),
                        )
                    for corner, target in pairs:
                        route = "straight"
                        if corner[1] == 0 and rect[1] + rect[3] / 2 < 0.5:
                            # Bottom corners of a lower inset must exit to
                            # either side before turning upward, otherwise
                            # the connector would cross the inset's data.
                            start_x = x0 + (rect[0] + corner[0] * rect[2]) * (x1 - x0)
                            start_y = bottom + rect[1] * (top - bottom)
                            elbow_x = start_x + (0.025 if corner[0] else -0.025) * (
                                x1 - x0
                            )
                            ax.plot(
                                [start_x, elbow_x, elbow_x, target[0]],
                                [start_y, start_y, target[1], target[1]],
                                color=".55",
                                linewidth=0.7,
                                alpha=0.85,
                                zorder=1.5,
                            )
                            route = "outside-right" if corner[0] else "outside-left"
                        else:
                            ax.add_artist(
                                ConnectionPatch(
                                    xyA=corner,
                                    coordsA=zoom.transAxes,
                                    xyB=target,
                                    coordsB=ax.transData,
                                    color=".55",
                                    linewidth=0.7,
                                    alpha=0.85,
                                    zorder=1.5,
                                    clip_on=False,
                                )
                            )
                        connectors.append(
                            dict(
                                inset_corner=list(corner),
                                main_point=list(target),
                                route=route,
                            )
                        )
            display[variable] = dict(
                panel_label=f"{letter})",
                main_years=main_years,
                inset_years=config["inset_years"],
                inset_axis_labels=inset_axis_labels,
                inset_tick_labels=inset_tick_labels,
                inset_title=inset_title,
                main_box_aspect=ax.get_box_aspect(),
                main_xlabel=ax.get_xlabel(),
                full_frame=config.get("full_frame", False),
                corner_connectors=connectors,
                line_style=config.get("line_style", {}),
                main_ylim=[bottom, top],
                inset_ylim=list(zoom.get_ylim()),
                inset_bounds_axes=list(rect),
                line_clearance_main_axis_fraction=boundary,
                inset_occludes_main_lines=False,
                main_window_box_xywh=window_box,
                inset_window_box_xywh=inset_window_box,
                units=STYLE[variable][1],
            )
        handles = [
            Line2D(
                [], [], color=s["color"], ls=s["linestyle"], lw=2.25, label=s["label"]
            )
            for s in config["sources"].values()
        ]
        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(
                0.52,
                config.get("legend_y", 0.105 if style_legend else 0.04),
            ),
            ncol=4,
            frameon=False,
            fontsize=10,
            handlelength=3.4,
            columnspacing=2.3,
        )
        if style_legend:
            fig.legend(
                handles=[
                    Line2D(
                        [], [], color=".2", lw=2.25, label="Ensemble mean / reanalysis"
                    ),
                    Line2D(
                        [],
                        [],
                        color=".5",
                        lw=0.8,
                        alpha=0.55,
                        label="Individual member",
                    ),
                    Patch(
                        facecolor=".91", edgecolor="none", label="Training: 1966–2006"
                    ),
                ],
                loc="lower center",
                bbox_to_anchor=(0.52, 0.037),
                ncol=3,
                frameon=False,
                fontsize=9,
                handlelength=2.8,
                columnspacing=2.6,
            )
        # Check the rendered line segments (not only annual sample points),
        # including a stroke-width margin around each inset's opaque frame.
        fig.canvas.draw()
        for variable, (zoom, traces) in inset_checks.items():
            margin = max(line.get_linewidth() for line in traces) * fig.dpi / 144 + 1
            bounds = zoom.bbox.padded(margin)
            intersections = sum(
                line.get_path()
                .transformed(line.get_transform())
                .intersects_bbox(bounds, filled=False)
                for line in traces
            )
            display[variable]["inset_occludes_main_lines"] = bool(intersections)
            display[variable]["main_trace_intersections_with_inset"] = int(
                intersections
            )
            if intersections and config.get("verify_no_inset_overlap", False):
                raise ValueError(
                    f"{variable}: inset overlaps {intersections} main-panel traces"
                )
        for extension in ("png", "pdf"):
            fig.savefig(
                root / f"global_mean_trajectories.{extension}",
                dpi=320,
                facecolor="white",
                bbox_inches="tight",
            )
        plt.close(fig)
    (root / "display.json").write_text(json.dumps(display, indent=2) + "\n")
