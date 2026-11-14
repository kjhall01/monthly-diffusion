"""The display must preserve absolute levels, complete years and each member."""
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from workflows.experiments import plot_global_trajectories_comparison as plotter

ROOT = Path(__file__).resolve().parents[1]


def test_absolute_units_and_complete_years():
    time = xr.date_range("1958-03-01", "1960-12-01", freq="MS")
    values = xr.DataArray(np.ones((2, len(time))), dims=("member", "time"),
                         coords=dict(time=time, member=[0, 1]))
    monthly = xr.Dataset(dict(t2m=values*288.15, sst=values*18., mtpr=values*(3/86400)))
    annual = plotter.annual_means(monthly)
    np.testing.assert_array_equal(annual.year, [1959, 1960])
    for variable, expected in (("t2m", 15.), ("sst", 18.), ("mtpr", 3.)):
        _, _, scale, offset, _, _ = plotter.STYLE[variable]
        np.testing.assert_allclose(annual[variable]*scale+offset, expected)


def test_members_and_arithmetic_ensemble_means_are_all_drawn():
    data = xr.DataArray([[11., 12.], [14., 19.], [18., 20.]], dims=("member", "year"),
                        coords=dict(member=[0, 1, 2], year=[2000, 2001]))
    sources = dict(e3sm_emulator=dict(color="blue", linestyle="dashed"),
                   era5_reference=dict(color="black", linestyle="solid"))
    fig, ax = plotter.plt.subplots()
    plotter.draw_lines(ax, dict(e3sm_emulator=data, era5_reference=data.isel(member=[0])), sources)
    assert len(ax.lines) == 5  # three members, their ensemble mean, one reference
    np.testing.assert_allclose(ax.lines[3].get_ydata(), data.mean("member"))
    assert ax.lines[3].get_linewidth() > ax.lines[0].get_linewidth()
    assert ax.lines[3].get_linestyle() == "--"
    assert ax.lines[4].get_color() == "black"
    plotter.plt.close(fig)


def test_reversed_view_uses_same_data_and_outlines_main_window(tmp_path):
    import json
    import yaml
    config = yaml.safe_load((ROOT/"examples/global-trajectories-comparison-recent-focus.yaml").read_text())
    config["data_directory"] = str(tmp_path)
    years = np.arange(1850, 2101)
    for i, name in enumerate(config["sources"]):
        values = np.tile((years-1850)/100., (2, 1))+i*.1
        ds = xr.Dataset({v: (("member", "year"), values+10) for v in config["variables"]},
                        coords=dict(member=[0, 1], year=years))
        ds.to_netcdf(tmp_path/f"{name}_annual.nc")
    plotter.plot(config, tmp_path)
    display = json.loads((tmp_path/"display.json").read_text())
    for variable, record in display.items():
        assert record["main_years"] == [1958, 2025]
        assert record["inset_years"] == [1850, 2100]
        assert record["inset_axis_labels"] is False
        x, y, width, height = record["main_window_box_xywh"]
        assert x == 1958 and width == 67
        np.testing.assert_allclose([y, y+height], record["main_ylim"])
        assert record["inset_ylim"][0] < y
        assert record["inset_ylim"][1] > y+height
    assert (tmp_path/"global_mean_trajectories.pdf").exists()


@pytest.mark.parametrize('filename', ['global-trajectories-comparison-wide-full-period.yaml',
                                     'global-trajectories-comparison-connected-insets.yaml',
                                     'global-trajectories-comparison-larger-insets.yaml',
                                     'global-trajectories-comparison-bottom-top-connectors.yaml',
                                     'global-trajectories-comparison-facing-connectors.yaml',
                                     'global-trajectories-comparison-t2m-precip.yaml'])
def test_wide_full_timeline_with_unlabeled_insets_and_outer_boxes(tmp_path, filename):
    import json
    import yaml
    config = yaml.safe_load((ROOT/"examples"/filename).read_text())
    config["data_directory"] = str(tmp_path)
    years = np.arange(1850, 2101)
    for i, name in enumerate(config["sources"]):
        trend = np.tile((years-1850)/100., (2, 1))
        ds = xr.Dataset({v: (("member", "year"), (10+i*.1)+trend*(-1 if v == 'mtpr' else 1))
                         for v in config["variables"]}, coords=dict(member=[0, 1], year=years))
        ds.to_netcdf(tmp_path/f"{name}_annual.nc")
    plotter.plot(config, tmp_path)
    display = json.loads((tmp_path/"display.json").read_text())
    assert list(display) == config['variables']
    assert [record['panel_label'] for record in display.values()] == [f'{letter})' for letter in 'abc'[:len(config['variables'])]]
    for variable, record in display.items():
        assert record["main_years"] == [1850, 2100]
        assert record["inset_years"] == [1958, 2025]
        assert not any(record[k] for k in ('inset_axis_labels', 'inset_tick_labels', 'inset_title'))
        assert record['main_window_box_xywh'] is None
        x, y, width, height = record['inset_window_box_xywh']
        assert x == 1958 and width == 67
        np.testing.assert_allclose([y, y+height], record['inset_ylim'])
        assert record['main_ylim'][0] < y < y+height < record['main_ylim'][1]
        assert record['main_box_aspect'] == config['main_box_aspect']
        if config.get('verify_no_inset_overlap'):
            assert not record['inset_occludes_main_lines']
            assert record['main_trace_intersections_with_inset'] == 0
            if not config.get('connect_inset_corners'):
                assert record['corner_connectors'] == []
            assert record['main_xlabel'] == '' and record['full_frame']
        if config.get('connect_inset_corners'):
            assert record['main_xlabel'] == '' and record['full_frame']
            style = config.get('inset_connection_styles', {}).get(variable, config.get('inset_connection_style'))
            if style == 'top_to_bottom':
                assert [c['inset_corner'] for c in record['corner_connectors']] == [[0, 1], [1, 1]]
                np.testing.assert_allclose(record['corner_connectors'][0]['main_point'], [x, y])
                np.testing.assert_allclose(record['corner_connectors'][1]['main_point'], [x+width, y])
                assert all(c['route'] == 'straight' for c in record['corner_connectors'])
            else:
                bottom_to_top = style == 'bottom_to_top'
                assert [c['inset_corner'] for c in record['corner_connectors']] == [[0, 0], [1, 0] if bottom_to_top else [1, 1]]
                np.testing.assert_allclose(record['corner_connectors'][0]['main_point'], [x, y+height if bottom_to_top else y])
                np.testing.assert_allclose(record['corner_connectors'][1]['main_point'], [x+width, y+height])
            assert record['line_style']['member_alpha'] > .27
    assert display['mtpr']['inset_bounds_axes'][1] < .1
    rect = display['t2m']['inset_bounds_axes']
    assert rect[1]+rect[3]/2 > .5
    if config.get('connect_inset_corners'):
        style = config.get('inset_connection_styles', {}).get('mtpr', config.get('inset_connection_style'))
        if style != 'top_to_bottom':
            assert display['mtpr']['corner_connectors'][0]['route'] == 'outside-left'
            if style == 'bottom_to_top':
                assert display['mtpr']['corner_connectors'][1]['route'] == 'outside-right'


def test_stronger_members_remain_thinner_than_the_mean():
    data = xr.DataArray([[11., 12.], [14., 19.]], dims=('member', 'year'),
                       coords=dict(member=[0, 1], year=[2000, 2001]))
    fig, ax = plotter.plt.subplots()
    plotter.draw_lines(ax, dict(ensemble=data), dict(ensemble=dict(color='blue', linestyle='solid')),
                       inset=True, style=dict(member_alpha=.55, inset_member_linewidth=.85, inset_mean_linewidth=1.8))
    assert ax.lines[0].get_alpha() == .55
    assert ax.lines[0].get_linewidth() == .85
    assert ax.lines[-1].get_linewidth() == 1.8
    plotter.plt.close(fig)
