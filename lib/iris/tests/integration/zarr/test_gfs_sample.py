# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""A published Zarr store, loaded and saved.

A subset of the NOAA GFS forecast archive published by dynamical.org, held in
``iris-test-data`` under ``Zarr/``. Every synthetic fixture in this package
was written by someone who already knew the rules; this store was not, and
the things it does differently from the synthetic one are the point of it.

The design document is ``docs/superpowers/specs/2026-09-21-zarr-io-design.md``;
section 7 says what the subset keeps and why.
"""

import warnings

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
xr = pytest.importorskip("xarray")

import iris
from iris.loading import LOAD_PROBLEMS
from iris.tests import _shared_utils
from iris.warnings import IrisLoadWarning

#: The shards and inner chunks the published store is written with.
SOURCE_SHARDS = (1, 4, 120, 120)
SOURCE_CHUNKS = (1, 2, 40, 40)


@pytest.fixture
def sample():
    return _shared_utils.get_data_path(("Zarr", "gfs_forecast_sample.zarr"))


@pytest.fixture(autouse=True)
def _assert_clean_load():
    """Every load in this module must be free of recorded problems.

    The published store loads silently, so anything recorded here is a
    change in Iris rather than a quirk of the data. Same shape as the
    fixture of this name in ``test_load.py``.
    """
    LOAD_PROBLEMS.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", IrisLoadWarning)
        yield
    assert LOAD_PROBLEMS.problems == []


@_shared_utils.skip_data
def test_the_sample_yields_four_cubes(sample):
    cubes = iris.load(sample)
    assert sorted(cube.name() for cube in cubes) == [
        "air_temperature",
        "eastward_wind",
        "northward_wind",
        "precipitation_flux",
    ]


@_shared_utils.skip_data
def test_the_sharded_variable_loads_lazily(sample):
    cube = iris.load_cube(sample, "air_temperature")
    assert cube.has_lazy_data()
    assert cube.shape == (2, 8, 120, 120)


@_shared_utils.skip_data
def test_the_two_dimensional_time_coordinate_spans_both_its_dimensions(sample):
    """``valid_time`` is a function of initialisation and lead time together.

    It carries ``standard_name: time``, so it shares its Iris name with the
    ``init_time`` dimension coordinate and ``var_name`` is what tells them
    apart.
    """
    cube = iris.load_cube(sample, "air_temperature")
    assert cube.coord_dims(cube.coord(var_name="valid_time")) == (0, 1)


@_shared_utils.skip_data
def test_the_base64_fill_value_is_read_and_masks_nothing(sample):
    """A NaN sentinel in xarray's base64 encoding, over data with no NaNs.

    Read wrongly the attribute raises rather than quietly doing nothing,
    which is why this asserts the unmasked result and not just that the
    load succeeded.
    """
    cube = iris.load_cube(sample, "air_temperature")
    assert not np.ma.is_masked(cube.data)


@_shared_utils.skip_data
def test_the_crs_variable_arrives_as_a_coordinate_not_a_coordinate_system(sample):
    """The store names ``spatial_ref`` in ``coordinates``, not ``grid_mapping``.

    So there is nothing for Iris to build a coordinate system from, and the
    CRS arrives as a scalar coordinate carrying ``crs_wkt``. CF section 5.6
    asks for a ``grid_mapping`` attribute; this publisher does not write one.
    Whether Iris should fall back to reading ``crs_wkt`` off a coordinate is
    :issue:`7352`.
    """
    cube = iris.load_cube(sample, "air_temperature")
    assert cube.coord("latitude").coord_system is None
    assert "crs_wkt" in cube.coord("spatial_ref").attributes


@_shared_utils.skip_data
def test_an_attribute_named_dtype_does_not_shadow_the_dtype(sample):
    """The publisher records the source NumPy dtype in an attribute of that name.

    ``dtype`` is one of the names that used to be shadowed on load, because
    it collides with a property of the backend variable object. Here the
    attribute says ``timedelta64[us]`` while the coordinate's own dtype is
    float64, and both survive.
    """
    cube = iris.load_cube(sample, "air_temperature")
    coord = cube.coord("forecast_period")
    assert coord.attributes["dtype"] == "timedelta64[us]"
    assert coord.dtype == np.dtype("f8")


@_shared_utils.skip_data
def test_a_structured_attribute_survives(sample):
    """CF cannot express a nested attribute; Zarr allows one. Carry it."""
    cube = iris.load_cube(sample, "air_temperature")
    statistics = cube.coord(var_name="valid_time").attributes["statistics_approximate"]
    assert set(statistics) == {"min", "max"}


@_shared_utils.skip_data
def test_the_sample_round_trips(tmp_path, sample):
    """Saving adds a ``Conventions`` global, and changes nothing else."""
    cube = iris.load_cube(sample, "air_temperature")
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(cube, tmp_path / "rt.zarr")

    back = iris.load_cube(str(tmp_path / "rt.zarr"))
    assert back.attributes.globals["Conventions"] == "CF-1.7"
    del back.attributes.globals["Conventions"]
    assert back == cube


@_shared_utils.skip_data
def test_an_existing_store_is_replaced_under_mode_w(tmp_path, sample):
    """``mode="w"`` is the third of the three modes, and the least pinned.

    ``test_save.py`` covers what ``mode="w-"`` refuses and what ``mode="a"``
    adds. This covers the one that replaces, which is what a script rerun
    over the same path depends on.
    """
    path = tmp_path / "replaced.zarr"
    first = iris.load_cube(sample, "air_temperature")
    second = iris.load_cube(sample, "eastward_wind")
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(first, path)
        iris.save(second, path, mode="w")

    names = [cube.name() for cube in iris.load(str(path))]
    assert names == ["eastward_wind"]


@_shared_utils.skip_data
def test_the_round_trip_does_not_keep_the_source_chunk_layout(tmp_path, sample):
    """Iris sizes chunks itself, so a store's own layout does not survive.

    The published arrays are sharded; the saved ones are not. Nothing in the
    design promises otherwise, and nothing warns either, so this is the test
    that would have to change before the behaviour could.
    """
    cube = iris.load_cube(sample, "air_temperature")
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(cube, tmp_path / "rt.zarr")

    source = zarr.open_group(sample, mode="r")["temperature_2m"]
    assert source.shards == SOURCE_SHARDS
    assert source.chunks == SOURCE_CHUNKS

    saved = zarr.open_group(str(tmp_path / "rt.zarr"), mode="r")["temperature_2m"]
    assert saved.shards is None
    assert saved.chunks != SOURCE_CHUNKS


@_shared_utils.skip_data
def test_xarray_opens_the_round_tripped_store(tmp_path, sample):
    """Asserted on the open, because a bad attribute fails the whole open."""
    cubes = iris.load(sample)
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(cubes, tmp_path / "rt.zarr")

    dataset = xr.open_zarr(str(tmp_path / "rt.zarr"))
    assert sorted(dataset.data_vars) == [
        "precipitation_surface",
        "temperature_2m",
        "wind_u_10m",
        "wind_v_10m",
    ]
