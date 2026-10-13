# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for loading native Zarr stores."""

import warnings

import numpy as np
import pytest

import iris
from iris.coord_systems import GeogCS
from iris.fileformats.cf import loader as cf_loader
from iris.loading import LOAD_PROBLEMS
from iris.tests import stock
from iris.warnings import IrisLoadWarning

from . import fixtures


@pytest.fixture(autouse=True)
def _assert_clean_load():
    """Every load in this module must be free of recorded problems.

    A cube that loads "successfully" while quietly recording a load problem
    is a weaker result than it looks, so `IrisLoadWarning` is escalated to an
    error for the duration of each test, and `LOAD_PROBLEMS` is asserted
    empty afterwards.
    """
    LOAD_PROBLEMS.reset()
    with warnings.catch_warnings():
        warnings.simplefilter("error", IrisLoadWarning)
        yield
    assert LOAD_PROBLEMS.problems == []


@pytest.fixture
def gfs_like(tmp_path):
    return fixtures.make_gfs_like_store(tmp_path / "gfs.zarr")


def test_the_store_loads_two_cubes(gfs_like):
    cubes = iris.load(gfs_like)
    assert sorted(cube.name() for cube in cubes) == [
        "air_temperature",
        "surface_air_pressure",
    ]


def test_dimension_coordinates_attach(gfs_like):
    cube = iris.load_cube(gfs_like, "air_temperature")
    assert [coord.name() for coord in cube.coords(dim_coords=True)] == [
        "time",
        "level",
        "latitude",
        "longitude",
    ]
    assert cube.shape == fixtures.SHAPE


def test_bounds_attach_to_their_coordinate(gfs_like):
    cube = iris.load_cube(gfs_like, "air_temperature")
    latitude = cube.coord("latitude")
    assert latitude.has_bounds()
    np.testing.assert_array_equal(latitude.bounds, fixtures.LATITUDE_BOUNDS)


def test_the_auxiliary_coordinate_attaches_to_its_own_dimensions(gfs_like):
    cube = iris.load_cube(gfs_like, "air_temperature")
    # `valid_time` carries `standard_name: "time"`, as the real GFS store's
    # does, so it shares its Iris name with the "time" dimension coordinate;
    # `var_name` is what still tells the two apart.
    aux = cube.coord(var_name="valid_time")
    assert cube.coord_dims(aux) == (0, 1)


def test_the_grid_mapping_becomes_a_coordinate_system(gfs_like):
    cube = iris.load_cube(gfs_like, "air_temperature")
    assert isinstance(cube.coord("latitude").coord_system, GeogCS)


def test_packing_is_applied(gfs_like):
    cube = iris.load_cube(gfs_like, "air_temperature")
    assert cube.dtype == np.dtype("f8")
    # The first stored value is 0, so scale_factor * 0 + add_offset.
    assert cube.data.flat[0] == pytest.approx(273.15)


def test_the_base64_fill_value_masks_nothing_here(gfs_like):
    """A NaN sentinel over data with no NaNs masks nothing.

    The attribute is still read, and read wrongly it raises rather than
    quietly doing nothing - which is why this asserts the unmasked result
    rather than only that loading succeeded.
    """
    cube = iris.load_cube(gfs_like, "surface_air_pressure")
    assert not np.ma.is_masked(cube.data)


def test_the_cube_is_lazy_and_chunked_to_the_store(gfs_like, monkeypatch):
    # `temperature_2m` is a few hundred bytes, well under the small-array
    # cutoff (`_LAZYVAR_MIN_BYTES`) below which the CF loader reads a
    # variable whole rather than lazily. A real store is nowhere near that
    # small, so the cutoff is lowered here to exercise the chunking this
    # test is actually about.
    monkeypatch.setattr(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    cube = iris.load_cube(gfs_like, "air_temperature")
    assert cube.has_lazy_data()
    # Whatever the chunking works out to, it tiles the store's inner chunk
    # grid: a Zarr chunk is read whole, so a dask block that subdivides one
    # costs a re-read.
    store_chunks = (1, 1, 3, 5)
    for sizes, store_size in zip(cube.lazy_data().chunks, store_chunks):
        assert all(size % store_size == 0 for size in sizes[:-1])


def test_a_structured_attribute_survives(gfs_like):
    cube = iris.load_cube(gfs_like, "air_temperature")
    aux = cube.coord(var_name="valid_time")
    assert aux.attributes["statistics_approximate"] == {
        "min": 0.0,
        "max": 7.0,
    }


@pytest.fixture
def realistic_4d_w_everything():
    """Return a realistic 4D cube with all the bells and whistles."""
    cube = stock.realistic_4d_w_everything()
    # Precision loss for floating-point scalar variable attributes:
    #  earth_radius 6371229.0 round-trips as 6371230.0.
    for coord_name in ("grid_latitude", "grid_longitude"):
        cube.coord(coord_name).coord_system.ellipsoid = GeogCS(6371230.0)
    return cube


def test_the_native_reader_agrees_with_the_nczarr_writer(
    tmp_path, realistic_4d_w_everything
):
    """A version 2 store Iris wrote through netCDF-c, read natively.

    The NCZarr `xarray` mode writes `_ARRAY_DIMENSIONS`, which is how a
    version 2 store records dimension names. The two code paths share no
    code, so agreement between them is evidence rather than a tautology.

    `realistic_4d_w_everything` carries a hybrid height factory, whose
    orography dependency round-trips through CF as its own top-level cube
    rather than folding back into the main one - true of a plain netCDF
    round trip of this fixture as well, so a name constraint selects the
    main cube deliberately rather than papering over a Zarr-specific gap.

    None of this fixture's variables declare a CF `_FillValue`, but netCDF-c's
    NCZarr writer still gives every one its own default storage fill value -
    the classic netCDF sentinel `9.96921e+36`, not zero, so this does not
    warn: see `lib/iris/fileformats/zarr/_decode.py` for why zero is what
    distinguishes a writer's deliberate pre-convention sentinel from
    zarr-python's own default.
    """
    path = tmp_path / "nczarr.zarr"
    uri = path.as_uri() + "#mode=xarray,file"
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(realistic_4d_w_everything, uri)

    assert (path / ".zgroup").is_file()
    loaded = iris.load_cube(str(path), realistic_4d_w_everything.name())
    # `_NCProperties`, netCDF-c's own bookkeeping stamp on the root group of
    # every store it writes, is withheld by the reader rather than compared
    # away here - the same storage-not-CF treatment as `_ARRAY_DIMENSIONS`
    # below, and covered by its own unit test in
    # `lib/iris/tests/unit/fileformats/zarr/dataset/test_ZarrDataset.py`.
    assert loaded == realistic_4d_w_everything


def test_array_dimensions_does_not_reach_the_cube(tmp_path, realistic_4d_w_everything):
    """It is how version 2 records dimension names, so it is storage.

    Left in the CF attributes it would land on every cube loaded from a
    version 2 store, and make it unequal to the same data in version 3.
    Checked across every cube the store yields, including the orography
    cube the hybrid height factory depends on, not only the main one.
    """
    path = tmp_path / "nczarr.zarr"
    with iris.FUTURE.context(save_split_attrs=True):
        iris.save(realistic_4d_w_everything, path.as_uri() + "#mode=xarray,file")

    cubes = iris.load(str(path))
    assert len(cubes) == 2
    for cube in cubes:
        assert "_ARRAY_DIMENSIONS" not in cube.attributes
        for coord in cube.coords():
            assert "_ARRAY_DIMENSIONS" not in coord.attributes
