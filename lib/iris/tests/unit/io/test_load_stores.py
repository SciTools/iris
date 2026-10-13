# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.io.load_stores`."""

import numpy as np
import pytest

import iris
from iris.io import _is_remote_store_uri, load_stores
from iris.tests import _shared_utils

zarr = pytest.importorskip("zarr", minversion="3.0.8")

from iris.tests.unit.fileformats.zarr import stores  # noqa: E402


def test_a_local_store_loads(tmp_path):
    location = stores.make_store(tmp_path / "s.zarr")
    cubes = list(load_stores([location], None))
    assert [cube.var_name for cube in cubes] == ["air_temperature"]


def test_several_stores_in_one_call(tmp_path):
    first = stores.make_store(tmp_path / "a.zarr")
    second = stores.make_store(tmp_path / "b.zarr")
    assert len(list(load_stores([first, second], None))) == 2


def test_constraints_reach_the_handler(tmp_path):
    """A constraint the handler can pre-filter on rules out a data variable.

    `load_stores` forwards `constraints` to a constraint-aware handler rather
    than applying it itself - the general `cube_func` form is only enforced
    later, by `_CubeFilter.add` - so the constraint used here has to be one
    `_translate_constraints_to_var_callback` can act on: a `NameConstraint`
    with no `STASH` component.
    """
    location = stores.make_store(tmp_path / "s.zarr")
    constraint = iris.NameConstraint(var_name="not_air_temperature")
    assert list(load_stores([location], None, constraint)) == []


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://bucket/data.zarr", True),
        ("gs://bucket/data.zarr", True),
        ("az://container/data.zarr", True),
        ("abfs://container/data.zarr", True),
        ("https://example.invalid/data.nc", False),
        ("/local/path/data.zarr", False),
        ("relative/data.zarr", False),
    ],
)
def test_which_uris_are_remote_stores(uri, expected):
    assert _is_remote_store_uri(uri) is expected


def test_iris_load_accepts_a_local_store(tmp_path):
    location = stores.make_store(tmp_path / "s.zarr")
    cubes = iris.load(location)
    assert len(cubes) == 1
    np.testing.assert_array_equal(cubes[0].data, stores.DATA)


def test_iris_load_accepts_a_remote_url_and_does_not_absolutise_it():
    """A remote URL must reach zarr-python intact.

    `expand_filespecs` would otherwise turn `s3://bucket/data.zarr` into
    `/<cwd>/s3:/bucket/data.zarr`. No network is touched: opening the store
    fails, and what matters is that the message still names the URL.
    """
    # The failure mode depends on what is installed to talk to S3, so no
    # single exception type or message is guaranteed - checked below instead.
    with pytest.raises(Exception) as caught:  # noqa: PT011
        iris.load("s3://bucket/data.zarr")
    assert "s3://bucket/data.zarr" in str(caught.value) or isinstance(
        caught.value, ImportError
    )


@_shared_utils.skip_data
def test_a_glob_matching_both_files_and_stores(tmp_path):
    """`iris.load("data/*")` must accept a mix.

    The file-or-store decision happens after glob expansion and before format
    dispatch. Mis-partitioning raises IsADirectoryError from `load_files` for
    the stores while the netCDF files load fine, which reads as an Iris bug
    to a user who did not expect the glob to match a store.
    """
    source = iris.load_cube(
        _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
    )
    iris.save(source, str(tmp_path / "a.nc"))
    stores.make_store(tmp_path / "b.zarr")

    cubes = iris.load(str(tmp_path / "*"))
    names = sorted(cube.var_name or cube.name() for cube in cubes)
    assert "air_temperature" in names
    assert len(cubes) == 2


def test_find_saver_tolerates_a_trailing_separator():
    from iris.io import find_saver

    assert find_saver("out.nc") is find_saver("out.nc/")


def test_the_cubelist_saver_set_names_the_netcdf_saver():
    """The substring test it replaces matched by accident of module naming.

    `"iris.fileformats.netcdf" not in saver.__module__` is true of any module
    whose path happens to contain that string, and says nothing a reader can
    check. The set is one line per format.
    """
    from iris.fileformats.netcdf import save as netcdf_save
    from iris.io import _CUBELIST_SAVER_MODULES

    assert netcdf_save.__module__ in _CUBELIST_SAVER_MODULES


@_shared_utils.skip_data
def test_saving_a_cubelist_to_netcdf_still_takes_one_call(tmp_path):
    cubes = iris.load(
        _shared_utils.get_data_path(
            ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
        )
    )
    iris.save(cubes, str(tmp_path / "out.nc"))
    assert len(iris.load(str(tmp_path / "out.nc"))) == len(cubes)


def test_load_meshes_skips_a_zarr_store_rather_than_raising(tmp_path):
    """Reaching a Zarr store is new; building a mesh from one is not.

    What must not happen is the `ValueError` the scheme dispatch raised for
    anything that was not a file.
    """
    from iris.mesh import load_meshes

    stores.make_store(tmp_path / "s.zarr")
    assert load_meshes(str(tmp_path / "s.zarr")) == {}
