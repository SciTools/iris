# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.io.load_stores`, and the load chain it joins."""

import numpy as np
import pytest

import iris
from iris.io import _is_remote_store_uri, load_stores
from iris.tests import _shared_utils

zarr = pytest.importorskip("zarr", minversion="3.0.8")

from iris.tests.unit.fileformats.zarr import stores  # noqa: E402


class Test_load_stores:
    def test_a_local_store_loads(self, tmp_path):
        location = stores.make_store(tmp_path / "s.zarr")
        cubes = list(load_stores([location], None))
        assert [cube.var_name for cube in cubes] == ["air_temperature"]

    def test_several_stores_in_one_call(self, tmp_path):
        first = stores.make_store(tmp_path / "a.zarr")
        second = stores.make_store(tmp_path / "b.zarr")
        assert len(list(load_stores([first, second], None))) == 2

    def test_constraints_reach_the_handler(self, tmp_path):
        """A constraint the handler can pre-filter on rules out a data variable.

        `load_stores` forwards `constraints` to a constraint-aware handler
        rather than applying it itself - the general `cube_func` form is only
        enforced later, by `_CubeFilter.add` - so the constraint used here has
        to be one `_translate_constraints_to_var_callback` can act on: a
        `NameConstraint` with no `STASH` component.
        """
        location = stores.make_store(tmp_path / "s.zarr")
        constraint = iris.NameConstraint(var_name="not_air_temperature")
        assert list(load_stores([location], None, constraint)) == []


class Test__is_remote_store_uri:
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
    def test_which_uris_are_remote_stores(self, uri, expected):
        assert _is_remote_store_uri(uri) is expected


class Test_iris_load:
    def test_accepts_a_local_store(self, tmp_path):
        location = stores.make_store(tmp_path / "s.zarr")
        cubes = iris.load(location)
        assert len(cubes) == 1
        np.testing.assert_array_equal(cubes[0].data, stores.DATA)

    def test_accepts_a_remote_url_and_does_not_absolutise_it(self, mocker):
        """A remote URL must reach `load_stores` unmangled, not as a local path.

        `expand_filespecs` would otherwise turn `s3://bucket/data.zarr` into
        `/<cwd>/s3:/bucket/data.zarr`. The dispatch is what is under test, so
        `load_stores` itself is mocked out: no store library is invoked, and
        no network is touched, whatever is installed alongside zarr-python.
        """
        mock_load_stores = mocker.patch("iris.io.load_stores", return_value=iter(()))
        iris.load("s3://bucket/data.zarr")
        mock_load_stores.assert_called_once_with(["s3://bucket/data.zarr"], None, None)

    @_shared_utils.skip_data
    def test_a_glob_matching_both_files_and_stores(self, tmp_path):
        """`iris.load("data/*")` must accept a mix.

        The file-or-store decision happens after glob expansion and before
        format dispatch. Mis-partitioning raises IsADirectoryError from
        `load_files` for the stores while the netCDF files load fine, which
        reads as an Iris bug to a user who did not expect the glob to match a
        store.
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


class Test_load_files:
    def test_still_expands_a_glob(self, tmp_path, mocker):
        """Public `load_files` keeps its promise: filenames may be globs.

        The duplicate-collapsing bug fixed alongside `load_stores` lives in
        calling `expand_filespecs` twice, once in a caller and again inside
        `load_files` - not in `load_files` expanding at all. Pinning this
        guards against quietly narrowing the public contract while fixing
        that.
        """
        from iris.io import load_files

        (tmp_path / "a.nc").write_bytes(b"")
        (tmp_path / "b.nc").write_bytes(b"")
        mock_load = mocker.patch("iris.io._load_files", return_value=iter(()))

        list(load_files([str(tmp_path / "*.nc")], None))

        mock_load.assert_called_once_with(
            [str(tmp_path / "a.nc"), str(tmp_path / "b.nc")], None, None
        )


class Test_saving:
    def test_the_zarr_saver_takes_a_cubelist_whole(self):
        """A Zarr store is opened once and holds many arrays.

        Registered by identity, beside the netCDF saver: were it missing,
        `iris.save` would call it once per cube, and every cube after the
        first would meet a store that the default `mode="w-"` refuses.
        `_zarr_save` shares `iris.io` with `_dot_save`, which must keep
        being handed one cube at a time, so the registry naming functions
        rather than modules is what keeps the two apart.
        """
        import iris.io

        iris.io._check_init_savers()
        assert iris.io._zarr_save in iris.io._savers.cubelist_savers
        assert iris.io._dot_save not in iris.io._savers.cubelist_savers

    def test_find_saver_tolerates_a_trailing_separator(self):
        from iris.io import find_saver

        assert find_saver("out.nc") is find_saver("out.nc/")

    @_shared_utils.skip_data
    def test_saving_a_cubelist_to_netcdf_still_takes_one_call(self, tmp_path):
        cubes = iris.load(
            _shared_utils.get_data_path(
                ("NetCDF", "global", "xyt", "SMALL_hires_wind_u_for_ipcc4.nc")
            )
        )
        iris.save(cubes, str(tmp_path / "out.nc"))
        assert len(iris.load(str(tmp_path / "out.nc"))) == len(cubes)


class Test_load_meshes:
    def test_a_store_holding_no_mesh_yields_no_meshes(self, tmp_path):
        """Reaching a Zarr store is what was new here.

        What must not happen is the `ValueError` the scheme dispatch raised
        for anything that was not a file. Reading a mesh *out* of a store is
        covered by `unit/fileformats/zarr/loader/test_load_cubes.py::TestMesh`.
        """
        from iris.mesh import load_meshes

        stores.make_store(tmp_path / "s.zarr")
        assert load_meshes(str(tmp_path / "s.zarr")) == {}
