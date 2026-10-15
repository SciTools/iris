# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for saving cubes to a Zarr store."""

import warnings

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import iris
from iris.cube import CubeList
import iris.fileformats.zarr
import iris.tests.stock as stock


@pytest.fixture
def cube():
    # The signature is `realistic_4d_w_everything(w_mesh=False)`: there is no
    # `lazy` keyword, and the cube it returns holds real data.
    #
    # Its hybrid-height factory depends on an orography coordinate, which the
    # CF data model has no way to fold back into the main cube: it round
    # trips as its own top-level cube, standard name "surface_altitude", the
    # same as a plain netCDF round trip of this fixture does. So a store this
    # fixture is saved into always holds two cubes, and a single-cube load
    # names the one it wants.
    return stock.realistic_4d_w_everything()


class TestRoundTrip:
    def test_the_cube_comes_back(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        loaded = iris.load_cube(path, cube.name())
        assert loaded.standard_name == cube.standard_name
        assert loaded.units == cube.units
        assert loaded.shape == cube.shape
        np.testing.assert_allclose(loaded.data, cube.data)

    def test_the_coordinates_come_back(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        loaded = iris.load_cube(path, cube.name())
        saved_names = sorted(coord.name() for coord in cube.coords())
        loaded_names = sorted(coord.name() for coord in loaded.coords())
        assert loaded_names == saved_names

    def test_the_bounds_come_back(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        loaded = iris.load_cube(path, cube.name())
        for coord in cube.coords():
            if coord.has_bounds():
                np.testing.assert_allclose(
                    loaded.coord(coord.name()).bounds, coord.bounds
                )

    def test_the_cell_methods_come_back(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        assert iris.load_cube(path, cube.name()).cell_methods == cube.cell_methods

    def test_the_coordinate_system_comes_back(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        loaded = iris.load_cube(path, cube.name())
        assert loaded.coord_system() == cube.coord_system()

    def test_attribute_values_come_back(self, tmp_path, cube):
        # Values, not dtypes: spec §4.5 trades the NumPy dtype of an
        # attribute for output every Zarr consumer can read.
        path = tmp_path / "out.zarr"
        cube.attributes["a_number"] = np.float32(1.5)
        cube.attributes["a_list"] = np.array([1, 2], dtype=np.int16)
        iris.save(cube, path)
        loaded = iris.load_cube(path, cube.name())
        assert loaded.attributes["a_number"] == 1.5
        assert list(loaded.attributes["a_list"]) == [1, 2]

    def test_a_masked_cube_keeps_its_mask(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        masked = cube.copy(np.ma.masked_greater(cube.data, cube.data.mean()))
        iris.save(masked, path)
        loaded = iris.load_cube(path, masked.name())
        np.testing.assert_array_equal(loaded.data.mask, masked.data.mask)

    def test_several_cubes_round_trip(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        other = cube[0].copy()
        other.rename("air_pressure")
        other.var_name = "air_pressure"
        iris.save(CubeList([cube, other]), path)
        loaded = iris.load(path)
        # Three, not two: `cube`'s hybrid-height factory always brings its
        # orography cube along as well (see the `cube` fixture above), so
        # both saved names are checked explicitly rather than trusting a
        # bare count to mean the right two cubes came back.
        assert len(loaded) == 3
        assert {one.name() for one in loaded} >= {cube.name(), other.name()}

    def test_a_group_round_trips(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        # A different name in the group than at the root: reading the
        # group and finding this name only works if `group=` was honoured,
        # where reading the root (which does not have this name) would not.
        renamed = cube.copy()
        renamed.rename("a_renamed_cube")
        renamed.var_name = "a_renamed_cube"
        iris.save(renamed, path, group="again")
        # `iris.load`/`iris.load_cube` read the root group only - see
        # "Groups" in `docs/src/user_manual/explanation/zarr_io.rst` - so a
        # named group is read through the Zarr loader directly.
        loaded = CubeList(
            iris.fileformats.zarr.load_cubes(path, group="again")
        ).extract_cube(renamed.name())
        assert loaded.shape == renamed.shape


class TestWhatTheFormatDoesNotPromise:
    # Spec §4.5: neither of these is fixable from inside Iris. They are
    # pinned so that a zarr-python release changing one fails here.
    def test_a_missing_chunk_reads_back_filled(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        # The data variable's name is the cube's `var_name`, which this
        # fixture sets to "air_temp" - not its standard name.
        chunks = sorted((path / cube.var_name / "c").rglob("*"))
        removed = [p for p in chunks if p.is_file()][0]
        removed.unlink()
        original_masked = np.ma.count_masked(cube.data)
        # Opens cleanly, and under-reports data, with no warning of any kind:
        # both halves of that are asserted below, not only the shape, which
        # is unchanged whether the gap is filled in silence or not.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reloaded = iris.load_cube(path, cube.name())
            data = reloaded.data
        assert caught == []
        assert data.shape == cube.shape
        # The deleted chunk reads back as the array's own fill, which this
        # masked cube's decode then masks - strictly more points than the
        # cube ever declared missing, and nothing said so.
        assert np.ma.count_masked(data) > original_masked

    def test_consolidated_metadata_goes_stale(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        # Add a group behind Iris's back, without re-consolidating.
        zarr.open_group(store=path, mode="a", path="sneaked").create_array(
            "data", shape=(2,), dtype="f4", dimension_names=("x",)
        )
        reader = zarr.open_group(store=path, mode="r")
        assert "sneaked" not in list(reader.group_keys())
        # Which is why Iris re-consolidates when it adds a group itself.
        iris.save(cube, path, group="properly")
        reader = zarr.open_group(store=path, mode="r")
        assert "properly" in list(reader.group_keys())


class TestRefusals:
    def test_an_existing_store_is_not_clobbered(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        with pytest.raises(FileExistsError, match="mode"):
            iris.save(cube, path)
        # And the first save is still intact.
        assert iris.load_cube(path, cube.name()).shape == cube.shape

    def test_adding_a_name_that_exists_is_refused(self, tmp_path, cube):
        path = tmp_path / "out.zarr"
        iris.save(cube, path)
        with pytest.raises(ValueError, match=cube.var_name):
            iris.save(cube, path, mode="a")


class TestInputsTheDesignImplies:
    def test_a_mesh_cube_writes_cf_ugrid_structure(self, tmp_path):
        # Iris supports unstructured meshes, and a mesh cube reaching the
        # Zarr saver is ordinary: the mesh variables go through the same
        # generic path as everything else, and nothing above has proved it.
        # This is the saver's own promise, independent of what a loader can
        # rebuild from it - see the xfail below for that.
        path = tmp_path / "mesh.zarr"
        mesh_cube = stock.realistic_4d_w_everything(w_mesh=True)
        iris.save(mesh_cube, path)
        store = zarr.open_group(store=path, mode="r")

        data_var = store[mesh_cube.var_name]
        assert data_var.attrs["location"] == mesh_cube.location

        topology = store[data_var.attrs["mesh"]]
        assert topology.attrs["cf_role"] == "mesh_topology"
        assert topology.attrs["topology_dimension"] == mesh_cube.mesh.topology_dimension

        connectivity = store[topology.attrs["face_node_connectivity"]]
        assert connectivity.attrs["cf_role"] == "face_node_connectivity"
        assert "start_index" in connectivity.attrs

        # The mesh's own dimensions carry no coordinate variable of their
        # own name - that is how UGRID keeps a mesh dimension distinct from
        # an ordinary CF one.
        assert mesh_cube.mesh.face_dimension not in store.array_keys()
        assert mesh_cube.mesh.node_dimension not in store.array_keys()

    @pytest.mark.xfail(
        reason=(
            "The Zarr loader does not build a MeshXY from a store, so a "
            "cube it loads never carries one, however completely the "
            "store describes it - see the passing test above for that."
        )
    )
    def test_a_mesh_cube_round_trips(self, tmp_path):
        path = tmp_path / "mesh.zarr"
        mesh_cube = stock.realistic_4d_w_everything(w_mesh=True)
        iris.save(mesh_cube, path)
        loaded = iris.load_cube(path, mesh_cube.name())
        assert loaded.mesh is not None
        assert loaded.location == mesh_cube.location
        assert loaded.mesh.topology_dimension == mesh_cube.mesh.topology_dimension

    def test_packed_data_round_trips(self, tmp_path, cube):
        # `packing` writes scale_factor and add_offset and narrows the dtype.
        # Both attributes are numbers the JSON conversion touches, and the
        # read side unpacks with them.
        path = tmp_path / "packed.zarr"
        iris.save(cube, path, packing="i2")
        # The data variable's name is the cube's `var_name`, which this
        # fixture sets to "air_temp" - not its standard name.
        stored = zarr.open_group(store=path, mode="r")[cube.var_name]
        assert stored.dtype == np.dtype("i2")
        assert "scale_factor" in dict(stored.attrs)
        assert "add_offset" in dict(stored.attrs)
        loaded = iris.load_cube(path, cube.name())
        # A tolerance of one stored count: loose enough for the quantisation
        # error correct packing is allowed to introduce (up to half a count),
        # tight enough that a systematic one-count bias - truncation instead
        # of rounding, say - would fail it, where the original `rtol=1e-3`
        # would not have.
        scale_factor = stored.attrs["scale_factor"]
        np.testing.assert_allclose(loaded.data, cube.data, atol=scale_factor, rtol=0)

    def test_int64_data_survives(self, tmp_path, cube):
        # The one dtype the saver has narrowing logic for. A Zarr store holds
        # int64, so `supports_dtype` answers True and nothing is narrowed -
        # which is a different answer from NETCDF3, and worth pinning.
        path = tmp_path / "int64.zarr"
        wide = cube.copy(cube.data.astype(np.int64))
        iris.save(wide, path)
        assert iris.load_cube(path, wide.name()).dtype == np.dtype("i8")

    def test_a_non_ascii_attribute_survives(self, tmp_path, cube):
        # A degree sign in a title is ordinary, and netCDF's ASCII-or-bytes
        # coercion does not apply here: JSON is text.
        path = tmp_path / "unicode.zarr"
        cube.attributes["title"] = "Temperature in °C, naïve grid"
        iris.save(cube, path)
        assert iris.load_cube(path, cube.name()).attributes["title"] == (
            "Temperature in °C, naïve grid"
        )

    def test_a_dataless_cube_saves(self, tmp_path, cube):
        # `Cube` supports a dataless form, and the saver has a branch for it
        # that chooses its own dtype and fill value. Nothing above reaches it.
        path = tmp_path / "dataless.zarr"
        dataless = iris.cube.Cube(
            None,
            shape=cube.shape,
            standard_name=cube.standard_name,
            units=cube.units,
            var_name=cube.var_name,
        )
        iris.save(dataless, path)
        assert iris.load_cube(path).shape == cube.shape
