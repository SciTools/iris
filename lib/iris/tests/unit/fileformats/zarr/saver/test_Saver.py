# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.zarr.saver.Saver`."""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

import iris
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.fileformats.cf.saver import _CFSaver
from iris.fileformats.zarr.saver import Saver


@pytest.fixture
def cube():
    cube = Cube(
        np.arange(6, dtype=np.float32).reshape(2, 3),
        standard_name="air_temperature",
        units="K",
        var_name="air_temperature",
    )
    cube.add_dim_coord(
        DimCoord([0.0, 1.0], standard_name="latitude", units="degrees"), 0
    )
    cube.add_dim_coord(
        DimCoord([0.0, 1.0, 2.0], standard_name="longitude", units="degrees"), 1
    )
    return cube


class TestTheClass:
    def test_it_is_a_cf_saver(self):
        assert issubclass(Saver, _CFSaver)

    def test_it_opens_a_store(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert sorted(group.array_keys()) == [
            "air_temperature",
            "latitude",
            "longitude",
        ]

    def test_the_default_mode_refuses_an_existing_store(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        with pytest.raises(FileExistsError):
            Saver(tmp_path / "out.zarr")

    def test_a_group_implies_adding(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        with Saver(tmp_path / "out.zarr", group="later") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert "later" in list(group.group_keys())
        assert "air_temperature" in list(group.array_keys())

    def test_an_explicit_mode_beats_the_group_default(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        with pytest.raises(FileExistsError):
            Saver(tmp_path / "out.zarr", mode="w-", group=None)


class TestStringVariables:
    def test_a_label_coordinate_is_a_string_array(self, tmp_path, cube):
        # netCDF stores text as bytes over an extra dimension. Zarr has a
        # string type, so the extra dimension would be an invention.
        cube.add_aux_coord(AuxCoord(np.array(["north", "south"]), long_name="place"), 0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        place = group["place"]
        assert list(place[:]) == ["north", "south"]
        assert place.metadata.dimension_names == ("latitude",)

    def test_no_string_dimension_is_invented(self, tmp_path, cube):
        cube.add_aux_coord(AuxCoord(np.array(["north", "south"]), long_name="place"), 0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        names = {
            name
            for array in group.arrays()
            for name in (array[1].metadata.dimension_names or ())
        }
        assert not {name for name in names if name.startswith("string")}


class TestScalarCubes:
    def test_a_scalar_cube_is_saved(self, tmp_path):
        # A scalar cube is ordinary, and saves to netCDF without complaint.
        cube = Cube(
            np.float32(3.0),
            standard_name="air_temperature",
            units="K",
            var_name="air_temperature",
        )
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert group["air_temperature"][()] == pytest.approx(3.0)


class TestGridMappings:
    def test_a_coordinate_system_is_written(self, tmp_path, cube):
        # `_grid_mapping_writer`'s base implementation, reached for real:
        # without it the parameters would be written onto a netCDF4 object
        # that a Zarr store does not have.
        from iris.coord_systems import GeogCS

        for coord in cube.coords():
            coord.coord_system = GeogCS(6371229.0)
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        group = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        grid = group["latitude_longitude"]
        assert grid.attrs["grid_mapping_name"] == "latitude_longitude"
        assert grid.attrs["earth_radius"] == pytest.approx(6371229.0)


class TestDeferredSaving:
    @pytest.fixture
    def lazy_cube(self, cube):
        import dask.array as da

        lazy = cube.copy(da.from_array(cube.data, chunks=(1, 3)))
        return lazy

    def test_compute_false_returns_a_real_delayed(self, tmp_path, lazy_cube):
        # SciTools/iris#7291: the inherited delayed_completion() returns a
        # plain dask.array.core.Array here, never a Delayed despite its own
        # annotation, and the documented `result.compute()` raises
        # AttributeError on it.
        from dask.delayed import Delayed

        from iris.fileformats.zarr.saver import save

        result = save(lazy_cube, tmp_path / "out.zarr", compute=False)
        assert isinstance(result, Delayed)

    def test_the_data_lands_when_it_is_computed(self, tmp_path, lazy_cube):
        from iris.fileformats.zarr.saver import save

        result = save(lazy_cube, tmp_path / "out.zarr", compute=False)
        result.compute()
        written = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        np.testing.assert_array_equal(written["air_temperature"][:], lazy_cube.data)

    def test_consolidation_follows_the_last_chunk(self, tmp_path, lazy_cube):
        # Spec §4.5: under compute=False the Delayed owns finalise(), so a
        # store that is never completed never claims to be.
        from iris.fileformats.zarr.saver import save

        result = save(lazy_cube, tmp_path / "out.zarr", compute=False)
        before = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert before.metadata.consolidated_metadata is None
        result.compute()
        after = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        assert after.metadata.consolidated_metadata is not None

    def test_the_whole_graph_survives_a_pickle(self, tmp_path, lazy_cube):
        # write_handle() returns the ZarrDatasetVariable, not the zarr.Array:
        # a distributed scheduler pickles the graph, so the substitution has
        # to be re-checked.
        import pickle

        from iris.fileformats.zarr.saver import save

        result = save(lazy_cube, tmp_path / "out.zarr", compute=False)
        revived = pickle.loads(pickle.dumps(result))
        revived.compute()
        written = zarr.open_group(store=tmp_path / "out.zarr", mode="r")
        np.testing.assert_array_equal(written["air_temperature"][:], lazy_cube.data)


class TestChunksAndShards:
    @pytest.fixture
    def awkward_cube(self):
        import dask.array as da

        # A lazy chunking that does not divide the shard below: ordinary,
        # since Dask chose it for memory and the user chose the shard for
        # storage, and the two have no reason to agree.
        return Cube(
            da.zeros((64, 64), chunks=(30, 30)),
            standard_name="air_temperature",
            units="K",
            var_name="air_temperature",
        )

    def test_shards_alone_does_not_inherit_the_cubes_chunking(
        self, tmp_path, awkward_cube
    ):
        # `chunks` defaults to the cube's own lazy chunking, which zarr then
        # refuses as an inner chunk the shard is not a multiple of - about a
        # number the user never supplied. With only `shards` given,
        # zarr-python picks the inner chunk itself.
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(awkward_cube, shards=(64, 64))
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert array.shards == (64, 64)
        np.testing.assert_array_equal(array[:], np.zeros((64, 64)))

    def test_chunks_still_default_to_the_cubes_own(self, tmp_path, awkward_cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(awkward_cube)
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert array.chunks == (30, 30)

    def test_both_given_are_both_honoured(self, tmp_path, awkward_cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(awkward_cube, chunks=(32, 32), shards=(64, 64))
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert array.chunks == (32, 32)
        assert array.shards == (64, 64)


class TestMaskedSaving:
    def test_a_masked_cube_declares_a_fill_value(self, tmp_path, cube):
        # Spec §4.5: without the attribute, every masked point reads back
        # unmasked - a round trip that silently loses the mask.
        masked = cube.copy(np.ma.masked_greater(cube.data, 3.0))
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(masked)
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert "_FillValue" in dict(array.attrs)

    def test_a_lazy_masked_cube_declares_one_too(self, tmp_path, cube):
        import dask.array as da

        masked = cube.copy(da.ma.masked_greater(da.from_array(cube.data), 3.0))
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(masked)
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert "_FillValue" in dict(array.attrs)

    def test_the_mask_round_trips(self, tmp_path, cube):
        masked = cube.copy(np.ma.masked_greater(cube.data, 3.0))
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(masked)
        loaded = iris.load_cube(tmp_path / "out.zarr")
        np.testing.assert_array_equal(loaded.data.mask, masked.data.mask)

    def test_an_unmasked_cube_declares_nothing(self, tmp_path, cube):
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(cube)
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert "_FillValue" not in dict(array.attrs)

    def test_a_maskless_masked_array_declares_nothing(self, tmp_path, cube):
        # np.ma.isMaskedArray is true of an np.ma array with an all-False
        # mask, and Iris produces those routinely - a cube round-tripped
        # from netCDF, say. Declaring a fill value for one invents metadata
        # the source never carried.
        unmasked = cube.copy(np.ma.masked_array(cube.data))
        with Saver(tmp_path / "out.zarr") as saver:
            saver.write(unmasked)
        array = zarr.open_group(store=tmp_path / "out.zarr", mode="r")[
            "air_temperature"
        ]
        assert "_FillValue" not in dict(array.attrs)

    def test_a_masked_string_variable_is_refused(self, tmp_path, cube):
        # A Zarr string array's only candidate sentinel is the empty string,
        # and `iris.fileformats.zarr._decode` does not restore a mask from
        # one - so writing it would lose the mask in silence, which is the
        # worse of the two outcomes. netCDF cannot save this either.
        labels = np.ma.masked_array(["north", "south"], mask=[False, True])
        cube.add_aux_coord(AuxCoord(labels, long_name="place", var_name="place"), 0)
        with pytest.raises(ValueError, match="place") as excinfo:
            with Saver(tmp_path / "out.zarr") as saver:
                saver.write(cube)
        message = str(excinfo.value)
        assert "string" in message
        # No advice to pass `fill_value`: `iris.save` offers it for the data
        # variable alone, and this is a coordinate.
        assert "fill_value" not in message
