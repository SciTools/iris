# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :func:`iris.fileformats.zarr.load_cubes`."""

import numpy as np
import pytest
import zarr

import iris
from iris.cube import Cube
from iris.exceptions import TranslationError
from iris.fileformats.zarr import load_cubes
from iris.loading import LOAD_PROBLEMS
from iris.warnings import IrisCfLoadWarning

from .. import stores


@pytest.fixture(autouse=True)
def _assert_clean_load():
    """Every load in this file must leave `LOAD_PROBLEMS` empty.

    A malformed grid mapping on the shared `stores.make_store` fixture once
    made every test that loaded through it record a load problem and raise
    the standing `IrisLoadWarning` that announces one - silently covering the
    one signal that would reveal a genuine Zarr load problem. This guards
    against that recurring, for every test in this file, not only the ones
    that use the canonical fixture.
    """
    LOAD_PROBLEMS.reset()
    yield
    assert LOAD_PROBLEMS.problems == []


def test_the_canonical_store_loads_its_one_data_variable(tmp_path):
    location = stores.make_store(tmp_path / "s.zarr")
    cubes = list(load_cubes(location))
    assert len(cubes) == 1
    cube = cubes[0]
    assert isinstance(cube, Cube)
    assert cube.var_name == "air_temperature"
    assert cube.units == "K"
    np.testing.assert_array_equal(cube.data, stores.DATA)


def test_one_cube_per_data_variable(tmp_path):
    root = zarr.open_group(store=tmp_path / "two.zarr", mode="w", zarr_format=3)
    for name in ("first", "second", "third"):
        array = stores.make_array(
            root, name, (2,), "f8", 3, dimensions=("p",), attributes={"units": "1"}
        )
        array[:] = [1.0, 2.0]

    cubes = list(load_cubes(str(tmp_path / "two.zarr")))
    assert sorted(cube.var_name for cube in cubes) == ["first", "second", "third"]


def test_a_version_2_store_loads_the_same_cube(tmp_path):
    v3 = list(load_cubes(stores.make_store(tmp_path / "v3.zarr")))[0]
    v2 = list(load_cubes(stores.make_store(tmp_path / "v2.zarr", zarr_format=2)))[0]
    assert v2 == v3


def test_a_name_constraint_skips_variables_before_they_are_built(tmp_path, mocker):
    """`constraints` is a pre-build filter, not a post-build one.

    `_translate_constraints_to_var_callback` only answers for a bare
    `NameConstraint`; any other constraint returns None and every variable is
    built. The spy is what distinguishes "filtered early" from "built then
    discarded" - the cube list alone cannot.
    """
    from iris.fileformats.zarr import loader as zarr_loader

    root = zarr.open_group(store=tmp_path / "two.zarr", mode="w", zarr_format=3)
    for name in ("wanted", "unwanted"):
        array = stores.make_array(
            root, name, (2,), "f8", 3, dimensions=("p",), attributes={"units": "1"}
        )
        array[:] = [1.0, 2.0]

    # The loader binds `_load_cube` at import, so the spy must go on its own
    #  module reference rather than on `cf.loader`'s.
    spy = mocker.spy(zarr_loader, "_load_cube")
    constraint = iris.NameConstraint(var_name="wanted")
    cubes = list(load_cubes(str(tmp_path / "two.zarr"), constraints=constraint))

    assert [cube.var_name for cube in cubes] == ["wanted"]
    assert spy.call_count == 1


def test_a_callback_can_reject_a_cube(tmp_path):
    from iris.exceptions import IgnoreCubeException

    def reject(cube, field, filename):
        raise IgnoreCubeException

    location = stores.make_store(tmp_path / "s.zarr")
    assert list(load_cubes(location, callback=reject)) == []


def test_the_group_keyword_selects_a_group(tmp_path):
    root = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
    sub = root.create_group("measurements")
    band = stores.make_array(
        sub, "b02", (3,), "u2", 3, dimensions=("p",), attributes={"units": "1"}
    )
    band[:] = [1, 2, 3]

    cubes = list(load_cubes(str(tmp_path / "n.zarr"), group="measurements"))
    assert [cube.var_name for cube in cubes] == ["b02"]


def test_several_stores_in_one_call(tmp_path):
    first = stores.make_store(tmp_path / "a.zarr")
    second = stores.make_store(tmp_path / "b.zarr")
    assert len(list(load_cubes([first, second]))) == 2


def test_a_group_with_no_arrays_names_the_groups_that_have_them(tmp_path):
    """A real store in this shape exists, and silence is the worst answer.

    The ESA EOPF Sentinel-2 products hold no arrays in the root group, only
    subgroups - `conditions/geometry`, `measurements/reflectance/r10m` and
    others. Iris loads one group per call, so loading the default root yields
    nothing; returning an empty CubeList from a store that plainly holds data
    leaves the user with no way to find out why.
    """
    root = zarr.open_group(store=tmp_path / "e.zarr", mode="w", zarr_format=3)
    root.create_group("conditions")
    measurements = root.create_group("measurements")
    stores.make_array(measurements, "b02", (2,), "u2", 3, dimensions=("p",))

    with pytest.raises(TranslationError, match="measurements"):
        list(load_cubes(str(tmp_path / "e.zarr")))


def test_the_message_says_so_when_there_are_no_groups_either(tmp_path):
    zarr.open_group(store=tmp_path / "empty.zarr", mode="w", zarr_format=3)
    with pytest.raises(TranslationError, match="no data variables"):
        list(load_cubes(str(tmp_path / "empty.zarr")))


def test_nested_groups_are_not_walked(tmp_path):
    """Iris loads one group per call, like xarray.

    A Zarr store can legitimately hold unrelated datasets, and merging them
    into one CubeList would be wrong.
    """
    root = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
    top = stores.make_array(
        root, "top", (2,), "f8", 3, dimensions=("p",), attributes={"units": "1"}
    )
    top[:] = [1.0, 2.0]
    inner = stores.make_array(
        root.create_group("sub"), "inner", (2,), "f8", 3, dimensions=("p",)
    )
    inner[:] = [3.0, 4.0]

    cubes = list(load_cubes(str(tmp_path / "n.zarr")))
    assert [cube.var_name for cube in cubes] == ["top"]


def test_the_cube_is_lazy(tmp_path, mocker):
    from iris.fileformats.cf import loader as cf_loader

    mocker.patch.object(cf_loader, "_LAZYVAR_MIN_BYTES", 1)
    cube = list(load_cubes(stores.make_store(tmp_path / "s.zarr")))[0]
    assert cube.has_lazy_data()


def test_the_grid_mapping_variable_does_not_become_a_cube(tmp_path):
    cubes = list(load_cubes(stores.make_store(tmp_path / "s.zarr")))
    assert "grid" not in [cube.var_name for cube in cubes]


def test_cf_reserved_attributes_do_not_reach_the_cube(tmp_path):
    """`units` becomes a property, so it must not also be an attribute.

    Which attributes survive is decided by which ones the loading rules did
    not read, so an attribute Iris consumes but fails to record as read
    arrives on the cube as well.
    """
    cube = list(load_cubes(stores.make_store(tmp_path / "s.zarr")))[0]
    assert "units" not in cube.attributes
    assert "long_name" not in cube.attributes
    assert cube.attributes["Conventions"] == "CF-1.7"


def test_a_structured_attribute_reaches_the_cube_unchanged(tmp_path):
    """CF cannot express a nested attribute; Zarr allows one. Carry it.

    Two independent producers write them - NOAA GFS gives `valid_time` a
    `statistics_approximate` object, and the ESA EOPF product root carries
    `stac_discovery`. Interpreting them needs the attribute-model discussion
    in SciTools/iris#7288; dropping them loses the publisher's metadata.
    """
    group = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group,
        "v",
        (2,),
        "f8",
        3,
        dimensions=("p",),
        attributes={"stac_discovery": {"id": "abc"}},
    )
    array[:] = [1.0, 2.0]

    cube = list(load_cubes(str(tmp_path / "n.zarr")))[0]
    assert cube.attributes["stac_discovery"] == {"id": "abc"}


def test_a_structured_attribute_cannot_then_be_saved_to_netcdf(tmp_path):
    """Known limitation, characterised so that it is not a surprise.

    Zarr allows any JSON literal as an attribute value and netCDF does not,
    so a store carrying one produces a cube Iris cannot save as netCDF.
    Checking attribute content before saving is SciTools/iris#7329, and
    fixing it means changing the netCDF saver. Strip the attribute, or save
    to Zarr once that is supported.
    """
    import iris

    group = zarr.open_group(store=tmp_path / "n.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group,
        "v",
        (2,),
        "f8",
        3,
        dimensions=("p",),
        attributes={"stac_discovery": {"id": "abc"}},
    )
    array[:] = [1.0, 2.0]
    cube = list(load_cubes(str(tmp_path / "n.zarr")))[0]

    with pytest.raises(TypeError, match="stac_discovery"):
        iris.save(cube, str(tmp_path / "out.nc"))

    # And the documented workaround works.
    del cube.attributes["stac_discovery"]
    iris.save(cube, str(tmp_path / "out.nc"))


def test_a_dimension_name_reused_at_two_lengths_still_places_a_coordinate(tmp_path):
    """Zarr dimension names carry none of netCDF's uniqueness guarantee.

    A data variable may name both of its dimensions `d` at two lengths, which
    makes the name alone useless for choosing an axis. `(name, size)` names
    one axis here, and the coordinate belongs on it; matching on the bare name
    picks the first `d` and the coordinate is dropped as the wrong length.
    """
    group = zarr.open_group(store=tmp_path / "r.zarr", mode="w", zarr_format=3)
    data = stores.make_array(
        group,
        "v",
        (2, 3),
        "f8",
        3,
        dimensions=("d", "d"),
        attributes={"units": "1", "coordinates": "lat"},
    )
    data[:] = np.zeros((2, 3))
    lat = stores.make_array(
        group,
        "lat",
        (3,),
        "f8",
        3,
        dimensions=("d",),
        attributes={"units": "degrees_north", "standard_name": "latitude"},
    )
    lat[:] = [10.0, 20.0, 30.0]

    cube = list(load_cubes(str(tmp_path / "r.zarr")))[0]
    assert cube.coord_dims("latitude") == (1,)


def test_version_3_zeros_are_not_masked(tmp_path):
    """An int16 [0, 1, 2] with no CF attributes loads with nothing masked.

    zarr-python defaults `fill_value` to zero, so reading the storage field
    as a sentinel masks valid data in the ordinary case.
    """
    group = zarr.open_group(store=tmp_path / "z3.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group, "v", (3,), "i2", 3, dimensions=("p",), attributes={"units": "1"}
    )
    array[:] = [0, 1, 2]

    cube = list(load_cubes(str(tmp_path / "z3.zarr")))[0]
    assert not np.ma.is_masked(cube.data)
    np.testing.assert_array_equal(cube.data, [0, 1, 2])


def test_version_2_zeros_are_masked(tmp_path):
    """The same array as version 2 does use `fill_value` as the mask source.

    Version 2 stores predate the CF `_FillValue` convention, and this is what
    their writers meant - which is what released xarray's
    `use_zarr_fill_value_as_mask` does. Iris warns about exactly this, since
    zarr-python 3 defaults a version 2 store's `fill_value` to zero too, and a
    store written today can lose a genuine zero this way.
    """
    group = zarr.open_group(store=tmp_path / "z2.zarr", mode="w", zarr_format=2)
    array = stores.make_array(
        group, "v", (3,), "i2", 2, dimensions=("p",), attributes={"units": "1"}
    )
    array[:] = [0, 1, 2]

    with pytest.warns(IrisCfLoadWarning, match="v"):
        cube = list(load_cubes(str(tmp_path / "z2.zarr")))[0]
    assert cube.data.mask.tolist() == [True, False, False]


def test_a_cf_fill_value_beats_the_storage_one(tmp_path):
    group = zarr.open_group(store=tmp_path / "z.zarr", mode="w", zarr_format=3)
    array = stores.make_array(
        group,
        "v",
        (3,),
        "i2",
        3,
        dimensions=("p",),
        fill_value=0,
        attributes={"units": "1", "_FillValue": -999},
    )
    array[:] = [0, -999, 2]

    cube = list(load_cubes(str(tmp_path / "z.zarr")))[0]
    assert cube.data.mask.tolist() == [False, True, False]


class TestMesh:
    """UGRID data, which a Zarr store holds exactly as a netCDF file does.

    Nothing in `iris.fileformats.netcdf.ugrid_load` is netCDF-specific: it
    reads a `CFReader`, and a `CFReader` over a `ZarrDataset` answers the
    same questions. These assert that, rather than re-testing the mesh
    machinery itself.
    """

    @pytest.fixture
    def cube(self, tmp_path):
        location = stores.make_mesh_store(tmp_path / "mesh.zarr")
        return list(load_cubes(location))[0]

    def test_the_cube_carries_a_mesh(self, cube):
        assert cube.mesh is not None
        assert cube.mesh.var_name == "mesh"
        assert cube.mesh.topology_dimension == 2

    def test_the_mesh_is_on_the_right_dimension(self, cube):
        assert cube.mesh_dim() == cube.coord_dims("time")[0] + 1

    def test_the_mesh_coords_are_attached(self, cube):
        names = sorted(coord.name() for coord in cube.coords(mesh_coords=True))
        assert names == ["latitude", "longitude"]

    def test_the_connectivity_is_read(self, cube):
        connectivity = cube.mesh.face_node_connectivity
        np.testing.assert_array_equal(
            connectivity.indices, stores.MESH_ARRAYS["face_nodes"][0]
        )

    def test_mesh_attributes_do_not_also_reach_the_cube(self, cube):
        """`mesh` and `location` are consumed by the mesh build, not carried.

        They reach the cube as stray attributes if the MeshCoords are built
        after it rather than before - which is why `_prepare_mesh_coords` and
        `_attach_mesh_coords` are two calls with `_load_cube` between them.
        """
        assert "mesh" not in cube.attributes
        assert "location" not in cube.attributes

    def test_a_version_2_store_gives_the_same_mesh(self, tmp_path):
        v3 = list(load_cubes(stores.make_mesh_store(tmp_path / "v3.zarr")))[0]
        v2 = list(
            load_cubes(stores.make_mesh_store(tmp_path / "v2.zarr", zarr_format=2))
        )[0]
        assert v2.mesh == v3.mesh
        assert v2 == v3

    def test_load_meshes_reads_a_zarr_store(self, tmp_path):
        """`iris.mesh.load_meshes` dispatches on format, so it reaches Zarr too.

        Its dispatch is a second copy of the main loader's; the two are meant
        to move together.
        """
        location = stores.make_mesh_store(tmp_path / "mesh.zarr")
        meshes = iris.mesh.load_meshes(location)
        assert list(meshes) == [location]
        assert [mesh.var_name for mesh in meshes[location]] == ["mesh"]

    def test_a_store_naming_an_absent_mesh_still_loads(self, tmp_path):
        """A dangling `mesh` reference is logged, not raised.

        Same treatment as netCDF: the cube is worth having without its mesh.
        """
        location = stores.make_mesh_store(tmp_path / "mesh.zarr")
        group = zarr.open_group(store=location, mode="r+")
        group["temp"].attrs["mesh"] = "absent"

        cube = list(load_cubes(location))[0]
        assert cube.mesh is None
