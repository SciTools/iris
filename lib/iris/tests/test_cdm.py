# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test cube indexing, slicing, and extracting, and also the dot graphs."""

import collections
import os
import re

import cf_units
import numpy as np
import numpy.ma as ma
import pytest

import iris
import iris.analysis
import iris.coords
import iris.cube
import iris.fileformats
import iris.fileformats.dot
from iris.tests import _shared_utils
import iris.tests.stock


class IrisDotTest:
    def check_dot(self, cube, reference_filename):
        test_string = iris.fileformats.dot.cube_text(cube)
        reference_path = _shared_utils.get_result_path(reference_filename)
        if os.path.isfile(reference_path):
            with open(reference_path, "r") as reference_fh:
                reference = "".join(reference_fh.readlines())
            _shared_utils._assert_str_same(
                reference,
                test_string,
                reference_filename,
                type_comparison_name="DOT files",
            )
        else:
            with open(reference_path, "w") as reference_fh:
                reference_fh.writelines(test_string)


class TestBasicCubeConstruction:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.cube.Cube(
            np.arange(12, dtype=np.int32).reshape((3, 4)),
            long_name="test cube",
        )
        self.x = iris.coords.DimCoord(np.array([-7.5, 7.5, 22.5, 37.5]), long_name="x")
        self.y = iris.coords.DimCoord(np.array([2.5, 7.5, 12.5]), long_name="y")
        self.xy = iris.coords.AuxCoord(
            np.arange(12).reshape((3, 4)) * 3.0, long_name="xy"
        )

    def test_add_dim_coord(self):
        # Lengths must match
        with pytest.raises(ValueError, match="Unequal lengths"):
            self.cube.add_dim_coord(self.y, 1)
        with pytest.raises(ValueError, match="Unequal lengths"):
            self.cube.add_dim_coord(self.x, 0)

        # Must specify a dimension
        with pytest.raises(TypeError):
            self.cube.add_dim_coord(self.y)

        # Add y
        self.cube.add_dim_coord(self.y, 0)
        assert self.cube.coords() == [self.y]
        assert self.cube.dim_coords == (self.y,)
        # Add x
        self.cube.add_dim_coord(self.x, 1)
        assert self.cube.coords() == [self.y, self.x]
        assert self.cube.dim_coords == (self.y, self.x)

        # Cannot add a coord twice
        with pytest.raises(ValueError, match="coordinate already exists"):
            self.cube.add_dim_coord(self.y, 0)
        # ... even to cube.aux_coords
        with pytest.raises(ValueError, match="Duplicate coordinates are not permitted"):
            self.cube.add_aux_coord(self.y, 0)

        # Can't add AuxCoord to dim_coords
        y_other = iris.coords.AuxCoord(np.array([2.5, 7.5, 12.5]), long_name="y_other")
        with pytest.raises(ValueError, match="dim_coord is already associated"):
            self.cube.add_dim_coord(y_other, 0)

    def test_add_scalar_coord(self):
        scalar_dim_coord = iris.coords.DimCoord(23, long_name="scalar_dim_coord")
        scalar_aux_coord = iris.coords.AuxCoord(23, long_name="scalar_aux_coord")
        match_common = "dimension must be a single number"
        # Scalars cannot be in cube.dim_coords
        with pytest.raises(TypeError):
            self.cube.add_dim_coord(scalar_dim_coord)
        with pytest.raises(TypeError):
            self.cube.add_dim_coord(scalar_dim_coord, None)
        with pytest.raises(ValueError, match=match_common):
            self.cube.add_dim_coord(scalar_dim_coord, [])
        with pytest.raises(ValueError, match=match_common):
            self.cube.add_dim_coord(scalar_dim_coord, ())

        # Make sure that's still the case for a 0-dimensional cube.
        cube = iris.cube.Cube(666)
        assert cube.ndim == 0
        with pytest.raises(TypeError):
            self.cube.add_dim_coord(scalar_dim_coord)
        with pytest.raises(TypeError):
            self.cube.add_dim_coord(scalar_dim_coord, None)
        with pytest.raises(ValueError, match=match_common):
            self.cube.add_dim_coord(scalar_dim_coord, [])
        with pytest.raises(ValueError, match=match_common):
            self.cube.add_dim_coord(scalar_dim_coord, ())

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord)
        cube.add_aux_coord(scalar_aux_coord)
        assert set(cube.aux_coords) == {scalar_dim_coord, scalar_aux_coord}

        # Various options for dims
        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord, [])
        assert cube.aux_coords == (scalar_dim_coord,)

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord, ())
        assert cube.aux_coords == (scalar_dim_coord,)

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord, None)
        assert cube.aux_coords == (scalar_dim_coord,)

        cube = self.cube.copy()
        cube.add_aux_coord(scalar_dim_coord)
        assert cube.aux_coords == (scalar_dim_coord,)

    def test_add_aux_coord(self):
        y_another = iris.coords.DimCoord(
            np.array([2.5, 7.5, 12.5]), long_name="y_another"
        )

        # DimCoords can live in cube.aux_coords
        self.cube.add_aux_coord(y_another, 0)
        assert self.cube.dim_coords == ()
        assert self.cube.coords() == [y_another]
        assert self.cube.aux_coords == (y_another,)

        # AuxCoords in cube.aux_coords
        self.cube.add_aux_coord(self.xy, [0, 1])
        assert self.cube.dim_coords == ()
        assert self.cube.coords() == [y_another, self.xy]
        assert set(self.cube.aux_coords) == {y_another, self.xy}

        # Lengths must match up
        cube = self.cube.copy()
        with pytest.raises(ValueError, match="Duplicate coordinates are not permitted"):
            cube.add_aux_coord(self.xy, [1, 0])

    def test_remove_coord(self):
        self.cube.add_dim_coord(self.y, 0)
        self.cube.add_dim_coord(self.x, 1)
        self.cube.add_aux_coord(self.xy, (0, 1))
        assert set(self.cube.coords()) == {self.y, self.x, self.xy}

        self.cube.remove_coord("xy")
        assert set(self.cube.coords()) == {self.y, self.x}

        self.cube.remove_coord("x")
        assert self.cube.coords() == [self.y]

        self.cube.remove_coord("y")
        assert self.cube.coords() == []

    def test_immutable_dimcoord_dims(self):
        # Add DimCoord to dimension 1
        dims = [1]
        self.cube.add_dim_coord(self.x, dims)
        assert self.cube.coord_dims(self.x) == (1,)

        # Change dims object
        dims[0] = 0
        # Check the cube is unchanged
        assert self.cube.coord_dims(self.x) == (1,)

        # Check coord_dims cannot be changed
        dims = self.cube.coord_dims(self.x)
        with pytest.raises(TypeError):
            dims[0] = 0

    def test_immutable_auxcoord_dims(self):
        # Add AuxCoord to dimensions (0, 1)
        dims = [0, 1]
        self.cube.add_aux_coord(self.xy, dims)
        assert self.cube.coord_dims(self.xy) == (0, 1)

        # Change dims object
        dims[0] = 1
        dims[1] = 0
        # Check the cube is unchanged
        assert self.cube.coord_dims(self.xy) == (0, 1)

        # Check coord_dims cannot be changed
        dims = self.cube.coord_dims(self.xy)
        with pytest.raises(TypeError):
            dims[0] = 1


@_shared_utils.skip_data
class TestStockCubeStringRepresentations:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.realistic_4d()

    def test_4d_str(self, request):
        _shared_utils.assert_string(request, str(self.cube))

    def test_4d_repr(self, request):
        _shared_utils.assert_string(request, repr(self.cube))

    def test_3d_str(self, request):
        _shared_utils.assert_string(request, str(self.cube[0]))

    def test_3d_repr(self, request):
        _shared_utils.assert_string(request, repr(self.cube[0]))

    def test_2d_str(self, request):
        _shared_utils.assert_string(request, str(self.cube[0, 0]))

    def test_2d_repr(self, request):
        _shared_utils.assert_string(request, repr(self.cube[0, 0]))

    def test_1d_str(self, request):
        _shared_utils.assert_string(request, str(self.cube[0, 0, 0]))

    def test_1d_repr(self, request):
        _shared_utils.assert_string(request, repr(self.cube[0, 0, 0]))

    def test_0d_str(self, request):
        _shared_utils.assert_string(request, str(self.cube[0, 0, 0, 0]))

    def test_0d_repr(self, request):
        _shared_utils.assert_string(request, repr(self.cube[0, 0, 0, 0]))


@_shared_utils.skip_data
class TestCubeStringRepresentations(IrisDotTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        path = _shared_utils.get_data_path(("PP", "simple_pp", "global.pp"))
        self.cube_2d = iris.load_cube(path)
        # Generate the unicode cube up here now it's used in two tests.
        unicode_str = chr(40960) + "abcd" + chr(1972)
        self.unicode_cube = iris.tests.stock.simple_1d()
        self.unicode_cube.attributes["source"] = unicode_str

    def test_dot_simple_pp(self):
        # Test dot output of a 2d cube loaded from pp.
        cube = self.cube_2d
        cube.attributes["my_attribute"] = "foobar"
        self.check_dot(cube, ("file_load", "global_pp.dot"))

        pt = cube.coord("time")
        # and with custom coord attributes
        pt.attributes["monty"] = "python"
        pt.attributes["brain"] = "hurts"
        self.check_dot(cube, ("file_load", "coord_attributes.dot"))

        del pt.attributes["monty"]
        del pt.attributes["brain"]
        del cube.attributes["my_attribute"]

    # TODO hybrid height and dot output - relatitionship links
    @_shared_utils.skip_data
    def test_dot_4d(self):
        cube = iris.tests.stock.realistic_4d()
        self.check_dot(cube, ("file_load", "4d_pp.dot"))

    @_shared_utils.skip_data
    def test_missing_coords(self, request):
        cube = iris.tests.stock.realistic_4d()
        cube.remove_coord("time")
        cube.remove_coord("model_level_number")
        _shared_utils.assert_string(
            request, repr(cube), ("cdm", "str_repr", "missing_coords_cube.repr.txt")
        )
        _shared_utils.assert_string(
            request, str(cube), ("cdm", "str_repr", "missing_coords_cube.str.txt")
        )

    @_shared_utils.skip_data
    def test_cubelist_string(self, request):
        cube_list = iris.cube.CubeList(
            [iris.tests.stock.realistic_4d(), iris.tests.stock.global_pp()]
        )
        _shared_utils.assert_string(
            request, str(cube_list), ("cdm", "str_repr", "cubelist.__str__.txt")
        )
        _shared_utils.assert_string(
            request, repr(cube_list), ("cdm", "str_repr", "cubelist.__repr__.txt")
        )

    def test_basic_0d_cube(self, request):
        _shared_utils.assert_string(
            request,
            repr(self.cube_2d[0, 0]),
            ("cdm", "str_repr", "0d_cube.__repr__.txt"),
        )
        _shared_utils.assert_string(
            request,
            str(self.cube_2d[0, 0]),
            ("cdm", "str_repr", "0d_cube.__unicode__.txt"),
        )
        _shared_utils.assert_string(
            request, str(self.cube_2d[0, 0]), ("cdm", "str_repr", "0d_cube.__str__.txt")
        )

    def test_similar_coord(self, request):
        cube = self.cube_2d.copy()

        lon = cube.coord("longitude")
        lon.attributes["flight"] = "218BX"
        lon.attributes["sensor_id"] = 808
        lon.attributes["status"] = 2
        lon2 = lon.copy()
        lon2.attributes["sensor_id"] = 810
        lon2.attributes["ref"] = "A8T-22"
        del lon2.attributes["status"]
        cube.add_aux_coord(lon2, [1])

        lat = cube.coord("latitude")
        lat2 = lat.copy()
        lat2.attributes["test"] = "True"
        cube.add_aux_coord(lat2, [0])

        _shared_utils.assert_string(
            request, str(cube), ("cdm", "str_repr", "similar.__str__.txt")
        )

    def test_cube_summary_cell_methods(self, request):
        cube = self.cube_2d.copy()

        # Create a list of values used to create cell methods
        test_values = (
            (
                ("mean",),
                ("longitude", "latitude"),
                ("6 minutes", "12 minutes"),
                ("This is a test comment",),
            ),
            (
                ("average",),
                ("longitude", "latitude"),
                ("6 minutes", "15 minutes"),
                ("This is another test comment", "This is another comment"),
            ),
            (("average",), ("longitude", "latitude"), (), ()),
            (
                ("percentile",),
                ("longitude",),
                ("6 minutes",),
                ("This is another test comment",),
            ),
        )

        for x in test_values:
            # Create a cell method
            cm = iris.coords.CellMethod(
                method=x[0][0], coords=x[1], intervals=x[2], comments=x[3]
            )
            cube.add_cell_method(cm)

        _shared_utils.assert_string(
            request, str(cube), ("cdm", "str_repr", "cell_methods.__str__.txt")
        )

    def test_cube_summary_alignment(self, request):
        # Test the cube summary dimension alignment and coord name clipping
        cube = iris.tests.stock.simple_1d()
        aux = iris.coords.AuxCoord(
            np.arange(11),
            long_name="This is a really, really, really, really long "
            "long_name that must be clipped because it is too long",
        )
        cube.add_aux_coord(aux, 0)
        aux = iris.coords.AuxCoord(np.arange(11), long_name="This is a short long_name")
        cube.add_aux_coord(aux, 0)
        _shared_utils.assert_string(
            request, str(cube), ("cdm", "str_repr", "simple.__str__.txt")
        )

    def test_unicode_attribute(self, request):
        _shared_utils.assert_string(
            request,
            str(self.unicode_cube),
            ("cdm", "str_repr", "unicode_attribute.__unicode__.txt"),
        )


@_shared_utils.skip_data
class TestValidity:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube_2d = iris.load_cube(
            _shared_utils.get_data_path(("PP", "simple_pp", "global.pp"))
        )

    def test_wrong_length_vector_coord(self):
        wobble = iris.coords.DimCoord(points=[1, 2], long_name="wobble", units="1")
        with pytest.raises(ValueError, match="Unequal lengths"):
            self.cube_2d.add_aux_coord(wobble, 0)

    def test_invalid_dimension_vector_coord(self):
        wobble = iris.coords.DimCoord(points=[1, 2], long_name="wobble", units="1")
        with pytest.raises(
            ValueError, match="cube does not have the specified dimension"
        ):
            self.cube_2d.add_dim_coord(wobble, 99)


class TestQueryCoord:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.t = iris.tests.stock.simple_2d_w_multidim_and_scalars()

    def test_name(self):
        coords = self.t.coords("dim1")
        assert [coord.name() for coord in coords] == ["dim1"]

        coords = self.t.coords("dim2")
        assert [coord.name() for coord in coords] == ["dim2"]

        coords = self.t.coords("an_other")
        assert [coord.name() for coord in coords] == ["an_other"]

        coords = self.t.coords("air_temperature")
        assert [coord.name() for coord in coords] == ["air_temperature"]

        coords = self.t.coords("wibble")
        assert coords == []

    def test_long_name(self):
        # Both standard_name and long_name defined
        coords = self.t.coords(long_name="custom long name")
        # coord.name() returns standard_name if available
        assert [coord.name() for coord in coords] == ["air_temperature"]

    def test_standard_name(self):
        # Both standard_name and long_name defined
        coords = self.t.coords(standard_name="custom long name")
        assert [coord.name() for coord in coords] == []
        coords = self.t.coords(standard_name="air_temperature")
        assert [coord.name() for coord in coords] == ["air_temperature"]

    def test_var_name(self):
        coords = self.t.coords(var_name="custom_var_name")
        # Matching coord in test cube has a standard_name of 'air_temperature'.
        assert [coord.name() for coord in coords] == ["air_temperature"]

    def test_axis(self):
        cube = self.t.copy()
        cube.coord("dim1").rename("latitude")
        cube.coord("dim2").rename("longitude")

        coords = cube.coords(axis="y")
        assert [coord.name() for coord in coords] == ["latitude"]

        coords = cube.coords(axis="x")
        assert [coord.name() for coord in coords] == ["longitude"]

        # Renaming shouldn't be enough
        cube.coord("an_other").rename("time")
        coords = cube.coords(axis="t")
        assert [coord.name() for coord in coords] == []
        # Change units to "hours since ..." as it's the presence of a
        # time unit that identifies a time axis.
        cube.coord("time").units = "hours since 1970-01-01 00:00:00"
        coords = cube.coords(axis="t")
        assert [coord.name() for coord in coords] == ["time"]

        coords = cube.coords(axis="z")
        assert coords == []

    def test_contains_dimension(self):
        coords = self.t.coords(contains_dimension=0)
        assert [coord.name() for coord in coords] == ["dim1", "my_multi_dim_coord"]

        coords = self.t.coords(contains_dimension=1)
        assert [coord.name() for coord in coords] == ["dim2", "my_multi_dim_coord"]

        coords = self.t.coords(contains_dimension=2)
        assert coords == []

    def test_dimensions(self):
        coords = self.t.coords(dimensions=0)
        assert [coord.name() for coord in coords] == ["dim1"]

        coords = self.t.coords(dimensions=1)
        assert [coord.name() for coord in coords] == ["dim2"]

        # find all coordinates which do not describe a dimension
        coords = self.t.coords(dimensions=[])
        assert [coord.name() for coord in coords] == ["air_temperature", "an_other"]

        coords = self.t.coords(dimensions=2)
        assert coords == []

        coords = self.t.coords(dimensions=[0, 1])
        assert [coord.name() for coord in coords] == ["my_multi_dim_coord"]

    def test_coord_dim_coords_keyword(self):
        coords = self.t.coords(dim_coords=True)
        assert set([coord.name() for coord in coords]) == {"dim1", "dim2"}

        coords = self.t.coords(dim_coords=False)
        assert set([coord.name() for coord in coords]) == {
            "an_other",
            "my_multi_dim_coord",
            "air_temperature",
        }

    def test_coords_empty(self):
        coords = self.t.coords()
        assert set([coord.name() for coord in coords]) == {
            "dim1",
            "dim2",
            "an_other",
            "my_multi_dim_coord",
            "air_temperature",
        }

    def test_coord(self):
        coords = self.t.coords(self.t.coord("dim1"))
        assert [coord.name() for coord in coords] == ["dim1"]
        # check for metadata look-up by modifying points
        coord = self.t.coord("dim1").copy()
        coord.points = np.arange(5) * 1.23
        coords = self.t.coords(coord)
        assert [coord.name() for coord in coords] == ["dim1"]

    def test_str_repr(self, request):
        # TODO consolidate with the TestCubeStringRepresentations class
        _shared_utils.assert_string(
            request, str(self.t), ("cdm", "str_repr", "multi_dim_coord.__str__.txt")
        )
        _shared_utils.assert_string(
            request, repr(self.t), ("cdm", "str_repr", "multi_dim_coord.__repr__.txt")
        )


class TestCube2d:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.t = iris.tests.stock.simple_2d_w_multidim_and_scalars()
        self.t.remove_coord("air_temperature")


class Test2dIndexing(TestCube2d):
    def test_indexing_of_0d_cube(self):
        c = self.t[0, 0]
        with pytest.raises(IndexError, match="More slices requested than dimensions"):
            _ = c[:]

    def test_cube_indexing_0d(self, request):
        _shared_utils.assert_CML(
            request, [self.t[0, 0]], ("cube_slice", "2d_to_0d_cube_slice.cml")
        )

    def test_cube_indexing_1d(self, request):
        _shared_utils.assert_CML(
            request, [self.t[0, 0:]], ("cube_slice", "2d_to_1d_cube_slice.cml")
        )

    def test_cube_indexing_1d_multi_slice(self, request):
        _shared_utils.assert_CML(
            request,
            [self.t[0, (0, 1)]],
            ("cube_slice", "2d_to_1d_cube_multi_slice.cml"),
        )
        _shared_utils.assert_CML(
            request,
            [self.t[0, np.array([0, 1])]],
            ("cube_slice", "2d_to_1d_cube_multi_slice.cml"),
        )

    def test_cube_indexing_1d_multi_slice2(self, request):
        _shared_utils.assert_CML(
            request,
            [self.t[(0, 2), (0, 1, 3)]],
            ("cube_slice", "2d_to_1d_cube_multi_slice2.cml"),
        )
        _shared_utils.assert_CML(
            request,
            [self.t[np.array([0, 2]), (0, 1, 3)]],
            ("cube_slice", "2d_to_1d_cube_multi_slice2.cml"),
        )
        _shared_utils.assert_CML(
            request,
            [self.t[np.array([0, 2]), np.array([0, 1, 3])]],
            ("cube_slice", "2d_to_1d_cube_multi_slice2.cml"),
        )

    def test_cube_indexing_1d_multi_slice3(self, request):
        _shared_utils.assert_CML(
            request,
            [self.t[(0, 2), :]],
            ("cube_slice", "2d_to_1d_cube_multi_slice3.cml"),
        )
        _shared_utils.assert_CML(
            request,
            [self.t[np.array([0, 2]), :]],
            ("cube_slice", "2d_to_1d_cube_multi_slice3.cml"),
        )

    def test_cube_indexing_no_change(self, request):
        _shared_utils.assert_CML(
            request, [self.t[0:, 0:]], ("cube_slice", "2d_orig.cml")
        )

    def test_cube_indexing_reverse_coords(self, request):
        _shared_utils.assert_CML(
            request, [self.t[::-1, ::-1]], ("cube_slice", "2d_to_2d_revesed.cml")
        )

    def test_cube_indexing_no_residual_change(self, request):
        self.t[0:3]
        _shared_utils.assert_CML(request, [self.t], ("cube_slice", "2d_orig.cml"))

    def test_overspecified(self):
        with pytest.raises(IndexError, match="More slices requested than dimensions"):
            _ = self.t[0, 0, Ellipsis, 0]
        with pytest.raises(IndexError, match="More slices requested than dimensions"):
            _ = self.t[0, 0, 0]

    def test_ellipsis(self, request):
        _shared_utils.assert_CML(
            request, [self.t[Ellipsis]], ("cube_slice", "2d_orig.cml")
        )
        _shared_utils.assert_CML(
            request, [self.t[:, :, :]], ("cube_slice", "2d_orig.cml")
        )
        _shared_utils.assert_CML(
            request, [self.t[Ellipsis, Ellipsis]], ("cube_slice", "2d_orig.cml")
        )
        _shared_utils.assert_CML(
            request,
            [self.t[Ellipsis, Ellipsis, Ellipsis]],
            ("cube_slice", "2d_orig.cml"),
        )

        _shared_utils.assert_CML(
            request, [self.t[Ellipsis, 0, 0]], ("cube_slice", "2d_to_0d_cube_slice.cml")
        )
        _shared_utils.assert_CML(
            request, [self.t[0, Ellipsis, 0]], ("cube_slice", "2d_to_0d_cube_slice.cml")
        )
        _shared_utils.assert_CML(
            request, [self.t[0, 0, Ellipsis]], ("cube_slice", "2d_to_0d_cube_slice.cml")
        )

        _shared_utils.assert_CML(
            request,
            [self.t[Ellipsis, (0, 2), :]],
            ("cube_slice", "2d_to_1d_cube_multi_slice3.cml"),
        )
        _shared_utils.assert_CML(
            request,
            [self.t[(0, 2), Ellipsis, :]],
            ("cube_slice", "2d_to_1d_cube_multi_slice3.cml"),
        )
        _shared_utils.assert_CML(
            request,
            [self.t[(0, 2), :, Ellipsis]],
            ("cube_slice", "2d_to_1d_cube_multi_slice3.cml"),
        )


class TestIteration(TestCube2d):
    def test_cube_iteration(self):
        with pytest.raises(TypeError):
            for subcube in self.t:
                pass

    def test_not_iterable(self):
        assert not isinstance(self.t, collections.abc.Iterable)


class Test2dSlicing(TestCube2d):
    def test_cube_slice_all_dimensions(self, request):
        for cube in self.t.slices(["dim1", "dim2"]):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_orig.cml"))

    def test_cube_slice_with_transpose(self, request):
        for cube in self.t.slices(["dim2", "dim1"]):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_transposed.cml"))

    def test_cube_slice_without_transpose(self, request):
        for cube in self.t.slices(["dim2", "dim1"], ordered=False):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_orig.cml"))

    def test_cube_slice_1dimension(self, request):
        # Result came from the equivalent test test_cube_indexing_1d which
        # does self.t[0, 0:]
        slices = [res for res in self.t.slices(["dim2"])]
        _shared_utils.assert_CML(
            request, slices[0], ("cube_slice", "2d_to_1d_cube_slice.cml")
        )

    def test_cube_slice_zero_len_slice(self):
        with pytest.raises(IndexError, match="Cannot index with zero length slice"):
            _ = self.t[0:0]

    def test_cube_slice_with_non_existant_coords(self):
        with pytest.raises(iris.exceptions.CoordinateNotFoundError):
            self.t.slices(["dim2", "dim1", "doesn't exist"])

    def test_cube_extract_coord_with_non_describing_coordinates(self):
        with pytest.raises(ValueError, match="does not describe a dimension"):
            self.t.slices(["an_other"])


class Test2dSlicing_ByDim(TestCube2d):
    def test_cube_slice_all_dimensions(self, request):
        for cube in self.t.slices([0, 1]):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_orig.cml"))

    def test_cube_slice_with_transpose(self, request):
        for cube in self.t.slices([1, 0]):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_transposed.cml"))

    def test_cube_slice_without_transpose(self, request):
        for cube in self.t.slices([1, 0], ordered=False):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_orig.cml"))

    def test_cube_slice_1dimension(self, request):
        # Result came from the equivalent test test_cube_indexing_1d which
        # does self.t[0, 0:]
        slices = [res for res in self.t.slices([1])]
        _shared_utils.assert_CML(
            request, slices[0], ("cube_slice", "2d_to_1d_cube_slice.cml")
        )

    def test_cube_slice_nodimension(self, request):
        slices = [res for res in self.t.slices([])]
        _shared_utils.assert_CML(
            request, slices[0], ("cube_slice", "2d_to_0d_cube_slice.cml")
        )

    def test_cube_slice_with_non_existant_dims(self):
        with pytest.raises(IndexError):
            self.t.slices([1, 0, 2])

    def test_cube_slice_duplicate_dimensions(self):
        with pytest.raises(ValueError, match="coordinates are not orthogonal"):
            self.t.slices([1, 1])


class Test2dSlicing_ByMix(TestCube2d):
    def test_cube_slice_all_dimensions(self, request):
        for cube in self.t.slices([0, "dim2"]):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_orig.cml"))

    def test_cube_slice_with_transpose(self, request):
        for cube in self.t.slices(["dim2", 0]):
            _shared_utils.assert_CML(request, cube, ("cube_slice", "2d_transposed.cml"))

    def test_cube_slice_with_non_existant_dims(self):
        with pytest.raises(ValueError, match="does not describe a dimension"):
            self.t.slices([1, 0, "an_other"])


class Test2dExtraction(TestCube2d):
    def test_cube_extract_0d(self, request):
        # Extract the first value from each of the coords in the cube
        # this result is shared with the self.t[0, 0] test
        _shared_utils.assert_CML(
            request,
            [
                self.t.extract(
                    iris.Constraint(dim1=3.0, dim2=iris.coords.Cell(0, (0, 1)))
                )
            ],
            ("cube_slice", "2d_to_0d_cube_slice.cml"),
        )

    def test_cube_extract_1d(self, request):
        # Extract the first value from the second coord in the cube
        # this result is shared with the self.t[0, 0:] test
        _shared_utils.assert_CML(
            request,
            [self.t.extract(iris.Constraint(dim1=3.0))],
            ("cube_slice", "2d_to_1d_cube_slice.cml"),
        )

    def test_cube_extract_2d(self, request):
        # Do nothing - return the original
        _shared_utils.assert_CML(
            request, [self.t.extract(iris.Constraint())], ("cube_slice", "2d_orig.cml")
        )

    def test_cube_extract_coord_which_does_not_exist(self):
        assert self.t.extract(iris.Constraint(doesnt_exist=8.1)) is None

    def test_cube_extract_coord_with_non_existant_values(self):
        assert self.t.extract(iris.Constraint(dim1=8)) is None


class Test2dExtractionByCoord(TestCube2d):
    def test_cube_extract_by_coord_advanced(self, request):
        # This test reverses the coordinate in the cube and also takes a subset of the original coordinate
        points = np.array([9, 8, 7, 5, 4, 3, 2, 1, 0], dtype=np.int32)
        bounds = np.array(
            [
                [18, 19],
                [16, 17],
                [14, 15],
                [10, 11],
                [8, 9],
                [6, 7],
                [4, 5],
                [2, 3],
                [0, 1],
            ],
            dtype=np.int32,
        )
        c = iris.coords.DimCoord(
            points, long_name="dim2", units="meters", bounds=bounds
        )
        _shared_utils.assert_CML(
            request, self.t.subset(c), ("cube_slice", "2d_intersect_and_reverse.cml")
        )


@_shared_utils.skip_data
class TestCubeExtract:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.single_cube = iris.load_cube(
            _shared_utils.get_data_path(("PP", "globClim1", "theta.pp")),
            "air_potential_temperature",
        )

    def test_simple(self, request):
        constraint = iris.Constraint(latitude=10)
        cube = self.single_cube.extract(constraint)
        _shared_utils.assert_CML(request, cube, ("cdm", "extract", "lat_eq_10.cml"))
        constraint = iris.Constraint(latitude=lambda c: c > 10)
        _shared_utils.assert_CML(
            request,
            self.single_cube.extract(constraint),
            ("cdm", "extract", "lat_gt_10.cml"),
        )

    def test_combined(self, request):
        constraint = iris.Constraint(
            latitude=lambda c: c > 10, longitude=lambda c: c >= 10
        )

        _shared_utils.assert_CML(
            request,
            self.single_cube.extract(constraint),
            ("cdm", "extract", "lat_gt_10_and_lon_ge_10.cml"),
        )

    def test_no_results(self):
        constraint = iris.Constraint(latitude=lambda c: c > 1000000)
        assert self.single_cube.extract(constraint) is None


class TestCubeAPI(TestCube2d):
    def test_getting_standard_name(self):
        assert self.t.name() == "test 2d dimensional cube"

    def test_rename(self):
        self.t.rename("foo")
        assert self.t.name() == "foo"

    def test_var_name(self):
        self.t.var_name = None
        assert self.t.var_name is None
        self.t.var_name = "bar"
        assert self.t.var_name == "bar"

    def test_default_name(self):
        self.t.long_name = ""
        assert self.t.name() == "unknown"
        assert self.t.name("a_default") == "a_default"

    def test_stash_name(self):
        self.t.long_name = ""
        self.t.attributes["STASH"] = iris.fileformats.pp.STASH(1, 2, 3)
        assert self.t.name() == "m01s02i003"

    def test_name_and_var_name(self):
        # Assign only var_name.
        self.t.standard_name = None
        self.t.long_name = None
        self.t.var_name = "foo"
        # name() should return var_name if standard_name and
        # long_name are None.
        assert self.t.name() == "foo"

    def test_rename_and_var_name(self):
        self.t.var_name = "bar"
        self.t.rename("foo")
        # Rename should clear var_name.
        assert self.t.var_name is None

    def test_setting_invalid_var_name(self):
        # Name with whitespace should raise an exception.
        with pytest.raises(ValueError, match="not a valid NetCDF variable name"):
            self.t.var_name = "foo bar"

    def test_setting_empty_var_name(self):
        # Empty string should raise an exception.
        with pytest.raises(ValueError, match="not a valid NetCDF variable name"):
            self.t.var_name = ""

    def test_getting_units(self):
        assert self.t.units == cf_units.Unit("meters")

    def test_setting_units(self):
        assert self.t.units == cf_units.Unit("meters")
        self.t.units = "kelvin"
        assert self.t.units == cf_units.Unit("kelvin")

    def test_clearing_units(self):
        self.t.units = None
        assert str(self.t.units) == "unknown"

    def test_convert_units(self):
        # Set to 'volt'
        self.t.units = cf_units.Unit("volt")
        data = self.t.data.copy()
        # Change to 'kV' - data should be scaled automatically.
        self.t.convert_units("kV")
        assert str(self.t.units) == "kV"
        _shared_utils.assert_array_almost_equal(self.t.data, data / 1000.0)

    def test_coords_are_copies(self):
        assert self.t.coord("dim1") is not self.t.copy().coord("dim1")

    def test_metadata_nop(self):
        self.t.metadata = self.t.metadata
        assert self.t.standard_name is None
        assert self.t.long_name == "test 2d dimensional cube"
        assert self.t.var_name is None
        assert self.t.units == "meters"
        assert self.t.attributes == {}
        assert self.t.cell_methods == ()

    def test_metadata_tuple(self):
        metadata = ("air_pressure", "foo", "bar", "", {"random": "12"}, ())
        self.t.metadata = metadata
        assert self.t.standard_name == "air_pressure"
        assert self.t.long_name == "foo"
        assert self.t.var_name == "bar"
        assert self.t.units == ""
        assert self.t.attributes == metadata[4]
        assert self.t.attributes is not metadata[4]
        assert self.t.cell_methods == ()

    def test_metadata_dict(self):
        metadata = {
            "standard_name": "air_pressure",
            "long_name": "foo",
            "var_name": "bar",
            "units": "",
            "attributes": {"random": "12"},
            "cell_methods": (),
        }
        self.t.metadata = metadata
        assert self.t.standard_name == "air_pressure"
        assert self.t.long_name == "foo"
        assert self.t.var_name == "bar"
        assert self.t.units == ""
        assert self.t.attributes == metadata["attributes"]
        assert self.t.attributes is not metadata["attributes"]
        assert self.t.cell_methods == ()

    def test_metadata_attrs(self):
        class Metadata:
            pass

        metadata = Metadata()
        metadata.standard_name = "air_pressure"
        metadata.long_name = "foo"
        metadata.var_name = "bar"
        metadata.units = ""
        metadata.attributes = {"random": "12"}
        metadata.cell_methods = ()
        metadata.cell_measures_and_dims = []
        self.t.metadata = metadata
        assert self.t.standard_name == "air_pressure"
        assert self.t.long_name == "foo"
        assert self.t.var_name == "bar"
        assert self.t.units == ""
        assert self.t.attributes == metadata.attributes
        assert self.t.attributes is not metadata.attributes
        assert self.t.cell_methods == ()
        assert self.t._cell_measures_and_dims == []

    def test_metadata_fail(self):
        with pytest.raises(TypeError):
            self.t.metadata = (
                "air_pressure",
                "foo",
                "bar",
                "",
                {"random": "12"},
            )
        with pytest.raises(TypeError):
            self.t.metadata = (
                "air_pressure",
                "foo",
                "bar",
                "",
                {"random": "12"},
                (),
                [],
                (),
                (),
            )

        class Metadata:
            pass

        metadata = Metadata()
        metadata.standard_name = "air_pressure"
        metadata.long_name = "foo"
        metadata.var_name = "bar"
        metadata.units = ""
        metadata.attributes = {"random": "12"}
        with pytest.raises(TypeError):
            self.t.metadata = metadata


class TestCubeEquality(TestCube2d):
    def test_simple_equality(self):
        assert self.t == self.t.copy()

    def test_data_inequality(self):
        assert self.t != self.t + 1

    def test_coords_inequality(self):
        r = self.t.copy()
        r.remove_coord(r.coord("an_other"))
        assert self.t != r

    def test_attributes_inequality(self):
        r = self.t.copy()
        r.attributes["new_thing"] = None
        assert self.t != r

    def test_array_attributes(self):
        r = self.t.copy()
        r.attributes["things"] = np.arange(3)
        s = r.copy()
        assert s == r

        s.attributes["things"] = np.arange(2)
        assert s != r

        del s.attributes["things"]
        assert s != r

    def test_cell_methods_inequality(self):
        r = self.t.copy()
        r.add_cell_method(iris.coords.CellMethod("mean"))
        assert self.t != r

    def test_not_compatible(self):
        r = self.t.copy()
        assert self.t.is_compatible(r)
        # The following changes should make the cubes incompatible.
        # Different units.
        r.units = "kelvin"
        assert not self.t.is_compatible(r)
        # Different cell_methods.
        r = self.t.copy()
        r.add_cell_method(iris.coords.CellMethod("mean", coords="dim1"))
        assert not self.t.is_compatible(r)
        # Different attributes.
        r = self.t.copy()
        self.t.attributes["source"] = "bob"
        r.attributes["source"] = "alice"
        assert not self.t.is_compatible(r)

    def test_compatible(self):
        r = self.t.copy()
        assert self.t.is_compatible(r)
        # The following changes should not affect compatibility.
        # Different non-common attributes.
        self.t.attributes["source"] = "bob"
        r.attributes["origin"] = "alice"
        assert self.t.is_compatible(r)
        # Different coordinates.
        r.remove_coord("dim1")
        assert self.t.is_compatible(r)
        # Different data.
        r.data = np.zeros(r.shape)
        assert self.t.is_compatible(r)
        # Different var_names (but equal name()).
        r.var_name = "foo"
        assert self.t.is_compatible(r)

    def test_is_compatible_ignore(self):
        r = self.t.copy()
        assert self.t.is_compatible(r)
        # Different histories.
        self.t.attributes["history"] = "One history."
        r.attributes["history"] = "An alternative history."
        assert not self.t.is_compatible(r)
        # Use ignore keyword.
        assert self.t.is_compatible(r, ignore="history")
        assert self.t.is_compatible(r, ignore=("history",))
        assert self.t.is_compatible(r, ignore=r.attributes)

    def test_is_compatible_metadata(self):
        metadata = self.t.metadata
        assert self.t.is_compatible(metadata)


@_shared_utils.skip_data
class TestDataManagerIndexing(TestCube2d):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.load_cube(
            _shared_utils.get_data_path(("PP", "aPProt1", "rotatedMHtimecube.pp"))
        )

    def assert_is_lazy(self, cube):
        assert cube.has_lazy_data()

    def assert_is_not_lazy(self, cube):
        assert not cube.has_lazy_data()

    def test_slices(self):
        lat_cube = next(self.cube.slices(["grid_latitude"]))
        assert lat_cube.has_lazy_data()
        assert self.cube.has_lazy_data()

    def test_cube_empty_indexing(self, request):
        test_filename = ("cube_slice", "real_empty_data_indexing.cml")
        r = self.cube[:5, ::-1][3]
        rshape = r.shape

        # Make sure we still have deferred data.
        assert r.has_lazy_data()
        # check the CML of this result
        _shared_utils.assert_CML(request, r, test_filename)
        # The CML was checked, meaning the data must have been loaded.
        # Check that the cube no longer has deferred data.
        assert not r.has_lazy_data()

        r_data = r.data

        # finally, load the data before indexing and check that it generates the same result
        c = self.cube
        c.data
        c = c[:5, ::-1][3]
        _shared_utils.assert_CML(request, c, test_filename)

        assert rshape == c.shape

        _shared_utils.assert_array_equal(r_data, c.data)

    def test_real_data_cube_indexing(self, request):
        cube = self.cube[(0, 4, 5, 2), 0, 0]
        _shared_utils.assert_CML(
            request, cube, ("cube_slice", "real_data_dual_tuple_indexing1.cml")
        )

        cube = self.cube[0, (0, 4, 5, 2), (3, 5, 5)]
        _shared_utils.assert_CML(
            request, cube, ("cube_slice", "real_data_dual_tuple_indexing2.cml")
        )

        cube = self.cube[(0, 4, 5, 2), 0, (3, 5, 5)]
        _shared_utils.assert_CML(
            request, cube, ("cube_slice", "real_data_dual_tuple_indexing3.cml")
        )

        with pytest.raises(IndexError, match="More slices requested than dimensions"):
            _ = self.cube[(0, 4, 5, 2), (3, 5, 5), 0, 0, 4]
        six_ellipsis = [Ellipsis] * 6
        with pytest.raises(IndexError, match="More slices requested than dimensions"):
            _ = self.cube[*six_ellipsis]

    def test_fancy_indexing_bool_array(self):
        cube = self.cube
        cube.data = np.ma.masked_array(cube.data, mask=cube.data > 100000)
        r = cube[:, cube.coord("grid_latitude").points > 1]
        assert r.shape == (10, 218, 720)
        data = cube.data[:, self.cube.coord("grid_latitude").points > 1, :]
        _shared_utils.assert_array_equal(data, r.data)
        _shared_utils.assert_array_equal(data.mask, r.data.mask)


class TestCubeCollapsed:
    def partial_compare(self, dual, single):
        result = iris.analysis._dimensional_metadata_comparison(dual, single)
        assert len(result["not_equal"]) == 0
        assert dual.name() == single.name(), (
            "dual and single stage standard_names differ"
        )
        assert dual.units == single.units, "dual and single stage units differ"
        assert dual.shape == single.shape, "dual and single stage shape differ"

    def collapse_test_common(self, request, cube, a_name, b_name, *args, **kwargs):
        # preserve filenames from before the introduction of "grid_" in rotated coord names.
        a_filename = a_name.replace("grid_", "")
        b_filename = b_name.replace("grid_", "")

        # compare dual and single stage collapsing
        dual_stage = cube.collapsed(a_name, iris.analysis.MEAN)
        dual_stage = dual_stage.collapsed(b_name, iris.analysis.MEAN)
        # np.ma.average doesn't apply type promotion rules in some versions,
        # and instead makes the result type float64. To ignore that case we
        # fix up the dtype here if it is promotable from cube.dtype. We still
        # want to catch cases where there is a loss of precision however.
        if dual_stage.dtype > cube.dtype:
            data = dual_stage.data.astype(cube.dtype)
            dual_stage.data = data
        _shared_utils.assert_CML(
            request,
            dual_stage,
            (
                "cube_collapsed",
                "%s_%s_dual_stage.cml" % (a_filename, b_filename),
            ),
            *args,
            **kwargs,
            approx_data=True,
        )

        single_stage = cube.collapsed([a_name, b_name], iris.analysis.MEAN)
        if single_stage.dtype > cube.dtype:
            data = single_stage.data.astype(cube.dtype)
            single_stage.data = data
        _shared_utils.assert_CML(
            request,
            single_stage,
            (
                "cube_collapsed",
                "%s_%s_single_stage.cml" % (a_filename, b_filename),
            ),
            *args,
            **kwargs,
            approx_data=True,
        )

        # Compare the cube bits that should match
        self.partial_compare(dual_stage, single_stage)

    @_shared_utils.skip_data
    def test_multi_d(self, request):
        cube = iris.tests.stock.realistic_4d()

        # TODO: Re-instate surface_altitude & hybrid-height once we're
        # using the post-CF test results.
        cube.remove_aux_factory(cube.aux_factories[0])
        cube.remove_coord("surface_altitude")

        _shared_utils.assert_CML(request, cube, ("cube_collapsed", "original.cml"))

        # Compare 2-stage collapsing with a single stage collapse
        # over 2 Coords.
        self.collapse_test_common(
            request, cube, "grid_latitude", "grid_longitude", rtol=1e-05
        )
        self.collapse_test_common(
            request, cube, "grid_longitude", "grid_latitude", rtol=1e-05
        )

        self.collapse_test_common(request, cube, "time", "grid_latitude", rtol=1e-05)
        self.collapse_test_common(request, cube, "grid_latitude", "time", rtol=1e-05)

        self.collapse_test_common(request, cube, "time", "grid_longitude", rtol=1e-05)
        self.collapse_test_common(request, cube, "grid_longitude", "time", rtol=1e-05)

        self.collapse_test_common(
            request, cube, "grid_latitude", "model_level_number", rtol=5e-04
        )
        self.collapse_test_common(
            request, cube, "model_level_number", "grid_latitude", rtol=5e-04
        )

        self.collapse_test_common(
            request, cube, "grid_longitude", "model_level_number", rtol=5e-04
        )
        self.collapse_test_common(
            request, cube, "model_level_number", "grid_longitude", rtol=5e-04
        )

        self.collapse_test_common(
            request, cube, "time", "model_level_number", rtol=5e-04
        )
        self.collapse_test_common(
            request, cube, "model_level_number", "time", rtol=5e-04
        )

        self.collapse_test_common(
            request, cube, "model_level_number", "time", rtol=5e-04
        )
        self.collapse_test_common(
            request, cube, "time", "model_level_number", rtol=5e-04
        )

        # Collapse 3 things at once.
        triple_collapse = cube.collapsed(
            ["model_level_number", "time", "grid_longitude"],
            iris.analysis.MEAN,
        )
        _shared_utils.assert_CML(
            request,
            triple_collapse,
            ("cube_collapsed", ("triple_collapse_ml_pt_lon.cml")),
            approx_data=True,
            rtol=5e-04,
        )

        triple_collapse = cube.collapsed(
            ["grid_latitude", "model_level_number", "time"], iris.analysis.MEAN
        )
        _shared_utils.assert_CML(
            request,
            triple_collapse,
            ("cube_collapsed", ("triple_collapse_lat_ml_pt.cml")),
            approx_data=True,
            rtol=0.05,
        )
        # KNOWN PROBLEM: the previous 'rtol' is very large.
        # Numpy 1.10 and 1.11 give significantly different results here.
        # This may relate to known problems with summing over large arrays,
        # which were largely fixed in numpy 1.9 but still occur in some cases,
        # as-of numpy 1.11.

        # Ensure no side effects
        _shared_utils.assert_CML(request, cube, ("cube_collapsed", "original.cml"))


@_shared_utils.skip_data
class TestTrimAttributes:
    def test_non_string_attributes(self):
        cube = iris.tests.stock.realistic_4d()
        attrib_key = "gorf"
        attrib_val = 23
        cube.attributes[attrib_key] = attrib_val

        summary = cube.summary()  # Get the cube summary

        # Check through the lines of the summary to see that our attribute is there
        attrib_re = re.compile("%s.*?%s" % (attrib_key, attrib_val))

        for line in summary.split("\n"):
            result = re.match(attrib_re, line.strip())
            if result:
                break
        else:  # No match found for our attribute
            pytest.fail("Attribute not found in summary output of cube.")


@_shared_utils.skip_data
class TestMaskedData:
    def _load_3d_cube(self):
        # This 3D data set has a missing a slice with SOME missing values.
        # The missing data is in the pressure = 1000 hPa, forcast_period = 0,
        # time = 1970-02-11 16:00:00 slice.
        return iris.load_cube(
            _shared_utils.get_data_path(["PP", "mdi_handmade_small", "*.pp"])
        )

    def test_complete_field(self):
        # This pp field has no missing data values
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ["PP", "mdi_handmade_small", "mdi_test_1000_3.pp"]
            )
        )

        assert isinstance(cube.data, np.ndarray)

    def test_masked_field(self):
        # This pp field has some missing data values
        cube = iris.load_cube(
            _shared_utils.get_data_path(
                ["PP", "mdi_handmade_small", "mdi_test_1000_0.pp"]
            )
        )
        assert isinstance(cube.data, ma.core.MaskedArray)

    def test_missing_file(self, request):
        cube = self._load_3d_cube()
        assert isinstance(cube.data, ma.core.MaskedArray)
        _shared_utils.assert_CML(request, cube, ("cdm", "masked_cube.cml"))

    def test_slicing(self):
        cube = self._load_3d_cube()

        # Test the slicing before deferred loading
        full_slice = cube[3]
        partial_slice = cube[0]
        assert isinstance(full_slice.data, np.ndarray)
        assert isinstance(partial_slice.data, ma.core.MaskedArray)
        assert ma.count_masked(partial_slice.data) == 25

        # Test the slicing is consistent after deferred loading
        full_slice = cube[3]
        partial_slice = cube[0]
        assert isinstance(full_slice.data, np.ndarray)
        assert isinstance(partial_slice.data, ma.core.MaskedArray)
        assert ma.count_masked(partial_slice.data) == 25

    def test_save_and_merge(self):
        cube = self._load_3d_cube()
        dtype = cube.dtype
        fill_value = 123456

        # extract the 2d field that has SOME missing values
        masked_slice = cube[0]
        masked_slice.data.fill_value = fill_value

        # test saving masked data
        reference_txt_path = _shared_utils.get_result_path(
            ("cdm", "masked_save_pp.txt")
        )
        with _shared_utils.pp_cube_save_test(
            reference_txt_path, reference_cubes=masked_slice
        ) as temp_pp_path:
            iris.save(masked_slice, temp_pp_path)

            # test merge keeps the mdi we just saved
            cube1 = iris.load_cube(temp_pp_path)
            assert cube1.dtype == dtype

            cube2 = cube1.copy()
            # make cube1 and cube2 differ on a scalar coord, to make them mergeable into a 3d cube
            cube2.coord("pressure").points = [1001.0]
            merged_cubes = iris.cube.CubeList([cube1, cube2]).merge()
            assert len(merged_cubes) == 1, "expected a single merged cube"
            merged_cube = merged_cubes[0]
            assert merged_cube.dtype == dtype
            # Check that the original masked-array fill-value is *ignored*.
            _shared_utils.assert_array_all_close(merged_cube.data.fill_value, -1e30)


@_shared_utils.skip_data
class TestConversionToCoordList:
    def test_coord_conversion(self):
        cube = iris.tests.stock.realistic_4d()

        # Single string
        assert len(cube._as_list_of_coords("grid_longitude")) == 1

        # List of string and unicode
        assert len(cube._as_list_of_coords(["grid_longitude", "grid_latitude"])) == 2

        # Coord object(s)
        lat = cube.coords("grid_latitude")[0]
        lon = cube.coords("grid_longitude")[0]
        assert len(cube._as_list_of_coords(lat)) == 1
        assert len(cube._as_list_of_coords([lat, lon])) == 2

        # Mix of string-like and coord
        assert len(cube._as_list_of_coords(["grid_latitude", lon])) == 2

        # Empty list
        assert len(cube._as_list_of_coords([])) == 0

        # Invalid coords
        invalid_choices = [
            iris.analysis.MEAN,  # Caused by mixing up argument order in call to cube.collapsed for example
            None,
            ["grid_latitude", None],
            [lat, None],
        ]

        for coords in invalid_choices:
            with pytest.raises(TypeError):
                cube._as_list_of_coords(coords)
