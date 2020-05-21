# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.


# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np
import numpy.ma as ma

import iris
import iris.aux_factory
import iris.coord_systems
import iris.coords
import iris.exceptions
from iris._data_manager import DataManager
import iris.tests.stock


@tests.skip_data
class TestCoordSlicing(tests.IrisTest):
    def setUp(self):
        cube = iris.tests.stock.realistic_4d()
        self.lat = cube.coord("grid_latitude")
        self.surface_altitude = cube.coord("surface_altitude")

    def test_slice_copy(self):
        a = self.lat
        b = a.copy()
        self.assertEqual(a, b)
        self.assertFalse(a is b)

        a = self.lat
        b = a[:]
        self.assertEqual(a, b)
        self.assertFalse(a is b)

    def test_slice_multiple_indices(self):
        aux_lat = iris.coords.AuxCoord.from_coord(self.lat)
        aux_sliced = aux_lat[(3, 4), :]
        dim_sliced = self.lat[(3, 4), :]

        self.assertEqual(dim_sliced, aux_sliced)

    def test_slice_reverse(self):
        b = self.lat[::-1]
        np.testing.assert_array_equal(b.points, self.lat.points[::-1])
        np.testing.assert_array_equal(b.bounds, self.lat.bounds[::-1, :])

        c = b[::-1]
        self.assertEqual(self.lat, c)

    def test_multidim(self):
        a = self.surface_altitude
        # make some arbitrary bounds
        bound_shape = a.shape + (2,)
        a.bounds = np.arange(np.prod(bound_shape)).reshape(bound_shape)
        b = a[(0, 2), (0, -1)]
        np.testing.assert_array_equal(
            b.points, a.points[(0, 2), :][:, (0, -1)]
        )
        np.testing.assert_array_equal(
            b.bounds, a.bounds[(0, 2), :, :][:, (0, -1), :]
        )


class TestCoordIntersection(tests.IrisTest):
    def setUp(self):
        self.a = iris.coords.DimCoord(
            np.arange(9.0, dtype=np.float32) * 3 + 9.0,
            long_name="foo",
            units="meter",
        )  # 0.75)
        self.a.guess_bounds(0.75)
        pts = np.array(
            [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0],
            dtype=np.float32,
        )
        bnds = np.array(
            [
                [0.75, 3.75],
                [3.75, 6.75],
                [6.75, 9.75],
                [9.75, 12.75],
                [12.75, 15.75],
                [15.75, 18.75],
                [18.75, 21.75],
                [21.75, 24.75],
                [24.75, 27.75],
                [27.75, 30.75],
            ],
            dtype=np.float32,
        )
        self.b = iris.coords.AuxCoord(
            pts, long_name="foo", units="meter", bounds=bnds
        )

    def test_basic_intersection(self):
        inds = self.a.intersect(self.b, return_indices=True)
        self.assertEqual((0, 1, 2, 3, 4, 5, 6, 7), tuple(inds))

        c = self.a.intersect(self.b)
        self.assertXMLElement(c, ("coord_api", "intersection.xml"))

    def test_intersection_reverse(self):
        inds = self.a.intersect(self.b[::-1], return_indices=True)
        self.assertEqual((7, 6, 5, 4, 3, 2, 1, 0), tuple(inds))

        c = self.a.intersect(self.b[::-1])
        self.assertXMLElement(c, ("coord_api", "intersection_reversed.xml"))

    def test_no_intersection_on_points(self):
        # Coordinates which do not share common points but with common
        # bounds should fail
        self.a.points = self.a.points + 200
        self.assertRaises(ValueError, self.a.intersect, self.b)

    def test_intersection_one_fewer_upper_bound_than_lower(self):
        self.b.bounds[4, 1] = self.b.bounds[0, 1]
        c = self.a.intersect(self.b)
        self.assertXMLElement(c, ("coord_api", "intersection_missing.xml"))

    def test_no_intersection_on_bounds(self):
        # Coordinates which do not share common bounds but with common
        # points should fail
        self.a.bounds = None
        a = self.a.copy()
        a.bounds = None
        a.guess_bounds(bound_position=0.25)
        self.assertRaises(ValueError, a.intersect, self.b)

    def test_no_intersection_on_name(self):
        # Coordinates which do not share the same name should fail
        self.a.long_name = "foobar"
        self.assertRaises(ValueError, self.a.intersect, self.b)

    def test_no_intersection_on_unit(self):
        # Coordinates which do not share the same unit should fail
        self.a.units = "kilometer"
        self.assertRaises(ValueError, self.a.intersect, self.b)

    @tests.skip_data
    def test_commutative(self):
        cube = iris.tests.stock.realistic_4d()
        coord = cube.coord("grid_longitude")
        offset_coord = coord.copy()
        offset_coord = offset_coord - (
            offset_coord.points[20] - offset_coord.points[0]
        )
        self.assertEqual(
            coord.intersect(offset_coord), offset_coord.intersect(coord)
        )


class TestXML(tests.IrisTest):
    def test_minimal(self):
        coord = iris.coords.DimCoord(np.arange(10, dtype=np.int32))
        self.assertXMLElement(coord, ("coord_api", "minimal.xml"))

    def test_complex(self):
        crs = iris.coord_systems.GeogCS(6370000)
        coord = iris.coords.AuxCoord(
            np.arange(4, dtype=np.float32),
            "air_temperature",
            "my_long_name",
            units="K",
            attributes={"foo": "bar", "count": 2},
            coord_system=crs,
        )
        coord.guess_bounds(0.5)
        self.assertXMLElement(coord, ("coord_api", "complex.xml"))


@tests.skip_data
class TestCoord_ReprStr_nontime(tests.IrisTest):
    def setUp(self):
        self.lat = iris.tests.stock.realistic_4d().coord("grid_latitude")[:10]

    def test_DimCoord_repr(self):
        self.assertRepr(
            self.lat, ("coord_api", "str_repr", "dim_nontime_repr.txt")
        )

    def test_AuxCoord_repr(self):
        self.assertRepr(
            self.lat, ("coord_api", "str_repr", "aux_nontime_repr.txt")
        )

    def test_DimCoord_str(self):
        self.assertString(
            str(self.lat), ("coord_api", "str_repr", "dim_nontime_str.txt")
        )

    def test_AuxCoord_str(self):
        self.assertString(
            str(self.lat), ("coord_api", "str_repr", "aux_nontime_str.txt")
        )


@tests.skip_data
class TestCoord_ReprStr_time(tests.IrisTest):
    def setUp(self):
        self.time = iris.tests.stock.realistic_4d().coord("time")

    def test_DimCoord_repr(self):
        self.assertRepr(
            self.time, ("coord_api", "str_repr", "dim_time_repr.txt")
        )

    def test_AuxCoord_repr(self):
        self.assertRepr(
            self.time, ("coord_api", "str_repr", "aux_time_repr.txt")
        )

    def test_DimCoord_str(self):
        self.assertString(
            str(self.time), ("coord_api", "str_repr", "dim_time_str.txt")
        )

    def test_AuxCoord_str(self):
        self.assertString(
            str(self.time), ("coord_api", "str_repr", "aux_time_str.txt")
        )


class TestAuxCoordCreation(tests.IrisTest):
    def test_basic(self):
        a = iris.coords.AuxCoord(
            np.arange(10), "air_temperature", units="kelvin"
        )
        result = (
            "AuxCoord("
            "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),"
            " standard_name='air_temperature',"
            " units=Unit('kelvin'))"
        )
        self.assertEqual(result, str(a))

        b = iris.coords.AuxCoord(
            list(range(10)), attributes={"monty": "python"}
        )
        result = (
            "AuxCoord("
            "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),"
            " standard_name=None,"
            " units=Unit('unknown'),"
            " attributes={'monty': 'python'})"
        )
        self.assertEqual(result, str(b))

    def test_excluded_attributes(self):
        with self.assertRaises(ValueError):
            iris.coords.AuxCoord(
                np.arange(10),
                "air_temperature",
                units="kelvin",
                attributes={"standard_name": "whoopsy"},
            )

        a = iris.coords.AuxCoord(
            np.arange(10), "air_temperature", units="kelvin"
        )
        with self.assertRaises(ValueError):
            a.attributes["standard_name"] = "whoopsy"
        with self.assertRaises(ValueError):
            a.attributes.update({"standard_name": "whoopsy"})

    def test_coord_system(self):
        a = iris.coords.AuxCoord(
            np.arange(10),
            "air_temperature",
            units="kelvin",
            coord_system=iris.coord_systems.GeogCS(6000),
        )
        result = (
            "AuxCoord("
            "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),"
            " standard_name='air_temperature',"
            " units=Unit('kelvin'),"
            " coord_system=GeogCS(6000.0))"
        )
        self.assertEqual(result, str(a))

    def test_bounded(self):
        a = iris.coords.AuxCoord(
            np.arange(10),
            "air_temperature",
            units="kelvin",
            bounds=np.arange(0, 20).reshape(10, 2),
        )
        result = (
            "AuxCoord(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
            ", bounds=array(["
            "[ 0,  1],\n       [ 2,  3],\n       [ 4,  5],\n       "
            "[ 6,  7],\n       [ 8,  9],\n       [10, 11],\n       "
            "[12, 13],\n       [14, 15],\n       [16, 17],\n       "
            "[18, 19]])"
            ", standard_name='air_temperature', units=Unit('kelvin'))"
        )
        self.assertEqual(result, str(a))

    def test_string_coord_equality(self):
        b = iris.coords.AuxCoord(["Jan", "Feb", "March"], units="no_unit")
        c = iris.coords.AuxCoord(["Jan", "Feb", "March"], units="no_unit")
        self.assertEqual(b, c)

    def test_AuxCoord_fromcoord(self):
        # Check the coordinate returned by `from_coord` doesn't reference the
        # same coordinate system as the source coordinate.
        crs = iris.coord_systems.GeogCS(6370000)
        a = iris.coords.DimCoord(10, coord_system=crs)
        b = iris.coords.AuxCoord.from_coord(a)
        self.assertIsNot(a.coord_system, b.coord_system)


class TestDimCoordCreation(tests.IrisTest):
    def test_basic(self):
        a = iris.coords.DimCoord(
            np.arange(10), "air_temperature", units="kelvin"
        )
        result = (
            "DimCoord("
            "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),"
            " standard_name='air_temperature',"
            " units=Unit('kelvin'))"
        )
        self.assertEqual(result, str(a))

        b = iris.coords.DimCoord(
            list(range(10)), attributes={"monty": "python"}
        )
        result = (
            "DimCoord("
            "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),"
            " standard_name=None,"
            " units=Unit('unknown'),"
            " attributes={'monty': 'python'})"
        )
        self.assertEqual(result, str(b))

    def test_excluded_attributes(self):
        with self.assertRaises(ValueError):
            iris.coords.DimCoord(
                np.arange(10),
                "air_temperature",
                units="kelvin",
                attributes={"standard_name": "whoopsy"},
            )

        a = iris.coords.DimCoord(
            np.arange(10), "air_temperature", units="kelvin"
        )
        with self.assertRaises(ValueError):
            a.attributes["standard_name"] = "whoopsy"
        with self.assertRaises(ValueError):
            a.attributes.update({"standard_name": "whoopsy"})

    def test_coord_system(self):
        a = iris.coords.DimCoord(
            np.arange(10),
            "air_temperature",
            units="kelvin",
            coord_system=iris.coord_systems.GeogCS(6000),
        )
        result = (
            "DimCoord("
            "array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),"
            " standard_name='air_temperature',"
            " units=Unit('kelvin'),"
            " coord_system=GeogCS(6000.0))"
        )
        self.assertEqual(result, str(a))

    def test_bounded(self):
        a = iris.coords.DimCoord(
            np.arange(10),
            "air_temperature",
            units="kelvin",
            bounds=np.arange(0, 20).reshape(10, 2),
        )
        result = (
            "DimCoord(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])"
            ", bounds=array(["
            "[ 0,  1],\n       [ 2,  3],\n       [ 4,  5],\n       "
            "[ 6,  7],\n       [ 8,  9],\n       [10, 11],\n       "
            "[12, 13],\n       [14, 15],\n       [16, 17],\n       "
            "[18, 19]])"
            ", standard_name='air_temperature', units=Unit('kelvin'))"
        )
        self.assertEqual(result, str(a))

    def test_dim_coord_restrictions(self):
        # 1d
        with self.assertRaisesRegex(ValueError, "must be scalar or 1-dim"):
            iris.coords.DimCoord([[1, 2, 3], [4, 5, 6]])
        # monotonic points
        with self.assertRaisesRegex(ValueError, "must be strictly monotonic"):
            iris.coords.DimCoord([1, 2, 99, 4, 5])
        # monotonic bounds
        with self.assertRaisesRegex(ValueError, "direction of monotonicity"):
            iris.coords.DimCoord([1, 2, 3], bounds=[[1, 12], [2, 9], [3, 6]])
        # masked points
        emsg = "points array must not be masked"
        with self.assertRaisesRegex(TypeError, emsg):
            iris.coords.DimCoord(ma.masked_array([0, 1, 2], mask=[0, 1, 0]))
        # masked bounds
        emsg = "bounds array must not be masked"
        with self.assertRaisesRegex(TypeError, emsg):
            iris.coords.DimCoord(
                [1], bounds=ma.masked_array([[0, 2]], mask=True)
            )
        # shapes of points and bounds
        msg = "The shape of the 'unknown' DimCoord bounds array should be"
        with self.assertRaisesRegex(ValueError, msg):
            iris.coords.DimCoord([1, 2, 3], bounds=[0.5, 1.5, 2.5, 3.5])
        # another example of shapes of points and bounds
        with self.assertRaisesRegex(ValueError, msg):
            iris.coords.DimCoord([1, 2, 3], bounds=[[0.5, 1.5], [1.5, 2.5]])

        # numeric
        with self.assertRaises(ValueError):
            iris.coords.DimCoord(["Jan", "Feb", "March"])

    def test_DimCoord_equality(self):
        # basic regular coord
        b = iris.coords.DimCoord([1, 2])
        c = iris.coords.DimCoord([1, 2.0])
        d = iris.coords.DimCoord([1, 2], circular=True)
        self.assertEqual(b, c)
        self.assertNotEqual(b, d)

    def test_Dim_to_Aux(self):
        a = iris.coords.DimCoord(
            np.arange(10),
            standard_name="air_temperature",
            long_name="custom air temp",
            units="kelvin",
            attributes={"monty": "python"},
            bounds=np.arange(20).reshape(10, 2),
            circular=True,
        )
        b = iris.coords.AuxCoord.from_coord(a)
        # Note - circular attribute is not a factor in equality comparison
        self.assertEqual(a, b)

    def test_DimCoord_fromcoord(self):
        # Check the coordinate returned by `from_coord` doesn't reference the
        # same coordinate system as the source coordinate.
        crs = iris.coord_systems.GeogCS(6370000)
        a = iris.coords.AuxCoord(10, coord_system=crs)
        b = iris.coords.DimCoord.from_coord(a)
        self.assertIsNot(a.coord_system, b.coord_system)

    def test_DimCoord_from_regular(self):
        zeroth = 10.0
        step = 20.0
        count = 100
        kwargs = dict(
            standard_name="latitude",
            long_name="latitude",
            var_name="lat",
            units="degrees",
            attributes=dict(fruit="pear"),
            coord_system=iris.coord_systems.GeogCS(6371229),
            circular=False,
        )

        coord = iris.coords.DimCoord.from_regular(
            zeroth, step, count, **kwargs
        )
        expected_points = np.arange(
            zeroth + step, zeroth + (count + 1) * step, step
        )
        expected = iris.coords.DimCoord(expected_points, **kwargs)
        self.assertIsInstance(coord, iris.coords.DimCoord)
        self.assertEqual(coord, expected)

    def test_DimCoord_from_regular_with_bounds(self):
        zeroth = 3.0
        step = 0.5
        count = 20
        kwargs = dict(
            standard_name="latitude",
            long_name="latitude",
            var_name="lat",
            units="degrees",
            attributes=dict(fruit="pear"),
            coord_system=iris.coord_systems.GeogCS(6371229),
            circular=False,
        )

        coord = iris.coords.DimCoord.from_regular(
            zeroth, step, count, with_bounds=True, **kwargs
        )
        expected_points = np.arange(
            zeroth + step, zeroth + (count + 1) * step, step
        )
        expected_bounds = np.transpose(
            [expected_points - 0.5 * step, expected_points + 0.5 * step]
        )
        expected = iris.coords.DimCoord(
            expected_points, bounds=expected_bounds, **kwargs
        )
        self.assertIsInstance(coord, iris.coords.DimCoord)
        self.assertEqual(coord, expected)


class TestCoordMaths(tests.IrisTest):
    def _build_coord(self, start=None, step=None, count=None):
        # Create points and bounds akin to an old RegularCoord.
        dtype = np.float32
        start = dtype(start or self.start)
        step = dtype(step or self.step)
        count = int(count or self.count)
        bound_position = dtype(0.5)
        points = np.arange(count, dtype=dtype) * step + start
        bounds = np.concatenate(
            [
                [points - bound_position * step],
                [points + (1 - bound_position) * step],
            ]
        ).T
        self.lon = iris.coords.AuxCoord(
            points, "latitude", units="degrees", bounds=bounds
        )
        self.rlon = iris.coords.AuxCoord(
            np.deg2rad(points),
            "latitude",
            units="radians",
            bounds=np.deg2rad(bounds),
        )

    def setUp(self):
        self.start = 0
        self.step = 2.3
        self.count = 20
        self._build_coord()


class TestCoordAdditionSubtract(TestCoordMaths):
    def test_subtract(self):
        r_expl = self.lon - 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "subtract_simple_expl.xml")
        )

    def test_subtract_in_place(self):
        r_expl = self.lon.copy()
        r_expl -= 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "subtract_simple_expl.xml")
        )

    def test_neg(self):
        self._build_coord(start=8)
        r_expl = -self.lon
        np.testing.assert_array_equal(r_expl.points, -(self.lon.points))
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "negate_expl.xml")
        )

    def test_right_subtract(self):
        r_expl = 10 - self.lon
        # XXX original xml was for regular case, not explicit.
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "r_subtract_simple_exl.xml")
        )

    def test_add(self):
        r_expl = self.lon + 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "add_simple_expl.xml")
        )

    def test_add_in_place(self):
        r_expl = self.lon.copy()
        r_expl += 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "add_simple_expl.xml")
        )

    def test_add_float(self):
        r_expl = self.lon + 10.321
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "add_float_expl.xml")
        )
        self.assertEqual(r_expl, 10.321 + self.lon.copy())


class TestCoordMultDivide(TestCoordMaths):
    def test_divide(self):
        r_expl = self.lon.copy() / 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "divide_simple_expl.xml")
        )

    def test_right_divide(self):
        self._build_coord(start=10)
        test_coord = self.lon.copy()

        r_expl = 1 / test_coord
        self.assertXMLElement(
            r_expl,
            ("coord_api", "coord_maths", "right_divide_simple_expl.xml"),
        )

    def test_divide_in_place(self):
        r_expl = self.lon.copy()
        r_expl /= 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "divide_simple_expl.xml")
        )

    def test_multiply(self):
        r_expl = self.lon.copy() * 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "multiply_simple_expl.xml")
        )

    def test_multiply_in_place_reg(self):
        r_expl = self.lon.copy()
        r_expl *= 10
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "multiply_simple_expl.xml")
        )

    def test_multiply_float(self):
        r_expl = self.lon.copy() * 10.321
        self.assertXMLElement(
            r_expl, ("coord_api", "coord_maths", "mult_float_expl.xml")
        )
        self.assertEqual(r_expl, 10.321 * self.lon.copy())


class TestCoordCollapsed(tests.IrisTest):
    def create_1d_coord(self, bounds=None, points=None, units="meter"):
        coord = iris.coords.DimCoord(
            points, long_name="test", units=units, bounds=bounds
        )
        return coord

    def test_explicit(self):
        orig_coord = self.create_1d_coord(
            points=list(range(10)), bounds=[(b, b + 1) for b in range(10)]
        )
        coord_expected = self.create_1d_coord(points=5, bounds=[(0, 10)])

        # test points & bounds
        self.assertEqual(coord_expected, orig_coord.collapsed())

        # test points only
        coord = orig_coord.copy()
        coord_expected = self.create_1d_coord(points=4, bounds=[(0, 9)])
        coord.bounds = None
        self.assertEqual(coord_expected, coord.collapsed())

    def test_circular_collapse(self):
        # set up a coordinate that wraps 360 degrees in points using the
        # circular flag
        coord = self.create_1d_coord(None, np.arange(10) * 36, "degrees")
        expected_coord = self.create_1d_coord([0.0, 360.0], [180.0], "degrees")
        coord.circular = True

        # test collapsing
        self.assertEqual(expected_coord, coord.collapsed())
        # the order of the points/bounds should not affect the resultant
        # bounded coordinate.
        coord = coord[::-1]
        self.assertEqual(expected_coord, coord.collapsed())

    def test_nd_bounds(self):
        cube = iris.tests.stock.simple_2d_w_multidim_coords(with_bounds=True)
        pcube = cube.collapsed(["bar", "foo"], iris.analysis.SUM)
        pcube.data = pcube.data.astype("i8")
        self.assertCML(pcube, ("coord_api", "nd_bounds.cml"))


@tests.skip_data
class TestGetterSetter(tests.IrisTest):
    def test_get_set_points_and_bounds(self):
        cube = iris.tests.stock.realistic_4d()
        coord = cube.coord("grid_latitude")

        # get bounds
        bounds = coord.bounds
        self.assertEqual(bounds.shape, (100, 2))

        self.assertEqual(bounds.shape[-1], coord.nbounds)

        # set bounds
        coord.bounds = bounds + 1

        np.testing.assert_array_equal(coord.bounds, bounds + 1)

        # set bounds - different length to existing points
        with self.assertRaises(ValueError):
            coord.bounds = bounds[::2, :]

        # set points/bounds to None
        with self.assertRaises(ValueError):
            coord.points = None
        coord.bounds = None

        # set bounds from non-numpy pair.
        # First reset the underlying shape of the coordinate.
        coord._values_dm = DataManager(1)
        coord.points = 1
        coord.bounds = [123, 456]
        self.assertEqual(coord.shape, (1,))
        self.assertEqual(coord.bounds.shape, (1, 2))

        # set bounds from non-numpy pairs
        # First reset the underlying shape of the coord's points and bounds.
        coord._values_dm = DataManager(np.arange(3))
        coord.bounds = None
        coord.bounds = [[123, 456], [234, 567], [345, 678]]
        self.assertEqual(coord.shape, (3,))
        self.assertEqual(coord.bounds.shape, (3, 2))


class TestGuessBounds(tests.IrisTest):
    def test_guess_bounds(self):
        coord = iris.coords.DimCoord(
            np.array([0, 10, 20, 30]), long_name="foo", units="1"
        )
        coord.guess_bounds()
        self.assertArrayEqual(
            coord.bounds, np.array([[-5, 5], [5, 15], [15, 25], [25, 35]])
        )

        coord.bounds = None
        coord.guess_bounds(0.25)
        self.assertArrayEqual(
            coord.bounds,
            np.array([[-5, 5], [5, 15], [15, 25], [25, 35]]) + 2.5,
        )

        coord.bounds = None
        coord.guess_bounds(0.75)
        self.assertArrayEqual(
            coord.bounds,
            np.array([[-5, 5], [5, 15], [15, 25], [25, 35]]) - 2.5,
        )

        points = coord.points.copy()
        points[2] = 25
        coord.points = points
        coord.bounds = None
        coord.guess_bounds()
        self.assertArrayEqual(
            coord.bounds,
            np.array([[-5.0, 5.0], [5.0, 17.5], [17.5, 27.5], [27.5, 32.5]]),
        )

        # if the points are not monotonic, then guess_bounds should fail
        points = coord.points.copy()
        points[2] = 32
        coord = iris.coords.AuxCoord.from_coord(coord)
        coord.points = points
        coord.bounds = None
        with self.assertRaises(ValueError):
            coord.guess_bounds()


class TestIsContiguous(tests.IrisTest):
    def test_scalar(self):
        coord = iris.coords.DimCoord(23.0, bounds=[20.0, 26.0])
        self.assertTrue(coord.is_contiguous())

    def test_equal_int(self):
        coord = iris.coords.DimCoord(
            [0, 10, 20], bounds=[[0, 10], [10, 20], [20, 30]]
        )
        self.assertTrue(coord.is_contiguous())

    def test_equal_float(self):
        coord = iris.coords.DimCoord(
            [0.0, 10.0, 20.0], bounds=[[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]]
        )
        self.assertTrue(coord.is_contiguous())

    def test_guessed_bounds(self):
        delta = np.float64(0.00001)
        lower = -1.0 + delta
        upper = 3.0 - delta
        points, step = np.linspace(
            lower, upper, 2, endpoint=False, retstep=True
        )
        points += step * 0.5
        coord = iris.coords.DimCoord(points)
        coord.guess_bounds()
        self.assertTrue(coord.is_contiguous())

    def test_nobounds(self):
        coord = iris.coords.DimCoord([0, 10, 20])
        self.assertFalse(coord.is_contiguous())

    def test_multidim(self):
        points = np.arange(12, dtype=np.float64).reshape(3, 4)
        bounds = np.array([points, points + 1.0]).transpose(1, 2, 0)
        coord = iris.coords.AuxCoord(points, bounds=bounds)
        with self.assertRaises(ValueError):
            coord.is_contiguous()

    def test_one_bound(self):
        coord = iris.coords.DimCoord([0, 10, 20], bounds=[[0], [10], [20]])
        with self.assertRaises(ValueError):
            coord.is_contiguous()

    def test_three_bound(self):
        coord = iris.coords.DimCoord(
            [0, 10, 20], bounds=[[0, 1, 2], [10, 11, 12], [20, 21, 22]]
        )
        with self.assertRaises(ValueError):
            coord.is_contiguous()

    def test_non_contiguous(self):
        # Large enough difference to exceed default tolerance.
        delta = 1e-3
        points = np.array([0.0, 10.0, 20.0])
        bounds = np.array([[0.0, 10.0], [10.0, 20], [20.0, 30.0]])
        coord = iris.coords.DimCoord(points, bounds=bounds)
        self.assertTrue(coord.is_contiguous())

        non_contig_bounds = bounds.copy()
        non_contig_bounds[0, 1] -= delta
        coord = iris.coords.DimCoord(points, bounds=non_contig_bounds)
        self.assertFalse(coord.is_contiguous())

        non_contig_bounds = bounds.copy()
        non_contig_bounds[1, 1] -= delta
        coord = iris.coords.DimCoord(points, bounds=non_contig_bounds)
        self.assertFalse(coord.is_contiguous())

        non_contig_bounds = bounds.copy()
        non_contig_bounds[1, 0] -= delta
        coord = iris.coords.DimCoord(points, bounds=non_contig_bounds)
        self.assertFalse(coord.is_contiguous())

        non_contig_bounds = bounds.copy()
        non_contig_bounds[1, 0] += delta
        coord = iris.coords.DimCoord(points, bounds=non_contig_bounds)
        self.assertFalse(coord.is_contiguous())

        non_contig_bounds = bounds.copy()
        non_contig_bounds[2, 0] -= delta
        coord = iris.coords.DimCoord(points, bounds=non_contig_bounds)
        self.assertFalse(coord.is_contiguous())

    def test_default_tol(self):
        # Smaller difference that default tolerance.
        delta = 1e-6
        points = np.array([0.0, 10.0, 20.0])
        bounds = np.array([[0.0, 10.0], [10.0, 20], [20.0, 30.0]])
        bounds[1, 0] -= delta
        coord = iris.coords.DimCoord(points, bounds=bounds)
        self.assertTrue(coord.is_contiguous())

    def test_specified_tol(self):
        delta = 1e-6
        points = np.array([0.0, 10.0, 20.0])
        bounds = np.array([[0.0, 10.0], [10.0, 20], [20.0, 30.0]])
        bounds[1, 0] += delta
        coord = iris.coords.DimCoord(points, bounds=bounds)
        self.assertTrue(coord.is_contiguous())
        # No tolerance.
        rtol = 0
        atol = 0
        self.assertFalse(coord.is_contiguous(rtol, atol))
        # Absolute only.
        rtol = 0
        atol = 1e-5  # larger than delta.
        self.assertTrue(coord.is_contiguous(rtol, atol))
        atol = 1e-7  # smaller than delta
        self.assertFalse(coord.is_contiguous(rtol, atol))
        # Relative only.
        atol = 0
        rtol = 1e-6  # is multiplied by upper bound (10.0) in comparison.
        self.assertTrue(coord.is_contiguous(rtol, atol))
        rtol = 1e-8
        self.assertFalse(coord.is_contiguous(rtol, atol))


class TestCoordCompatibility(tests.IrisTest):
    def setUp(self):
        self.aux_coord = iris.coords.AuxCoord(
            [1.0, 2.0, 3.0],
            standard_name="longitude",
            var_name="lon",
            units="degrees",
        )
        self.dim_coord = iris.coords.DimCoord(
            np.arange(0, 360, dtype=np.float64),
            standard_name="longitude",
            var_name="lon",
            units="degrees",
            circular=True,
        )

    def test_not_compatible(self):
        r = self.aux_coord.copy()
        self.assertTrue(self.aux_coord.is_compatible(r))
        # The following changes should make the coords incompatible.
        # Different units.
        r.units = "radians"
        self.assertFalse(self.aux_coord.is_compatible(r))
        # Different coord_systems.
        r = self.aux_coord.copy()
        r.coord_system = iris.coord_systems.GeogCS(6371229)
        self.assertFalse(self.aux_coord.is_compatible(r))
        # Different attributes.
        r = self.aux_coord.copy()
        self.aux_coord.attributes["source"] = "bob"
        r.attributes["source"] = "alice"
        self.assertFalse(self.aux_coord.is_compatible(r))

    def test_compatible(self):
        # The following changes should not affect compatibility.
        # Different non-common attributes.
        r = self.aux_coord.copy()
        self.aux_coord.attributes["source"] = "bob"
        r.attributes["origin"] = "alice"
        self.assertTrue(self.aux_coord.is_compatible(r))
        # Different points.
        r.points = np.zeros(r.points.shape)
        self.assertTrue(self.aux_coord.is_compatible(r))
        # Different var_names (but equal name()).
        r.var_name = "foo"
        self.assertTrue(self.aux_coord.is_compatible(r))
        # With/without bounds.
        r.bounds = np.array([[0.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
        self.assertTrue(self.aux_coord.is_compatible(r))

    def test_circular(self):
        # Test that circular has no effect on compatibility.
        # AuxCoord and circular DimCoord.
        self.assertTrue(self.aux_coord.is_compatible(self.dim_coord))
        # circular and non-circular DimCoord.
        r = self.dim_coord.copy()
        r.circular = False
        self.assertTrue(r.is_compatible(self.dim_coord))

    def test_defn(self):
        coord_defn = self.aux_coord._as_defn()
        self.assertTrue(self.aux_coord.is_compatible(coord_defn))
        coord_defn = self.dim_coord._as_defn()
        self.assertTrue(self.dim_coord.is_compatible(coord_defn))

    def test_is_ignore(self):
        r = self.aux_coord.copy()
        self.aux_coord.attributes["source"] = "bob"
        r.attributes["source"] = "alice"
        self.assertFalse(self.aux_coord.is_compatible(r))
        # Use ignore keyword.
        self.assertTrue(self.aux_coord.is_compatible(r, ignore="source"))
        self.assertTrue(self.aux_coord.is_compatible(r, ignore=("source",)))
        self.assertTrue(self.aux_coord.is_compatible(r, ignore=r.attributes))


class TestAuxCoordEquality(tests.IrisTest):
    def test_not_implmemented(self):
        class Terry:
            pass

        aux = iris.coords.AuxCoord(0)
        self.assertIs(aux.__eq__(Terry()), NotImplemented)
        self.assertIs(aux.__ne__(Terry()), NotImplemented)


class TestDimCoordEquality(tests.IrisTest):
    def test_not_implmemented(self):
        class Terry:
            pass

        dim = iris.coords.DimCoord(0)
        aux = iris.coords.AuxCoord(0)
        self.assertIs(dim.__eq__(Terry()), NotImplemented)
        self.assertIs(dim.__ne__(Terry()), NotImplemented)
        self.assertIs(dim.__eq__(aux), NotImplemented)
        self.assertIs(dim.__ne__(aux), NotImplemented)

    def test_climatological(self):
        co1 = iris.coords.DimCoord(
            [0], bounds=[[0, 1]], units="days since 1970-01-01"
        )
        co2 = co1.copy()
        co2.climatological = True
        self.assertNotEqual(co1, co2)
        co2.climatological = False
        self.assertEqual(co1, co2)


if __name__ == "__main__":
    tests.main()
