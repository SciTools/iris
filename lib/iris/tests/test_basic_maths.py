# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import math
import operator

import cf_units
import numpy as np
import numpy.ma as ma
import pytest

import iris
import iris.analysis.maths
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.exceptions import NotYetImplementedError
from iris.tests import _shared_utils
import iris.tests.stock


@_shared_utils.skip_data
class TestBasicMaths:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.global_pp()
        self.cube.data = self.cube.data - 260

    def test_abs(self, request):
        a = self.cube

        b = iris.analysis.maths.abs(a, in_place=False)
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))
        _shared_utils.assert_CML(request, b, ("analysis", "abs.cml"))

        iris.analysis.maths.abs(a, in_place=True)
        _shared_utils.assert_CML(request, b, ("analysis", "abs.cml"))
        _shared_utils.assert_CML(request, a, ("analysis", "abs.cml"))

    def test_minus(self, request):
        a = self.cube
        e = self.cube.copy()

        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        d = a - a
        _shared_utils.assert_CML(request, d, ("analysis", "subtract.cml"))

        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        c = iris.analysis.maths.subtract(e, e)
        _shared_utils.assert_CML(request, c, ("analysis", "subtract.cml"))

        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, e, ("analysis", "maths_original.cml"))

    def test_minus_with_data_describing_coordinate(self):
        a = self.cube
        e = self.cube.copy()
        lat = e.coord("latitude")
        lat.points = lat.points + 100

        # Cannot ignore a axis describing coordinate
        pytest.raises(ValueError, iris.analysis.maths.subtract, a, e)

    def test_minus_scalar(self, request):
        a = self.cube

        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        b = a - 200
        _shared_utils.assert_CML(request, b, ("analysis", "subtract_scalar.cml"))
        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_minus_array(self, request):
        a = self.cube
        data_array = self.cube.copy().data

        # check that the file has not changed (avoids false positives by failing early)
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        # subtract an array of exactly the same shape as the original
        b = a - data_array
        _shared_utils.assert_array_equal(b.data, np.array(0, dtype=np.float32))
        _shared_utils.assert_CML(
            request, b, ("analysis", "subtract_array.cml"), checksum=False
        )

        # subtract an array of the same number of dimensions, but with one of the dimensions having len 1
        b = a - data_array[:, 0:1]
        _shared_utils.assert_array_equal(b.data[:, 0:1], np.array(0, dtype=np.float32))
        _shared_utils.assert_array_equal(b.data[:, 1:2], b.data[:, 1:2])

        # subtract an array of 1 dimension fewer than the cube
        b = a - data_array[0, :]
        _shared_utils.assert_array_equal(b.data[0, :], np.array(0, dtype=np.float32))
        _shared_utils.assert_array_equal(b.data[:, 1:2], b.data[:, 1:2])

        # subtract an array of 1 dimension more than the cube
        d_array = data_array.reshape(data_array.shape[0], data_array.shape[1], 1)
        pytest.raises(ValueError, iris.analysis.maths.subtract, a, d_array)

        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_minus_coord(self, request):
        a = self.cube

        xdim = a.ndim - 1
        ydim = a.ndim - 2
        c_x = DimCoord(
            points=np.arange(a.shape[xdim]),
            long_name="x_coord",
            units=self.cube.units,
        )
        c_y = AuxCoord(
            points=np.arange(a.shape[ydim]),
            long_name="y_coord",
            units=self.cube.units,
        )

        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        b = iris.analysis.maths.subtract(a, c_x, dim=1)
        _shared_utils.assert_CML(request, b, ("analysis", "subtract_coord_x.cml"))
        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        b = iris.analysis.maths.subtract(a, c_y, dim=0)
        _shared_utils.assert_CML(request, b, ("analysis", "subtract_coord_y.cml"))
        # Check that the subtraction has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_addition_scalar(self, request):
        a = self.cube

        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        b = a + 200
        _shared_utils.assert_CML(request, b, ("analysis", "addition_scalar.cml"))
        # Check that the addition has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_addition_coord(self, request):
        a = self.cube

        xdim = a.ndim - 1
        ydim = a.ndim - 2
        c_x = DimCoord(
            points=np.arange(a.shape[xdim]),
            long_name="x_coord",
            units=self.cube.units,
        )
        c_y = AuxCoord(
            points=np.arange(a.shape[ydim]),
            long_name="y_coord",
            units=self.cube.units,
        )

        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        b = iris.analysis.maths.add(a, c_x, dim=1)
        _shared_utils.assert_CML(request, b, ("analysis", "addition_coord_x.cml"))
        # Check that the addition has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

        b = iris.analysis.maths.add(a, c_y, dim=0)
        _shared_utils.assert_CML(request, b, ("analysis", "addition_coord_y.cml"))
        # Check that the addition has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_addition(self, request):
        a = self.cube

        c = a + a
        _shared_utils.assert_CML(request, c, ("analysis", "addition.cml"))
        # Check that the addition has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_addition_different_standard_name(self, request):
        a = self.cube.copy()
        b = self.cube.copy()
        b.rename("my cube data")
        c = a + b
        _shared_utils.assert_CML(
            request, c, ("analysis", "addition_different_std_name.cml"), checksum=False
        )

    def test_addition_fail(self):
        a = self.cube

        xdim = a.ndim - 1
        ydim = a.ndim - 2
        c_axis_length_fail = DimCoord(
            points=np.arange(a.shape[ydim]),
            long_name="x_coord",
            units=self.cube.units,
        )
        c_unit_fail = AuxCoord(
            points=np.arange(a.shape[xdim]), long_name="x_coord", units="volts"
        )

        pytest.raises(ValueError, iris.analysis.maths.add, a, c_axis_length_fail)
        pytest.raises(
            NotYetImplementedError,
            iris.analysis.maths.add,
            a,
            c_unit_fail,
        )

    def test_addition_in_place(self, request):
        a = self.cube

        b = iris.analysis.maths.add(a, self.cube, in_place=True)
        assert b is a
        _shared_utils.assert_CML(request, a, ("analysis", "addition_in_place.cml"))

    def test_addition_in_place_coord(self, request):
        a = self.cube

        # scalar is promoted to a coordinate internally
        b = iris.analysis.maths.add(a, 1000, in_place=True)
        assert b is a
        _shared_utils.assert_CML(
            request, a, ("analysis", "addition_in_place_coord.cml")
        )

    def test_addition_different_attributes(self):
        a = self.cube.copy()
        b = self.cube.copy()
        b.attributes["my attribute"] = "foobar"
        c = a + b
        assert c.standard_name is None
        expected = {
            "my attribute": "foobar",
            "source": "Data from Met Office Unified Model",
        }
        assert expected == c.attributes

    def test_apply_ufunc(self, request):
        a = self.cube

        b = iris.analysis.maths.apply_ufunc(
            np.square,
            a,
            new_name="squared temperature",
            new_unit=a.units**2,
            in_place=False,
        )
        _shared_utils.assert_CML(
            request, a, ("analysis", "apply_ufunc_original.cml"), approx_data=True
        )
        _shared_utils.assert_CML(
            request, b, ("analysis", "apply_ufunc.cml"), approx_data=True
        )

        b = iris.analysis.maths.apply_ufunc(
            np.square,
            a,
            new_name="squared temperature",
            new_unit=a.units**2,
            in_place=True,
        )
        _shared_utils.assert_CML(
            request, b, ("analysis", "apply_ufunc.cml"), approx_data=True
        )
        _shared_utils.assert_CML(
            request, a, ("analysis", "apply_ufunc.cml"), approx_data=True
        )

        def vec_mag(u, v):
            return math.sqrt(u**2 + v**2)

        c = a.copy() + 2

        vec_mag_ufunc = np.frompyfunc(vec_mag, 2, 1)
        b = iris.analysis.maths.apply_ufunc(vec_mag_ufunc, a, c)
        _shared_utils.assert_CML(
            request, b, ("analysis", "apply_ufunc_frompyfunc.cml"), approx_data=True
        )

    def test_apply_ufunc_fail(self):
        a = self.cube

        # should fail because 'blah' is a string, not a np.ufunc
        pytest.raises(TypeError, iris.analysis.maths.apply_ufunc, "blah", a)

        # should fail because math.sqrt is not a np.ufunc
        pytest.raises(TypeError, iris.analysis.maths.apply_ufunc, math.sqrt, a)

        # should fail because np.frexp gives 2 arrays as output
        pytest.raises(ValueError, iris.analysis.maths.apply_ufunc, np.frexp, a)

    def test_ifunc(self, request):
        a = self.cube

        my_ifunc = iris.analysis.maths.IFunc(np.square, lambda a: a.units**2)
        b = my_ifunc(a, new_name="squared temperature", in_place=False)

        _shared_utils.assert_CML(
            request, a, ("analysis", "apply_ifunc_original.cml"), approx_data=True
        )
        _shared_utils.assert_CML(
            request, b, ("analysis", "apply_ifunc.cml"), approx_data=True
        )

        b = my_ifunc(a, new_name="squared temperature", in_place=True)

        _shared_utils.assert_CML(
            request, b, ("analysis", "apply_ifunc.cml"), approx_data=True
        )
        _shared_utils.assert_CML(
            request, a, ("analysis", "apply_ifunc.cml"), approx_data=True
        )

        def vec_mag(u, v):
            return math.sqrt(u**2 + v**2)

        c = a.copy() + 2

        vec_mag_ufunc = np.frompyfunc(vec_mag, 2, 1)
        my_ifunc = iris.analysis.maths.IFunc(vec_mag_ufunc, lambda a, b: (a + b).units)

        b = my_ifunc(a, c)
        _shared_utils.assert_CML(
            request, b, ("analysis", "apply_ifunc_frompyfunc.cml"), approx_data=True
        )

    def test_ifunc_init_fail(self):
        # should fail because 'blah' is a string not a python function
        pytest.raises(
            TypeError,
            iris.analysis.maths.IFunc,
            "blah",
            lambda cube: cf_units.Unit("1"),
        )

        # should fail because np.frexp gives 2 arrays as output
        pytest.raises(
            ValueError,
            iris.analysis.maths.IFunc,
            np.frexp,
            lambda cube: cf_units.Unit("1"),
        )

        # should fail because data function has 3 arguments
        pytest.raises(
            ValueError,
            iris.analysis.maths.IFunc,
            lambda a, b, c: a + b + c,
            lambda cube: cf_units.Unit("1"),
        )

    def test_ifunc_call_fail(self):
        a = self.cube

        my_ifunc = iris.analysis.maths.IFunc(np.square, lambda a: a.units**2)

        # should now NOT fail because giving 2 arguments to an ifunc that
        # expects only one will now ignore the surplus argument and raise
        # a logging message instead, and go on to perform the operation.
        try:
            _ = my_ifunc(a, a)
        except ValueError as exc:
            pytest.fail(f"ValueError raised: {exc}")

        my_ifunc = iris.analysis.maths.IFunc(np.multiply, lambda a: cf_units.Unit("1"))

        # should fail because giving 1 arguments to an ifunc that expects
        # 2
        with pytest.raises(ValueError, match="multiply requires two arguments"):
            my_ifunc(a)

        my_ifunc = iris.analysis.maths.IFunc(
            lambda a: (a, a**2.0), lambda cube: cf_units.Unit("1")
        )

        # should fail because data function returns a tuple
        with pytest.raises(ValueError, match="Require data with shape"):
            my_ifunc(a)

        my_ifunc = iris.analysis.maths.IFunc(
            lambda a: math.sqrt(a), lambda cube: cf_units.Unit("1")
        )

        # should fail because data function does not work when its argument
        # is a numpy array
        with pytest.raises(TypeError):
            my_ifunc(a)

    def test_type_error(self):
        with pytest.raises(TypeError):
            iris.analysis.maths.add("not a cube", 123)


@_shared_utils.skip_data
class TestDivideAndMultiply:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.global_pp()
        self.cube.data = self.cube.data - 260

    def test_divide(self, request):
        a = self.cube

        c = a / a

        _shared_utils.assert_array_almost_equal(a.data / a.data, c.data)
        _shared_utils.assert_CML(
            request, c, ("analysis", "division.cml"), checksum=False
        )

        # Check that the division has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_divide_by_scalar(self, request):
        a = self.cube

        c = a / 10

        _shared_utils.assert_array_almost_equal(a.data / 10, c.data)
        _shared_utils.assert_CML(
            request, c, ("analysis", "division_scalar.cml"), checksum=False
        )

        # Check that the division has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_divide_by_coordinate(self, request):
        a = self.cube

        c = a / a.coord("latitude")
        _shared_utils.assert_CML(request, c, ("analysis", "division_by_latitude.cml"))

        # Check that the division has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_divide_by_array(self, request):
        a = self.cube
        data_array = self.cube.copy().data

        # test division by exactly the same shape data
        c = a / data_array
        _shared_utils.assert_array_equal(c.data, np.array(1, dtype=np.float32))
        _shared_utils.assert_CML(
            request, c, ("analysis", "division_by_array.cml"), checksum=False
        )

        # test division by array of fewer dimensions
        c = a / data_array[0, :]
        _shared_utils.assert_array_equal(c.data[0, :], np.array(1, dtype=np.float32))

        # test division by array of more dimensions
        d_array = data_array.reshape(-1, data_array.shape[1], 1, 1)
        pytest.raises(ValueError, iris.analysis.maths.divide, c, d_array)

        # Check that the division has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_divide_by_coordinate_dim2(self, request):
        a = self.cube

        # Prevent divide-by-zero warning
        a.coord("longitude").points = a.coord("longitude").points + 0.5

        c = a / a.coord("longitude")
        _shared_utils.assert_CML(request, c, ("analysis", "division_by_longitude.cml"))

        # Reset to allow comparison with original
        a.coord("longitude").points = a.coord("longitude").points - 0.5

        # Check that the division has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_divide_by_singular_coordinate(self, request):
        a = self.cube

        coord = DimCoord(points=2, long_name="foo", units="1")
        c = iris.analysis.maths.divide(a, coord)
        _shared_utils.assert_CML(
            request, c, ("analysis", "division_by_singular_coord.cml")
        )

        # Check that the division is equivalent to dividing the whole of the data by 2
        _shared_utils.assert_array_equal(c.data, a.data / 2.0)

    def test_divide_by_different_len_coord(self):
        a = self.cube

        coord = DimCoord(
            points=np.arange(10) * 2 + 5,
            standard_name="longitude",
            units="degrees",
        )

        pytest.raises(ValueError, iris.analysis.maths.divide, a, coord)

    def test_divide_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.divide(a, 5, in_place=True)
        assert a is b

    def test_divide_not_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.divide(a, 5, in_place=False)
        assert a is not b

    def test_multiply(self, request):
        a = self.cube

        c = a * a
        _shared_utils.assert_CML(request, c, ("analysis", "multiply.cml"))

        # Check that the multiplication has had no effect on the original
        _shared_utils.assert_CML(request, a, ("analysis", "maths_original.cml"))

    def test_multiplication_different_standard_name(self, request):
        a = self.cube.copy()
        b = self.cube.copy()
        b.rename("my cube data")
        c = a * b
        _shared_utils.assert_CML(
            request, c, ("analysis", "multiply_different_std_name.cml"), checksum=False
        )

    def test_multiplication_different_attributes(self):
        a = self.cube.copy()
        b = self.cube.copy()
        b.attributes["my attribute"] = "foobar"
        c = a * b
        assert c.standard_name is None
        expected = {
            "source": "Data from Met Office Unified Model",
            "my attribute": "foobar",
        }
        assert expected == c.attributes

    def test_multiplication_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.multiply(a, 5, in_place=True)
        assert a is b

    def test_multiplication_not_in_place(self):
        a = self.cube.copy()
        b = iris.analysis.maths.multiply(a, 5, in_place=False)
        assert a is not b


@_shared_utils.skip_data
class TestExponentiate:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.global_pp()
        # Increase dtype from float32 to float64 in order
        # to avoid dtype quantization errors during maths.
        self.cube.data = self.cube.data.astype(np.float64) - 260.0

    def test_exponentiate(self, request):
        a = self.cube
        e = pow(a, 4)
        _shared_utils.assert_CML(
            request, e, ("analysis", "exponentiate.cml"), approx_data=True
        )

    def test_square_root(self, request):
        # Make sure we have something which we can take the root of.
        a = self.cube
        a.data = abs(a.data)
        a.units **= 2

        e = a**0.5

        _shared_utils.assert_array_all_close(e.data, a.data**0.5)
        _shared_utils.assert_CML(request, e, ("analysis", "sqrt.cml"), checksum=False)
        pytest.raises(ValueError, iris.analysis.maths.exponentiate, a, 0.3)

    def test_type_error(self):
        with pytest.raises(TypeError):
            iris.analysis.maths.exponentiate("not a cube", 2)


class TestExponential:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.simple_1d()

    def test_exp(self, request):
        e = iris.analysis.maths.exp(self.cube)
        _shared_utils.assert_CML(request, e, ("analysis", "exp.cml"), approx_data=True)


class TestApplyUfunc:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.simple_2d()

    def test_apply_ufunc(self):
        a = self.cube
        a.units = cf_units.Unit("meters")

        b = iris.analysis.maths.apply_ufunc(
            np.square,
            a,
            new_name="more_thingness",
            new_unit=a.units**2,
            in_place=False,
        )

        answer = a.data**2

        _shared_utils.assert_array_equal(b.data, answer)
        assert b.name() == "more_thingness"
        assert b.units == cf_units.Unit("m^2")

        def vec_mag(u, v):
            return math.sqrt(u**2 + v**2)

        c = a.copy() + 2

        vec_mag_ufunc = np.frompyfunc(vec_mag, 2, 1)
        b = iris.analysis.maths.apply_ufunc(vec_mag_ufunc, a, c)

        answer = a.data**2 + c.data**2
        b2 = b**2

        _shared_utils.assert_array_almost_equal(b2.data, answer)


class TestIFunc:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.simple_2d()

    def test_ifunc(self):
        a = self.cube
        a.units = cf_units.Unit("meters")

        my_ifunc = iris.analysis.maths.IFunc(np.square, lambda x: x.units**2)
        b = my_ifunc(a, new_name="more_thingness", in_place=False)

        answer = a.data**2

        _shared_utils.assert_array_equal(b.data, answer)
        assert b.name() == "more_thingness"
        assert b.units == cf_units.Unit("m^2")

        def vec_mag(u, v):
            return math.sqrt(u**2 + v**2)

        c = a.copy() + 2

        vec_mag_ufunc = np.frompyfunc(vec_mag, 2, 1)
        my_ifunc = iris.analysis.maths.IFunc(vec_mag_ufunc, lambda x, y: (x + y).units)
        b = my_ifunc(a, c)

        answer = (a.data**2 + c.data**2) ** 0.5

        _shared_utils.assert_array_almost_equal(b.data, answer)

        def vec_mag_data_func(u_data, v_data):
            return np.sqrt(u_data**2 + v_data**2)

        vec_mag_ifunc = iris.analysis.maths.IFunc(
            vec_mag_data_func, lambda a, b: (a + b).units
        )
        b2 = vec_mag_ifunc(a, c)

        _shared_utils.assert_array_almost_equal(b.data, b2.data)

        cs_ifunc = iris.analysis.maths.IFunc(np.cumsum, lambda a: a.units)

        b = cs_ifunc(a, axis=1)
        answer = a.data.copy()
        answer = np.cumsum(answer, axis=1)

        _shared_utils.assert_array_almost_equal(b.data, answer)


@_shared_utils.skip_data
class TestLog:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.global_pp()

    def test_log(self, request):
        e = iris.analysis.maths.log(self.cube)
        _shared_utils.assert_CML(request, e, ("analysis", "log.cml"), approx_data=True)

    def test_log2(self, request):
        e = iris.analysis.maths.log2(self.cube)
        _shared_utils.assert_CML(request, e, ("analysis", "log2.cml"), approx_data=True)

    def test_log10(self, request):
        e = iris.analysis.maths.log10(self.cube)
        _shared_utils.assert_CML(
            request, e, ("analysis", "log10.cml"), approx_data=True, rtol=1e-6
        )


class TestMathOperations:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data_1f = np.array([[9, 9, 9], [8, 8, 8]], dtype=np.float64)
        self.data_2f = np.array([[3, 3, 3], [2, 2, 2]], dtype=np.float64)
        self.data_1i = np.array([[9, 9, 9], [8, 8, 8]], dtype=np.int64)
        self.data_2i = np.array([[3, 3, 3], [2, 2, 2]], dtype=np.int64)
        self.data_1u = np.array([[9, 9, 9], [8, 8, 8]], dtype=np.uint64)
        self.data_2u = np.array([[3, 3, 3], [2, 2, 2]], dtype=np.uint64)

        self.cube_1f = Cube(self.data_1f.copy())
        self.cube_2f = Cube(self.data_2f.copy())
        self.cube_1i = Cube(self.data_1i.copy())
        self.cube_2i = Cube(self.data_2i.copy())
        self.cube_1u = Cube(self.data_1u.copy())
        self.cube_2u = Cube(self.data_2u.copy())

        self.ops = (operator.add, operator.sub, operator.mul, operator.truediv)
        self.iops = (
            operator.iadd,
            operator.isub,
            operator.imul,
            operator.itruediv,
        )

    def common_neg(self, cube, data):
        result1 = -cube
        result2 = -data
        assert isinstance(result1, Cube)
        assert result1 is not cube
        _shared_utils.assert_array_almost_equal(result1.data, result2)

    def test_neg_f(self):
        self.common_neg(self.cube_1f, self.data_1f)

    def test_neg_i(self):
        self.common_neg(self.cube_1i, self.data_1i)

    def test_neg_u(self):
        self.common_neg(self.cube_1u, self.data_1u)

    def test_operator(self):
        for test_op in self.ops:
            result1 = test_op(self.cube_1f, self.cube_2f)
            result2 = test_op(self.data_1f, self.data_2f)
            _shared_utils.assert_array_almost_equal(result1.data, result2)
            result3 = test_op(self.cube_1i, self.cube_2i)
            result4 = test_op(self.data_1i, self.data_2i)
            _shared_utils.assert_array_almost_equal(result3.data, result4)
            result3 = test_op(self.cube_1u, self.cube_2u)
            result4 = test_op(self.data_1u, self.data_2u)
            _shared_utils.assert_array_almost_equal(result3.data, result4)

    def test_operator__mixed_floats_ints(self):
        for test_op in self.ops:
            result1 = test_op(self.cube_1f, self.cube_2i)
            result2 = test_op(self.data_1f, self.data_2i)
            _shared_utils.assert_array_almost_equal(result1.data, result2)
            result3 = test_op(self.cube_1i, self.cube_2f)
            result4 = test_op(self.data_1i, self.data_2f)
            _shared_utils.assert_array_almost_equal(result3.data, result4)

    def test_operator__inplace(self):
        for test_op in self.iops:
            test_op(self.cube_1f, self.cube_2f)
            test_op(self.data_1f, self.data_2f)
            _shared_utils.assert_array_almost_equal(self.cube_1f.data, self.data_1f)
            if test_op != operator.itruediv:
                # You can't safely cast floats from true-dividing back to ints
                # to satisfy the in-place requirement.
                test_op(self.cube_1i, self.cube_2i)
                expected = test_op(self.data_1i, self.data_2i)
                _shared_utils.assert_array_almost_equal(self.cube_1i.data, expected)
                test_op(self.cube_1u, self.cube_2u)
                expected = test_op(self.data_1u, self.data_2u)
                _shared_utils.assert_array_almost_equal(self.cube_1u.data, expected)

    def test_operator__inplace__mixed_floats_ints_uints(self):
        # Can only test float-type numerators as int-type numerators
        # would require casting ints to floats to satisfy the in-place
        # requirement, which cannot be done 'safely'.
        for test_op in self.iops:
            result1 = test_op(self.cube_1f, self.cube_2i)
            result2 = test_op(self.data_1f, self.data_2i)
            _shared_utils.assert_array_almost_equal(result1.data, result2)
            result1 = test_op(self.cube_1f, self.cube_2u)
            result2 = test_op(self.data_1f, self.data_2u)
            _shared_utils.assert_array_almost_equal(result1.data, result2)

    def test_operator__inplace__mixed_ints_uints_floats(self):
        # Math operations on int-type numerators float-type denominators
        # would require casting floats to ints to satisfy the in-place
        # requirement.
        for test_op in self.iops:
            test_emsg = "Cannot perform inplace"
            with pytest.raises(ArithmeticError, match=test_emsg):
                test_op(self.cube_1i, self.cube_2f)
            with pytest.raises(ArithmeticError, match=test_emsg):
                test_op(self.cube_1u, self.cube_2f)

    def test_operator__inplace__scalar_int(self):
        # Math operations on int-type numerators float-type denominators
        # would require casting floats to ints to satisfy the in-place
        # requirement.
        scalar = 2
        for test_op in self.iops:
            if test_op != operator.itruediv:
                # You can't safely cast floats from true-dividing back to ints
                # to satisfy the in-place requirement.
                test_op(self.cube_1i, scalar)
                test_op(self.data_1i, scalar)
                _shared_utils.assert_array_almost_equal(self.cube_1i.data, self.data_1i)
                test_op(self.cube_1u, scalar)
                test_op(self.data_1u, scalar)
                _shared_utils.assert_array_almost_equal(self.cube_1u.data, self.data_1u)

    def test_operator__inplace__scalar_float(self):
        # Math operations on int-type numerators float-type denominators
        # would require casting floats to ints to satisfy the in-place
        # requirement.
        scalar = 2.5
        for test_op in self.iops:
            test_emsg = "Cannot perform inplace"
            with pytest.raises(ArithmeticError, match=test_emsg):
                test_op(self.cube_1i, scalar)
            with pytest.raises(ArithmeticError, match=test_emsg):
                test_op(self.cube_1u, scalar)

    def test_operator__scalar_float(self):
        scalar = 2.5
        for test_op in self.ops:
            result1 = test_op(self.cube_1f, scalar)
            result2 = test_op(self.data_1f, scalar)
            _shared_utils.assert_array_almost_equal(result1.data, result2)
            result3 = test_op(self.cube_1i, scalar)
            result4 = test_op(self.data_1i, scalar)
            _shared_utils.assert_array_almost_equal(result3.data, result4)

    def test_operator_scalar__int(self):
        scalar = 2
        for test_op in self.ops:
            result1 = test_op(self.cube_1f, scalar)
            result2 = test_op(self.data_1f, scalar)
            _shared_utils.assert_array_almost_equal(result1.data, result2)
            result3 = test_op(self.cube_1i, scalar)
            result4 = test_op(self.data_1i, scalar)
            _shared_utils.assert_array_almost_equal(result3.data, result4)

    def test_operator_array(self):
        for test_op in self.ops:
            result1 = test_op(self.cube_1f, self.data_2f)
            result2 = test_op(self.data_1f, self.data_2f)
            _shared_utils.assert_array_almost_equal(result1.data, result2)
            result3 = test_op(self.cube_1i, self.data_2i)
            result4 = test_op(self.data_1i, self.data_2i)
            _shared_utils.assert_array_almost_equal(result3.data, result4)

    def test_cube_itruediv__int(self):
        with pytest.raises(ArithmeticError, match="Cannot perform inplace division"):
            operator.itruediv(self.cube_1i, self.cube_2i)

    def test_cube_itruediv__uint(self):
        with pytest.raises(ArithmeticError, match="Cannot perform inplace division"):
            operator.itruediv(self.cube_1u, self.cube_2u)

    def test_int_cube_itruediv__scalar(self):
        with pytest.raises(ArithmeticError, match="Cannot perform inplace division"):
            operator.itruediv(self.cube_1i, 5)

    def test_uint_cube_itruediv__scalar(self):
        with pytest.raises(ArithmeticError, match="Cannot perform inplace division"):
            operator.itruediv(self.cube_1u, 5)


class TestMaskedArrays:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.data = ma.MaskedArray(
            [[9, 9, 9], [8, 8, 8]],
            mask=[[0, 1, 0], [0, 0, 1]],
            dtype=np.float64,
        )
        self.cube = Cube(self.data)

    def test_incompatible_dimensions(self):
        data3 = ma.MaskedArray(
            [[3, 3, 3, 4], [2, 2, 2, 2]], mask=[[0, 1, 0, 0], [0, 1, 1, 1]]
        )
        with pytest.raises(
            ValueError, match="operands could not be broadcast together"
        ):
            # Incompatible dimensions.
            self.cube + data3

    def test_increase_cube_dimensionality(self):
        cube_x = Cube(ma.MaskedArray([[9]], mask=[[0]]))

        with pytest.raises(ValueError, match="Require data with shape"):
            # This would increase the dimensionality of the cube
            # due to auto-broadcasting.
            cube_x + ma.MaskedArray([[3, 3, 3, 3]], mask=[[0, 1, 0, 1]])


class TestCoordMathOperations:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.value_cube = 100
        self.value = 10
        self.cube = Cube([self.value_cube])
        self.dim = DimCoord([self.value])
        self.aux = AuxCoord([self.value])
        self.i = int(self.value)
        self.f = float(self.value)
        self.np_i = np.int32(self.value)
        self.np_f = np.float32(self.value)
        self.numbers = (self.i, self.f, self.np_i, self.np_f)
        self.ops = (operator.add, operator.sub, operator.mul, operator.truediv)
        self.symbols = ("+", "-", "*", "/")

    def test_coord_op_coord__fail(self):
        for op, symbol in zip(self.ops, self.symbols):
            emsg = rf"AuxCoord \{symbol} DimCoord"
            with pytest.raises(NotYetImplementedError, match=emsg):
                _ = op(self.aux, self.dim)

    def test_coord_op_number(self):
        for op in self.ops:
            for number in self.numbers:
                actual = op(self.aux, number)
                expected = op(self.value, number)
                assert isinstance(actual, AuxCoord)
                assert expected == actual.points

    def test_number_op_coord(self):
        for op in self.ops:
            for number in self.numbers:
                actual = op(number, self.dim)
                expected = op(number, self.value)
                assert isinstance(actual, DimCoord)
                assert expected == actual.points

    def test_coord_op_cube(self):
        for op in self.ops:
            actual = op(self.aux, self.cube)
            expected = op(self.value, self.value_cube)
            assert isinstance(actual, Cube)
            assert expected == actual.data

    def test_cube_op_coord(self):
        for op in self.ops:
            actual = op(self.cube, self.dim)
            expected = op(self.value_cube, self.value)
            assert isinstance(actual, Cube)
            assert actual.data == expected
