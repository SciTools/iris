# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the cube concatenate mechanism."""

import dask.array as da
import numpy as np
import numpy.ma as ma
import pytest

from iris.aux_factory import HybridHeightFactory
from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
import iris.cube
from iris.tests import _shared_utils
import iris.tests.stock as stock
from iris.warnings import IrisUserWarning


def _make_cube(
    x,
    y,
    data,
    aux=None,
    cell_measure=None,
    ancil=None,
    derived=None,
    offset=0,
    scalar=None,
):
    """A convenience test function that creates a custom 2D cube.

    Parameters
    ----------
    x :
        A (start, stop, step) tuple for specifying the x-axis dimensional
        coordinate points. Bounds are automatically guessed.
    y :
        A (start, stop, step) tuple for specifying the y-axis dimensional
        coordinate points. Bounds are automatically guessed.
    data :
        The data payload for the cube.
    aux : optional, default=None
        A CSV string specifying which points only auxiliary
        coordinates to create. Accepts either of 'x', 'y', 'xy'.
    cell_measure : optional, default=None
        A CSV string specifying which points only cell measures
        coordinates to create. Accepts either of 'x', 'y', 'xy'.
    ancil : optional, default=None
        A CSV string specifying which points only ancillary variables
        coordinates to create. Accepts either of 'x', 'y', 'xy'.
    derived : optional, default=None
        A CSV string specifying which points only derived coordinates
        coordinates to create. Accepts either of 'x', 'y', 'xy'.
    offset : int, optional, default=0
        Offset value to be added to the 'xy' auxiliary coordinate
        points.
    scalar : optional, default=None
        Create a 'height' scalar coordinate with the given value.

    Returns
    -------
    The newly created 2D :class:`iris.cube.Cube`.

    """
    x_range = np.arange(*x, dtype=np.float32)
    y_range = np.arange(*y, dtype=np.float32)
    x_size = len(x_range)
    y_size = len(y_range)

    cube_data = np.empty((y_size, x_size), dtype=np.float32)
    cube_data[:] = data
    cube = iris.cube.Cube(cube_data)
    coord = DimCoord(y_range, long_name="y", units="1")
    coord.guess_bounds()
    cube.add_dim_coord(coord, 0)
    coord = DimCoord(x_range, long_name="x", units="1")
    coord.guess_bounds()
    cube.add_dim_coord(coord, 1)

    if aux is not None:
        aux = aux.split(",")
        if "y" in aux:
            coord = AuxCoord(y_range * 10, long_name="y-aux", units="1")
            cube.add_aux_coord(coord, (0,))
        if "x" in aux:
            coord = AuxCoord(x_range * 10, long_name="x-aux", units="1")
            cube.add_aux_coord(coord, (1,))
        if "xy" in aux:
            payload = np.arange(y_size * x_size, dtype=np.float32).reshape(
                y_size, x_size
            )
            coord = AuxCoord(payload * 100 + offset, long_name="xy-aux", units="1")
            cube.add_aux_coord(coord, (0, 1))

    if cell_measure is not None:
        cell_measure = cell_measure.split(",")
        if "y" in cell_measure:
            cm = CellMeasure(y_range * 10, long_name="y-aux", units="1")
            cube.add_cell_measure(cm, (0,))
        if "x" in cell_measure:
            cm = CellMeasure(x_range * 10, long_name="x-aux", units="1")
            cube.add_cell_measure(cm, (1,))
        if "xy" in cell_measure:
            payload = x_range + y_range[:, np.newaxis]
            cm = CellMeasure(payload * 100 + offset, long_name="xy-aux", units="1")
            cube.add_cell_measure(cm, (0, 1))

    if ancil is not None:
        ancil = ancil.split(",")
        if "y" in ancil:
            av = AncillaryVariable(y_range * 10, long_name="y-aux", units="1")
            cube.add_ancillary_variable(av, (0,))
        if "x" in ancil:
            av = AncillaryVariable(x_range * 10, long_name="x-aux", units="1")
            cube.add_ancillary_variable(av, (1,))
        if "xy" in ancil:
            payload = x_range + y_range[:, np.newaxis]
            av = AncillaryVariable(
                payload * 100 + offset, long_name="xy-aux", units="1"
            )
            cube.add_ancillary_variable(av, (0, 1))

    if derived is not None:
        derived = derived.split(",")
        delta = AuxCoord(0.0, var_name="delta", units="m")
        sigma = AuxCoord(1.0, var_name="sigma", units="1")
        cube.add_aux_coord(delta, ())
        cube.add_aux_coord(sigma, ())
        if "y" in derived:
            orog = AuxCoord(y_range * 10, long_name="orog", units="m")
            cube.add_aux_coord(orog, 0)
        elif "x" in derived:
            orog = AuxCoord(x_range * 10, long_name="orog", units="m")
            cube.add_aux_coord(orog, 1)
        elif "xy" in derived:
            payload = np.arange(y_size * x_size, dtype=np.float32).reshape(
                y_size, x_size
            )
            orog = AuxCoord(payload * 100 + offset, long_name="orog", units="m")
            cube.add_aux_coord(orog, (0, 1))
        else:
            raise NotImplementedError()
        cube.add_aux_factory(HybridHeightFactory(delta, sigma, orog))

    if scalar is not None:
        data = np.array([scalar], dtype=np.float32)
        coord = AuxCoord(data, long_name="height", units="m")
        cube.add_aux_coord(coord, ())

    return cube


def _make_cube_3d(x, y, z, data, aux=None, offset=0):
    """A convenience test function that creates a custom 3D cube.

    Parameters
    ----------
    x :
        A (start, stop, step) tuple for specifying the x-axis dimensional
        coordinate points. Bounds are automatically guessed.
    y :
        A (start, stop, step) tuple for specifying the y-axis dimensional
        coordinate points. Bounds are automatically guessed.
    z :
        A (start, stop, step) tuple for specifying the z-axis dimensional
        coordinate points. Bounds are automatically guessed.
    data :
        The data payload for the cube.
    aux : optional, default=None
        A CSV string specifying which points only auxiliary
        coordinates to create. Accepts either of 'x', 'y', 'z',
        'xy', 'xz', 'yz', 'xyz'.
    offset : int, optional, default=0
        Offset value to be added to non-1D auxiliary coordinate
        points.

    Returns
    -------
    The newly created 3D :class:`iris.cube.Cube`.

    """
    x_range = np.arange(*x, dtype=np.float32)
    y_range = np.arange(*y, dtype=np.float32)
    z_range = np.arange(*z, dtype=np.float32)
    x_size, y_size, z_size = len(x_range), len(y_range), len(z_range)

    cube_data = np.empty((x_size, y_size, z_size), dtype=np.float32)
    cube_data[:] = data
    cube = iris.cube.Cube(cube_data)
    coord = DimCoord(z_range, long_name="z", units="1")
    coord.guess_bounds()
    cube.add_dim_coord(coord, 0)
    coord = DimCoord(y_range, long_name="y", units="1")
    coord.guess_bounds()
    cube.add_dim_coord(coord, 1)
    coord = DimCoord(x_range, long_name="x", units="1")
    coord.guess_bounds()
    cube.add_dim_coord(coord, 2)

    if aux is not None:
        aux = aux.split(",")
        if "z" in aux:
            coord = AuxCoord(z_range * 10, long_name="z-aux", units="1")
            cube.add_aux_coord(coord, (0,))
        if "y" in aux:
            coord = AuxCoord(y_range * 10, long_name="y-aux", units="1")
            cube.add_aux_coord(coord, (1,))
        if "x" in aux:
            coord = AuxCoord(x_range * 10, long_name="x-aux", units="1")
            cube.add_aux_coord(coord, (2,))
        if "xy" in aux:
            payload = np.arange(x_size * y_size, dtype=np.float32).reshape(
                y_size, x_size
            )
            coord = AuxCoord(payload + offset, long_name="xy-aux", units="1")
            cube.add_aux_coord(coord, (1, 2))
        if "xz" in aux:
            payload = np.arange(x_size * z_size, dtype=np.float32).reshape(
                z_size, x_size
            )
            coord = AuxCoord(payload * 10 + offset, long_name="xz-aux", units="1")
            cube.add_aux_coord(coord, (0, 2))
        if "yz" in aux:
            payload = np.arange(y_size * z_size, dtype=np.float32).reshape(
                z_size, y_size
            )
            coord = AuxCoord(payload * 100 + offset, long_name="yz-aux", units="1")
            cube.add_aux_coord(coord, (0, 1))
        if "xyz" in aux:
            payload = np.arange(x_size * y_size * z_size, dtype=np.float32).reshape(
                z_size, y_size, x_size
            )
            coord = AuxCoord(payload * 1000 + offset, long_name="xyz-aux", units="1")
            cube.add_aux_coord(coord, (0, 1, 2))

    return cube


def concatenate(cubes, order=None):
    """Explicitly force the contiguous major order of cube data
    alignment to ensure consistent CML crc32 checksums.

    Defaults to contiguous 'C' row-major order.

    """
    if order is None:
        order = "C"

    cubelist = iris.cube.CubeList(cubes)
    result = cubelist.concatenate()

    for cube in result:
        if ma.isMaskedArray(cube.data):
            # cube.data = ma.copy(cube.data, order=order)
            data = np.array(cube.data.data, copy=True, order=order)
            mask = np.array(cube.data.mask, copy=True, order=order)
            fill_value = cube.data.fill_value
            cube.data = ma.array(data, mask=mask, fill_value=fill_value)
        else:
            # cube.data = np.copy(cube.data, order=order)
            cube.data = np.array(cube.data, copy=True, order=order)

    return result


class TestSimple:
    def test_empty(self):
        cubes = iris.cube.CubeList()
        assert concatenate(cubes) == iris.cube.CubeList()

    def test_single(self):
        cubes = [stock.simple_2d()]
        assert concatenate(cubes) == cubes

    def test_multi_equal(self):
        cubes = [stock.simple_2d()] * 2
        assert concatenate(cubes) == cubes


class TestNoConcat:
    def test_one_cube_has_anon_dim(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1))
        cubes.append(_make_cube((2, 4), y, 2))
        cube = _make_cube((4, 6), y, 3)
        cube.remove_coord("x")
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_anonymous.cml")
        )
        assert len(result) == 2
        assert result[0].shape == (2, 4)
        assert result[1].shape == (2, 2)

    def test_points_overlap_increasing(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1))
        cubes.append(_make_cube((1, 3), y, 2))
        with pytest.warns(
            IrisUserWarning,
            match="Found cubes with overlap on concatenate axis",
        ):
            result = concatenate(cubes)
        assert len(result) == 2

    def test_points_overlap_decreasing(self):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (3, 0, -1), 1))
        cubes.append(_make_cube(x, (1, -1, -1), 2))
        with pytest.warns(
            IrisUserWarning,
            match="Found cubes with overlap on concatenate axis",
        ):
            result = concatenate(cubes)
        assert len(result) == 2

    def test_bounds_overlap_increasing(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1))
        cube = _make_cube((2, 4), y, 1)
        cube.coord("x").bounds = np.array([[0.5, 2.5], [2.5, 3.5]], dtype=np.float32)
        cubes.append(cube)
        with pytest.warns(
            IrisUserWarning,
            match="Found cubes with overlap on concatenate axis",
        ):
            result = concatenate(cubes)
        assert len(result) == 2

    def test_bounds_overlap_decreasing(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((3, 1, -1), y, 1))
        cube = _make_cube((1, -1, -1), y, 2)
        cube.coord("x").bounds = np.array([[2.5, 0.5], [0.5, -0.5]], dtype=np.float32)
        cubes.append(cube)
        with pytest.warns(
            IrisUserWarning,
            match="Found cubes with overlap on concatenate axis",
        ):
            result = concatenate(cubes)
        assert len(result) == 2

    def test_scalar_difference(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, scalar=10))
        cubes.append(_make_cube((2, 4), y, 2, scalar=20))
        result = concatenate(cubes)
        assert len(result) == 2

    def test_uncommon(self):
        cubes = []
        cubes.append(_make_cube((0, 2), (0, 2), 1))
        cubes.append(_make_cube((2, 4), (2, 4), 2))
        result = concatenate(cubes)
        assert len(result) == 2

    def test_order_difference(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1))
        cubes.append(_make_cube((6, 1, -1), y, 2))
        result = concatenate(cubes)
        assert len(result) == 2

    def test_cell_measure_missing(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, cell_measure="x"))
        cubes.append(_make_cube((2, 4), y, 2))
        result = concatenate(cubes)
        assert len(result) == 2

    def test_ancil_missing(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, ancil="x"))
        cubes.append(_make_cube((2, 4), y, 2))
        result = concatenate(cubes)
        assert len(result) == 2

    def test_derived_coord_missing(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, derived="x"))
        cubes.append(_make_cube((2, 4), y, 2))
        result = concatenate(cubes)
        assert len(result) == 2


class Test2D:
    def test_masked_and_unmasked(self):
        cubes = []
        y = (0, 2)
        cube = _make_cube((2, 4), y, 2)
        cube.data = ma.asarray(cube.data)
        cubes.append(cube)
        cubes.append(_make_cube((0, 2), y, 1))
        result = concatenate(cubes)
        assert len(result) == 1

    def test_unmasked_and_masked(self):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1))
        cube = _make_cube((2, 4), y, 2)
        cube.data = ma.asarray(cube.data)
        cubes.append(cube)
        result = concatenate(cubes)
        assert len(result) == 1

    def test_masked_fill_value(self):
        cubes = []
        y = (0, 2)
        cube = _make_cube((0, 2), y, 1)
        cube.data = ma.asarray(cube.data)
        cube.data.fill_value = 10
        cubes.append(cube)
        cube = _make_cube((2, 4), y, 1)
        cube.data = ma.asarray(cube.data)
        cube.data.fill_value = 20
        cubes.append(cube)
        result = concatenate(cubes)
        assert len(result) == 1

    def test_concat_masked_2x2d(self, request):
        cubes = []
        y = (0, 2)
        cube = _make_cube((0, 2), y, 1)
        cube.data = ma.asarray(cube.data)
        cube.data[(0, 1), (0, 1)] = ma.masked
        cubes.append(cube)
        cube = _make_cube((2, 4), y, 2)
        cube.data = ma.asarray(cube.data)
        cube.data[(0, 1), (1, 0)] = ma.masked
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_masked_2x2d.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 4)
        mask = np.array(
            [[True, False, False, True], [False, True, True, False]],
            dtype=np.bool_,
        )
        _shared_utils.assert_array_equal(result[0].data.mask, mask)

    def test_concat_masked_2y2d(self, request):
        cubes = []
        x = (0, 2)
        cube = _make_cube(x, (0, 2), 1)
        cube.data = np.ma.asarray(cube.data)
        cube.data[(0, 1), (0, 1)] = ma.masked
        cubes.append(cube)
        cube = _make_cube(x, (2, 4), 2)
        cube.data = ma.asarray(cube.data)
        cube.data[(0, 1), (1, 0)] = ma.masked
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_masked_2y2d.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 2)
        mask = np.array(
            [[True, False], [False, True], [False, True], [True, False]],
            dtype=np.bool_,
        )
        _shared_utils.assert_array_equal(result[0].data.mask, mask)

    def test_concat_masked_2y2d_with_concrete_and_lazy(self, request):
        cubes = []
        x = (0, 2)
        cube = _make_cube(x, (0, 2), 1)
        cube.data = np.ma.asarray(cube.data)
        cube.data[(0, 1), (0, 1)] = ma.masked
        cubes.append(cube)
        cube = _make_cube(x, (2, 4), 2)
        cube.data = ma.asarray(cube.data)
        cube.data[(0, 1), (1, 0)] = ma.masked
        cube.data = cube.lazy_data()
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_masked_2y2d.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 2)
        mask = np.array(
            [[True, False], [False, True], [False, True], [True, False]],
            dtype=np.bool_,
        )
        _shared_utils.assert_array_equal(result[0].data.mask, mask)

    def test_concat_masked_2y2d_with_lazy_and_concrete(self, request):
        cubes = []
        x = (0, 2)
        cube = _make_cube(x, (0, 2), 1)
        cube.data = np.ma.asarray(cube.data)
        cube.data[(0, 1), (0, 1)] = ma.masked
        cube.data = cube.lazy_data()
        cubes.append(cube)
        cube = _make_cube(x, (2, 4), 2)
        cube.data = ma.asarray(cube.data)
        cube.data[(0, 1), (1, 0)] = ma.masked
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_masked_2y2d.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 2)
        mask = np.array(
            [[True, False], [False, True], [False, True], [True, False]],
            dtype=np.bool_,
        )
        _shared_utils.assert_array_equal(result[0].data.mask, mask)

    def test_concat_2x2d(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1))
        cubes.append(_make_cube((4, 6), y, 2))
        result = concatenate(cubes)
        _shared_utils.assert_CML(request, result, ("concatenate", "concat_2x2d.cml"))
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1))
        cubes.append(_make_cube(x, (4, 6), 2))
        result = concatenate(cubes)
        _shared_utils.assert_CML(request, result, ("concatenate", "concat_2y2d.cml"))
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_x(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="x"))
        cubes.append(_make_cube((4, 6), y, 2, aux="x"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_x.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_x(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="x"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="x"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_x.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_y(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="y"))
        cubes.append(_make_cube((4, 6), y, 2, aux="y"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_y.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_y(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="y"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="y"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_y.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_x_y(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="x,y"))
        cubes.append(_make_cube((4, 6), y, 2, aux="x,y"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_x_y.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_x_y(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="x,y"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="x,y"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_x_y.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_xy(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="xy"))
        cubes.append(_make_cube((4, 6), y, 2, aux="xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_xy(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="xy"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_x_xy(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="x,xy"))
        cubes.append(_make_cube((4, 6), y, 2, aux="x,xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_x_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_x_xy(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="x,xy"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="x,xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_x_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_y_xy(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="y,xy"))
        cubes.append(_make_cube((4, 6), y, 2, aux="y,xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_y_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_y_xy(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="y,xy"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="y,xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_y_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_x_y_xy(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 4), y, 1, aux="x,y,xy"))
        cubes.append(_make_cube((4, 6), y, 2, aux="x,y,xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_x_y_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2y2d_aux_x_y_xy(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, aux="x,y,xy"))
        cubes.append(_make_cube(x, (4, 6), 2, aux="x,y,xy"))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2y2d_aux_x_y_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 2)

    def test_concat_2x2d_aux_x_bounds(self, request):
        cubes = []
        y = (0, 2)
        cube = _make_cube((0, 4), y, 1, aux="x")
        cube.coord("x-aux").guess_bounds()
        cubes.append(cube)
        cube = _make_cube((4, 6), y, 2, aux="x")
        cube.coord("x-aux").guess_bounds()
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_x_bounds.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 6)

    def test_concat_2x2d_aux_xy_bounds(self, request):
        cubes = []
        y = (0, 2)
        cube = _make_cube((0, 2), y, 1, aux="xy", offset=1)
        coord = cube.coord("xy-aux")
        coord.bounds = np.array(
            [
                1,
                2,
                3,
                4,
                101,
                102,
                103,
                104,
                201,
                202,
                203,
                204,
                301,
                302,
                303,
                304,
            ]
        ).reshape(2, 2, 4)
        cubes.append(cube)
        cube = _make_cube((2, 4), y, 2, aux="xy", offset=2)
        coord = cube.coord("xy-aux")
        coord.bounds = np.array(
            [
                2,
                3,
                4,
                5,
                102,
                103,
                104,
                105,
                202,
                203,
                204,
                205,
                302,
                303,
                304,
                305,
            ]
        ).reshape(2, 2, 4)
        cubes.append(cube)
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_2x2d_aux_xy_bounds.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 4)

    def test_concat_2y2d_cell_measure_x_y_xy(self):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, cell_measure="x,y,xy"))
        cubes.append(_make_cube(x, (4, 6), 1, cell_measure="x,y,xy"))
        result = concatenate(cubes)
        com = _make_cube(x, (0, 6), 1, cell_measure="x,y,xy")
        assert len(result) == 1
        assert result[0].shape == (6, 2)
        assert result[0] == com

    def test_concat_2y2d_ancil_x_y_xy(self):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, ancil="x,y,xy"))
        cubes.append(_make_cube(x, (4, 6), 1, ancil="x,y,xy"))
        result = concatenate(cubes)
        com = _make_cube(x, (0, 6), 1, ancil="x,y,xy")
        assert len(result) == 1
        assert result[0].shape == (6, 2)
        assert result[0] == com

    def test_concat_2y2d_derived_x_y_xy(self):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 4), 1, derived="x,y,xy"))
        cubes.append(_make_cube(x, (4, 6), 1, derived="x,y,xy"))
        result = concatenate(cubes)
        com = _make_cube(x, (0, 6), 1, derived="x,y,xy")
        assert len(result) == 1
        assert result[0].shape == (6, 2)
        assert result[0] == com

    def test_concat_lazy_aux_coords(self):
        cubes = []
        y = (0, 2)
        cube = _make_cube((2, 4), y, 2, aux="xy")
        cubes.append(cube)
        cubes.append(_make_cube((0, 2), y, 1, aux="xy"))
        for cube in cubes:
            cube.data = cube.lazy_data()
            cube.coord("xy-aux").points = cube.coord("xy-aux").lazy_points()
            bounds = da.arange(4 * cube.coord("xy-aux").core_points().size).reshape(
                cube.shape + (4,)
            )
            cube.coord("xy-aux").bounds = bounds
        result = concatenate(cubes)

        assert cubes[0].coord("xy-aux").has_lazy_points()
        assert cubes[0].coord("xy-aux").has_lazy_bounds()

        assert cubes[1].coord("xy-aux").has_lazy_points()
        assert cubes[1].coord("xy-aux").has_lazy_bounds()

        assert result[0].coord("xy-aux").has_lazy_points()
        assert result[0].coord("xy-aux").has_lazy_bounds()


class TestMulti2D:
    def test_concat_4x2d_aux_xy(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, aux="xy", offset=1))
        cubes.append(_make_cube((2, 4), y, 2, aux="xy", offset=2))
        y = (2, 4)
        cubes.append(_make_cube((0, 2), y, 3, aux="xy", offset=3))
        cubes.append(_make_cube((2, 4), y, 4, aux="xy", offset=4))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_4x2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 4)

    def test_concat_4y2d_aux_xy(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 2), 1, aux="xy", offset=1))
        cubes.append(_make_cube(x, (2, 4), 2, aux="xy", offset=2))
        x = (2, 4)
        cubes.append(_make_cube(x, (0, 2), 3, aux="xy", offset=3))
        cubes.append(_make_cube(x, (2, 4), 4, aux="xy", offset=4))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_4y2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 4)

    def test_concat_4mix2d_aux_xy(self, request):
        cubes = []
        cubes.append(_make_cube((0, 2), (0, 2), 1, aux="xy", offset=1))
        cubes.append(_make_cube((2, 4), (2, 4), 2, aux="xy", offset=2))
        cubes.append(_make_cube((2, 4), (0, 2), 3, aux="xy", offset=3))
        cubes.append(_make_cube((0, 2), (2, 4), 4, aux="xy", offset=4))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_4mix2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 4)

    def test_concat_9x2d_aux_xy(self, request):
        cubes = []
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, aux="xy", offset=1))
        cubes.append(_make_cube((2, 4), y, 2, aux="xy", offset=2))
        cubes.append(_make_cube((4, 6), y, 3, aux="xy", offset=3))
        y = (2, 4)
        cubes.append(_make_cube((0, 2), y, 4, aux="xy", offset=4))
        cubes.append(_make_cube((2, 4), y, 5, aux="xy", offset=5))
        cubes.append(_make_cube((4, 6), y, 6, aux="xy", offset=6))
        y = (4, 6)
        cubes.append(_make_cube((0, 2), y, 7, aux="xy", offset=7))
        cubes.append(_make_cube((2, 4), y, 8, aux="xy", offset=8))
        cubes.append(_make_cube((4, 6), y, 9, aux="xy", offset=9))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_9x2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 6)

    def test_concat_9y2d_aux_xy(self, request):
        cubes = []
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 2), 1, aux="xy", offset=1))
        cubes.append(_make_cube(x, (2, 4), 2, aux="xy", offset=2))
        cubes.append(_make_cube(x, (4, 6), 3, aux="xy", offset=3))
        x = (2, 4)
        cubes.append(_make_cube(x, (0, 2), 4, aux="xy", offset=4))
        cubes.append(_make_cube(x, (2, 4), 5, aux="xy", offset=5))
        cubes.append(_make_cube(x, (4, 6), 6, aux="xy", offset=6))
        x = (4, 6)
        cubes.append(_make_cube(x, (0, 2), 7, aux="xy", offset=7))
        cubes.append(_make_cube(x, (2, 4), 8, aux="xy", offset=8))
        cubes.append(_make_cube(x, (4, 6), 9, aux="xy", offset=9))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_9y2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 6)

    def test_concat_9mix2d_aux_xy(self, request):
        cubes = []
        cubes.append(_make_cube((0, 2), (0, 2), 1, aux="xy", offset=1))
        cubes.append(_make_cube((2, 4), (2, 4), 2, aux="xy", offset=2))
        cubes.append(_make_cube((4, 6), (4, 6), 3, aux="xy", offset=3))
        cubes.append(_make_cube((4, 6), (0, 2), 4, aux="xy", offset=4))
        cubes.append(_make_cube((0, 2), (4, 6), 5, aux="xy", offset=5))
        cubes.append(_make_cube((0, 2), (2, 4), 6, aux="xy", offset=6))
        cubes.append(_make_cube((2, 4), (0, 2), 7, aux="xy", offset=7))
        cubes.append(_make_cube((4, 6), (2, 4), 8, aux="xy", offset=8))
        cubes.append(_make_cube((2, 4), (4, 6), 9, aux="xy", offset=9))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_9mix2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (6, 6)


class TestMulti2DScalar:
    def test_concat_scalar_4x2d_aux_xy(self, request):
        cubes = iris.cube.CubeList()
        # Level 1.
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 1, aux="xy", offset=1, scalar=10))
        cubes.append(_make_cube((2, 4), y, 2, aux="xy", offset=2, scalar=10))
        y = (2, 4)
        cubes.append(_make_cube((0, 2), y, 3, aux="xy", offset=3, scalar=10))
        cubes.append(_make_cube((2, 4), y, 4, aux="xy", offset=4, scalar=10))
        # Level 2.
        y = (0, 2)
        cubes.append(_make_cube((0, 2), y, 5, aux="xy", offset=1, scalar=20))
        cubes.append(_make_cube((2, 4), y, 6, aux="xy", offset=2, scalar=20))
        y = (2, 4)
        cubes.append(_make_cube((0, 2), y, 7, aux="xy", offset=3, scalar=20))
        cubes.append(_make_cube((2, 4), y, 8, aux="xy", offset=4, scalar=20))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_scalar_4x2d_aux_xy.cml")
        )
        assert len(result) == 2
        for cube in result:
            assert cube.shape == (4, 4)

        merged = result.merge()
        _shared_utils.assert_CML(
            request, merged, ("concatenate", "concat_merged_scalar_4x2d_aux_xy.cml")
        )
        assert len(merged) == 1
        assert merged[0].shape == (2, 4, 4)

        # Test concatenate and merge are commutative operations.
        merged = cubes.merge()
        _shared_utils.assert_CML(
            request, merged, ("concatenate", "concat_pre_merged_scalar_4x2_aux_xy.cml")
        )
        assert len(merged) == 4

        result = concatenate(merged)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_merged_scalar_4x2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 4, 4)

    def test_concat_scalar_4y2d_aux_xy(self, request):
        cubes = iris.cube.CubeList()
        # Level 1.
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 2), 1, aux="xy", offset=1, scalar=10))
        cubes.append(_make_cube(x, (2, 4), 2, aux="xy", offset=2, scalar=10))
        x = (2, 4)
        cubes.append(_make_cube(x, (0, 2), 3, aux="xy", offset=3, scalar=10))
        cubes.append(_make_cube(x, (2, 4), 4, aux="xy", offset=4, scalar=10))
        # Level 2.
        x = (0, 2)
        cubes.append(_make_cube(x, (0, 2), 5, aux="xy", offset=1, scalar=20))
        cubes.append(_make_cube(x, (2, 4), 6, aux="xy", offset=2, scalar=20))
        x = (2, 4)
        cubes.append(_make_cube(x, (0, 2), 7, aux="xy", offset=3, scalar=20))
        cubes.append(_make_cube(x, (2, 4), 8, aux="xy", offset=4, scalar=20))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_scalar_4y2d_aux_xy.cml")
        )
        assert len(result) == 2
        for cube in result:
            assert cube.shape == (4, 4)

        merged = result.merge()
        _shared_utils.assert_CML(
            request, merged, ("concatenate", "concat_merged_scalar_4y2d_aux_xy.cml")
        )
        assert len(merged) == 1
        assert merged[0].shape == (2, 4, 4)

        # Test concatenate and merge are commutative operations.
        merged = cubes.merge()
        _shared_utils.assert_CML(
            request, merged, ("concatenate", "concat_pre_merged_scalar_4y2d_aux_xy.cml")
        )
        assert len(merged) == 4

        result = concatenate(merged)
        assert len(result) == 1
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_merged_scalar_4y2d_aux_xy.cml")
        )
        assert result[0].shape == (2, 4, 4)

    def test_concat_scalar_4mix2d_aux_xy(self, request):
        cubes = iris.cube.CubeList()
        cubes.append(_make_cube((0, 2), (0, 2), 1, aux="xy", offset=1, scalar=10))
        cubes.append(_make_cube((2, 4), (2, 4), 8, aux="xy", offset=4, scalar=20))
        cubes.append(_make_cube((0, 2), (0, 2), 5, aux="xy", offset=1, scalar=20))
        cubes.append(_make_cube((2, 4), (0, 2), 2, aux="xy", offset=2, scalar=10))
        cubes.append(_make_cube((0, 2), (2, 4), 7, aux="xy", offset=3, scalar=20))
        cubes.append(_make_cube((0, 2), (2, 4), 3, aux="xy", offset=3, scalar=10))
        cubes.append(_make_cube((2, 4), (2, 4), 4, aux="xy", offset=4, scalar=10))
        cubes.append(_make_cube((2, 4), (0, 2), 6, aux="xy", offset=2, scalar=20))
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_scalar_4mix2d_aux_xy.cml")
        )
        assert len(result) == 2
        for cube in result:
            assert cube.shape == (4, 4)

        merged = result.merge()
        _shared_utils.assert_CML(
            request, merged, ("concatenate", "concat_merged_scalar_4mix2d_aux_xy.cml")
        )
        assert len(merged) == 1
        assert merged[0].shape == (2, 4, 4)

        # Test concatenate and merge are commutative operations.
        merged = cubes.merge()
        _shared_utils.assert_CML(
            request,
            merged,
            ("concatenate", "concat_pre_merged_scalar_4mix2d_aux_xy.cml"),
        )
        assert len(merged) == 4

        result = concatenate(merged)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_merged_scalar_4mix2d_aux_xy.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (2, 4, 4)


class Test3D:
    def _make_group(self, xoff=0, yoff=0, zoff=0, doff=0):
        xoff *= 4
        yoff *= 4
        zoff *= 4
        doff *= 8
        cubes = []
        cubes.append(
            _make_cube_3d(
                (0 + xoff, 2 + xoff),
                (0 + yoff, 2 + yoff),
                (0 + zoff, 2 + zoff),
                1 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )
        cubes.append(
            _make_cube_3d(
                (2 + xoff, 4 + xoff),
                (0 + yoff, 2 + yoff),
                (0 + zoff, 2 + zoff),
                2 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )
        cubes.append(
            _make_cube_3d(
                (0 + xoff, 2 + xoff),
                (2 + yoff, 4 + yoff),
                (0 + zoff, 2 + zoff),
                3 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )
        cubes.append(
            _make_cube_3d(
                (2 + xoff, 4 + xoff),
                (2 + yoff, 4 + yoff),
                (0 + zoff, 2 + zoff),
                4 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )

        cubes.append(
            _make_cube_3d(
                (0 + xoff, 2 + xoff),
                (0 + yoff, 2 + yoff),
                (2 + zoff, 4 + zoff),
                5 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )
        cubes.append(
            _make_cube_3d(
                (2 + xoff, 4 + xoff),
                (0 + yoff, 2 + yoff),
                (2 + zoff, 4 + zoff),
                6 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )
        cubes.append(
            _make_cube_3d(
                (0 + xoff, 2 + xoff),
                (2 + yoff, 4 + yoff),
                (2 + zoff, 4 + zoff),
                7 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )
        cubes.append(
            _make_cube_3d(
                (2 + xoff, 4 + xoff),
                (2 + yoff, 4 + yoff),
                (2 + zoff, 4 + zoff),
                8 + doff,
                aux="x,y,z,xy,xz,yz,xyz",
            )
        )

        return cubes

    def test_concat_3d_simple(self, request):
        cubes = self._make_group()
        result = concatenate(cubes)
        _shared_utils.assert_CML(
            request, result, ("concatenate", "concat_3d_simple.cml")
        )
        assert len(result) == 1
        assert result[0].shape == (4, 4, 4)

    def test_concat_3d_mega(self):
        cubes = []
        cubes.extend(self._make_group(xoff=0, doff=0))
        cubes.extend(self._make_group(xoff=1, doff=1))
        cubes.extend(self._make_group(yoff=1, doff=2))
        cubes.extend(self._make_group(xoff=1, yoff=1, doff=3))

        cubes.extend(self._make_group(xoff=0, zoff=1, doff=4))
        cubes.extend(self._make_group(xoff=1, zoff=1, doff=5))
        cubes.extend(self._make_group(yoff=1, zoff=1, doff=6))
        cubes.extend(self._make_group(xoff=1, yoff=1, zoff=1, doff=7))
        result = concatenate(cubes)

        assert len(result) == 1
        assert result[0].shape == (8, 8, 8)
