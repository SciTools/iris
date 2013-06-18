# (C) British Crown Copyright 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Test the experimental transport functionality.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import glob
from itertools import izip
import os
import os.path
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import iris
import iris.experimental.transport as transport
import iris.experimental.transport.top_edge as top_edge


T_x_points = np.array([[2.5, 7.5, 12.5, 17.5],
                       [10.0, 17.5, 27.5, 42.5],
                       [15.0, 22.5, 32.5, 50.0]])

U_x_point_offsets = np.array([[2.50, 2.5, 2.5, 2.5],
                              [3.75, 5.0, 7.5, 5.0],
                              [3.75, 5.0, 9.5, 8.0]])

T_y_points = np.array([[-7.5, 7.5, 22.5, 37.5],
                       [-12.5, 4.0, 26.5, 47.5],
                       [2.5, 14.0, 36.5, 44.0]])

V_y_point_offsets = np.array([[-2.5, -1.75, 2.0, 5.0],
                              [7.5, 5.0, 5.0, -1.75],
                              [7.5, 5.0, 5.0, 5.0]])

V_y_points = T_y_points - V_y_point_offsets
U_x_points = T_x_points + U_x_point_offsets


def make_cube(long_name, units, x_points, y_points,
              depth_points=None, offset=0):
    """Convenience function to generate a cube."""
    assert x_points.shape == y_points.shape
    if depth_points is not None:
        depth_points = np.asarray(depth_points, dtype=np.float32)
        shape = depth_points.shape + x_points.shape
        size = depth_points.size * x_points.size
    else:
        shape = x_points.shape
        size = x_points.size

    data = np.arange(size, dtype=np.int32).reshape(shape) + offset
    cube = iris.cube.Cube(data)
    cube.long_name = long_name
    cube.units = units

    y_coord = iris.coords.AuxCoord(points=y_points, long_name='latitude',
                                   units='degrees')
    x_coord = iris.coords.AuxCoord(points=x_points, long_name='longitude',
                                   units='degrees')
    if depth_points is not None:
        depth_coord = iris.coords.DimCoord(points=depth_points,
                                           long_name='depth',
                                           units='m')
        cube.add_dim_coord(depth_coord, 0)
        cube.add_aux_coord(y_coord, (1, 2))
        cube.add_aux_coord(x_coord, (1, 2))
    else:
        cube.add_aux_coord(y_coord, (0, 1))
        cube.add_aux_coord(x_coord, (0, 1))
    return cube


def make_mask(cells):
    """Convenience function to generate a square T-cell mask."""
    cells = np.asarray(cells)
    square = cells.max() + 1
    shape = (square, square)
    mask = np.ones((shape), dtype=np.bool)
    for cell in cells:
        mask[tuple(cell)] = False
    return mask


class TestTCell(tests.IrisTest):
    def setUp(self):
        fname = tests.get_data_path(('NetCDF', 'ocean', 'tcell_lat60.nc'))
        self.cube = iris.load_cube(fname)

    def test_mask_single(self):
        tcell = top_edge.TCell(self.cube)
        mask = tcell.latitude(60)
        fname = tests.get_result_path(('experimental',
                                       'transport', 'tcell_lat60.npy'))
        self.assertArrayEqual(mask, fname)

    def test_mask_multi(self):
        tcell = top_edge.TCell(self.cube)
        mask = tcell.latitude([59.8, 60])
        fname = tests.get_result_path(('experimental',
                                       'transport', 'tcell_lat_multi.npy'))
        self.assertArrayEqual(mask, fname)

    def test_cache(self):
        dname = tempfile.mkdtemp()
        tcell = top_edge.TCell(self.cube, cache_dir=dname)
        mask = tcell.latitude(60)
        fname = tests.get_result_path(('experimental',
                                       'transport', 'tcell_lat60.npy'))
        self.assertArrayEqual(mask, fname)

        cache = set(tcell._cache)
        self.assertEqual(len(cache), 2)
        files = {f for f in glob.glob(os.path.join(dname, '*.tcell'))}
        self.assertEqual(files, cache)

        mask = tcell.latitude(60)
        self.assertArrayEqual(mask, fname)

        tcell._purge()
        cache = set(tcell._cache)
        self.assertEqual(len(cache), 0)
        files = {f for f in glob.glob(os.path.join(dname, '*.tcell'))}
        self.assertEqual(len(files), 0)
        os.rmdir(dname)


class TestPath(tests.IrisTest):
    def test_top_edge_parabola(self):
        fname_cube = tests.get_data_path(('NetCDF', 'ocean', 'tcell_lat60.nc'))
        cube = iris.load_cube(fname_cube)
        tcell = top_edge.TCell(cube)
        mask = tcell.latitude(60)
        path = np.array(top_edge.top_edge_path(mask))
        fname_path = tests.get_result_path(('experimental',
                                            'transport', 'path_lat60.npy'))
        self.assertArrayEqual(path, fname_path)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()

        top_edge.plot_ll(cube, mask, path)
        self.check_graphic()

    def test_top_edge_line_upper(self):
        cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4)]
        mask = make_mask(cells)
        path = np.array(top_edge.top_edge_path(mask))
        fname = tests.get_result_path(('experimental',
                                       'transport', 'top_edge_line_upper.npy'))
        self.assertArrayEqual(path, fname)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()

    def test_top_edge_line_lower(self):
        cells = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
        mask = make_mask(cells)
        path = np.array(top_edge.top_edge_path(mask))
        fname = tests.get_result_path(('experimental',
                                       'transport', 'top_edge_line_lower.npy'))
        self.assertArrayEqual(path, fname)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()

    def test_top_edge_line_left(self):
        cells = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
        mask = make_mask(cells)
        path = np.array(top_edge.top_edge_path(mask))
        fname = tests.get_result_path(('experimental',
                                       'transport', 'top_edge_line_left.npy'))
        self.assertArrayEqual(path, fname)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()

    def test_top_edge_line_right(self):
        cells = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]
        mask = make_mask(cells)
        path = np.array(top_edge.top_edge_path(mask))
        fname = tests.get_result_path(('experimental',
                                       'transport', 'top_edge_line_right.npy'))
        self.assertArrayEqual(path, fname)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()

    def test_top_edge_hoop_upper(self):
        cells = [(4, 0), (3, 1), (2, 2), (3, 3), (4, 4)]
        mask = make_mask(cells)
        path = np.array(top_edge.top_edge_path(mask))
        fname = tests.get_result_path(('experimental',
                                       'transport', 'top_edge_hoop_upper.npy'))
        self.assertArrayEqual(path, fname)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()

    def test_top_edge_hoop_lower(self):
        cells = [(0, 0), (1, 1), (2, 2), (1, 3), (0, 4)]
        mask = make_mask(cells)
        path = np.array(top_edge.top_edge_path(mask))
        fname = tests.get_result_path(('experimental',
                                       'transport', 'top_edge_hoop_lower.npy'))
        self.assertArrayEqual(path, fname)

        top_edge.plot_ij(mask, path, lw=3)
        self.check_graphic()


class TestPathData(tests.IrisTest):
    def setUp(self):
        depth_points = [10.0, 30.0]
        self.t = make_cube('sea_water_potential_temperature', 'degC',
                           T_x_points, T_y_points,
                           depth_points=depth_points)
        self.u = make_cube('x_wind', 'm s-1',
                           U_x_points, T_y_points,
                           depth_points=depth_points)
        self.v = make_cube('y_wind', 'm s-1',
                           T_x_points, V_y_points,
                           depth_points=depth_points)
        self.dx = make_cube('x step', '1', U_x_points, T_y_points)
        self.dy = make_cube('y step', '1', T_x_points, V_y_points,
                            offset=12)
        self.dzu = make_cube('dz on u grid', 'm',
                             U_x_points, T_y_points,
                             depth_points=depth_points)
        self.dzv = make_cube('dz on v grid', 'm',
                             T_x_points, V_y_points,
                             depth_points=depth_points,
                             offset=24)
        self.data = transport.Data(self.t, self.u, self.v,
                                   self.dx, self.dy, self.dzu, self.dzv)

    def test_simple(self):
        path = [[(0, 1), (1, 1), (1, 2), (0, 2)]]
        path_data = transport.path_data(self.data, path)

        expected_indices = [[0, 0], [0, 1], [0, 1]]
        expected_uv_cubes = [self.u, self.v, self.u]
        expected_dxdy_cubes = [self.dy, self.dx, self.dy]
        expected_dz_cubes = [self.dzu, self.dzv, self.dzu]
        expected_sign = [1, -1, -1]

        zipper = izip(expected_indices, expected_uv_cubes, expected_sign)
        uv_data = [sign * (c[:, j, i].data) for (j, i), c, sign in zipper]

        zipper = izip(expected_indices, expected_dxdy_cubes)
        dxdy_data = [(c[j, i].data) for (j, i), c, in zipper]

        zipper = izip(expected_indices, expected_dz_cubes)
        dz_data = [(c[:, j, i].data) for (j, i), c, in zipper]

        uv = np.vstack(uv_data).transpose()
        np.testing.assert_array_equal(uv, path_data.uv)

        dxdy = np.array(dxdy_data)
        np.testing.assert_array_equal(dxdy, path_data.dxdy)

        dz = np.vstack(dz_data).transpose()
        np.testing.assert_array_equal(dz, path_data.dz)

    def test_multi_path(self):
        path = [[(0, 1), (1, 1), (1, 2), (0, 2)],
                [(0, 1), (1, 1), (1, 2), (0, 2)][::-1]]
        path_data = transport.path_data(self.data, path)

        expected_indices = [[0, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 0]]
        expected_uv_cubes = [self.u, self.v, self.u, self.u, self.v, self.u]
        expected_dxdy_cubes = [self.dy, self.dx, self.dy,
                               self.dy, self.dx, self.dy]
        expected_dz_cubes = [self.dzu, self.dzv, self.dzu,
                             self.dzu, self.dzv, self.dzu]
        expected_sign = [1, -1, -1, 1, 1, -1]

        zipper = izip(expected_indices, expected_uv_cubes, expected_sign)
        uv_data = [sign * (c[:, j, i].data) for (j, i), c, sign in zipper]

        zipper = izip(expected_indices, expected_dxdy_cubes)
        dxdy_data = [(c[j, i].data) for (j, i), c, in zipper]

        zipper = izip(expected_indices, expected_dz_cubes)
        dz_data = [(c[:, j, i].data) for (j, i), c, in zipper]

        uv = np.vstack(uv_data).transpose()
        np.testing.assert_array_equal(uv, path_data.uv)

        dxdy = np.array(dxdy_data)
        np.testing.assert_array_equal(dxdy, path_data.dxdy)

        dz = np.vstack(dz_data).transpose()
        np.testing.assert_array_equal(dz, path_data.dz)

    def test_bad_path(self):
        path = [[(-1, 1), (0, 1), (1, 1), (0, 1)]]
        with self.assertRaises(ValueError):
            transport.path_data(self.data, path)

        path = [[(self.dx.shape[1], self.dx.shape[0]),
                 (self.dx.shape[1] - 1, self.dx.shape[0])]]
        with self.assertRaises(ValueError):
            transport.path_data(self.data, path)

        path = [[(self.dx.shape[1], 0),
                 (self.dx.shape[1] - 1, 0)]]
        with self.assertRaises(ValueError):
            transport.path_data(self.data, path)

        path = [[(self.dx.shape[1], 0)]]
        path_data = transport.path_data(self.data, path)
        empty = np.array([], ndmin=2, dtype=np.float64).reshape(2, 0)
        target = transport.PathData(empty,
                                    np.array([], dtype=np.float64),
                                    empty)
        for actual, expected in zip(path_data, target):
            np.testing.assert_array_equal(actual, expected)


class TestTransport(tests.IrisTest):
    def setUp(self):
        depth_points = [10.0, 30.0]
        self.t = make_cube('sea_water_potential_temperature', 'degC',
                           T_x_points, T_y_points,
                           depth_points=depth_points)
        self.u = make_cube('x_wind', 'm s-1',
                           U_x_points, T_y_points,
                           depth_points=depth_points)
        self.v = make_cube('y_wind', 'm s-1',
                           T_x_points, V_y_points,
                           depth_points=depth_points)
        self.dx = make_cube('x step', '1', U_x_points, T_y_points)
        self.dy = make_cube('y step', '1', T_x_points, V_y_points,
                            offset=12)
        self.dzu = make_cube('dz on u grid', 'm',
                             U_x_points, T_y_points,
                             depth_points=depth_points)
        self.dzv = make_cube('dz on v grid', 'm',
                             T_x_points, V_y_points,
                             depth_points=depth_points,
                             offset=24)
        self.data = transport.Data(self.t, self.u, self.v,
                                   self.dx, self.dy, self.dzu, self.dzv)

    def test_simple_path_transport(self):
        path = [[(0, 1), (1, 1), (1, 2), (0, 2)]]
        actual = transport.path_transport(self.data, path)
        expected = ma.asarray([0.0, -480.0])
        np.testing.assert_array_equal(actual, expected)

    def test_simple_stream_function(self):
        path = [[(0, 1), (1, 1), (1, 2), (0, 2)]]
        actual = transport.stream_function(self.data, path)
        expected = ma.asarray([0.0, -480.0])
        np.testing.assert_array_equal(actual, expected)

    def test_simple_net_transport(self):
        path = [[(0, 1), (1, 1), (1, 2), (0, 2)]]
        actual = transport.net_transport(self.data, path)
        np.testing.assert_array_equal(actual, -480.0)


if __name__ == '__main__':
    tests.main()
