# (C) British Crown Copyright 2014, Met Office
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
Unit tests for
:func:`iris.experimental.regrid.regrid_bilinear_rectilinear_src_and_grid`.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import numpy as np

from iris.coord_systems import GeogCS, OSGB
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.experimental.regrid import \
    regrid_bilinear_rectilinear_src_and_grid as regrid
from iris.tests.stock import lat_lon_cube


class TestInvalidTypes(tests.IrisTest):
    def test_src_as_array(self):
        with self.assertRaises(TypeError):
            regrid(np.zeros((3, 4)), Cube())

    def test_grid_as_array(self):
        with self.assertRaises(TypeError):
            regrid(Cube(), np.zeros((3, 4)))

    def test_src_as_int(self):
        with self.assertRaises(TypeError):
            regrid(42, Cube())

    def test_grid_as_int(self):
        with self.assertRaises(TypeError):
            regrid(Cube(), 42)


class TestMissingCoords(tests.IrisTest):
    def ok_bad(self, coord_names):
        # Deletes the named coords from `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        for name in coord_names:
            bad.remove_coord(name)
        return ok, bad

    def test_src_missing_lat(self):
        ok, bad = self.ok_bad(['latitude'])
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_missing_lat(self):
        ok, bad = self.ok_bad(['latitude'])
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_missing_lon(self):
        ok, bad = self.ok_bad(['longitude'])
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_missing_lon(self):
        ok, bad = self.ok_bad(['longitude'])
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_missing_lat_lon(self):
        ok, bad = self.ok_bad(['latitude', 'longitude'])
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_missing_lat_lon(self):
        ok, bad = self.ok_bad(['latitude', 'longitude'])
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestNotDimCoord(tests.IrisTest):
    def ok_bad(self, coord_name):
        # Demotes the named DimCoord on `bad` to an AuxCoord.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        coord = bad.coord(coord_name)
        dims = bad.coord_dims(coord)
        bad.remove_coord(coord_name)
        aux_coord = AuxCoord.from_coord(coord)
        bad.add_aux_coord(aux_coord, dims)
        return ok, bad

    def test_src_with_aux_lat(self):
        ok, bad = self.ok_bad('latitude')
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_with_aux_lat(self):
        ok, bad = self.ok_bad('latitude')
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_with_aux_lon(self):
        ok, bad = self.ok_bad('longitude')
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_with_aux_lon(self):
        ok, bad = self.ok_bad('longitude')
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestNotDimCoord(tests.IrisTest):
    def ok_bad(self):
        # Make lat/lon share a single dimension on `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        lat = bad.coord('latitude')
        bad = bad[0, :lat.shape[0]]
        bad.remove_coord('latitude')
        bad.add_aux_coord(lat, 0)
        return ok, bad

    def test_src_shares_dim(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_shares_dim(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestBadGeoreference(tests.IrisTest):
    def ok_bad(self, lat_cs, lon_cs):
        # Updates `bad` to use the given coordinate systems.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        bad.coord('latitude').coord_system = lat_cs
        bad.coord('longitude').coord_system = lon_cs
        return ok, bad

    def test_src_no_cs(self):
        ok, bad = self.ok_bad(None, None)
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_no_cs(self):
        ok, bad = self.ok_bad(None, None)
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_one_cs(self):
        ok, bad = self.ok_bad(None, GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_one_cs(self):
        ok, bad = self.ok_bad(None, GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(ok, bad)

    def test_src_inconsistent_cs(self):
        ok, bad = self.ok_bad(GeogCS(6370000), GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_inconsistent_cs(self):
        ok, bad = self.ok_bad(GeogCS(6370000), GeogCS(6371000))
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestBadAngularUnits(tests.IrisTest):
    def ok_bad(self):
        # Changes the longitude coord to radians on `bad`.
        ok = lat_lon_cube()
        bad = lat_lon_cube()
        bad.coord('longitude').units = 'radians'
        return ok, bad

    def test_src_radians(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_radians(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(ok, bad)


class TestBadLinearUnits(tests.IrisTest):
    def ok_bad(self):
        # Defines `bad` with an x coordinate in km.
        ok = lat_lon_cube()
        bad = Cube(np.arange(12, dtype=np.float32).reshape(3, 4))
        cs = OSGB()
        y_coord = DimCoord(range(3), 'projection_y_coordinate', units='m',
                           coord_system=cs)
        x_coord = DimCoord(range(4), 'projection_x_coordinate', units='km',
                           coord_system=cs)
        bad.add_dim_coord(y_coord, 0)
        bad.add_dim_coord(x_coord, 1)
        return ok, bad

    def test_src_km(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(bad, ok)

    def test_grid_km(self):
        ok, bad = self.ok_bad()
        with self.assertRaises(ValueError):
            regrid(ok, bad)


if __name__ == '__main__':
    tests.main()
