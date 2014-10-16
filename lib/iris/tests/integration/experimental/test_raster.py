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

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris
import iris.tests.stock
try:
    from osgeo import gdal
    from iris.experimental.raster import import_raster, export_geotiff
except ImportError:
    gdal = None


@tests.skip_gdal
class TestTiff(tests.IrisTest):
    def setUp(self):
        self.cube = tests.stock.lat_lon_cube()
        self.cube.coord(axis='x').guess_bounds()
        self.cube.coord(axis='y').guess_bounds()

    def test_unmasked_data(self):
        tmp_filename = iris.util.create_temp_filename(suffix='.tiff')
        export_geotiff(self.cube, tmp_filename)

        result = import_raster(tmp_filename)[0]
        self.assertArrayEqual(self.cube.coord(axis='x').points,
                              result.coord(axis='x').points)
        self.assertArrayEqual(self.cube.coord(axis='y').points,
                              result.coord(axis='y').points)
        self.assertArrayEqual(self.cube.data, result.data)

    def test_masked_data(self):
        tmp_filename = iris.util.create_temp_filename(suffix='.tiff')
        self.cube.data = np.ma.masked_greater(self.cube.data, 5)
        export_geotiff(self.cube, tmp_filename)
        result = import_raster(tmp_filename)[0]
        self.assertMaskedArrayEqual(self.cube.data,
                                    result.data)

    def test_expected_metadata(self):
        # CML check.
        tmp_filename = iris.util.create_temp_filename(suffix='.tiff')
        export_geotiff(self.cube, tmp_filename)

        result = import_raster(tmp_filename)[0]
        self.assertCML(result, ('experimental', 'raster_import.cml'))


if __name__ == "__main__":
    tests.main()
