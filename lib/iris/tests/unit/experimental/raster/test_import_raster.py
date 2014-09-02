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
"""Unit tests for the `iris.experimental.raster.import_raster` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock
import numpy as np
try:
    from osgeo import gdal
    from iris.experimental.raster import import_raster
except ImportError:
    gdal = None


@tests.skip_gdal
class TestTiff(tests.IrisTest):
    def setUp(self):
        self.dataset = mock.Mock(name='dataset')
        self.dataset.RasterCount = 1
        self.dataset.GetGeoTransform.return_value = (-200000, 50000, 0,
                                                     -200000, 0, 50000)
        self.dataset.RasterXSize = 5
        self.dataset.RasterYSize = 5
        self.dataset.GetProjection.return_value = ''
        getdata = mock.Mock()
        getdata.ReadAsArray.return_value = np.arange(5*5).reshape(5, 5)
        getdata.GetNoDataValue.return_value = -999
        self.dataset.GetRasterBand.return_value = getdata
        self.gdal_patch = mock.patch('osgeo.gdal.Open',
                                     return_value=self.dataset)
        self.gdal_patch.start()
        self.addCleanup(self.gdal_patch.stop)

    def test_dataset_is_none(self):
        with mock.patch('osgeo.gdal.Open', return_value=None):
            with self.assertRaisesRegexp(IOError, 'gdal failed to open raster '
                                         'image'):
                import_raster('some_filename')

    def test_unsupported_projection(self):
        self.dataset.GetProjection.return_value = 'some projection'
        msg = ('Currently the following projection information is not '
               'interpreted: some projection')
        with mock.patch('warnings.warn') as warn:
            import_raster('some_filename')
        warn.assert_called_once_with(msg)

    def test_multiple_raster_bands(self):
        # Ensure that CubeList of length corresponding to the number of bands
        # is returned and that each has associated coordinates.
        self.dataset.RasterCount = 3
        with mock.patch('osgeo.gdal.Open', return_value=self.dataset), \
                mock.patch('warnings.warn') as warn:
            cubes = import_raster('some_filename')
            self.assertEqual(len(cubes), self.dataset.RasterCount)
            self.assertCML(cubes, ('experimental',
                                   'raster_multiple_band_import.cml'))
        warn.assert_called_once_with('Multiple raster band support ({}) has '
                                     'yet to be validated, use at your own '
                                     'risk'.format(self.dataset.RasterCount))

    def test_no_raster_bands(self):
        self.dataset.RasterCount = 0
        with mock.patch('osgeo.gdal.Open', return_value=self.dataset):
            self.assertIs(import_raster('some_filename'), None)

    def test_rotated_raster(self):
        # Rotated is where a non north-up image is defined.
        # No test data to develop interpretation of rotation so an exception
        # is raised.
        rotation = [1, 1]
        self.dataset.GetGeoTransform.return_value = (
            -200000, 50000, rotation[0], -200000, rotation[1], 50000)
        msg = ('Rotation not supported: \({}, {}\)'.format(rotation[0],
                                                           rotation[1]))
        with mock.patch('osgeo.gdal.Open', return_value=self.dataset), \
                self.assertRaisesRegexp(ValueError, msg):
            import_raster('some_filename')


if __name__ == "__main__":
    tests.main()
