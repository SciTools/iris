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
Unit tests for the `iris.fileformats.cf.CFReader` class.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import numpy as np

import iris
from iris.fileformats.cf import CFReader


def netcdf_variable(name, dimensions, dtype, ancillary_variables=None,
                    coordinates=None, bounds=None, climatology=None,
                    formula_terms=None, grid_mapping=None,
                    cell_measures=None):
    """Return a mock NetCDF4 variable."""
    ndim = 0
    if dimensions is not None:
        dimensions = dimensions.split()
        ndim = len(dimensions)
    else:
        dimensions = []
    ncvar = mock.Mock(name=name, dimensions=dimensions,
                      ncattrs=mock.Mock(return_value=[]),
                      ndim=ndim, dtype=dtype,
                      ancillary_variables=ancillary_variables,
                      coordinates=coordinates,
                      bounds=bounds, climatology=climatology,
                      formula_terms=formula_terms,
                      grid_mapping=grid_mapping, cell_measures=cell_measures)
    return ncvar


class Test_translate__formula_terms(tests.IrisTest):
    def setUp(self):
        self.delta = netcdf_variable('delta', 'height', np.float)
        self.sigma = netcdf_variable('sigma', 'height', np.float)
        self.orography = netcdf_variable('orography', 'lat lon', np.float)
        formula_terms = 'a: delta b: sigma orog: orography'
        self.height = netcdf_variable('height', 'height', np.float,
                                      formula_terms=formula_terms)
        self.lat = netcdf_variable('lat', 'lat', np.float)
        self.lon = netcdf_variable('lon', 'lon', np.float)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable('temp', 'height lat lon', np.float,
                                    coordinates='lat lon')

        self.variables = dict(delta=self.delta, sigma=self.sigma,
                              orography=self.orography, height=self.height,
                              lat=self.lat, lon=self.lon, temp=self.temp)
        ncattrs = mock.Mock(return_value=[])
        self.dataset = mock.Mock(file_format='NetCDF4',
                                 variables=self.variables,
                                 ncattrs=ncattrs)
        # Restrict the CFReader functionality to only performing translations.
        cls = 'iris.fileformats.cf.CFReader'
        build_patch = mock.patch(
            'iris.fileformats.cf.CFReader._build_cf_groups')
        reset_patch = mock.patch('iris.fileformats.cf.CFReader._reset')
        build_patch.start()
        reset_patch.start()
        self.addCleanup(build_patch.stop)
        self.addCleanup(reset_patch.stop)

    def test_create_formula_terms(self):
        with mock.patch('netCDF4.Dataset', return_value=self.dataset):
            cf_group = CFReader('dummy').cf_group
            self.assertEqual(len(cf_group), len(self.variables))
            # Check there is a singular data variable.
            group = cf_group.data_variables
            self.assertEqual(len(group), 1)
            self.assertEqual(group.keys(), ['temp'])
            self.assertIs(group['temp'].cf_data, self.temp)
            # Check there are three coordinates.
            group = cf_group.coordinates
            self.assertEqual(len(group), 3)
            self.assertEqual(group.viewkeys(), set(['height', 'lat', 'lon']))
            self.assertIs(group['height'].cf_data, self.height)
            self.assertIs(group['lat'].cf_data, self.lat)
            self.assertIs(group['lon'].cf_data, self.lon)
            # Check there are three auxiliary coordinates.
            group = cf_group.auxiliary_coordinates
            self.assertEqual(len(group), 3)
            self.assertEqual(group.viewkeys(),
                             set(['delta', 'sigma', 'orography']))
            self.assertIs(group['delta'].cf_data, self.delta)
            self.assertIs(group['sigma'].cf_data, self.sigma)
            self.assertIs(group['orography'].cf_data, self.orography)
            # Check all the auxiliary coordinates are formula terms.
            formula_terms = cf_group.formula_terms
            self.assertEqual(group.viewitems(), formula_terms.viewitems())


class Test_build_cf_groups__formula_terms(tests.IrisTest):
    def setUp(self):
        self.delta = netcdf_variable('delta', 'height', np.float)
        self.sigma = netcdf_variable('sigma', 'height', np.float)
        self.orography = netcdf_variable('orography', 'lat lon', np.float)
        formula_terms = 'a: delta b: sigma orog: orography'
        self.height = netcdf_variable('height', 'height', np.float,
                                      formula_terms=formula_terms)
        self.lat = netcdf_variable('lat', 'lat', np.float)
        self.lon = netcdf_variable('lon', 'lon', np.float)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable('temp', 'height lat lon', np.float,
                                    coordinates='lat lon')

        self.variables = dict(delta=self.delta, sigma=self.sigma,
                              orography=self.orography, height=self.height,
                              lat=self.lat, lon=self.lon, temp=self.temp)
        ncattrs = mock.Mock(return_value=[])
        self.dataset = mock.Mock(file_format='NetCDF4',
                                 variables=self.variables,
                                 ncattrs=ncattrs)
        # Restrict the CFReader functionality to only performing translations
        # and building first level cf-groups for variables.
        patcher = mock.patch('iris.fileformats.cf.CFReader._reset')
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_associate_formula_terms_with_data_variable(self):
        with mock.patch('netCDF4.Dataset', return_value=self.dataset):
            cf_group = CFReader('dummy').cf_group
            self.assertEqual(len(cf_group), len(self.variables))
            # Check the cf-group associated with the data variable.
            temp_cf_group = cf_group['temp'].cf_group
            # Check the data variable is associated with six variables.
            self.assertEqual(len(temp_cf_group), 6)
            # Check there are three coordinates.
            group = temp_cf_group.coordinates
            self.assertEqual(len(group), 3)
            self.assertEqual(group.viewkeys(), set(['height', 'lat', 'lon']))
            self.assertIs(group['height'].cf_data, self.height)
            self.assertIs(group['lat'].cf_data, self.lat)
            self.assertIs(group['lon'].cf_data, self.lon)
            # Check there are three auxiliary coordinates.
            group = temp_cf_group.auxiliary_coordinates
            self.assertEqual(len(group), 3)
            self.assertEqual(group.viewkeys(),
                             set(['delta', 'sigma', 'orography']))
            self.assertIs(group['delta'].cf_data, self.delta)
            self.assertIs(group['sigma'].cf_data, self.sigma)
            self.assertIs(group['orography'].cf_data, self.orography)
            # Check all the auxiliary coordinates are formula terms.
            formula_terms = cf_group.formula_terms
            self.assertEqual(group.viewitems(), formula_terms.viewitems())
            # Check the terms by root.
            self.assertEqual(formula_terms['delta'].cf_terms_by_root,
                             dict(height='a'))
            self.assertEqual(formula_terms['sigma'].cf_terms_by_root,
                             dict(height='b'))
            self.assertEqual(formula_terms['orography'].cf_terms_by_root,
                             dict(height='orog'))


if __name__ == '__main__':
    tests.main()
