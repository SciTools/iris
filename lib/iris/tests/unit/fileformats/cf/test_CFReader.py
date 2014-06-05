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
import warnings

import numpy as np

import iris
from iris.fileformats.cf import CFReader


def netcdf_variable(name, dimensions, dtype, ancillary_variables=None,
                    coordinates='', bounds=None, climatology=None,
                    formula_terms=None, grid_mapping=None,
                    cell_measures=None, standard_name=None):
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
                      grid_mapping=grid_mapping, cell_measures=cell_measures,
                      standard_name=standard_name)
    return ncvar


class Test_translate__formula_terms(tests.IrisTest):
    def setUp(self):
        self.delta = netcdf_variable('delta', 'height', np.float,
                                     bounds='delta_bnds')
        self.delta_bnds = netcdf_variable('delta_bnds', 'height bnds',
                                          np.float)
        self.sigma = netcdf_variable('sigma', 'height', np.float,
                                     bounds='sigma_bnds')
        self.sigma_bnds = netcdf_variable('sigma_bnds', 'height bnds',
                                          np.float)
        self.orography = netcdf_variable('orography', 'lat lon', np.float)
        formula_terms = 'a: delta b: sigma orog: orography'
        standard_name = 'atmosphere_hybrid_height_coordinate'
        self.height = netcdf_variable('height', 'height', np.float,
                                      formula_terms=formula_terms,
                                      bounds='height_bnds',
                                      standard_name=standard_name)
        # Over-specify the formula terms on the bounds variable,
        # which will be ignored by the cf loader.
        formula_terms = 'a: delta_bnds b: sigma_bnds orog: orography'
        self.height_bnds = netcdf_variable('height_bnds', 'height bnds',
                                           np.float,
                                           formula_terms=formula_terms)
        self.lat = netcdf_variable('lat', 'lat', np.float)
        self.lon = netcdf_variable('lon', 'lon', np.float)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable('temp', 'height lat lon', np.float,
                                    coordinates='lat lon')

        self.variables = dict(delta=self.delta, sigma=self.sigma,
                              orography=self.orography, height=self.height,
                              lat=self.lat, lon=self.lon, temp=self.temp,
                              delta_bnds=self.delta_bnds,
                              sigma_bnds=self.sigma_bnds,
                              height_bnds=self.height_bnds)
        ncattrs = mock.Mock(return_value=[])
        self.dataset = mock.Mock(file_format='NetCDF4',
                                 variables=self.variables,
                                 ncattrs=ncattrs)
        # Restrict the CFReader functionality to only performing translations.
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
            coordinates = ['height', 'lat', 'lon']
            self.assertEqual(group.viewkeys(), set(coordinates))
            for name in coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check there are three auxiliary coordinates.
            group = cf_group.auxiliary_coordinates
            self.assertEqual(len(group), 3)
            aux_coordinates = ['delta', 'sigma', 'orography']
            self.assertEqual(group.viewkeys(), set(aux_coordinates))
            for name in aux_coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check all the auxiliary coordinates are formula terms.
            formula_terms = cf_group.formula_terms
            self.assertEqual(group.viewitems(), formula_terms.viewitems())
            # Check there are three bounds.
            group = cf_group.bounds
            self.assertEqual(len(group), 3)
            bounds = ['height_bnds', 'delta_bnds', 'sigma_bnds']
            self.assertEqual(group.viewkeys(), set(bounds))
            for name in bounds:
                self.assertEqual(group[name].cf_data, getattr(self, name))


class Test_build_cf_groups__formula_terms(tests.IrisTest):
    def setUp(self):
        self.delta = netcdf_variable('delta', 'height', np.float,
                                     bounds='delta_bnds')
        self.delta_bnds = netcdf_variable('delta_bnds', 'height bnds',
                                          np.float)
        self.sigma = netcdf_variable('sigma', 'height', np.float,
                                     bounds='sigma_bnds')
        self.sigma_bnds = netcdf_variable('sigma_bnds', 'height bnds',
                                          np.float)
        self.orography = netcdf_variable('orography', 'lat lon', np.float)
        formula_terms = 'a: delta b: sigma orog: orography'
        standard_name = 'atmosphere_hybrid_height_coordinate'
        self.height = netcdf_variable('height', 'height', np.float,
                                      formula_terms=formula_terms,
                                      bounds='height_bnds',
                                      standard_name=standard_name)
        # Over-specify the formula terms on the bounds variable,
        # which will be ignored by the cf loader.
        formula_terms = 'a: delta_bnds b: sigma_bnds orog: orography'
        self.height_bnds = netcdf_variable('height_bnds', 'height bnds',
                                           np.float,
                                           formula_terms=formula_terms)
        self.lat = netcdf_variable('lat', 'lat', np.float)
        self.lon = netcdf_variable('lon', 'lon', np.float)
        # Note that, only lat and lon are explicitly associated as coordinates.
        self.temp = netcdf_variable('temp', 'height lat lon', np.float,
                                    coordinates='lat lon')

        self.variables = dict(delta=self.delta, sigma=self.sigma,
                              orography=self.orography, height=self.height,
                              lat=self.lat, lon=self.lon, temp=self.temp,
                              delta_bnds=self.delta_bnds,
                              sigma_bnds=self.sigma_bnds,
                              height_bnds=self.height_bnds)
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
            coordinates = ['height', 'lat', 'lon']
            self.assertEqual(group.viewkeys(), set(coordinates))
            for name in coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check the height coordinate is bounded.
            group = group['height'].cf_group
            self.assertEqual(len(group.bounds), 1)
            self.assertIn('height_bnds', group.bounds)
            self.assertIs(group['height_bnds'].cf_data, self.height_bnds)
            # Check there are three auxiliary coordinates.
            group = temp_cf_group.auxiliary_coordinates
            self.assertEqual(len(group), 3)
            aux_coordinates = ['delta', 'sigma', 'orography']
            self.assertEqual(group.viewkeys(), set(aux_coordinates))
            for name in aux_coordinates:
                self.assertIs(group[name].cf_data, getattr(self, name))
            # Check all the auxiliary coordinates are formula terms.
            formula_terms = cf_group.formula_terms
            self.assertEqual(group.viewitems(), formula_terms.viewitems())
            # Check the terms by root.
            for name, term in zip(aux_coordinates, ['a', 'b', 'orog']):
                self.assertEqual(formula_terms[name].cf_terms_by_root,
                                 dict(height=term))
            # Check the bounded auxiliary coordinates.
            for name, name_bnds in zip(['delta', 'sigma'],
                                       ['delta_bnds', 'sigma_bnds']):
                aux_coord_group = group[name].cf_group
                self.assertEqual(len(aux_coord_group.bounds), 1)
                self.assertIn(name_bnds, aux_coord_group.bounds)
                self.assertIs(aux_coord_group[name_bnds].cf_data,
                              getattr(self, name_bnds))

    def test_future_promote(self):
        with mock.patch('netCDF4.Dataset', return_value=self.dataset):
            with iris.FUTURE.context(netcdf_promote=True):
                cf_group = CFReader('dummy').cf_group
                self.assertEqual(len(cf_group), len(self.variables))
                # Check the number of data variables.
                self.assertEqual(len(cf_group.data_variables), 1)
                self.assertEqual(cf_group.data_variables.keys(), ['temp'])
                # Check the number of promoted variables.
                self.assertEqual(len(cf_group.promoted), 1)
                self.assertEqual(cf_group.promoted.keys(), ['orography'])
                # Check the promoted variable dependencies.
                group = cf_group.promoted['orography'].cf_group.coordinates
                self.assertEqual(len(group), 2)
                coordinates = ('lat', 'lon')
                self.assertEqual(group.viewkeys(), set(coordinates))
                for name in coordinates:
                    self.assertIs(group[name].cf_data, getattr(self, name))

    def test_formula_terms_dimension_mismatch(self):
        self.orography.dimensions = 'lat wibble'
        with mock.patch('netCDF4.Dataset', return_value=self.dataset):
            with warnings.catch_warnings(record=True) as warn:
                warnings.simplefilter('always')
                CFReader('dummy')
                self.assertEqual(len(warn), 1)
                self.assertIn('dimension mis-match', warn[0].message.message)


if __name__ == '__main__':
    tests.main()
