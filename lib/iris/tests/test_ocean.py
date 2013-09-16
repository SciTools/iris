# (C) British Crown Copyright 2010 - 2013, Met Office
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
Test the ocean dimesnionless vertical coordinate representations.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import warnings

import numpy as np

import iris
from iris.aux_factory import OceanSigmaZFactory
import iris.tests.stock


class TestOceanSigmaZ(tests.IrisTest):
    def setUp(self):
        self.cube = iris.tests.stock.ocean_sigma_z()
        self.derived_name = 'sea_surface_height_above_reference_ellipsoid'
        coords = self.cube.aux_factory().dependencies
        self.sigma = coords['sigma']
        self.eta = coords['eta']
        self.depth = coords['depth']
        self.depth_c = coords['depth_c']
        self.nsigma = coords['nsigma']
        self.zlev = coords['zlev']

    def test_metadata(self):
        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.units, 'm')
        self.assertIsNone(derived.coord_system)
        self.assertEqual(derived.attributes, dict(positive='up'))

    def test_points(self):
        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), -102.0)
        self.assertEqual(derived.points.max(), 4.0)

    def test_bounds(self):
        self.zlev.guess_bounds()
        self.sigma.guess_bounds()
        fname = ('derived', 'ocean', 'ocean_sigma_z_bounds.cml')
        self.assertCML(self.cube, fname)
        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.bounds.min(), -128.0)
        self.assertEqual(derived.bounds.max(), 4.5)

    def test_transpose(self):
        self.assertCML(self.cube, ('stock', 'ocean_sigma_z.cml'))
        self.cube.transpose()
        fname = ('derived', 'ocean', 'ocean_sigma_z_transpose.cml')
        self.assertCML(self.cube, fname)
        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), -102.0)
        self.assertEqual(derived.points.max(), 4.0)

    def test_indexing(self):
        cube = self.cube[:, :, :, 0]
        # Ensure the 3d derived coordinate can be realised.
        derived = cube.coord(self.derived_name)
        self.assertCML(cube, ('derived', 'ocean', 'ocean_sigma_z_3d.cml'))
        self.assertString(str(cube),
                          ('derived', 'ocean', 'ocean_sigma_z_3d.__str__.txt'))

        cube = self.cube[:, :, 0, 0]
        # Ensure the 2d derived coordinate can be realised.
        derived = cube.coord(self.derived_name)
        self.assertCML(cube, ('derived', 'ocean', 'ocean_sigma_z_2d.cml'))
        self.assertString(str(cube),
                          ('derived', 'ocean', 'ocean_sigma_z_2d.__str__.txt'))

        cube = self.cube[:, 0, 0, 0]
        # Ensure the 1d derived coordinate can be realised.
        derived = cube.coord(self.derived_name)
        self.assertCML(cube, ('derived', 'ocean', 'ocean_sigma_z_1d.cml'))
        self.assertString(str(cube),
                          ('derived', 'ocean', 'ocean_sigma_z_1d.__str__.txt'))

        cube = self.cube[0, 0, 0, 0]
        # Ensure the 0d derived coordinate can be realised.
        derived = cube.coord(self.derived_name)
        self.assertCML(cube, ('derived', 'ocean', 'ocean_sigma_z_0d.cml'))
        self.assertString(str(cube),
                          ('derived', 'ocean', 'ocean_sigma_z_0d.__str__.txt'))

    def test_remove_sigma(self):
        self.cube.remove_coord('sigma')
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_sigma.cml')
        self.assertCML(self.cube, fname)
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_sigma.__str__.txt')
        self.assertString(str(self.cube), fname)

        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), 1.0)
        self.assertEqual(derived.points.max(), 4.0)

        factory = self.cube.aux_factory()
        terms = [term for term, coord in factory.dependencies.items()
                 if coord is not None]
        self.assertItemsEqual(terms,
                              ['eta', 'depth', 'depth_c', 'nsigma', 'zlev'])

        dependency_dims = factory._dependency_dims(self.cube.coord_dims)
        self.assertEqual(dependency_dims, dict(eta=(0, 2, 3),
                                               depth=(2, 3),
                                               depth_c=(),
                                               nsigma=(),
                                               zlev=(1,)))
        derived_dims = factory.derived_dims(self.cube.coord_dims)
        self.assertEqual(derived_dims, (0, 1, 2, 3))
        nd_points = factory._remap(dependency_dims, derived_dims)
        nsigma_slice = factory._make_nsigma_slice(dependency_dims,
                                                  derived_dims,
                                                  nd_points['nsigma'])
        self.assertEqual(nsigma_slice, [slice(None),
                                        slice(0, 2),
                                        slice(None),
                                        slice(None)])

    def test_remove_eta(self):
        self.cube.remove_coord('eta')
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_eta.cml')
        self.assertCML(self.cube, fname)
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_eta.__str__.txt')
        self.assertString(str(self.cube), fname)

        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), -100.0)
        self.assertEqual(derived.points.max(), 4.0)

        factory = self.cube.aux_factory()
        terms = [term for term, coord in factory.dependencies.items()
                 if coord is not None]
        self.assertItemsEqual(terms,
                              ['sigma', 'depth', 'depth_c', 'nsigma', 'zlev'])

        dependency_dims = factory._dependency_dims(self.cube.coord_dims)
        self.assertEqual(dependency_dims, dict(sigma=(1,),
                                               depth=(2, 3),
                                               depth_c=(),
                                               nsigma=(),
                                               zlev=(1,)))
        derived_dims = factory.derived_dims(self.cube.coord_dims)
        self.assertEqual(derived_dims, (1, 2, 3))
        nd_points = factory._remap(dependency_dims, derived_dims)
        nsigma_slice = factory._make_nsigma_slice(dependency_dims,
                                                  derived_dims,
                                                  nd_points['nsigma'])
        self.assertEqual(nsigma_slice, [slice(0, 2),
                                        slice(None),
                                        slice(None)])

    def test_remove_depth(self):
        self.cube.remove_coord('depth')
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_depth.cml')
        self.assertCML(self.cube, fname)
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_depth.__str__.txt')
        self.assertString(str(self.cube), fname)

        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), -2.0)
        self.assertEqual(derived.points.max(), 4.0)

        factory = self.cube.aux_factory()
        terms = [term for term, coord in factory.dependencies.items()
                 if coord is not None]
        self.assertItemsEqual(terms,
                              ['sigma', 'eta', 'depth_c', 'nsigma', 'zlev'])

        dependency_dims = factory._dependency_dims(self.cube.coord_dims)
        self.assertEqual(dependency_dims, dict(sigma=(1,),
                                               eta=(0, 2, 3),
                                               depth_c=(),
                                               nsigma=(),
                                               zlev=(1,)))
        derived_dims = factory.derived_dims(self.cube.coord_dims)
        self.assertEqual(derived_dims, (0, 1, 2, 3))
        nd_points = factory._remap(dependency_dims, derived_dims)
        nsigma_slice = factory._make_nsigma_slice(dependency_dims,
                                                  derived_dims,
                                                  nd_points['nsigma'])
        self.assertEqual(nsigma_slice, [slice(None),
                                        slice(0, 2),
                                        slice(None),
                                        slice(None)])

    def test_remove_depth_c(self):
        self.cube.remove_coord('depth_c')
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_depth_c.cml')
        self.assertCML(self.cube, fname)
        fname = ('derived', 'ocean',
                 'ocean_sigma_z_remove_depth_c.__str__.txt')
        self.assertString(str(self.cube), fname)

        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), -2.0)
        self.assertEqual(derived.points.max(), 4.0)

        factory = self.cube.aux_factory()
        terms = [term for term, coord in factory.dependencies.items()
                 if coord is not None]
        self.assertItemsEqual(terms,
                              ['sigma', 'eta', 'depth', 'nsigma', 'zlev'])

        dependency_dims = factory._dependency_dims(self.cube.coord_dims)
        self.assertEqual(dependency_dims, dict(sigma=(1,),
                                               eta=(0, 2, 3),
                                               depth=(2, 3),
                                               nsigma=(),
                                               zlev=(1,)))
        derived_dims = factory.derived_dims(self.cube.coord_dims)
        self.assertEqual(derived_dims, (0, 1, 2, 3))
        nd_points = factory._remap(dependency_dims, derived_dims)
        nsigma_slice = factory._make_nsigma_slice(dependency_dims,
                                                  derived_dims,
                                                  nd_points['nsigma'])
        self.assertEqual(nsigma_slice, [slice(None),
                                        slice(0, 2),
                                        slice(None),
                                        slice(None)])

    def test_remove_nsigma(self):
        self.cube.remove_coord('nsigma')
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_nsigma.cml')
        self.assertCML(self.cube, fname)
        fname = ('derived', 'ocean', 'ocean_sigma_z_remove_nsigma.__str__.txt')
        self.assertString(str(self.cube), fname)

        derived = self.cube.coord(self.derived_name)
        self.assertEqual(derived.points.min(), 0.0)
        self.assertEqual(derived.points.max(), 4.0)

        factory = self.cube.aux_factory()
        terms = [term for term, coord in factory.dependencies.items()
                 if coord is not None]
        self.assertItemsEqual(terms,
                              ['sigma', 'eta', 'depth', 'depth_c', 'zlev'])

        dependency_dims = factory._dependency_dims(self.cube.coord_dims)
        self.assertEqual(dependency_dims, dict(sigma=(1,),
                                               eta=(0, 2, 3),
                                               depth=(2, 3),
                                               depth_c=(),
                                               zlev=(1,)))
        derived_dims = factory.derived_dims(self.cube.coord_dims)
        self.assertEqual(derived_dims, (0, 1, 2, 3))
        nd_points = factory._remap(dependency_dims, derived_dims)
        nsigma_slice = factory._make_nsigma_slice(dependency_dims,
                                                  derived_dims,
                                                  nd_points['nsigma'])
        self.assertEqual(nsigma_slice, [slice(None),
                                        slice(0, 0),
                                        slice(None),
                                        slice(None)])

    def test_remove_zlev(self):
        with self.assertRaises(ValueError):
            self.cube.remove_coord('zlev')

    def test_derived_coords(self):
        derived_coords = self.cube.derived_coords
        self.assertEqual(len(derived_coords), 1)
        derived, = derived_coords
        self.assertEqual(derived.standard_name, self.derived_name)

    def test_invalid_no_zlev(self):
        # Requires a zlev.
        with self.assertRaises(ValueError):
            OceanSigmaZFactory()

    def test_invalid_no_optional(self):
        # Requires either a eta, depth, depth_c.
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev)

    def test_invalid_zlev_bounds(self):
        self.zlev.bounds = np.zeros(self.zlev.shape + (4,))
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev, eta=self.eta)

    def test_invalid_sigma_bounds(self):
        self.sigma.bounds = np.zeros(self.sigma.shape + (4,))
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta,
                               sigma=self.sigma)

    def test_invalid_zlev_sigma_bounds(self):
        self.zlev.bounds = np.zeros(self.zlev.shape + (4,))
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta,
                               sigma=self.sigma)

    def test_invalid_depth_c_non_scalar(self):
        depth_c = self.depth_c.copy(points=[1, 2])
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta,
                               depth_c=depth_c)

    def test_invalid_nsigma_non_scalar(self):
        nsigma = self.nsigma.copy(points=[1, 2])
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta,
                               nsigma=nsigma)

    def test_invalid_zlev_depth_units_match(self):
        self.depth.units = '1'
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta,
                               depth=self.depth)

    def test_invalid_zlev_depth_c_units_match(self):
        self.depth_c.units = 'K'
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta,
                               depth_c=self.depth_c)

    def test_invalid_zlev_eta_units_match(self):
        self.eta.units = 'volts'
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta)

    def test_invalid_zlev_units(self):
        self.zlev.units = None
        with self.assertRaises(ValueError):
            OceanSigmaZFactory(zlev=self.zlev,
                               eta=self.eta)

    def test_warning_eta_bounds(self):
        self.eta.bounds = np.zeros(self.eta.shape + (4,))
        # Ignore eta bounds.
        with warnings.catch_warnings():
            # Promote warnings to exception status.
            warnings.simplefilter('error')
            with self.assertRaises(UserWarning):
                OceanSigmaZFactory(zlev=self.zlev,
                                   eta=self.eta)

    def test_warning_depth_bounds(self):
        self.depth.bounds = np.zeros(self.depth.shape + (4,))
        # Ignore depth bounds.
        with warnings.catch_warnings():
            # Promote warnings to exception status.
            warnings.simplefilter('error')
            with self.assertRaises(UserWarning):
                OceanSigmaZFactory(zlev=self.zlev,
                                   eta=self.eta,
                                   depth=self.depth)

    def test_netcdf_save_load(self):
        with self.temp_filename(suffix='.nc') as temp:
            iris.save(self.cube, temp)
            cube = iris.load_cube(temp)
            fname = ('derived', 'ocean', 'ocean_sigma_z_save_load.cml')
            self.assertCML(cube, fname)
            self.assertEqual(cube, self.cube)


if __name__ == '__main__':
    tests.main()
