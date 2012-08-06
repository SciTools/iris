# (C) British Crown Copyright 2010 - 2012, Met Office
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
Test the hybrid height representation.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import warnings
import zlib

import numpy

import iris
import iris.tests.stock


@iris.tests.skip_data
class TestHybridHeight(tests.IrisTest):
    def test_colpex(self):
        # Load the COLPEX data => TZYX
        path = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog.pp'))
        
        phenom = iris.load_strict(path, 'air_potential_temperature')

        # Select a ZX cross-section.
        cross_section = phenom[0, :, 0, :]

        # Obtain the real-world heights
        altitude = cross_section.coord('altitude')
        self.assertEqual(altitude.shape, (70, 412))
        self.assertEqual(cross_section.coord_dims(altitude), (0, 1))
        self.assertEqual(zlib.crc32(altitude.points), -306406502)


class TestAbstract(tests.GraphicsTest):
    def setUp(self):
        self.cube = iris.tests.stock.realistic_4d()
        self.altitude = self.cube.coord('altitude')

    def test_metadata(self):
        self.assertEqual(self.altitude.units, 'm')
        self.assertIsNone(self.altitude.coord_system)
        self.assertEqual(self.altitude.attributes, {'positive': 'up'})

    def test_points(self):
        self.assertAlmostEqual(self.altitude.points.min(), numpy.float32(191.84892))
        self.assertAlmostEqual(self.altitude.points.max(), numpy.float32(40000))

    def test_transpose(self):
        self.assertCML(self.cube, ('stock', 'realistic_4d.cml'))
        self.cube.transpose()
        self.assertCML(self.cube, ('derived', 'transposed.cml'))

    def test_indexing(self):
        cube = self.cube[:, :, 0, 0]
        # Make sure the derived 'altitude' coordinate survived the indexing.
        altitude = cube.coord('altitude')
        self.assertCML(cube, ('derived', 'column.cml'))

    def test_removing_sigma(self):
        # Check the cube remains OK when sigma is removed.
        cube = self.cube
        cube.remove_coord('sigma')
        self.assertCML(cube, ('derived', 'removed_sigma.cml'))
        self.assertString(str(cube), ('derived', 'removed_sigma.__str__.txt'))

        # Check the factory now only has surface_altitude and delta dependencies.
        factory = cube.aux_factory(name='altitude')
        t = [key for key, coord in factory.dependencies.iteritems() if coord is not None]
        self.assertItemsEqual(t, ['orography', 'delta'])

    def test_removing_orography(self):
        # Check the cube remains OK when the orography is removed.
        cube = self.cube
        cube.remove_coord('surface_altitude')
        self.assertCML(cube, ('derived', 'removed_orog.cml'))
        self.assertString(str(cube), ('derived', 'removed_orog.__str__.txt'))

        # Check the factory now only has sigma and delta dependencies.
        factory = cube.aux_factory(name='altitude')
        t = [key for key, coord in factory.dependencies.iteritems() if coord is not None]
        self.assertItemsEqual(t, ['sigma', 'delta'])

    def test_derived_coords(self):
        derived_coords = self.cube.derived_coords
        self.assertEqual(len(derived_coords), 1)
        altitude = derived_coords[0]
        self.assertEqual(altitude.standard_name, 'altitude')
        self.assertEqual(altitude.attributes, {'positive': 'up'})

    def test_aux_factory(self):
        factory = self.cube.aux_factory(name='altitude')
        self.assertEqual(factory.standard_name, 'altitude')
        self.assertEqual(factory.attributes, {'positive': 'up'})

    def test_no_orography(self):
        # Get rid of the normal hybrid-height factory.
        cube = self.cube
        factory = cube.aux_factory(name='altitude')
        cube.remove_aux_factory(factory)

        # Add a new one which only references level_height & sigma.
        delta = cube.coord('level_height')
        sigma = cube.coord('sigma')
        factory = iris.aux_factory.HybridHeightFactory(delta, sigma)
        cube.add_aux_factory(factory)

        self.assertEqual(len(cube.aux_factories), 1)
        self.assertEqual(len(cube.derived_coords), 1)
        self.assertString(str(cube), ('derived', 'no_orog.__str__.txt'))
        self.assertCML(cube, ('derived', 'no_orog.cml'))

    def test_invalid_dependencies(self):
        # Must have either delta or orography
        with self.assertRaises(ValueError):
            factory = iris.aux_factory.HybridHeightFactory()
        sigma = self.cube.coord('sigma')
        with self.assertRaises(ValueError):
            factory = iris.aux_factory.HybridHeightFactory(sigma=sigma)

        # Orography must not have bounds
        with warnings.catch_warnings():
            warnings.simplefilter("error")    # Cause all warnings to raise Exceptions
            with self.assertRaises(UserWarning):
                factory = iris.aux_factory.HybridHeightFactory(orography=sigma)

    def test_bounded_orography(self):
        # Start with everything normal
        orog = self.cube.coord('surface_altitude')
        altitude = self.cube.coord('altitude')
        self.assertIsInstance(altitude.bounds, numpy.ndarray)

        # Make sure altitude still works OK if orography was messed
        # with *after* altitude was created.
        altitude = self.cube.coord('altitude')
        orog.bounds = numpy.zeros(orog.shape + (4,))
        self.assertIsInstance(altitude.bounds, numpy.ndarray)

        # Make sure altitude.bounds now raises an error.
        altitude = self.cube.coord('altitude')
        with self.assertRaises(ValueError):
            bounds = altitude.bounds


if __name__ == "__main__":
    tests.main()
