# (C) British Crown Copyright 2010 - 2014, Met Office
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
Test metadata translation rules.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import types

from iris.aux_factory import HybridHeightFactory
from iris.cube import Cube
from iris.fileformats.rules import ConcreteReferenceTarget, Factory, Loader, \
                                   Reference, ReferenceTarget, load_cubes
import iris.tests.stock as stock


class Mock(object):
    def __repr__(self):
        return '<Mock {!r}>'.format(self.__dict__)


class TestConcreteReferenceTarget(tests.IrisTest):
    def test_attributes(self):
        with self.assertRaises(TypeError):
            target = ConcreteReferenceTarget()

        target = ConcreteReferenceTarget('foo')
        self.assertEqual(target.name, 'foo')
        self.assertIsNone(target.transform)

        transform = lambda _: _
        target = ConcreteReferenceTarget('foo', transform)
        self.assertEqual(target.name, 'foo')
        self.assertIs(target.transform, transform)

    def test_single_cube_no_transform(self):
        target = ConcreteReferenceTarget('foo')
        src = stock.simple_2d()
        target.add_cube(src)
        self.assertIs(target.as_cube(), src)

    def test_single_cube_with_transform(self):
        transform = lambda cube: {'long_name': 'wibble'}
        target = ConcreteReferenceTarget('foo', transform)
        src = stock.simple_2d()
        target.add_cube(src)
        dest = target.as_cube()
        self.assertEqual(dest.long_name, 'wibble')
        self.assertNotEqual(dest, src)
        dest.long_name = src.long_name
        self.assertEqual(dest, src)

    def test_multiple_cubes_no_transform(self):
        target = ConcreteReferenceTarget('foo')
        src = stock.realistic_4d()
        for i in range(src.shape[0]):
            target.add_cube(src[i])
        dest = target.as_cube()
        self.assertIsNot(dest, src)
        self.assertEqual(dest, src)

    def test_multiple_cubes_with_transform(self):
        transform = lambda cube: {'long_name': 'wibble'}
        target = ConcreteReferenceTarget('foo', transform)
        src = stock.realistic_4d()
        for i in range(src.shape[0]):
            target.add_cube(src[i])
        dest = target.as_cube()
        self.assertEqual(dest.long_name, 'wibble')
        self.assertNotEqual(dest, src)
        dest.long_name = src.long_name
        self.assertEqual(dest, src)


class TestLoadCubes(tests.IrisTest):
    def test_simple_factory(self):
        # Test the creation process for a factory definition which only
        # uses simple dict arguments.

        # The fake PPField which will be supplied to our converter.
        field = Mock()
        field.data = None
        field_generator = lambda filename: [field]
        # A fake conversion function returning:
        #   1) A parameter cube needing a simple factory construction.
        aux_factory = Mock()
        factory = Mock()
        factory.args = [{'name': 'foo'}]
        factory.factory_class = lambda *args: \
            setattr(aux_factory, 'fake_args', args) or aux_factory
        def converter(field):
            return ([factory], [], '', '', '', {}, [], [], [])
        # Finish by making a fake Loader
        fake_loader = Loader(field_generator, {}, converter, None)
        cubes = load_cubes(['fake_filename'], None, fake_loader)

        # Check the result is a generator with a single entry.
        self.assertIsInstance(cubes, types.GeneratorType)
        try:
            # Suppress the normal Cube.coord() and Cube.add_aux_factory()
            # methods.
            coord_method = Cube.coord
            add_aux_factory_method = Cube.add_aux_factory
            Cube.coord = lambda self, **args: args
            Cube.add_aux_factory = lambda self, aux_factory: \
                setattr(self, 'fake_aux_factory', aux_factory)

            cubes = list(cubes)
        finally:
            Cube.coord = coord_method
            Cube.add_aux_factory = add_aux_factory_method
        self.assertEqual(len(cubes), 1)
        # Check the "cube" has an "aux_factory" added, which itself
        # must have been created with the correct arguments.
        self.assertTrue(hasattr(cubes[0], 'fake_aux_factory'))
        self.assertIs(cubes[0].fake_aux_factory, aux_factory)
        self.assertTrue(hasattr(aux_factory, 'fake_args'))
        self.assertEqual(aux_factory.fake_args, ({'name': 'foo'},))

    def test_cross_reference(self):
        # Test the creation process for a factory definition which uses
        # a cross-reference.

        param_cube = stock.realistic_4d_no_derived()
        orog_coord = param_cube.coord('surface_altitude')
        param_cube.remove_coord(orog_coord)

        orog_cube = param_cube[0, 0, :, :]
        orog_cube.data = orog_coord.points
        orog_cube.rename('surface_altitude')
        orog_cube.units = orog_coord.units
        orog_cube.attributes = orog_coord.attributes

        # We're going to test for the presence of the hybrid height
        # stuff later, so let's make sure it's not already there!
        assert len(param_cube.aux_factories) == 0
        assert not param_cube.coords('surface_altitude')

        # The fake PPFields which will be supplied to our converter.
        press_field = Mock()
        press_field.data = param_cube.data
        orog_field = Mock()
        orog_field.data = orog_cube.data
        field_generator = lambda filename: [press_field, orog_field]
        # A fake rule set returning:
        #   1) A parameter cube needing an "orography" reference
        #   2) An "orography" cube
        def converter(field):
            if field is press_field:
                src = param_cube
                factories = [Factory(HybridHeightFactory,
                                     [Reference('orography')])]
                references = []
            else:
                src = orog_cube
                factories = []
                references = [ReferenceTarget('orography', None)]
            dim_coords_and_dims = [(coord, src.coord_dims(coord)[0])
                                   for coord in src.dim_coords]
            aux_coords_and_dims = [(coord, src.coord_dims(coord))
                                   for coord in src.aux_coords]
            return (factories, references, src.standard_name, src.long_name,
                    src.units, src.attributes, src.cell_methods,
                    dim_coords_and_dims, aux_coords_and_dims)
        # Finish by making a fake Loader
        fake_loader = Loader(field_generator, {}, converter, None)
        cubes = load_cubes(['fake_filename'], None, fake_loader)

        # Check the result is a generator containing two Cubes.
        self.assertIsInstance(cubes, types.GeneratorType)
        cubes = list(cubes)
        self.assertEqual(len(cubes), 2)
        # Check the "cube" has an "aux_factory" added, which itself
        # must have been created with the correct arguments.
        self.assertEqual(len(cubes[1].aux_factories), 1)
        self.assertEqual(len(cubes[1].coords('surface_altitude')), 1)


if __name__ == "__main__":
    tests.main()
