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
Test metadata translation rules.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import types

from iris.aux_factory import HybridHeightFactory
from iris.fileformats.rules import ConcreteReferenceTarget, Factory, Loader, \
                                   Reference, ReferenceTarget, RuleResult, \
                                   load_cubes
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
        field = Mock()
        field_generator = lambda filename: [field]
        # A fake rule set returning:
        #   1) A parameter cube needing a simple factory construction.
        src_cube = Mock()
        src_cube.coord = lambda **args: args
        src_cube.add_aux_factory = lambda aux_factory: \
            setattr(src_cube, 'fake_aux_factory', aux_factory)
        aux_factory = Mock()
        factory = Mock()
        factory.args = [{'name': 'foo'}]
        factory.factory_class = lambda *args: \
            setattr(aux_factory, 'fake_args', args) or aux_factory
        rule_result = RuleResult(src_cube, Mock(), [factory])
        rules = Mock()
        rules.result = lambda field: rule_result
        # A fake cross-reference rule set
        xref_rules = Mock()
        xref_rules.matching_rules = lambda field: []
        # Finish by making a fake Loader
        name = 'FAKE_PP'
        fake_loader = Loader(field_generator, rules, xref_rules, name)
        cubes = load_cubes(['fake_filename'], None, fake_loader)
        # Check the result is a generator with our "cube" as the only
        # entry.
        self.assertIsInstance(cubes, types.GeneratorType)
        cubes = list(cubes)
        self.assertEqual(len(cubes), 1)
        self.assertIs(cubes[0], src_cube)
        # Check the "cube" has an "aux_factory" added, which itself
        # must have been created with the correct arguments.
        self.assertTrue(hasattr(src_cube, 'fake_aux_factory'))
        self.assertIs(src_cube.fake_aux_factory, aux_factory)
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

        press_field = Mock()
        orog_field = Mock()
        field_generator = lambda filename: [press_field, orog_field]
        # A fake rule set returning:
        #   1) A parameter cube needing an "orography" reference
        #   2) An "orography" cube
        factory = Factory(HybridHeightFactory, [Reference('orography')])
        press_rule_result = RuleResult(param_cube, Mock(), [factory])
        orog_rule_result= RuleResult(orog_cube, Mock(), [])
        rules = Mock()
        rules.result = lambda field: \
            press_rule_result if field is press_field else orog_rule_result
        # A fake cross-reference rule set
        ref = ReferenceTarget('orography', None)
        orog_xref_rule = Mock()
        orog_xref_rule.run_actions = lambda cube, field: (ref,)
        xref_rules = Mock()
        xref_rules.matching_rules = lambda field: \
            [orog_xref_rule] if field is orog_field else []
        # Finish by making a fake Loader
        name = 'FAKE_PP'
        fake_loader = Loader(field_generator, rules, xref_rules, name)
        cubes = load_cubes(['fake_filename'], None, fake_loader)
        # Check the result is a generator containing both of our cubes.
        self.assertIsInstance(cubes, types.GeneratorType)
        cubes = list(cubes)
        self.assertEqual(len(cubes), 2)
        self.assertIs(cubes[0], orog_cube)
        self.assertIs(cubes[1], param_cube)
        # Check the "cube" has an "aux_factory" added, which itself
        # must have been created with the correct arguments.
        self.assertEqual(len(param_cube.aux_factories), 1)
        self.assertEqual(len(param_cube.coords('surface_altitude')), 1)


if __name__ == "__main__":
    tests.main()
