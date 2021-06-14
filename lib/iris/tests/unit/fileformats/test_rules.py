# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test iris.fileformats.rules.py - metadata translation rules.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import types
from unittest import mock

import numpy as np

from iris.aux_factory import HybridHeightFactory
from iris.coords import CellMethod
from iris.cube import Cube
from iris.fileformats.rules import (
    ConcreteReferenceTarget,
    ConversionMetadata,
    Factory,
    Loader,
    Reference,
    ReferenceTarget,
    load_cubes,
    scalar_cell_method,
)
import iris.tests.stock as stock


class TestConcreteReferenceTarget(tests.IrisTest):
    def test_attributes(self):
        with self.assertRaises(TypeError):
            target = ConcreteReferenceTarget()

        target = ConcreteReferenceTarget("foo")
        self.assertEqual(target.name, "foo")
        self.assertIsNone(target.transform)

        def transform(_):
            return _

        target = ConcreteReferenceTarget("foo", transform)
        self.assertEqual(target.name, "foo")
        self.assertIs(target.transform, transform)

    def test_single_cube_no_transform(self):
        target = ConcreteReferenceTarget("foo")
        src = stock.simple_2d()
        target.add_cube(src)
        self.assertIs(target.as_cube(), src)

    def test_single_cube_with_transform(self):
        def transform(cube):
            return {"long_name": "wibble"}

        target = ConcreteReferenceTarget("foo", transform)
        src = stock.simple_2d()
        target.add_cube(src)
        dest = target.as_cube()
        self.assertEqual(dest.long_name, "wibble")
        self.assertNotEqual(dest, src)
        dest.long_name = src.long_name
        self.assertEqual(dest, src)

    @tests.skip_data
    def test_multiple_cubes_no_transform(self):
        target = ConcreteReferenceTarget("foo")
        src = stock.realistic_4d()
        for i in range(src.shape[0]):
            target.add_cube(src[i])
        dest = target.as_cube()
        self.assertIsNot(dest, src)
        self.assertEqual(dest, src)

    @tests.skip_data
    def test_multiple_cubes_with_transform(self):
        def transform(cube):
            return {"long_name": "wibble"}

        target = ConcreteReferenceTarget("foo", transform)
        src = stock.realistic_4d()
        for i in range(src.shape[0]):
            target.add_cube(src[i])
        dest = target.as_cube()
        self.assertEqual(dest.long_name, "wibble")
        self.assertNotEqual(dest, src)
        dest.long_name = src.long_name
        self.assertEqual(dest, src)


class TestLoadCubes(tests.IrisTest):
    def test_simple_factory(self):
        # Test the creation process for a factory definition which only
        # uses simple dict arguments.

        # Make a minimal fake data object that passes as lazy data.
        core_data_array = mock.Mock(compute=None, dtype=np.dtype("f4"))
        # Make a fake PPField which will be supplied to our converter.
        field = mock.Mock(
            core_data=mock.Mock(return_value=core_data_array),
            realised_dtype=np.dtype("f4"),
            bmdi=None,
        )

        def field_generator(filename):
            return [field]

        # A fake conversion function returning:
        #   1) A parameter cube needing a simple factory construction.
        aux_factory = mock.Mock()
        factory = mock.Mock()
        factory.args = [{"name": "foo"}]
        factory.factory_class = (
            lambda *args: setattr(aux_factory, "fake_args", args)
            or aux_factory
        )

        def converter(field):
            return ConversionMetadata(
                [factory], [], "", "", "", {}, [], [], []
            )

        # Finish by making a fake Loader
        fake_loader = Loader(field_generator, {}, converter)
        cubes = load_cubes(["fake_filename"], None, fake_loader)

        # Check the result is a generator with a single entry.
        self.assertIsInstance(cubes, types.GeneratorType)
        try:
            # Suppress the normal Cube.coord() and Cube.add_aux_factory()
            # methods.
            coord_method = Cube.coord
            add_aux_factory_method = Cube.add_aux_factory
            Cube.coord = lambda self, **args: args
            Cube.add_aux_factory = lambda self, aux_factory: setattr(
                self, "fake_aux_factory", aux_factory
            )

            cubes = list(cubes)
        finally:
            Cube.coord = coord_method
            Cube.add_aux_factory = add_aux_factory_method
        self.assertEqual(len(cubes), 1)
        # Check the "cube" has an "aux_factory" added, which itself
        # must have been created with the correct arguments.
        self.assertTrue(hasattr(cubes[0], "fake_aux_factory"))
        self.assertIs(cubes[0].fake_aux_factory, aux_factory)
        self.assertTrue(hasattr(aux_factory, "fake_args"))
        self.assertEqual(aux_factory.fake_args, ({"name": "foo"},))

    @tests.skip_data
    def test_cross_reference(self):
        # Test the creation process for a factory definition which uses
        # a cross-reference.

        param_cube = stock.realistic_4d_no_derived()
        orog_coord = param_cube.coord("surface_altitude")
        param_cube.remove_coord(orog_coord)

        orog_cube = param_cube[0, 0, :, :]
        orog_cube.data = orog_coord.points
        orog_cube.rename("surface_altitude")
        orog_cube.units = orog_coord.units
        orog_cube.attributes = orog_coord.attributes

        # We're going to test for the presence of the hybrid height
        # stuff later, so let's make sure it's not already there!
        assert len(param_cube.aux_factories) == 0
        assert not param_cube.coords("surface_altitude")

        # The fake PPFields which will be supplied to our converter.
        press_field = mock.Mock(
            core_data=mock.Mock(return_value=param_cube.data),
            bmdi=-1e20,
            realised_dtype=param_cube.dtype,
        )

        orog_field = mock.Mock(
            core_data=mock.Mock(return_value=orog_cube.data),
            bmdi=-1e20,
            realised_dtype=orog_cube.dtype,
        )

        def field_generator(filename):
            return [press_field, orog_field]

        # A fake rule set returning:
        #   1) A parameter cube needing an "orography" reference
        #   2) An "orography" cube

        def converter(field):
            if field is press_field:
                src = param_cube
                factories = [
                    Factory(HybridHeightFactory, [Reference("orography")])
                ]
                references = []
            else:
                src = orog_cube
                factories = []
                references = [ReferenceTarget("orography", None)]
            dim_coords_and_dims = [
                (coord, src.coord_dims(coord)[0]) for coord in src.dim_coords
            ]
            aux_coords_and_dims = [
                (coord, src.coord_dims(coord)) for coord in src.aux_coords
            ]
            return ConversionMetadata(
                factories,
                references,
                src.standard_name,
                src.long_name,
                src.units,
                src.attributes,
                src.cell_methods,
                dim_coords_and_dims,
                aux_coords_and_dims,
            )

        # Finish by making a fake Loader
        fake_loader = Loader(field_generator, {}, converter)
        cubes = load_cubes(["fake_filename"], None, fake_loader)

        # Check the result is a generator containing two Cubes.
        self.assertIsInstance(cubes, types.GeneratorType)
        cubes = list(cubes)
        self.assertEqual(len(cubes), 2)
        # Check the "cube" has an "aux_factory" added, which itself
        # must have been created with the correct arguments.
        self.assertEqual(len(cubes[1].aux_factories), 1)
        self.assertEqual(len(cubes[1].coords("surface_altitude")), 1)


class Test_scalar_cell_method(tests.IrisTest):
    """Tests for iris.fileformats.rules.scalar_cell_method() function"""

    def setUp(self):
        self.cube = stock.simple_2d()
        self.cm = CellMethod("mean", "foo", "1 hour")
        self.cube.cell_methods = (self.cm,)

    def test_cell_method_found(self):
        actual = scalar_cell_method(self.cube, "mean", "foo")
        self.assertEqual(actual, self.cm)

    def test_method_different(self):
        actual = scalar_cell_method(self.cube, "average", "foo")
        self.assertIsNone(actual)

    def test_coord_name_different(self):
        actual = scalar_cell_method(self.cube, "average", "bar")
        self.assertIsNone(actual)

    def test_double_coord_fails(self):
        self.cube.cell_methods = (
            CellMethod("mean", ("foo", "bar"), ("1 hour", "1 hour")),
        )
        actual = scalar_cell_method(self.cube, "mean", "foo")
        self.assertIsNone(actual)


if __name__ == "__main__":
    tests.main()
