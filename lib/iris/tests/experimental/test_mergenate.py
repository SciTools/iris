# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the mergenation of cubes within iris.

"""

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests  # isort:skip

import numpy as np

from iris.coords import AncillaryVariable, AuxCoord, CellMeasure, DimCoord
from iris.cube import CubeList
from iris.exceptions import MergeError
from iris.experimental.mergenate import mergenate
from iris.tests import stock
from iris.util import promote_aux_coord_to_dim_coord


class TestBasics(tests.IrisTest):

    # Merge
    def test_merge(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, "wibble")

        self.assertEqual(expected, result)

    # Concatenate
    def test_concatenate(self):
        expected = stock.simple_3d()

        cube_0 = expected[:1]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, "wibble")

        self.assertEqual(expected, result)

    # Combo
    def test_combination(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, "wibble")

        self.assertEqual(expected, result)

    # More than 2 pieces
    def test_three_pieces(self):
        # Invert to make latitude axis ascending, as ascending is default
        expected = stock.simple_3d()[:, ::-1]

        cube_0 = expected[:, 0]
        cube_1 = expected[:, 1]
        cube_2 = expected[:, 2:]
        cubelist = CubeList([cube_0, cube_1, cube_2])
        result = mergenate(cubelist, "latitude")

        self.assertEqual(expected, result)

    # Disordered pieces
    def test_disordered_pieces(self):
        expected = stock.realistic_3d()

        cube_0 = expected[0]
        cube_1 = expected[1]
        cube_2 = expected[2:]
        cubelist = CubeList([cube_0, cube_2, cube_1])
        result = mergenate(cubelist, "time", extend_coords=True)

        self.assertEqual(expected, result)

    # Test descending merge (which assumes ascending as it lacks other info)
    def test_descending_merge(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1]
        cubelist = CubeList([cube_1, cube_0])
        result = mergenate(cubelist, "wibble")

        self.assertEqual(expected, result)

    # Test descending concatenate
    def test_descending_concatenate(self):
        expected = stock.realistic_3d()[::-1]

        cube_0 = expected[:3]
        cube_1 = expected[3:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, "time")

        self.assertEqual(expected, result)

    # Test descending combo
    def test_descending_combo(self):
        expected = stock.realistic_3d()[::-1]

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, "time", extend_coords=True)

        self.assertEqual(expected, result)

    # Anonymous dim
    def test_anon_dim(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, extend_coords=True)

        promote_aux_coord_to_dim_coord(result, "wibble")

        self.assertEqual(expected, result)

    # Other dim coord
    def test_non_leading_dim(self):
        expected = stock.simple_3d()

        cube_0 = expected[:, 0]
        cube_1 = expected[:, 1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, "latitude")

        self.assertEqual(expected, result)

    # Multiple coords
    def test_multiple_coords(self):
        expected = stock.simple_3d()

        expected.coord("longitude").circular = False

        cube_0 = expected[:, 0, 0]
        cube_1 = expected[:, 0, 1:]
        cube_2 = expected[:, 1:, 0]
        cube_3 = expected[:, 1:, 1:]
        cubelist = CubeList([cube_0, cube_2, cube_3, cube_1])

        result1 = mergenate(cubelist, ["latitude", "longitude"])
        self.assertEqual(expected, result1)

        result2 = mergenate(cubelist, ["longitude", "latitude"])
        self.assertEqual(expected, result2)


class TestArgs(tests.IrisTest):
    # Coord object
    def test_object(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, expected.coord("wibble"))

        self.assertEqual(expected, result)

    # String list
    def test_string_list(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, ["wibble"])

        self.assertEqual(expected, result)

    # Coord object list
    def test_object_list(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, [expected.coord("wibble")])

        self.assertEqual(expected, result)


class TestAuxCoords(tests.IrisTest):
    def run_test(self, expected, coord_name="wibble"):
        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])
        result = mergenate(cubelist, coord_name, extend_coords=True)

        print(expected)
        for coord in result.coords():
            print(coord)

        self.assertEqual(expected, result)

    # AuxCoords
    def test_aux_coord(self):
        test_cube = stock.simple_3d()
        extra_coord = AuxCoord(
            [20, 40],
            long_name="foo",
        )
        test_cube.add_aux_coord(
            extra_coord,
            0,
        )

        self.run_test(test_cube)

    # AuxCoord that's a DimCoord
    def test_aux_dim_coord(self):
        test_cube = stock.simple_3d()
        extra_coord = DimCoord(
            [20, 40],
            long_name="foo",
        )
        test_cube.add_aux_coord(
            extra_coord,
            0,
        )

        self.run_test(test_cube)

    # CellMeasures
    def test_cell_measure(self):
        test_cube = stock.simple_3d()
        extra_coord = CellMeasure(
            [20, 40],
            long_name="foo",
        )
        test_cube.add_cell_measure(
            extra_coord,
            0,
        )

        self.run_test(test_cube)

    # AncillaryVariables
    def test_ancillary_variable(self):
        test_cube = stock.simple_3d()
        extra_coord = AncillaryVariable(
            [20, 40],
            long_name="foo",
        )
        test_cube.add_ancillary_variable(
            extra_coord,
            0,
        )

        self.run_test(test_cube)

    # AuxFactories
    def test_aux_factory(self):
        test_cube = stock.simple_4d_with_hybrid_height()

        self.run_test(test_cube, "time")


class TestCoordExtension(tests.IrisTest):

    # Extending each coord type
    # Here the dim coord is demoted to aux coord on extension because it's no
    # longer monotonic (or 1D)
    def test_extend_dim_coord(self):
        test_cube = stock.realistic_3d()

        cube_0 = test_cube[0]
        cube_1 = test_cube[1:2]
        cube_2 = test_cube[2:]

        coord_data = np.arange(test_cube.shape[1])
        coord_0 = DimCoord(
            coord_data,
            long_name="foo",
        )
        coord_1 = DimCoord(
            coord_data + 1,
            long_name="foo",
        )
        coord_2 = DimCoord(
            coord_data + 2,
            long_name="foo",
        )

        cube_0.add_aux_coord(coord_0, (0,))
        cube_1.add_aux_coord(coord_1, (1,))
        cube_2.add_aux_coord(coord_2, (1,))

        result_cube = mergenate(
            CubeList([cube_0, cube_1, cube_2]),
            coords="time",
            extend_coords=True,
        )

        expected_cross_section = np.array([0, 1, 2, 2, 2, 2, 2])
        expected_coord_data = (
            coord_data[np.newaxis, :] + expected_cross_section[:, np.newaxis]
        )
        expected_coord = AuxCoord(
            expected_coord_data,
            long_name="foo",
        )

        self.assertEqual(result_cube.coord("foo"), expected_coord)

    def test_extend_aux_coord(self):
        test_cube = stock.realistic_3d()

        cube_0 = test_cube[0]
        cube_1 = test_cube[1:2]
        cube_2 = test_cube[2:]

        coord_data = np.arange(test_cube.shape[1])
        coord_0 = AuxCoord(
            coord_data,
            long_name="foo",
        )
        coord_1 = AuxCoord(
            coord_data + 1,
            long_name="foo",
        )
        coord_2 = AuxCoord(
            coord_data + 2,
            long_name="foo",
        )

        cube_0.add_aux_coord(coord_0, (0,))
        cube_1.add_aux_coord(coord_1, (1,))
        cube_2.add_aux_coord(coord_2, (1,))

        result_cube = mergenate(
            CubeList([cube_0, cube_1, cube_2]),
            coords="time",
            extend_coords=True,
        )

        expected_cross_section = np.array([0, 1, 2, 2, 2, 2, 2])
        expected_coord_data = (
            coord_data[np.newaxis, :] + expected_cross_section[:, np.newaxis]
        )
        expected_coord = AuxCoord(
            expected_coord_data,
            long_name="foo",
        )

        self.assertEqual(result_cube.coord("foo"), expected_coord)

    def test_extend_ancillary_var(self):
        test_cube = stock.realistic_3d()

        cube_0 = test_cube[0]
        cube_1 = test_cube[1:2]
        cube_2 = test_cube[2:]

        coord_data = np.arange(test_cube.shape[1])
        coord_0 = AncillaryVariable(
            coord_data,
            long_name="foo",
        )
        coord_1 = AncillaryVariable(
            coord_data + 1,
            long_name="foo",
        )
        coord_2 = AncillaryVariable(
            coord_data + 2,
            long_name="foo",
        )

        cube_0.add_ancillary_variable(coord_0, (0,))
        cube_1.add_ancillary_variable(coord_1, (1,))
        cube_2.add_ancillary_variable(coord_2, (1,))

        result_cube = mergenate(
            CubeList([cube_0, cube_1, cube_2]),
            coords="time",
            extend_coords=True,
        )

        expected_cross_section = np.array([0, 1, 2, 2, 2, 2, 2])
        expected_coord_data = (
            coord_data[np.newaxis, :] + expected_cross_section[:, np.newaxis]
        )
        expected_coord = AncillaryVariable(
            expected_coord_data,
            long_name="foo",
        )

        self.assertEqual(result_cube.ancillary_variable("foo"), expected_coord)

    def test_extend_cell_measure(self):
        test_cube = stock.realistic_3d()

        cube_0 = test_cube[0]
        cube_1 = test_cube[1:2]
        cube_2 = test_cube[2:]

        coord_data = np.arange(test_cube.shape[1])
        coord_0 = CellMeasure(
            coord_data,
            long_name="foo",
        )
        coord_1 = CellMeasure(
            coord_data + 1,
            long_name="foo",
        )
        coord_2 = CellMeasure(
            coord_data + 2,
            long_name="foo",
        )

        cube_0.add_cell_measure(coord_0, (0,))
        cube_1.add_cell_measure(coord_1, (1,))
        cube_2.add_cell_measure(coord_2, (1,))

        result_cube = mergenate(
            CubeList([cube_0, cube_1, cube_2]),
            coords="time",
            extend_coords=True,
        )

        expected_cross_section = np.array([0, 1, 2, 2, 2, 2, 2])
        expected_coord_data = (
            coord_data[np.newaxis, :] + expected_cross_section[:, np.newaxis]
        )
        expected_coord = CellMeasure(
            expected_coord_data,
            long_name="foo",
        )

        self.assertEqual(result_cube.cell_measure("foo"), expected_coord)

    # On different dims
    def test_extend_lead_dim(self):
        test_cube = stock.realistic_3d()

        cube_0 = test_cube[:, 0]
        cube_1 = test_cube[:, 1:2]
        cube_2 = test_cube[:, 2:]

        coord_data = np.arange(test_cube.shape[0])
        coord_0 = AuxCoord(
            coord_data,
            long_name="foo",
        )
        coord_1 = AuxCoord(
            coord_data + 1,
            long_name="foo",
        )
        coord_2 = AuxCoord(
            coord_data + 2,
            long_name="foo",
        )

        cube_0.add_aux_coord(coord_0, (0,))
        cube_1.add_aux_coord(coord_1, (0,))
        cube_2.add_aux_coord(coord_2, (0,))

        result_cube = mergenate(
            CubeList([cube_0, cube_1, cube_2]),
            coords="grid_latitude",
            extend_coords=True,
        )

        expected_cross_section = np.array([0, 1, 2, 2, 2, 2, 2, 2, 2])
        expected_coord_data = (
            coord_data[np.newaxis, :] + expected_cross_section[:, np.newaxis]
        )
        expected_coord = AuxCoord(
            expected_coord_data,
            long_name="foo",
        )

        self.assertEqual(result_cube.coord("foo"), expected_coord)

    # Extend 2D to 3D
    def test_extend_2D_to_3D(self):
        test_cube = stock.realistic_3d()

        cube_0 = test_cube[0]
        cube_1 = test_cube[1:2]
        cube_2 = test_cube[2:]

        coord_data = np.arange(
            test_cube.shape[1] * test_cube.shape[2]
        ).reshape(test_cube.shape[1:3])
        coord_0 = AuxCoord(
            coord_data,
            long_name="foo",
        )
        coord_1 = AuxCoord(
            coord_data + 1,
            long_name="foo",
        )
        coord_2 = AuxCoord(
            coord_data + 2,
            long_name="foo",
        )

        cube_0.add_aux_coord(coord_0, (0, 1))
        cube_1.add_aux_coord(coord_1, (1, 2))
        cube_2.add_aux_coord(coord_2, (1, 2))

        result_cube = mergenate(
            CubeList([cube_0, cube_1, cube_2]),
            coords="time",
            extend_coords=True,
        )

        expected_cross_section = np.array([0, 1, 2, 2, 2, 2, 2])
        expected_coord_data = (
            coord_data[np.newaxis, :]
            + expected_cross_section[:, np.newaxis, np.newaxis]
        )
        expected_coord = AuxCoord(
            expected_coord_data,
            long_name="foo",
        )

        self.assertEqual(result_cube.coord("foo"), expected_coord)


class TestErrors(tests.IrisTest):
    def test_extra_coord_in_first(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])

        spare_coord = AuxCoord(
            [0],
            long_name="foo",
        )
        cube_0.add_aux_coord(spare_coord, ())

        msg = "foo in cube 0 doesn't match any coord in cube 1"
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(cubelist, "wibble")

    def test_extra_coord_in_second(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])

        spare_coord = AuxCoord(
            [0],
            long_name="foo",
        )
        cube_1.add_aux_coord(spare_coord, ())

        msg = "foo in cube 1 doesn't match any coord in cube 0"
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(cubelist[::-1], "wibble")

    def test_metadata_conflict_cube(self):
        expected = stock.simple_3d()

        cube_0 = expected[0]
        cube_1 = expected[1:]
        cubelist = CubeList([cube_0, cube_1])

        cube_0.attributes["foo"] = "bar"
        cube_1.attributes["foo"] = "barbar"

        msg = "Inconsistent metadata between merging cubes.\nDifference is CubeMetadata\(attributes=\(\{'foo': 'bar'\}, \{'foo': 'barbar'\}\)\)"  # noqa: W605
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(cubelist, "wibble")

    def test_cant_extend_aux_coord(self):
        test_cube = stock.simple_3d()

        cube_0 = test_cube[0]
        cube_1 = test_cube[1:]

        coord_data = np.arange(test_cube.shape[1])
        coord_0 = AuxCoord(
            coord_data,
            long_name="foo",
        )
        coord_1 = AuxCoord(
            coord_data + 1,
            long_name="foo",
        )

        cube_0.add_aux_coord(coord_0, (0,))
        cube_1.add_aux_coord(coord_1, (1,))

        msg = (
            "Different points or bounds in foo coords, but not allowed to "
            "extend coords. Consider trying again with extend_coords=True"
        )
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(CubeList([cube_0, cube_1]), coords="wibble")

    def test_ascending_descending(self):
        test_cube = stock.simple_3d()

        cube_0 = test_cube[:, :, 0:2]
        cube_1 = test_cube[:, :, 2:4]
        cube_1 = cube_1[:, :, ::-1]

        msg = "Mixture of ascending and descending coordinate points"
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(CubeList([cube_0, cube_1]), "longitude")

    def test_ambiguous_order(self):
        test_cube = stock.simple_3d()

        cube_0 = test_cube[:, :, 0:2]
        cube_1 = test_cube[:, :, 1:4]

        msg = "Coordinate points overlap so correct merge order is ambiguous"
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(CubeList([cube_0, cube_1]), "longitude")

        cube_2 = test_cube[:, :, ::2]
        cube_3 = test_cube[:, :, 1::2]

        msg = "Coordinate points overlap so correct merge order is ambiguous"
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(CubeList([cube_2, cube_3]), "longitude")

    def test_different_axes(self):
        test_cube = stock.simple_3d()

        cube_0 = test_cube[0:1]
        cube_1 = test_cube[1:]
        cube_1.transpose((1, 0, 2))

        msg = "Coord lies on different axes on different cubes"
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(CubeList([cube_0, cube_1]), "wibble")

    def test_inconsistent_shapes(self):
        test_cube = stock.simple_3d()

        cube_0 = test_cube[0:1]
        cube_1 = test_cube[1:, 1:]

        msg = (
            "The shapes of cubes to be concatenated can only differ on the "
            "affected dimensions"
        )
        with self.assertRaisesRegex(MergeError, msg):
            mergenate(CubeList([cube_0, cube_1]), "wibble")

        # TODO: Test for the following errors:
        # Inconsistent aux factories (I can't work out how to make these)
        # Cube datas that can't be concatenated (but pass shape test)
        # Unknown type of non-dimensional coordinate
        # Can only merge on 1D or 0D coordinates


if __name__ == "__main__":
    tests.main()
