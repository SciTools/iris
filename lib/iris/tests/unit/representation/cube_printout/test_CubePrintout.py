# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :mod:`iris._representation.cube_summary` module."""

import iris.tests as tests

import numpy as np
import iris
from iris.cube import Cube
from iris.coords import AuxCoord, DimCoord
from iris._representation.cube_summary import CubeSummary
import iris.tests.stock as istk
from iris.util import new_axis

from iris._representation.cube_printout import CubePrinter, Table


class TestTable(tests.IrisTest):
    def setUp(self):
        table = Table()
        table.add_row(["one", "b", "three"], aligns=["left", "right", "left"])
        table.add_row(["a", "two", "c"], aligns=["right", "left", "right"])
        self.simple_table = table

    def test_empty(self):
        table = Table()
        self.assertIsNone(table.n_columns)
        self.assertEqual(len(table.rows), 0)
        self.assertIsNone(table.col_widths)
        # Check other methods : should be ok but do nothing.
        table.set_min_column_widths()  # Ok but does nothing.
        self.assertIsNone(table.col_widths)
        self.assertEqual(table.formatted_as_strings(), [])
        self.assertEqual(str(table), "")

    def test_basic_content(self):
        # Mirror the above 'empty' tests on a small basic table.
        table = self.simple_table
        self.assertEqual(table.n_columns, 3)
        self.assertEqual(len(table.rows), 2)
        self.assertIsNone(table.col_widths)
        table.set_min_column_widths()  # Ok but does nothing.
        self.assertEqual(table.col_widths, [3, 3, 5])
        self.assertEqual(
            table.formatted_as_strings(), ["one   b three", "  a two     c"]
        )
        self.assertEqual(str(table), "one   b three\n  a two     c")

    def test_copy(self):
        table = self.simple_table
        # Add some detail information
        table.rows[1].i_col_unlimited = 77  # Doesn't actually affect anything
        table.col_widths = [10, 15, 12]
        # Make the copy
        table2 = table.copy()
        self.assertIsNot(table2, table)
        self.assertNotEqual(table2, table)  # Note: equality is not implemented
        # Check the parts match the original.
        self.assertEqual(len(table2.rows), len(table.rows))
        for row2, row in zip(table2.rows, table.rows):
            self.assertEqual(row2.cols, row.cols)
            self.assertEqual(row2.aligns, row.aligns)
            self.assertEqual(row2.i_col_unlimited, row.i_col_unlimited)

    def test_add_row(self):
        table = Table()
        self.assertEqual(table.n_columns, None)
        # Add onw row.
        table.add_row(["one", "two", "three"], aligns=["left", "left", "left"])
        self.assertEqual(len(table.rows), 1)
        self.assertEqual(table.n_columns, 3)
        self.assertIsNone(table.rows[0].i_col_unlimited)
        # Second row ok.
        table.add_row(
            ["x", "y", "z"],
            aligns=["right", "right", "right"],
            i_col_unlimited=199,
        )
        self.assertEqual(len(table.rows), 2)
        self.assertEqual(table.rows[-1].i_col_unlimited, 199)
        # Fails with bad number of columns
        with self.assertRaisesRegex(ValueError, "columns.*!=.*existing"):
            table.add_row(["one"], ["left"])
        # Fails with bad number of aligns
        with self.assertRaisesRegex(ValueError, "aligns.*!=.*col"):
            table.add_row(["one", "two"], ["left"])


def example_cube(n_extra_dims=0):
    cube = istk.realistic_4d()  # this one has a derived coord

    # Optionally : add multiple extra dimensions to test the width controls
    if n_extra_dims > 0:

        new_cube = cube.copy()
        # Add n extra scalar *1 coords
        for i_dim in range(n_extra_dims):
            dim_name = "long_name_dim_{}".format(i_dim + cube.ndim)
            dimco = DimCoord([0], long_name=dim_name)
            new_cube.add_aux_coord(dimco)
            # Promote to dim coord
            new_cube = new_axis(new_cube, dim_name)

        # Put them all at the back
        dim_order = list(range(new_cube.ndim))
        dim_order = dim_order[n_extra_dims:] + dim_order[:n_extra_dims]
        new_cube.transpose(dim_order)  # dontcha hate this inplace way ??

        # Replace the original test cube
        cube = new_cube

    # Add extra things to test all aspects
    rotlats_1d, rotlons_1d = (
        cube.coord("grid_latitude").points,
        cube.coord("grid_longitude").points,
    )
    rotlons_2d, rotlats_2d = np.meshgrid(rotlons_1d, rotlats_1d)

    cs = cube.coord_system()
    trulons, trulats = iris.analysis.cartography.unrotate_pole(
        rotlons_2d,
        rotlats_2d,
        cs.grid_north_pole_longitude,
        cs.grid_north_pole_latitude,
    )
    co_lat, co_lon = cube.coord(axis="y"), cube.coord(axis="x")
    latlon_dims = cube.coord_dims(co_lat) + cube.coord_dims(co_lon)
    cube.add_aux_coord(
        AuxCoord(trulons, standard_name="longitude", units="degrees"),
        latlon_dims,
    )
    cube.add_aux_coord(
        AuxCoord(trulats, standard_name="latitude", units="degrees"),
        latlon_dims,
    )

    cube.attributes[
        "history"
    ] = "Exceedingly and annoying long message with many sentences.  And more and more.  And more and more."

    cube.add_cell_method(iris.coords.CellMethod("mean", ["time"]))
    cube.add_cell_method(
        iris.coords.CellMethod(
            "max", ["latitude"], intervals="3 hour", comments="remark"
        )
    )
    latlons_shape = [cube.shape[i_dim] for i_dim in latlon_dims]
    cube.add_cell_measure(
        iris.coords.CellMeasure(
            np.zeros(latlons_shape), long_name="cell-timings", units="s"
        ),
        latlon_dims,
    )
    cube.add_cell_measure(
        iris.coords.CellMeasure(
            [4.3], long_name="whole_cell_factor", units="m^2"
        ),
        (),
    )  # a SCALAR cell-measure

    time_dim = cube.coord_dims(cube.coord(axis="t"))
    cube.add_ancillary_variable(
        iris.coords.AncillaryVariable(
            np.zeros(cube.shape[0]), long_name="time_scalings", units="ppm"
        ),
        time_dim,
    )
    cube.add_ancillary_variable(
        iris.coords.AncillaryVariable(
            [3.2], long_name="whole_cube_area_factor", units="m^2"
        ),
        (),
    )  # a SCALAR ancillary

    # Add some duplicate-named coords (not possible for dim-coords)
    vector_duplicate_name = "level_height"
    co_orig = cube.coord(vector_duplicate_name)
    dim_orig = cube.coord_dims(co_orig)
    co_new = co_orig.copy()
    co_new.attributes.update(dict(a=1, b=2))
    cube.add_aux_coord(co_new, dim_orig)

    vector_different_name = "sigma"
    co_orig = cube.coord(vector_different_name)
    co_orig.attributes["setting"] = "a"
    dim_orig = cube.coord_dims(co_orig)
    co_new = co_orig.copy()
    co_new.attributes["setting"] = "B"
    cube.add_aux_coord(co_new, dim_orig)

    # Also need to test this with a SCALAR coord
    scalar_duplicate_name = "forecast_period"
    co_orig = cube.coord(scalar_duplicate_name)
    co_new = co_orig.copy()
    co_new.points = co_new.points + 2.3
    co_new.attributes["different"] = "True"
    cube.add_aux_coord(co_new)

    # Add a scalar coord with a *really* long name, to challenge the column width formatting
    long_name = "long_long_long_long_long_long_long_long_long_long_long_name"
    cube.add_aux_coord(DimCoord([0], long_name=long_name))
    return cube


def cube_repr(cube, **kwargs):
    return CubePrinter(CubeSummary(cube)).to_string(**kwargs)


class TestCubePrintout__cubefeatures_summaries(tests.IrisTest):
    def test_empty(self):
        cube = Cube([0])
        repr = cube_repr(cube)
        self.assertEqual(repr, "unknown / (unknown)                 (-- : 1)")
        repr = cube_repr(cube, oneline=True)
        self.assertEqual(repr, "unknown / (unknown) (-- : 1)")

    def test_scalar_cube(self):
        cube = Cube(0)
        repr = cube_repr(cube)
        self.assertEqual(
            repr, "unknown / (unknown)                 (scalar cube)"
        )
        repr = cube_repr(cube, oneline=True)
        self.assertEqual(repr, "unknown / (unknown) (scalar cube)")

    def test_name_padding(self):
        cube = Cube([1, 2], long_name="cube_accel", units="ms-2")
        repr = cube_repr(cube)
        self.assertEqual(repr, "cube_accel / (ms-2)                 (-- : 2)")
        repr = cube_repr(cube, name_padding=0)
        self.assertEqual(repr, "cube_accel / (ms-2) (-- : 2)")
        repr = cube_repr(cube, name_padding=25)
        self.assertEqual(repr, "cube_accel / (ms-2)       (-- : 2)")

    def test_long_coordname_columns(self):
        cube = Cube([0], long_name="short", units=1)
        coord = AuxCoord(
            [0], long_name="very_very_very_very_very_long_coord_name"
        )
        cube.add_aux_coord(coord, 0)
        repr = cube_repr(cube)
        expected = (
            "short / (1)                                      (-- : 1)\n"
            "    Auxiliary coordinates:\n"
            "        very_very_very_very_very_long_coord_name       x"
        )
        self.assertEqual(repr, expected)
        repr = cube_repr(cube, oneline=True)
        self.assertEqual(repr, "short / (1) (-- : 1)")

    def test_long_attribute_columns(self):
        cube = Cube([0], long_name="short", units=1)
        cube.attributes[
            "very_very_very_very_very_long_name"
        ] = "longish string extends beyond dim columns"
        repr = cube_repr(cube)
        expected = (
            "short / (1)                                (-- : 1)\n"
            "    Attributes:\n"
            "        very_very_very_very_very_long_name "
            "longish string extends beyond dim columns"
        )
        self.assertEqual(repr, expected)


class TestCubePrintout__integration_examples(tests.IrisTest):
    def _exercise_methods(self, cube):
        summ = CubeSummary(cube)
        printer = CubePrinter(summ)
        has_scalar_ancils = any(
            len(anc.cube_dims(cube)) == 0 for anc in cube.ancillary_variables()
        )
        unprintable = has_scalar_ancils and cube.ndim == 0
        print("EXISTING full :")
        if unprintable:
            print("  ( would fail, due to scalar-cube with scalar-ancils )")
        else:
            print(cube)
        print("---full--- :")
        print(printer.to_string())
        print("")
        print("EXISTING oneline :")
        print(repr(cube))
        print("---oneline--- :")
        print(printer.to_string(oneline=True))
        print("")
        print("original table form:")
        tb = printer.table
        print(tb)
        print("")
        print("")

    def test_basic(self):
        cube = example_cube(
            n_extra_dims=4
        )  # NB does not yet work with factories.
        self._exercise_methods(cube)

    def test_scalar_cube(self):
        cube = example_cube()[0, 0, 0, 0]
        self._exercise_methods(cube)


if __name__ == "__main__":
    tests.main()
