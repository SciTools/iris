# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for :class:`iris._representation.cube_printout.Table`."""
import iris.tests as tests

from iris._representation.cube_printout import Table


class TestTable(tests.IrisTest):
    # Note: this is just barely an independent definition, not *strictly* part
    # of CubePrinter, but effectively more-or-less so.
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


if __name__ == "__main__":
    tests.main()
