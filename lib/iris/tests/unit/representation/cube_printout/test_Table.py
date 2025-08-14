# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris._representation.cube_printout.Table`."""

import pytest

from iris._representation.cube_printout import Table


class TestTable:
    # Note: this is just barely an independent definition, not *strictly* part
    # of CubePrinter, but effectively more-or-less so.
    @pytest.fixture(autouse=True)
    def _setup(self):
        table = Table()
        table.add_row(["one", "b", "three"], aligns=["left", "right", "left"])
        table.add_row(["a", "two", "c"], aligns=["right", "left", "right"])
        self.simple_table = table

    def test_empty(self):
        table = Table()
        assert table.n_columns is None
        assert len(table.rows) == 0
        assert table.col_widths is None
        # Check other methods : should be ok but do nothing.
        table.set_min_column_widths()  # Ok but does nothing.
        assert table.col_widths is None
        assert table.formatted_as_strings() == []
        assert str(table) == ""

    def test_basic_content(self):
        # Mirror the above 'empty' tests on a small basic table.
        table = self.simple_table
        assert table.n_columns == 3
        assert len(table.rows) == 2
        assert table.col_widths is None
        table.set_min_column_widths()  # Ok but does nothing.
        assert table.col_widths == [3, 3, 5]
        assert table.formatted_as_strings() == ["one   b three", "  a two     c"]
        assert str(table) == "one   b three\n  a two     c"

    def test_copy(self):
        table = self.simple_table
        # Add some detail information
        table.rows[1].i_col_unlimited = 77  # Doesn't actually affect anything
        table.col_widths = [10, 15, 12]
        # Make the copy
        table2 = table.copy()
        assert table2 is not table
        assert table2 != table  # Note: equality is not implemented
        # Check the parts match the original.
        assert len(table2.rows) == len(table.rows)
        for row2, row in zip(table2.rows, table.rows):
            assert row2.cols == row.cols
            assert row2.aligns == row.aligns
            assert row2.i_col_unlimited == row.i_col_unlimited

    def test_add_row(self):
        table = Table()
        assert table.n_columns is None
        # Add one row.
        table.add_row(["one", "two", "three"], aligns=["left", "left", "left"])
        assert len(table.rows) == 1
        assert table.n_columns == 3
        assert table.rows[0].i_col_unlimited is None
        # Second row ok.
        table.add_row(
            ["x", "y", "z"],
            aligns=["right", "right", "right"],
            i_col_unlimited=199,
        )
        assert len(table.rows) == 2
        assert table.rows[-1].i_col_unlimited == 199

        # Fails with bad number of columns
        regex = "columns.*!=.*existing"
        with pytest.raises(ValueError, match=regex):
            table.add_row(["1", "2"], ["left", "right"])

        # Fails with bad number of aligns
        regex = "aligns.*!=.*col"
        with pytest.raises(ValueError, match=regex):
            table.add_row(["1", "2", "3"], ["left", "left", "left", "left"])

    def test_formatted_as_strings(self):
        # Test simple self-print is same as
        table = Table()
        aligns = ["left", "right", "left"]
        table.add_row(["1", "266", "32"], aligns)
        table.add_row(["123", "2", "3"], aligns)

        # Check that printing calculates default column widths, and result..
        assert table.col_widths is None
        result = table.formatted_as_strings()
        assert result == ["1   266 32", "123   2 3"]
        assert table.col_widths == [3, 3, 2]

    def test_fail_bad_alignments(self):
        # Invalid 'aligns' content : only detected when printed
        table = Table()
        table.add_row(["1", "2", "3"], ["left", "right", "BAD"])
        regex = 'Unknown alignment "BAD"'
        with pytest.raises(ValueError, match=regex):
            str(table)

    def test_table_set_width(self):
        # Check that changes do *not* affect pre-existing widths.
        table = Table()
        aligns = ["left", "right", "left"]
        table.col_widths = [3, 3, 2]
        table.add_row(["333", "333", "22"], aligns)
        table.add_row(["a", "b", "c"], aligns)
        table.add_row(["12345", "12345", "12345"], aligns)
        result = table.formatted_as_strings()
        assert table.col_widths == [3, 3, 2]

        # These are exceeding the given widths.
        assert result == [
            "333 333 22",
            "a     b c",
            "12345 12345 12345",
        ]

    def test_unlimited_column(self):
        table = Table()
        aligns = ["left", "right", "left"]
        table.add_row(["a", "beee", "c"], aligns)
        table.add_row(["abcd", "any-longer-stuff", "this"], aligns, i_col_unlimited=1)
        table.add_row(["12", "x", "yy"], aligns)
        result = table.formatted_as_strings()

        # NOTE: the widths-calc is ignoring cols 1-2, but
        # entry#0 *is* extending the width of col#0
        assert result == [
            "a    beee c",
            "abcd any-longer-stuff this",
            "12      x yy",
        ]

    def test_str(self):
        # Check that str returns the formatted_as_strings() output.
        table = Table()
        aligns = ["left", "left", "left"]
        table.add_row(["one", "two", "three"], aligns=aligns)
        table.add_row(["1", "2", "3"], aligns=aligns)
        expected = "\n".join(table.formatted_as_strings())
        result = str(table)
        assert result == expected
