# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides text printouts of Iris cubes.

"""
from copy import deepcopy

from iris._representation.cube_summary import CubeSummary


class Table:
    """
    A container of text strings in rows + columns, that can format its content
    into a string per row, with contents in columns of fixed width.

    Supports left- or right- aligned columns, alignment being set "per row".
    A column may also be set, beyond which output is printed without further
    formatting, and without affecting any subsequent column widths.
    This is used as a crude alternative to column spanning.

    """

    def __init__(self, rows=None, col_widths=None):
        if rows is None:
            rows = []
        self.rows = [deepcopy(row) for row in rows]
        self.col_widths = col_widths

    def copy(self):
        return Table(self.rows, col_widths=self.col_widths)

    @property
    def n_columns(self):
        if self.rows:
            result = len(self.rows[0].cols)
        else:
            result = None
        return result

    class Row:
        """A set of column info, plus per-row formatting controls."""

        def __init__(self, cols, aligns, i_col_unlimited=None):
            assert len(cols) == len(aligns)
            self.cols = cols
            self.aligns = aligns
            self.i_col_unlimited = i_col_unlimited
            # This col + those after do not add to width
            # - a crude alternative to proper column spanning

    def add_row(self, cols, aligns, i_col_unlimited=None):
        """
        Create a new row at the bottom.

        Args:
        * cols (list of string):
            Per-column content.  Length must match the other rows (if any).
        * aligns (list of {'left', 'right'}):
            Per-column alignments.  Length must match 'cols'.
        * i_col_unlimited (int or None):
            Column beyond which content does not affect the column widths.
            ( meaning contents will print without limit ).

        """
        n_cols = len(cols)
        if len(aligns) != n_cols:
            msg = (
                f"Number of aligns ({len(aligns)})"
                f" != number of cols ({n_cols})"
            )
            raise ValueError(msg)
        if self.n_columns is not None:
            # For now, all rows must have same number of columns
            if n_cols != self.n_columns:
                msg = (
                    f"Number of columns ({n_cols})"
                    f" != existing table.n_columns ({self.n_columns})"
                )
                raise ValueError(msg)
        row = self.Row(cols, aligns, i_col_unlimited)
        self.rows.append(row)

    def set_min_column_widths(self):
        """Set all column widths to minimum required for current content."""
        if self.rows:
            widths = [0] * self.n_columns
            for row in self.rows:
                cols, lim = row.cols, row.i_col_unlimited
                if lim is not None:
                    cols = cols[:lim]  # Ignore "unlimited" columns
                for i_col, col in enumerate(cols):
                    widths[i_col] = max(widths[i_col], len(col))

            self.col_widths = widths

    def formatted_as_strings(self):
        """Return lines formatted to the set column widths."""
        if self.col_widths is None:
            # If not set, calculate minimum widths.
            self.set_min_column_widths()
        result_lines = []
        for row in self.rows:
            col_texts = []
            for col, align, width in zip(
                row.cols, row.aligns, self.col_widths
            ):
                if align == "left":
                    col_text = col.ljust(width)
                elif align == "right":
                    col_text = col.rjust(width)
                else:
                    msg = (
                        f'Unknown alignment "{align}" '
                        'not in ("left", "right")'
                    )
                    raise ValueError(msg)
                col_texts.append(col_text)

            row_line = " ".join(col_texts).rstrip()
            result_lines.append(row_line)
        return result_lines

    def __str__(self):
        return "\n".join(self.formatted_as_strings())


class CubePrinter:
    """
    An object created from a
    :class:`iris._representation.CubeSummary`, which provides
    text printout of a :class:`iris.cube.Cube`.

    This class has no internal knowledge of :class:`iris.cube.Cube`, but only
    of :class:`iris._representation.CubeSummary`.

    """

    N_INDENT_SECTION = 4
    N_INDENT_ITEM = 4
    N_INDENT_EXTRA = 4

    def __init__(self, cube_or_summary):
        """
        An object that provides a printout of a cube.

        Args:

        * cube_or_summary (Cube or CubeSummary):
            If a cube, first create a CubeSummary from it.


        .. note::
            The CubePrinter is based on a digest of a CubeSummary, but does
            not reference or store it.

        """
        # Create our internal table from the summary, to produce the printouts.
        if isinstance(cube_or_summary, CubeSummary):
            cube_summary = cube_or_summary
        else:
            cube_summary = CubeSummary(cube_or_summary)
        self.table = self._ingest_summary(cube_summary)

    def _ingest_summary(self, cube_summary):
        """Make a table of strings representing the cube-summary."""
        sect_indent = " " * self.N_INDENT_SECTION
        item_indent = sect_indent + " " * self.N_INDENT_ITEM
        item_to_extra_indent = " " * self.N_INDENT_EXTRA
        extra_indent = item_indent + item_to_extra_indent

        fullheader = cube_summary.header
        nameunits_string = fullheader.nameunit
        dimheader = fullheader.dimension_header
        cube_is_scalar = dimheader.scalar

        cube_shape = dimheader.shape  # may be empty
        dim_names = dimheader.dim_names  # may be empty
        n_dims = len(dim_names)
        assert len(cube_shape) == n_dims

        # First setup the columns
        #   - x1 @0 column-1 content : main title; headings; elements-names
        #   - x1 @1 "value" content (for scalar items)
        #   - OR x2n @1.. (name, length) for each of n dimensions
        column_header_texts = [nameunits_string]  # Note extra spacer here

        if cube_is_scalar:
            # We will put this in the column-1 position (replacing the dim-map)
            column_header_texts.append("(scalar cube)")
        else:
            for dim_name, length in zip(dim_names, cube_shape):
                column_header_texts.append(f"{dim_name}:")
                column_header_texts.append(f"{length:d}")

        n_cols = len(column_header_texts)

        # Create a table : a (n_rows) list of (n_cols) strings

        table = Table()

        # Code for adding a row, with control options.
        scalar_column_aligns = ["left"] * n_cols
        vector_column_aligns = deepcopy(scalar_column_aligns)
        if cube_is_scalar:
            vector_column_aligns[1] = "left"
        else:
            vector_column_aligns[1:] = n_dims * ["right", "left"]

        def add_row(col_texts, scalar=False):
            aligns = scalar_column_aligns if scalar else vector_column_aligns
            i_col_unlimited = 1 if scalar else None
            n_missing = n_cols - len(col_texts)
            col_texts += [" "] * n_missing
            table.add_row(col_texts, aligns, i_col_unlimited=i_col_unlimited)

        # Start with the header line
        add_row(column_header_texts)

        # Add rows from all the vector sections
        for sect in cube_summary.vector_sections.values():
            if sect.contents:
                sect_name = sect.title
                column_texts = [sect_indent + sect_name]
                add_row(column_texts)
                for vec_summary in sect.contents:
                    element_name = vec_summary.name
                    dim_chars = vec_summary.dim_chars
                    extra_string = vec_summary.extra
                    column_texts = [item_indent + element_name]
                    for dim_char in dim_chars:
                        column_texts += [dim_char, ""]
                    add_row(column_texts)
                    if extra_string:
                        column_texts = [extra_indent + extra_string]
                        add_row(column_texts)

        # Similar for scalar sections
        for sect in cube_summary.scalar_sections.values():
            if sect.contents:
                # Add a row for the "section title" text.
                sect_name = sect.title
                add_row([sect_indent + sect_name])

                def add_scalar_row(name, value=""):
                    column_texts = [item_indent + name, value]
                    add_row(column_texts, scalar=True)

                # Add a row for each item
                # NOTE: different section types need different handling
                title = sect_name.lower()
                if "scalar coordinate" in title:
                    for item in sect.contents:
                        add_scalar_row(item.name, item.content)
                        if item.extra:
                            add_scalar_row(item_to_extra_indent + item.extra)
                elif "attribute" in title or "cell method" in title:
                    for title, value in zip(sect.names, sect.values):
                        add_scalar_row(title, value)
                elif "scalar cell measure" in title:
                    # These are just strings: nothing in the 'value' column.
                    for name in sect.contents:
                        add_scalar_row(name)
                else:
                    msg = f"Unknown section type : {type(sect)}"
                    raise ValueError(msg)

        return table

    @staticmethod
    def _decorated_table(table, name_padding=None):
        """
        Return a modified table with added characters in the header.

        Note: 'name_padding' sets a minimum width for the name column (#0).

        """

        # Copy the input table + extract the header + its columns.
        table = table.copy()
        header = table.rows[0]
        cols = header.cols

        if name_padding:
            # Extend header column#0 to a given minimum width.
            cols[0] = cols[0].ljust(name_padding)

        # Add parentheses around the dim column texts.
        # -- unless already present, e.g. "(scalar cube)".
        if len(cols) > 1 and not cols[1].startswith("("):
            # Add parentheses around the dim columns
            cols[1] = "(" + cols[1]
            cols[-1] = cols[-1] + ")"

        # Add semicolons as dim column spacers
        for i_col in range(2, len(cols) - 1, 2):
            cols[i_col] += ";"

        # Modify the new table to be returned, invalidate any stored widths.
        header.cols = cols
        table.rows[0] = header

        # Recalc widths
        table.set_min_column_widths()

        return table

    def _oneline_string(self, name_padding):
        """Produce a one-line summary string."""
        # Copy existing content -- just the header line.
        table = Table(rows=[self.table.rows[0]])
        # Note: by excluding other columns, we get a minimum-width result.

        # Add standard decorations.
        table = self._decorated_table(table, name_padding=name_padding)

        # Format (with no extra spacing) --> one-line result
        (oneline_result,) = table.formatted_as_strings()
        return oneline_result

    def _multiline_summary(self, name_padding):
        """Produce a multi-line summary string."""
        # Get a derived table with standard 'decorations' added.
        table = self._decorated_table(self.table, name_padding=name_padding)
        result_lines = table.formatted_as_strings()
        result = "\n".join(result_lines)
        return result

    def to_string(self, oneline=False, name_padding=35):
        """
        Produce a printable summary.

        Args:
        * oneline (bool):
            If set, produce a one-line summary.
            Default is False = produce full (multiline) summary.
        * name_padding (int):
            The minimum width for the "name" (#0) column.

        Returns:
            result (string)

        """
        if oneline:
            result = self._oneline_string(name_padding)
        else:
            result = self._multiline_summary(name_padding)

        return result

    def __str__(self):
        """Printout of self, as a full multiline string."""
        return self.to_string()
