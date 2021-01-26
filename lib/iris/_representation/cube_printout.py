# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides text printouts of Iris cubes.

"""

import beautifultable as bt


class CubePrinter:
    """
    An object created from a
    :class:`iris._representation.cube_summary.CubeSummary`, which provides
    text printout of a :class:`iris.cube.Cube`.

    This is the type of object now returned by :meth:`iris.cube.Cube.summary`
    (when 'oneline=False') :  Hence it needs to be printable, so it has a
    :meth:`__str__` method which calls its :meth:`to_string`.

    The cube :meth:`iris.cube.Cube.__str__`  and
    :meth:`iris.cube.Cube.__repr__`  methods, and
    :meth:`iris.cube.Cube.summary` with 'oneline=True', also use this to
    produce cube summary strings.

    It's "table" property is a :class:`beautifultable.BeautifulTable`, which
    provides a representation of cube content as a flexible table object.
    However, but this cannot currently produce output identical to the
    :meth:`to_string` method, which uses additional techniques.

    In principle, this class does not have any internal knowledge of
    :class:`iris.cube.Cube`, but only of
    :class:`iris._representation.cube_summary.CubeSummary`.

    """

    def __init__(self, cube_summary, max_width=None):
        # Extract what we need from the cube summary, to produce printouts.

        if max_width is None:
            max_width = 120  # Our magic best guess
        self.max_width = max_width

        # Create a table to produce the printouts.
        self.table = self._make_table(cube_summary, max_width)
        # NOTE: although beautifultable is useful and provides a flexible output
        # form, its formatting features are not yet adequate to produce our
        # desired "standard cube summary" appearance.
        # (It really needs column-spanning, at least).
        # So +make_table the table is useful, it does not encode the whole of the
        # state / object info to produce
        # So the 'normal' cube summary is produced by "CubePrinter.to_string()",
        # which must also use information *not* stored in the table.
        # in :meth:`to_string`.

    def _make_table(
        self,
        cube_summary,
        max_width,
        n_indent_section=4,
        n_indent_item=4,
        n_indent_extra=4,
    ):
        """Make a beautifultable representing the cube-summary."""
        # NOTE: although beautifultable is useful and provides a flexible output
        # form, its formatting features are not yet adequate to produce our
        # desired "standard cube summary" appearance :  For that, we still need
        # column spanning (at least).
        # So a 'normal' cube summary is produced by "CubePrinter.to_string()",
        # which must also use information *not* stored in the table.
        extra_indent = " " * n_indent_extra
        sect_indent = " " * n_indent_section
        item_indent = sect_indent + " " * n_indent_item
        summ = cube_summary

        fullheader = summ.header
        nameunits_string = fullheader.nameunit
        dimheader = fullheader.dimension_header
        cube_is_scalar = dimheader.scalar
        assert not cube_is_scalar  # Just for now...

        cube_shape = dimheader.shape  # may be empty
        dim_names = dimheader.dim_names  # may be empty
        n_dims = len(dim_names)
        assert len(cube_shape) == n_dims

        tb = bt.BeautifulTable(maxwidth=max_width)

        # First setup the columns
        #   - x1 @0 column-1 content : main title; headings; elements-names
        #   - x1 @1 "value" content (for scalar items)
        #   - x2n @2 (name, length) for each of n dimensions
        column_texts = [nameunits_string, ""]
        for dim_name, length in zip(dim_names, cube_shape):
            column_texts.append(f"{dim_name}:")
            column_texts.append(f"{length:d}")

        tb.columns.header = column_texts[:]  # Making copy, in case (!)

        # Add rows from all the vector sections
        for sect in summ.vector_sections.values():
            if sect.contents:
                sect_name = sect.title
                column_texts = [sect_indent + sect_name, ""]
                column_texts += [""] * (2 * n_dims)
                tb.rows.append(column_texts)
                for vec_summary in sect.contents:
                    element_name = vec_summary.name
                    dim_chars = vec_summary.dim_chars
                    extra_string = vec_summary.extra
                    column_texts = [item_indent + element_name, ""]
                    for dim_char in dim_chars:
                        column_texts += ["", dim_char]
                    tb.rows.append(column_texts)
                    if extra_string:
                        column_texts = [""] * len(column_texts)
                        column_texts[1] = extra_indent + extra_string

        # Record where the 'scalar' part starts.
        self.i_first_scalar_row = len(tb.rows)

        # Similar for scalar sections
        for sect in summ.scalar_sections.values():
            if sect.contents:
                # Add a row for the "section title" text.
                sect_name = sect.title
                column_texts = [sect_indent + sect_name, ""]
                column_texts += [""] * (2 * n_dims)
                tb.rows.append(column_texts)
                title = sect_name.lower()

                def add_scalar(name, value):
                    column_texts = [item_indent + name, value]
                    column_texts += [""] * (2 * n_dims)
                    tb.rows.append(column_texts)

                # Add a row for each item
                # NOTE: different section types handle differently
                if "scalar coordinate" in title:
                    for item in sect.contents:
                        add_scalar(item.name, item.content)
                elif "attribute" in title:
                    for title, value in zip(sect.names, sect.values):
                        add_scalar(title, value)
                elif "scalar cell measure" in title or "cell method" in title:
                    # These are just strings: nothing in the 'value' column.
                    for name in sect.contents:
                        add_scalar(name, "")
                # elif "mesh" in title:
                #     for line in sect.contents()
                #         add_scalar(line, "")
                else:
                    msg = f"Unknown section type : {type(sect)}"
                    raise ValueError(msg)

        # Setup our "standard" style options, which is important because the
        # column alignment is very helpful to readability.
        CubePrinter._set_table_style(tb)
        # .. but adopt a 'normal' overall style showing the boxes.
        tb.set_style(bt.STYLE_DEFAULT)
        return tb

    @staticmethod
    def _set_table_style(tb, no_values_column=False):
        # Fix all the column paddings and alignments.
        # tb.maxwidth = 9999  # don't curtail or wrap *anything* (initially)
        tb.columns.alignment[0] = bt.ALIGN_LEFT
        if no_values_column:
            # Columns are: 1*(section/entry) + 2*(dim, dim-length)
            dim_cols = range(1, len(tb.columns) - 1, 2)
        else:
            # Columns are: 1*(section/entry) + 1*(value) + 2*(dim, dim-length)
            tb.columns.alignment[1] = bt.ALIGN_LEFT
            dim_cols = range(2, len(tb.columns) - 1, 2)
        for i_col in dim_cols:
            tb.columns.alignment[i_col] = bt.ALIGN_RIGHT
            tb.columns.alignment[i_col] = bt.ALIGN_LEFT
            tb.columns.padding_left[i_col] = 2
            tb.columns.padding_right[i_col] = 0
            tb.columns.padding_left[i_col + 1] = 0

        # Default style uses no decoration at all.
        tb.set_style(bt.STYLE_NONE)

    def _oneline_string(self):
        """Produce a one-line summary string."""
        # Make a copy of the table, with no spacing and doctored columns.
        tb = bt.BeautifulTable()  # start from a new table

        # Add column headers, with extra text modifications.
        column_headers = self.table.column_headers[:]
        column_headers[0] = "<iris 'Cube' of " + column_headers[0]
        column_headers[2] = "(" + column_headers[2]
        column_headers[-1] = column_headers[-1] + ")>"
        # Add semicolons as column spacers
        for i_col in range(3, len(column_headers) - 1, 2):
            column_headers[i_col] += ";"
            # NOTE: it would be "nice" use `table.columns.separator` to do
            # this, but bt doesn't currently support that :  Setting it
            # affects the header-underscore/separator line instead.
        tb.column_headers = column_headers

        # Add a single row matching the header (or nothing will print).
        # -- as used inside bt.BeautifulTable._get_string().
        tb.rows.append(tb.columns.header)

        # Setup all our normal column styling options
        CubePrinter._set_table_style(tb)

        # Adjust all column paddings for minimal spacing.
        tb.columns.padding_left = 0
        tb.columns.padding_right = 1

        # Print with no width restriction
        tb.maxwidth = 9999

        # Return only the top (header) line.
        result = next(tb._get_string())
        return result

    def _multiline_string_OLD(self, max_width):
        """Produce a one-line summary string."""
        # pre-render with no width limitation whatsoever.
        tb = self.table
        tb.maxwidth = 9999
        str(tb)

        # Force wraps in the 'value column' (== column #1)
        widths = tb.columns.width[:]
        widths[1] = 0
        widths[1] = max_width - sum(widths)
        tb.columns.width = widths
        tb.columns.width_exceed_policy = bt.WEP_WRAP
        # Also must re-establish the style.
        # Hmmm, none of this is that obvious, is it ??
        tb.set_style(bt.STYLE_NONE)

        # Finally, use _get_string to reprint *without* recalulting widths.
        summary_lines = list(tb._get_string(recalculate_width=False))
        result = "\n".join(summary_lines)

        return result

    def _multiline_summary(self, max_width):
        """
        Produce a one-line summary string.

        Note: 'max_width' controls wrapping of the values column. but the
        However, the sections-titles/item-names column and dim map are produced
        *without* any width restriction. The max_width

        """
        # First print the vector sections.

        # Make a copy, but omitting column 1 (the scalar "values" column)
        cols = list(self.table.columns.header)
        del cols[1]
        tb = bt.BeautifulTable()
        tb.columns.header = cols

        # Copy vector rows only, removing column#1 (which should be blank)
        # - which puts the dim-map columns in the column#1 place.
        for i_row in range(self.i_first_scalar_row):
            row = list(self.table.rows[i_row])
            del row[1]
            tb.rows.append(row)

        # Establish our standard style settings (alignment etc).
        self._set_table_style(tb, no_values_column=True)

        # Add parentheses around the dim column texts.
        column_headers = tb.columns.header
        column_headers[1] = "(" + column_headers[1]
        column_headers[-1] = column_headers[-1] + ")"
        tb.columns.header = column_headers

        # Use no width limitation.
        tb.maxwidth = 9999
        # Use _get_string to fetch a list of lines.
        summary_lines = list(tb._get_string())

        # Now add the "scalar rows".
        # For this part, we have only 2 columns + we force wrapping of the
        # second column at a specific width.

        tb = self.table.rows[self.i_first_scalar_row :]
        CubePrinter._set_table_style(tb)
        # Pre-render with no width restriction, to pre-calculate widths
        tb.maxwidth = 9999
        str(tb)

        # Force any wrapping needed in the 'value column' (== column #1)
        widths = tb.columns.width[:]
        widths[1] = max_width - widths[0]
        # widths[2:] = 0
        tb.columns.width = widths
        tb.columns.width_exceed_policy = bt.WEP_WRAP

        # Get rows for the scalar part
        scalar_lines = tb._get_string(recalculate_width=False)
        # discard first line (header)
        next(scalar_lines)
        # add the rest to the summary lines
        summary_lines += list(scalar_lines)

        result = "\n".join(summary_lines)
        return result

    def to_string(self, oneline=False, max_width=None):
        """
        Produce a printable summary.

        Args:
        * oneline (bool):
            If set, produce a one-line summary (without any extra spacings).
            Default is False  = produce full (multiline) summary.
        * max_width (int):
            If set, override the default maximum output width.
            Default is None = use the default established at object creation.

        Returns:
            result (string)

        """
        if max_width is None:
            max_width = self.max_width

        if oneline:
            result = self._oneline_string()
        else:
            result = self._multiline_summary(max_width)

        return result

    def __str__(self):
        """Printout of self is the full multiline string."""
        return self.to_string()
