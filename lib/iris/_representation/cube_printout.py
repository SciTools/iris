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
        self.table = self._make_table(cube_summary)
        # NOTE: owing to some problems with column width control, this does not
        # encode our 'max_width' : For now, that is implemented by special code
        # in :meth:`to_string`.

    def _make_table(
        self,
        cube_summary,
        n_indent_section=5,
        n_indent_item=5,
        n_indent_extra=5,
    ):
        # Construct a beautifultable to represent the cube-summary info.
        #
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

        tb = bt.BeautifulTable(max_width=self.max_width)

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
                else:
                    msg = f"Unknown section type : {type(sect)}"
                    raise ValueError(msg)

        return tb

    def _set_table_style(self, tb):
        # Fix all the column widths and alignments
        tb.maxwidth = 9999  # don't curtail or wrap *anything* (initially)
        tb.columns.alignment[0] = bt.ALIGN_LEFT
        tb.columns.alignment[1] = bt.ALIGN_LEFT
        for i_col in range(2, len(tb.columns) - 1, 2):
            tb.columns.alignment[i_col] = bt.ALIGN_RIGHT
            tb.columns.padding_left[i_col] = 2
            tb.columns.padding_right[i_col] = 0
            tb.columns.padding_left[i_col + 1] = 0
        tb.set_style(bt.STYLE_NONE)

    def _oneline_printout(self):
        # Make a copy of the table, with no spacing and doctored columns.
        tb = bt.BeautifulTable()  # use new table, as copy() is deprecated

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
        # Add a single row matching the header (or nothing will print)
        # ( as used inside bt.BeatifulTable._get_string() ).
        tb.rows.append(tb.columns.header)

        # Set all the normal style options
        self._set_table_style(tb)

        # Adjust the column paddings for minimal spacing.
        tb.columns.padding_left = 0
        tb.columns.padding_right = 1

        # Return only the top (header) line.
        result = next(tb._get_string())
        return result

    def _full_multiline_summary(self, max_width):
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
            result = self._oneline_printout()
        else:
            result = self._full_multiline_summary(max_width)

        return result

    def __str__(self):
        # Return a full cube summary as the printed form.
        return self.to_string()
