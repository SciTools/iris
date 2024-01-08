# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Definitions of how Iris objects should be represented."""

from html import escape
import re


class CubeRepresentation:
    """Produce representations of a :class:`~iris.cube.Cube`.

    This includes:

    * ``_html_repr_``: a representation of the cube as an html object,
      available in Jupyter notebooks. Specifically, this is presented as an
      html table.

    """

    _template = """
<style>
  a.iris {{
      text-decoration: none !important;
  }}
  table.iris {{
      white-space: pre;
      border: 1px solid;
      border-color: #9c9c9c;
      font-family: monaco, monospace;
  }}
  th.iris {{
      background: #303f3f;
      color: #e0e0e0;
      border-left: 1px solid;
      border-color: #9c9c9c;
      font-size: 1.05em;
      min-width: 50px;
      max-width: 125px;
  }}
  tr.iris :first-child {{
      border-right: 1px solid #9c9c9c !important;
  }}
  td.iris-title {{
      background: #d5dcdf;
      border-top: 1px solid #9c9c9c;
      font-weight: bold;
  }}
  .iris-word-cell {{
      text-align: left !important;
      white-space: pre;
  }}
  .iris-subheading-cell {{
      padding-left: 2em !important;
  }}
  .iris-inclusion-cell {{
      padding-right: 1em !important;
  }}
  .iris-panel-body {{
      padding-top: 0px;
  }}
  .iris-panel-title {{
      padding-left: 3em;
  }}
  .iris-panel-title {{
      margin-top: 7px;
  }}
</style>
<table class="iris" id="{id}">
    {header}
    {shape}
    {content}
</table>
        """

    def __init__(self, cube):
        self.cube = cube
        self.cube_id = id(self.cube)
        self.cube_str = escape(str(self.cube))

        # Define the expected vector and scalar sections in output, in expected
        # order of appearance.
        # NOTE: if we recoded this to use a CubeSummary, these section titles
        # would be available from that.
        self.vector_section_names = [
            "Dimension coordinates:",
            "Mesh coordinates:",
            "Auxiliary coordinates:",
            "Derived coordinates:",
            "Cell measures:",
            "Ancillary variables:",
        ]
        self.scalar_section_names = [
            "Mesh:",
            "Scalar coordinates:",
            "Scalar cell measures:",
            "Cell methods:",
            "Attributes:",
        ]
        self.sections_data = {
            name: None for name in self.vector_section_names + self.scalar_section_names
        }
        # 'Scalar-cell-measures' is currently alone amongst the scalar sections,
        # in displaying only a 'name' and no 'value' field.
        self.single_cell_section_names = ["Scalar cell measures:"]

        # Important content that summarises a cube is defined here.
        self.shapes = self.cube.shape
        self.scalar_cube = self.shapes == ()
        self.ndims = self.cube.ndim

        self.name = escape(self.cube.name().title().replace("_", " "))
        self.names = [escape(dim_name) for dim_name in self._dim_names()]
        self.units = escape(str(self.cube.units))

    def _get_dim_names(self):
        """Get dimension-describing coordinate names.

        Get dimension-describing coordinate names, or '--' if no coordinate]
        describes the dimension.

        Note: borrows from `cube.summary`.

        """
        # Create a set to contain the axis names for each data dimension.
        dim_names = list(range(len(self.cube.shape)))

        # Add the dim_coord names that participate in the associated data
        # dimensions.
        for dim in range(len(self.cube.shape)):
            dim_coords = self.cube.coords(contains_dimension=dim, dim_coords=True)
            if dim_coords:
                dim_names[dim] = dim_coords[0].name()
            else:
                dim_names[dim] = "--"
        return dim_names

    def _dim_names(self):
        if self.scalar_cube:
            dim_names = ["(scalar cube)"]
        else:
            dim_names = self._get_dim_names()
        return dim_names

    def _get_lines(self):
        return self.cube_str.split("\n")

    def _get_bits(self, bits):
        """Parse the body content (`bits`) of the cube string.

        Parse the body content (`bits`) of the cube string in preparation for
        being converted into table rows.

        """
        left_indent = re.split(r"\w+", bits[1])[0]

        # Get heading indices within the printout.
        start_inds = []
        for hdg in self.sections_data.keys():
            heading = "{}{}".format(left_indent, hdg)
            try:
                start_ind = bits.index(heading)
            except ValueError:
                continue
            else:
                start_inds.append(start_ind)
        # Mark the end of the file.
        start_inds.append(0)

        # Retrieve info for each heading from the printout.
        for i0, i1 in zip(start_inds[:-1], start_inds[1:]):
            str_heading_name = bits[i0].strip()
            if i1 != 0:
                content = bits[i0 + 1 : i1]
            else:
                content = bits[i0 + 1 :]
            self.sections_data[str_heading_name] = content

    def _make_header(self):
        """Make the table header.

        Make the table header. This is similar to the summary of the cube,
        but does not include dim shapes. These are included on the next table
        row down, and produced with `make_shapes_row`.

        """
        # Header row.
        tlc_template = '<th class="iris iris-word-cell">{self.name} ({self.units})</th>'
        top_left_cell = tlc_template.format(self=self)
        cells = ['<tr class="iris">', top_left_cell]
        for dim_name in self.names:
            cells.append('<th class="iris iris-word-cell">{}</th>'.format(dim_name))
        cells.append("</tr>")
        return "\n".join(cell for cell in cells)

    def _make_shapes_row(self):
        """Add a row to show data / dimensions shape."""
        title_cell = '<td class="iris-word-cell iris-subheading-cell">Shape</td>'
        cells = ['<tr class="iris">', title_cell]
        for shape in self.shapes:
            cells.append('<td class="iris iris-inclusion-cell">{}</td>'.format(shape))
        cells.append("</tr>")
        return "\n".join(cell for cell in cells)

    def _make_row(self, title, body=None, col_span=0):
        """Produce one row for the table body.

        For example::

            <tr><td>Coord name</td><td>x</td><td>-</td>...</tr>.

        * `body` contains the content for each cell not in the left-most (title)
          column.
          If None, indicates this row is a title row (see below).
        * `title` contains the row heading. If `body` is None, indicates
          that the row contains a sub-heading;
          e.g. 'Dimension coordinates:'.
        * `col_span` indicates how many columns the string should span.

        """
        row = ['<tr class="iris">']
        template = "    <td{html_cls}>{content}</td>"
        if body is None:
            # This is a title row.
            # Strip off the trailing ':' from the title string.
            title = title.strip()[:-1]
            row.append(
                template.format(
                    html_cls=' class="iris-title iris-word-cell"',
                    content=title,
                )
            )
            # Add blank cells for the rest of the rows.
            for _ in range(self.ndims):
                row.append(template.format(html_cls=' class="iris-title"', content=""))
        else:
            # This is not a title row.
            # Deal with name of coord/attr etc. first.
            sub_title = "\t{}".format(title)
            row.append(
                template.format(
                    html_cls=' class="iris-word-cell iris-subheading-cell"',
                    content=sub_title,
                )
            )
            # One further item or more than that?
            if col_span != 0:
                html_cls = ' class="{}" colspan="{}"'.format("iris-word-cell", col_span)
                row.append(template.format(html_cls=html_cls, content=body))
            else:
                # "Inclusion" - `x` or `-`.
                for itm in body:
                    row.append(
                        template.format(
                            html_cls=' class="iris-inclusion-cell"',
                            content=itm,
                        )
                    )
        row.append("</tr>")
        return row

    def _make_content(self):
        elements = []
        for k, v in self.sections_data.items():
            if v is not None:
                # Add the sub-heading title.
                elements.extend(self._make_row(k))
                for line in v:
                    # Add every other row in the sub-heading.
                    if k in self.vector_section_names:
                        body = re.findall(r"[\w-]+", line)
                        title = body.pop(0)
                        colspan = 0
                    else:
                        colspan = self.ndims
                        if k in self.single_cell_section_names:
                            title = line.strip()
                            body = ""
                        else:
                            line = line.strip()
                            split_point = line.index(" ")
                            title = line[:split_point].strip()
                            body = line[split_point + 2 :].strip()

                    elements.extend(self._make_row(title, body=body, col_span=colspan))
        return "\n".join(element for element in elements)

    def repr_html(self):
        """The `repr` interface for Jupyter."""
        # Deal with the header first.
        header = self._make_header()

        # Check if we have a scalar cube.
        if self.scalar_cube:
            shape = ""
            # We still need a single content column!
            self.ndims = 1
        else:
            shape = self._make_shapes_row()

        # Now deal with the rest of the content.
        lines = self._get_lines()
        # If we only have a single line `cube_str` we have no coords / attrs!
        # We need to handle this case specially.
        if len(lines) == 1:
            content = ""
        else:
            self._get_bits(lines)
            content = self._make_content()

        return self._template.format(
            header=header, id=self.cube_id, shape=shape, content=content
        )


class CubeListRepresentation:
    _template = """
<style>
    .accordion-{uid} {{
        color: var(--jp-ui-font-color2);
        background: var(--jp-layout-color2);
        cursor: pointer;
        padding: 10px;
        border: 1px solid var(--jp-border-color0);
        width: 100%;
        text-align: left;
        font-size: 14px;
        font-family: var(--jp-code-font-family);
        font-weight: normal;
        outline: none;
        transition: 0.4s;
    }}
    .active {{
        background: var(--jp-layout-color1);
        font-weight: 900;
    }}
    .accordion-{uid}.active {{
        border: 1px solid var(--jp-brand-color1) !important;
    }}
    .accordion-{uid}:hover {{
        box-shadow: var(--jp-input-box-shadow);
        border: 2px solid var(--jp-brand-color1);
    }}
    .panel-{uid} {{
        padding: 0 18px;
        margin-bottom: 5px;
        background-color: var(--jp-layout-color1);
        display: none;
        overflow: hidden;
        border: 1px solid var(--jp-brand-color2);
    }}
</style>
<script type="text/javascript">
    var accordion = document.getElementsByClassName("accordion-{uid}");
    var i;

    for (i = 0; i < accordion.length; i++) {{
        accordion[i].addEventListener("click", function() {{
            this.classList.toggle("active");

            var panel = this.nextElementSibling;
            if (panel.style.display === "block") {{
                panel.style.display = "none";
            }} else {{
                panel.style.display = "block";
            }}
        }});
    }}
</script>
{contents}
    """

    _accordian_panel = """
<button class="accordion-{uid}">{title}</button>
<div class="panel-{uid}">
    <p>{content}</p>
</div>
    """

    def __init__(self, cubelist):
        self.cubelist = cubelist
        self.cubelist_id = id(self.cubelist)

    def make_content(self):
        html = []
        for i, cube in enumerate(self.cubelist):
            title = "{i}: {summary}".format(i=i, summary=cube.summary(shorten=True))
            title = escape(title)
            content = cube._repr_html_()
            html.append(
                self._accordian_panel.format(
                    uid=self.cubelist_id, title=title, content=content
                )
            )
        return html

    def repr_html(self):
        contents = self.make_content()
        contents_str = "\n".join(contents)
        return self._template.format(uid=self.cubelist_id, contents=contents_str)
