# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Definitions of how Iris objects should be represented."""

from html import escape

from iris._representation.cube_summary import CubeSummary


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
        self.summary = CubeSummary(cube)
        self.cube_id = id(self.cube)

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

        Parameters
        ----------
        title : str, optional
            Contains the row heading. If `body` is None, indicates
            that the row contains a sub-heading;
            e.g. 'Dimension coordinates:'.
        body : str, optional
            Contains the content for each cell not in the left-most (title) column.
            If None, indicates this row is a title row (see below).
        col_span : int, default=0
            Indicates how many columns the string should span.

        Examples
        --------
        ::

            <tr><td>Coord name</td><td>x</td><td>-</td>...</tr>.

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
            if not isinstance(body, list):
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
        INDENT = 4 * "&nbsp;"
        for sect in self.summary.vector_sections.values():
            if sect.contents:
                sect_title = sect.title
                elements.extend(self._make_row(sect_title))

            for content in sect.contents:
                body = content.dim_chars

                title = escape(content.name)
                if content.extra:
                    title = title + "<br>" + INDENT + escape(content.extra)
                elements.extend(self._make_row(title, body=body, col_span=0))
        for sect in self.summary.scalar_sections.values():
            if sect.contents:
                sect_title = sect.title
                elements.extend(self._make_row(sect_title))
                st = sect_title.lower()
                if st == "scalar coordinates:":
                    for item in sect.contents:
                        body = escape(item.content)
                        title = escape(item.name)
                        if item.extra:
                            title = title + "<br>" + INDENT + escape(item.extra)
                        elements.extend(
                            self._make_row(title, body=body, col_span=self.ndims)
                        )
                elif st in ("attributes:", "cell methods:", "mesh:"):
                    for title, body in zip(sect.names, sect.values):
                        title = escape(title)
                        body = escape(body)
                        elements.extend(
                            self._make_row(title, body=body, col_span=self.ndims)
                        )
                        pass
                elif st in (
                    "scalar ancillary variables:",
                    "scalar cell measures:",
                ):
                    body = ""
                    # These are just strings: nothing in the 'value' column.
                    for title in sect.contents:
                        title = escape(title)
                        elements.extend(
                            self._make_row(title, body=body, col_span=self.ndims)
                        )
                else:
                    msg = f"Unknown section type : {type(sect)}"
                    raise ValueError(msg)
        return "\n".join(element for element in elements)

    def repr_html(self):
        """Represent html, the `repr` interface for Jupyter."""
        # Deal with the header first.
        header = self._make_header()

        # Check if we have a scalar cube.
        if self.scalar_cube:
            shape = ""
            # We still need a single content column!
            self.ndims = 1
        else:
            shape = self._make_shapes_row()
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
