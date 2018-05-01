# (C) British Crown Copyright 2018, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.

"""
Definitions of how Iris objects should be represented.

"""


class CubeRepresentation(object):
    """
    Produce representations of a :class:`~iris.cube.Cube`.

    This includes:

    * ``_html_repr_``: a representation of the cube as an html object,
      available in jupyter notebooks.

    """
    _template = """
<style>
  a.iris {{
      text-decoration: none !important;
  }}
  .iris {{
      white-space: pre;
  }}
  .iris-panel-group {{
      display: block;
      overflow: visible;
      width: max-content;
      font-family: monaco, monospace;
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
<div class="panel-group iris-panel-group">
  <div class="panel panel-default">
    <div class="panel-heading">
      <h4 class="panel-title">
        <a class="iris" data-toggle="collapse" href="#collapse1-{obj_id}">
{summary}
        </a>
      </h4>
    </div>
    <div id="collapse1-{obj_id}" class="panel-collapse collapse in">
      {content}
    </div>
  </div>
</div>
        """

    # Need to format the keywords:
    #     `emt_id`, `obj_id`, `str_heading`, `opened`, `content`.
    _insert_content = """
      <div class="panel-body iris-panel-body">
        <h4 class="panel-title iris-panel-title">
          <a class="iris" data-toggle="collapse" href="#{emt_id}-{obj_id}">
{str_heading}
          </a>
        </h4>
      </div>
      <div id="{emt_id}-{obj_id}" class="panel-collapse collapse{opened}">
          <div class="panel-body iris-panel-body">
              <p class="iris">{content}</p>
          </div>
      </div>
    """

    def __init__(self, cube):
        """
        Produce different representations of a :class:`~iris.cube.Cube`.

        Args:

        * cube
            the cube to produce representations of.

        """

        self.cube = cube
        self.cube_id = id(self.cube)
        self.cube_str = str(self.cube)

        self.summary = None
        self.str_headings = {
            'Dimension coordinates:': None,
            'Auxiliary coordinates:': None,
            'Derived coordinates:': None,
            'Scalar coordinates:': None,
            'Attributes:': None,
            'Cell methods:': None,
        }
        self.major_headings = ['Dimension coordinates:',
                               'Auxiliary coordinates:',
                               'Attributes:']

    def _get_bits(self):
        """
        Parse the str representation of the cube to retrieve the elements
        to add to an html representation of the cube.

        """
        bits = self.cube_str.split('\n')
        self.summary = bits[0]
        left_indent = bits[1].split('D')[0]

        # Get heading indices within the printout.
        start_inds = []
        for hdg in self.str_headings.keys():
            heading = '{}{}'.format(left_indent, hdg)
            try:
                start_ind = bits.index(heading)
            except ValueError:
                continue
            else:
                start_inds.append(start_ind)
        # Make sure the indices are in order.
        start_inds = sorted(start_inds)
        # Mark the end of the file.
        start_inds.append(None)

        # Retrieve info for each heading from the printout.
        for i0, i1 in zip(start_inds[:-1], start_inds[1:]):
            str_heading_name = bits[i0].strip()
            if i1 is not None:
                content = bits[i0 + 1: i1]
            else:
                content = bits[i0 + 1:]
            self.str_headings[str_heading_name] = content

    def _make_content(self):
        elements = []
        for k, v in self.str_headings.items():
            if v is not None:
                html_id = k.split(' ')[0].lower().strip(':')
                content = '\n'.join(line for line in v)
                collapse = ' in' if k in self.major_headings else ''
                element = self._insert_content.format(emt_id=html_id,
                                                      obj_id=self.cube_id,
                                                      str_heading=k,
                                                      opened=collapse,
                                                      content=content)
                elements.append(element)
        return '\n'.join(element for element in elements)

    def repr_html(self):
        """Produce an html representation of a cube and return it."""
        self._get_bits()
        summary = self.summary
        content = self._make_content()
        return self._template.format(summary=summary,
                                     content=content,
                                     obj_id=self.cube_id,
                                     )
