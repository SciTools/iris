# (C) British Crown Copyright 2014, Met Office
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

from __future__ import (absolute_import, division, print_function)

import os
from docutils import nodes


def auto_label_figures(app, doctree):
    """
    Add a label on every figure.
    """

    for fig in doctree.traverse(condition=nodes.figure):
        for img in fig.traverse(condition=nodes.image):
            fname, ext = os.path.splitext(img['uri'])
            if ext == '.png':
                fname = os.path.basename(fname).replace('_', '-')
                fig['ids'].append(fname)


def setup(app):
    app.connect('doctree-read', auto_label_figures)
