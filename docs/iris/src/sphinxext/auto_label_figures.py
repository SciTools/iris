# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from six.moves import (filter, input, map, range, zip)  # noqa

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
