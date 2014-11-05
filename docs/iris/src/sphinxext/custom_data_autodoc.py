# (C) British Crown Copyright 2010 - 2014, Met Office
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

from sphinx.ext.autodoc import DataDocumenter, ModuleLevelDocumenter
from sphinx.util.inspect import safe_repr

from iris.analysis import Aggregator


class IrisDataDocumenter(DataDocumenter):
    priority = 100

    def add_directive_header(self, sig):
        ModuleLevelDocumenter.add_directive_header(self, sig)
        if not self.options.annotation:
            try:
                objrepr = safe_repr(self.object)
            except ValueError:
                pass
            else:
                self.add_line(u'   :annotation:', '<autodoc>')
        elif self.options.annotation is object():
            pass
        else:
            self.add_line(u'   :annotation: %s' % self.options.annotation,
                          '<autodoc>')


def handler(app, what, name, obj, options, signature, return_annotation):
    if what == 'data':
        if isinstance(obj, object) and issubclass(obj.__class__, Aggregator):
            signature = '()'
            return_annotation = '{} instance.'.format(obj.__class__.__name__)
    return signature, return_annotation


def setup(app):
    app.add_autodocumenter(IrisDataDocumenter)
    app.connect('autodoc-process-signature', handler)
