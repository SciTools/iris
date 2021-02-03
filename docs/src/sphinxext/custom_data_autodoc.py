# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from sphinx.ext.autodoc import DataDocumenter, ModuleLevelDocumenter
try:
    # Use 'object_description' in place of the former 'safe_repr' function.
    from sphinx.util.inspect import object_description as safe_repr
except ImportError:
    # 'safe_repr' is the old usage, for Sphinx<1.3.
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
            self.add_line(
                u'   :annotation: {}'.format(self.options.annotation),
                '<autodoc>')


def handler(app, what, name, obj, options, signature, return_annotation):
    if what == 'data':
        if isinstance(obj, object) and issubclass(obj.__class__, Aggregator):
            signature = '()'
            return_annotation = '{} instance.'.format(obj.__class__.__name__)
    return signature, return_annotation


def setup(app):
    app.add_autodocumenter(IrisDataDocumenter, override=True)
    app.connect('autodoc-process-signature', handler)
