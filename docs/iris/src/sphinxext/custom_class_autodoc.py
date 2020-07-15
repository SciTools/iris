# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from sphinx.ext import autodoc
from sphinx.ext.autodoc import *
from sphinx.util import force_decode
from sphinx.util.docstrings import prepare_docstring
import inspect

# stop warnings cluttering the make output
import warnings
warnings.filterwarnings("ignore")


class ClassWithConstructorDocumenter(autodoc.ClassDocumenter):
    priority = 1000000

    def get_object_members(self, want_all):
        return autodoc.ClassDocumenter.get_object_members(self, want_all)

    @staticmethod
    def can_document_member(member, mname, isattr, self):
        return autodoc.ClassDocumenter.can_document_member(member, mname,
                                                           isattr, self)

    def get_doc(self, encoding=None):
        content = self.env.config.autoclass_content

        docstrings = []
        docstring = self.get_attr(self.object, '__doc__', None)
        if docstring:
            docstrings.append(docstring)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is only the class docstring
        if content in ('both', 'init'):
            constructor = self.get_constructor()
            if constructor:
                initdocstring = self.get_attr(constructor, '__doc__', None)
            else:
                initdocstring = None
            if initdocstring:
                if content == 'init':
                    docstrings = [initdocstring]
                else:
                    docstrings.append(initdocstring)

        return [prepare_docstring(force_decode(docstring, encoding))
                for docstring in docstrings]

    def get_constructor(self):
        # for classes, the relevant signature is the __init__ method's
        initmeth = self.get_attr(self.object, '__new__', None)

        if initmeth is None or initmeth is object.__new__ or not \
                (inspect.ismethod(initmeth) or inspect.isfunction(initmeth)):
            initmeth = None

        if initmeth is None:
            initmeth = self.get_attr(self.object, '__init__', None)

        if initmeth is None or initmeth is object.__init__ or \
                initmeth is object.__new__ or not \
                (inspect.ismethod(initmeth) or inspect.isfunction(initmeth)):
            initmeth = None

        return initmeth

    def format_args(self):
        initmeth = self.get_constructor()
        try:
            argspec = inspect.getargspec(initmeth)
        except TypeError:
            # still not possible: happens e.g. for old-style classes
            # with __init__ in C
            return None
        if argspec[0] and argspec[0][0] in ('cls', 'self'):
            del argspec[0][0]
        return inspect.formatargspec(*argspec)


def setup(app):
    app.add_autodocumenter(ClassWithConstructorDocumenter, override=True)
