# (C) British Crown Copyright 2010 - 2013, Met Office
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


from sphinx.ext import autodoc
from sphinx.ext.autodoc import *

import inspect


class ClassWithConstructorDocumenter(autodoc.ClassDocumenter):
    priority = 1000000

    def get_object_members(self, want_all):
        r,f = autodoc.ClassDocumenter.get_object_members(self, want_all)
        #print 'CALLED OBJECT MEMBERS....', r, f
        return r, f

#   def filter_members(self, members, want_all):
#      res = autodoc.ClassDocumenter.filter_members(self, members, want_all)
#      for (membername, member) in members:
#         if membername in ['__init__', '__new__']:
#            # final argument is "isattr" - no its not an attribute it is a contructor method.
#            res.insert(0, [membername, member, False])
#      return res

    @staticmethod
    def can_document_member(member, mname, isattr, self):
        #print ' asked me if I can document....', member, mname, isattr, self
        #print ' gave them :', autodoc.ClassDocumenter.can_document_member(member, mname, isattr, self)
        return autodoc.ClassDocumenter.can_document_member(member, mname, isattr, self)

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
                initdocstring = self.get_attr( constructor, '__doc__', None)
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

        if initmeth is None or initmeth is object.__init__ or initmeth is object.__new__ or not \
               (inspect.ismethod(initmeth) or inspect.isfunction(initmeth)):
            initmeth = None

        return initmeth


    def format_args(self):
        initmeth = self.get_constructor()
        #print 'DOING FORMAT ARGS: ', initmeth, self.object
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
    app.add_autodocumenter(ClassWithConstructorDocumenter)
