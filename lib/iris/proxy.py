# (C) British Crown Copyright 2010 - 2015, Met Office
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
Provision of a service to handle missing packages at runtime.
Current just a very thin layer but gives the option to extend
handling as much as needed

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import sys


class FakeModule(object):
    __slots__ = ('_name',)

    def __init__(self, name):
        self._name = name

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        raise AttributeError(
            'Module "{}" not available or not installed'.format(self._name))


def apply_proxy(module_name, dic):
    """
    Attempt the import else use the proxy module.
    It is important to note that '__import__()' must be used
    instead of the higher-level 'import' as we need to
    ensure the scope of the import can be propagated out of this package.
    Also, note the splitting of name - this is because '__import__()'
    requires full package path, unlike 'import' (this issue is
    explicitly seen in lib/iris/fileformats/pp.py importing pp_packing)

    """
    name = module_name.split('.')[-1]
    try:
        __import__(module_name)
        dic[name] = sys.modules[module_name]
    except ImportError:
        dic[name] = sys.modules[name] = FakeModule(name)
