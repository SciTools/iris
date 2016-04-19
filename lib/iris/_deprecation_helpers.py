# (C) British Crown Copyright 2010 - 2016, Met Office
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
Utilities for producing runtime deprecation messages.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

import warnings


# A Mixin for a wrapper class that mimics the underlying original, but also
# emits deprecation warnings when it is instantiated.
class ClassDeprecationWrapper(type):
    def __new__(metacls, classname, bases, class_dict):
        # Patch the subclass to duplicate docstrings from the parent class, and
        # provide an __init__ that issues a deprecation warning and then calls
        # the parent constructor.
        parent_class = bases[0]
        # Copy the original class docstring.
        class_dict['__doc__'] = parent_class.__doc__

        # Get a warning message from the underlying class.
        depr_warnstring = class_dict.get('_DEPRECATION_WARNING')

        # Save the parent class *on* the wrapper class, so we can chain to its
        #  __init__ call.
        class_dict['_target_parent_class'] = parent_class

        # Create a wrapper init function which issues the deprecation.
        def initfn(self, *args, **kwargs):
            print(repr(depr_warnstring))
            warnings.warn(depr_warnstring)
            self._target_parent_class.__init__(self, *args, **kwargs)

        # Set this as the init for the wrapper class.
        initfn.func_name = '__init__'
        # Also copy the original docstring.
        initfn.__doc__ = parent_class.__init__.__doc__
        class_dict['__init__'] = initfn

        # Return the result.
        return super(ClassDeprecationWrapper, metacls).__new__(
            metacls, classname, bases, class_dict)
