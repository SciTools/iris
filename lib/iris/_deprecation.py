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

from iris.exceptions import IrisError


class IrisDeprecation(UserWarning):
    """An Iris deprecation warning."""
    pass


def warn_deprecated(msg, **kwargs):
    warnings.warn(msg, IrisDeprecation, **kwargs)


# A Mixin for a wrapper class that copies the docstring of the wrapped class
# into the wrapper.
# This is useful in producing wrapper classes that need to mimic the original
# but emit deprecation warnings when used.
class ClassWrapperSameDocstring(type):
    def __new__(metacls, classname, bases, class_dict):
        # Patch the subclass to duplicate the class docstring from the wrapped
        # class, and give it a special '__new__' that issues a deprecation
        # warning when creating an instance.
        parent_class = bases[0]

        # Copy the original class docstring.
        class_dict['__doc__'] = parent_class.__doc__

        # Return the result.
        return super(ClassWrapperSameDocstring, metacls).__new__(
            metacls, classname, bases, class_dict)
