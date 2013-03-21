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
"""
Experimental code can be introduced to Iris through this package.

Changes to experimental code may be more extensive than in the rest of the
codebase. The code is expected to graduate, eventually, to "full status".

"""

from functools import wraps

import concatenate


__all__ = ['concatenate']


def _experimental(func_call):
    """
    Decorator function that provides convenience to experimental
    functionality.

    Args:

    * func_call:
        The actual experimental function to be invoked.

    Returns:
        Closure wrapper function.

    """
    def _outer(func):
        """
        Closure that unwraps the dummy convenience function.

        Args:

        * func:
            Top level dummy convenience function exposed at
            the experimental module level.

        Returns:
            Closure wrapper function.

        """
        @wraps(func_call)
        def _inner(*args, **kwargs):
            """Performs the actual experimental function call"""

            return func_call(*args, **kwargs)

        return _inner

    return _outer


@_experimental(concatenate.concatenate)
def concatenate(*args, **kwargs):
    """Convenience wrapper function for concatenate"""
