# (C) British Crown Copyright 2017, Met Office
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
Routines for lazy data handling.

To avoid replicating implementation-dependent test and conversion code.

"""
from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import dask.array as da


def is_lazy_data(data):
    """
    Return whether the argument is an Iris 'lazy' data array.

    At present, this means simply a Dask array.
    We determine this by checking for a "compute" property.

    """
    return hasattr(data, 'compute')


def as_concrete_data(data):
    """
    Return the actual content of the argument, as a numpy array.

    If lazy, return the realised data, otherwise return the argument unchanged.

    """
    if is_lazy_data(data):
        data = data.compute()
    return data


# A magic value, borrowed from biggus
_MAX_CHUNK_SIZE = 8 * 1024 * 1024 * 2


def as_lazy_data(data):
    """
    Return a lazy equivalent of the argument, as a lazy array.

    For an existing dask array, return it unchanged.
    Otherwise, return the argument wrapped with dask.array.from_array.
    This assumes the underlying object has numpy-array-like properties.

    .. Note::

        For now at least, chunksize is set to an arbitrary fixed value.

    """
    if not is_lazy_data(data):
        data = da.from_array(data, chunks=_MAX_CHUNK_SIZE)
    return data
