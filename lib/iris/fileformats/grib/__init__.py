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
Conversion of cubes to/from GRIB.

See also: `ECMWF GRIB API
           <http://www.ecmwf.int/publications/manuals/grib_api/index.html>`_.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import datetime
import math  # for fmod
import warnings

import biggus
import cartopy.crs as ccrs
import numpy as np
import numpy.ma as ma
import scipy.interpolate

import iris.proxy
iris.proxy.apply_proxy('gribapi', globals())

from iris.analysis.interpolate import Linear1dExtrapolator
import iris.coord_systems as coord_systems
from iris.exceptions import TranslationError
# NOTE: careful here, to avoid circular imports (as iris imports grib)
from iris.fileformats.grib import grib_phenom_translation as gptx
from iris.fileformats.grib import _save_rules
import iris.fileformats.grib._load_convert
from iris.fileformats.grib._message import _GribMessage
import iris.fileformats.grib.load_rules
import iris.unit
from iris.fileformats.grib._gribwrapper import (GribWrapper, grib_generator,
                                                GribDataProxy)

__all__ = ['load_cubes', 'reset_load_rules', 'save_grib2']


#: Set this flag to True to enable support of negative forecast periods
#: when loading and saving GRIB files.
hindcast_workaround = False

# rules for converting a grib message to a cm cube
_load_rules = None


def reset_load_rules():
    """
    Resets the GRIB load process to use only the standard conversion rules.

    .. deprecated:: 1.7

    """
    # Uses this module-level variable
    global _load_rules

    warnings.warn('reset_load_rules was deprecated in v1.7.')

    _load_rules = None


def load_cubes(filenames, callback=None, auto_regularise=True):
    """
    Returns a generator of cubes from the given list of filenames.

    Args:

    * filenames (string/list):
        One or more GRIB filenames to load from.

    Kwargs:

    * callback (callable function):
        Function which can be passed on to :func:`iris.io.run_callback`.

    * auto_regularise (*True* | *False*):
        If *True*, any cube defined on a reduced grid will be interpolated
        to an equivalent regular grid. If *False*, any cube defined on a
        reduced grid will be loaded on the raw reduced grid with no shape
        information. If `iris.FUTURE.strict_grib_load` is `True` then this
        keyword has no effect, raw grids are always used. If the older GRIB
        loader is in use then the default behaviour is to interpolate cubes
        on a reduced grid to an equivalent regular grid.

        .. deprecated:: 1.8. Please use strict_grib_load and regrid instead.


    """
    if iris.FUTURE.strict_grib_load:
        grib_loader = iris.fileformats.rules.Loader(
            _GribMessage.messages_from_filename,
            {},
            iris.fileformats.grib._load_convert.convert, None)
    else:
        if auto_regularise is not None:
            # The old loader supports the auto_regularise keyword, but in
            # deprecation mode, so warning if it is found.
            warnings.warn('the`auto_regularise` kwarg is deprecated and '
                          'will be removed in a future release. Resampling '
                          'quasi-regular grids on load will no longer be '
                          'available.  Resampling should be done on the '
                          'loaded cube instead using Cube.regrid.')

        grib_loader = iris.fileformats.rules.Loader(
            grib_generator, {'auto_regularise': auto_regularise,
                             'hindcast_workaround': hindcast_workaround},
            iris.fileformats.grib.load_rules.convert,
            _load_rules)
    return iris.fileformats.rules.load_cubes(filenames, callback, grib_loader)


def save_grib2(cube, target, append=False, **kwargs):
    """
    Save a cube to a GRIB2 file.

    Args:

        * cube      - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or
                      list of cubes.
        * target    - A filename or open file handle.

    Kwargs:

        * append    - Whether to start a new file afresh or add the cube(s) to
                      the end of the file.
                      Only applicable when target is a filename, not a file
                      handle. Default is False.

    See also :func:`iris.io.save`.

    """
    messages = as_messages(cube)
    save_messages(messages, target, append=append)


def as_pairs(cube):
    """
    Convert one or more cubes to (2D cube, GRIB message) pairs.
    Returns an iterable of tuples each consisting of one 2D cube and
    one GRIB message ID, the result of the 2D cube being processed by the GRIB
    save rules.

    Args:
        * cube      - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or
        list of cubes.

    """
    x_coords = cube.coords(axis='x', dim_coords=True)
    y_coords = cube.coords(axis='y', dim_coords=True)
    if len(x_coords) != 1 or len(y_coords) != 1:
        raise TranslationError("Did not find one (and only one) x or y coord")

    # Save each latlon slice2D in the cube
    for slice2D in cube.slices([y_coords[0], x_coords[0]]):
        grib_message = gribapi.grib_new_from_samples("GRIB2")
        _save_rules.run(slice2D, grib_message)
        yield (slice2D, grib_message)


def as_messages(cube):
    """
    Convert one or more cubes to GRIB messages.
    Returns an iterable of grib_api GRIB messages.

    Args:
        * cube      - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or
                      list of cubes.

    """
    return (message for cube, message in as_pairs(cube))


def save_messages(messages, target, append=False):
    """
    Save messages to a GRIB2 file.
    The messages will be released as part of the save.

    Args:

        * messages  - An iterable of grib_api message IDs.
        * target    - A filename or open file handle.

    Kwargs:

        * append    - Whether to start a new file afresh or add the cube(s) to
                      the end of the file.
                      Only applicable when target is a filename, not a file
                      handle. Default is False.

    """
    # grib file (this bit is common to the pp and grib savers...)
    if isinstance(target, basestring):
        grib_file = open(target, "ab" if append else "wb")
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" not in target.mode:
            raise ValueError("Target not binary")
        grib_file = target
    else:
        raise ValueError("Can only save grib to filename or writable")

    for message in messages:
        gribapi.grib_write(message, grib_file)
        gribapi.grib_release(message)

    # (this bit is common to the pp and grib savers...)
    if isinstance(target, basestring):
        grib_file.close()
