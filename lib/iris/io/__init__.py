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
Provides an interface to manage URI scheme support in iris.

"""

from __future__ import (absolute_import, division, print_function)

import glob
import os.path
import types
import re
import collections

import iris.fileformats
import iris.fileformats.dot
import iris.cube
import iris.exceptions


# Saving routines, indexed by file extension.
class _SaversDict(dict):
    """A dictionary that can only have string keys with no overlap."""
    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise ValueError("key is not a string")
        if key in self.keys():
            raise ValueError("A saver already exists for", key)
        for k in self.keys():
            if k.endswith(key) or key.endswith(k):
                raise ValueError("key %s conflicts with existing key %s" % (key, k))
        dict.__setitem__(self, key, value)


_savers = _SaversDict()


def run_callback(callback, cube, field, filename):
    """
    Runs the callback mechanism given the appropriate arguments.

    Args:

    * callback:
        A function to add metadata from the originating field and/or URI which
        obeys the following rules:

            1. Function signature must be: ``(cube, field, filename)``.
            2. Modifies the given cube inplace, unless a new cube is
               returned by the function.
            3. If the cube is to be rejected the callback must raise
               an :class:`iris.exceptions.IgnoreCubeException`.

    .. note::

        It is possible that this function returns None for certain callbacks,
        the caller of this function should handle this case.

    """
    if callback is None:
        return cube

    # Call the callback function on the cube, generally the function will
    # operate on the cube in place, but it is also possible that the function
    # will return a completely new cube instance.
    try:
        result = callback(cube, field, filename)
    except iris.exceptions.IgnoreCubeException:
        result = None
    else:
        if result is None:
            result = cube
        elif not isinstance(result, iris.cube.Cube):
                raise TypeError("Callback function returned an "
                                "unhandled data type.")
    return result


def decode_uri(uri, default='file'):
    r'''
    Decodes a single URI into scheme and scheme-specific parts.

    In addition to well-formed URIs, it also supports bare file paths.
    Both Windows and UNIX style paths are accepted.

    .. testsetup::

        from iris.io import *

    Examples:
        >>> from iris.io import decode_uri
        >>> print(decode_uri('http://www.thing.com:8080/resource?id=a:b'))
        ('http', '//www.thing.com:8080/resource?id=a:b')

        >>> print(decode_uri('file:///data/local/dataZoo/...'))
        ('file', '///data/local/dataZoo/...')

        >>> print(decode_uri('/data/local/dataZoo/...'))
        ('file', '/data/local/dataZoo/...')

        >>> print(decode_uri('file:///C:\data\local\dataZoo\...'))
        ('file', '///C:\\data\\local\\dataZoo\\...')

        >>> print(decode_uri('C:\data\local\dataZoo\...'))
        ('file', 'C:\\data\\local\\dataZoo\\...')

        >>> print(decode_uri('dataZoo/...'))
        ('file', 'dataZoo/...')

    '''
    # make sure scheme has at least 2 letters to avoid windows drives
    # put - last in the brackets so it refers to the character, not a range
    # reference on valid schemes: http://tools.ietf.org/html/std66#section-3.1
    match = re.match(r"^([a-zA-Z][a-zA-Z0-9+.-]+):(.+)", uri)
    if match:
        scheme = match.group(1)
        part = match.group(2)
    else:
        # Catch bare UNIX and Windows paths
        scheme = default
        part = uri
    return scheme, part


def expand_filespecs(file_specs):
    """
    Find all matching file paths from a list of file-specs.

    Args:

    * file_specs (iterable of string):
        File paths which may contain '~' elements or wildcards.

    Returns:
        A list of matching file paths.  If any of the file-specs matches no
        existing files, an exception is raised.

    """
    # Remove any hostname component - currently unused
    filenames = [os.path.expanduser(fn[2:] if fn.startswith('//') else fn)
                 for fn in file_specs]

    # Try to expand all filenames as globs
    glob_expanded = {fn : sorted(glob.glob(fn)) for fn in filenames}

    # If any of the specs expanded to an empty list then raise an error
    value_lists = glob_expanded.viewvalues()
    if not all(value_lists):
        raise IOError("One or more of the files specified did not exist %s." %
        ["%s expanded to %s" % (pattern, expanded if expanded else "empty")
         for pattern, expanded in glob_expanded.iteritems()])

    return sum(value_lists, [])


def load_files(filenames, callback, constraints=None):
    """
    Takes a list of filenames which may also be globs, and optionally a
    constraint set and a callback function, and returns a
    generator of Cubes from the given files.

    .. note::

        Typically, this function should not be called directly; instead, the
        intended interface for loading is :func:`iris.load`.

    """
    all_file_paths = expand_filespecs(filenames)

    # Create default dict mapping iris format handler to its associated filenames
    handler_map = collections.defaultdict(list)
    for fn in all_file_paths:
        with open(fn, 'rb') as fh:
            handling_format_spec = iris.fileformats.FORMAT_AGENT.get_spec(os.path.basename(fn), fh)
            handler_map[handling_format_spec].append(fn)

    # Call each iris format handler with the approriate filenames
    for handling_format_spec, fnames in handler_map.iteritems():
        if handling_format_spec.constraint_aware_handler:
            for cube in handling_format_spec.handler(fnames, callback,
                                                     constraints):
                yield cube
        else:
            for cube in handling_format_spec.handler(fnames, callback):
                yield cube


def load_http(urls, callback):
    """
    Takes a list of urls and a callback function, and returns a generator
    of Cubes from the given URLs.

    .. note::

        Typically, this function should not be called directly; instead, the
        intended interface for loading is :func:`iris.load`.

    """
    # Create default dict mapping iris format handler to its associated filenames
    handler_map = collections.defaultdict(list)
    for url in urls:
        handling_format_spec = iris.fileformats.FORMAT_AGENT.get_spec(url, None)
        handler_map[handling_format_spec].append(url)

    # Call each iris format handler with the appropriate filenames
    for handling_format_spec, fnames in handler_map.iteritems():
        for cube in handling_format_spec.handler(fnames, callback):
            yield cube


def _check_init_savers():
    # TODO: Raise a ticket to resolve the cyclic import error that requires
    # us to initialise this on first use. Probably merge io and fileformats.
    if "pp" not in _savers:
        _savers.update({"pp": iris.fileformats.pp.save,
                        "nc": iris.fileformats.netcdf.save,
                        "dot": iris.fileformats.dot.save,
                        "dotpng": iris.fileformats.dot.save_png,
                        "grib2": iris.fileformats.grib.save_grib2})


def add_saver(file_extension, new_saver):
    """
    Add a custom saver to the Iris session.

    Args:

        * file_extension - A string such as "pp" or "my_format".
        * new_saver      - A function of the form ``my_saver(cube, target)``.

    See also :func:`iris.io.save`

    """
    # Make sure it's a func with 2+ args
    if not hasattr(new_saver, "__call__") or new_saver.__code__.co_argcount < 2:
        raise ValueError("Saver routines must be callable with 2+ arguments.")

    # Try to add this saver. Invalid keys will be rejected.
    _savers[file_extension] = new_saver


def find_saver(filespec):
    """
    Find the saver function appropriate to the given filename or extension.

    Args:

        * filespec - A string such as "my_file.pp" or "PP".

    Returns:
        A save function or None.
        Save functions can be passed to :func:`iris.io.save`.

    """
    _check_init_savers()
    matches = [ext for ext in _savers if filespec.lower().endswith('.' + ext) or
                                         filespec.lower() == ext]
    # Multiple matches could occur if one of the savers included a '.':
    #   e.g. _savers = {'.dot.png': dot_png_saver, '.png': png_saver}
    if len(matches) > 1:
        fmt = "Multiple savers found for %r: %s"
        matches = ', '.join(map(repr, matches))
        raise ValueError(fmt % (filespec, matches))
    return _savers[matches[0]] if matches else None


def save(source, target, saver=None, **kwargs):
    """
    Save one or more Cubes to file (or other writable).

    Iris currently supports three file formats for saving, which it can
    recognise by filename extension:

        * netCDF - the Unidata network Common Data Format:
            * see :func:`iris.fileformats.netcdf.save`
        * GRIB2  - the WMO GRIdded Binary data format;
            * see :func:`iris.fileformats.grib.save_grib2`
        * PP     - the Met Office UM Post Processing Format.
            * see :func:`iris.fileformats.pp.save`

    A custom saver can be provided to the function to write to a different
    file format.

    Args:

        * source    - A :class:`iris.cube.Cube`, :class:`iris.cube.CubeList` or
                      sequence of cubes.
        * target    - A filename (or writable, depending on file format).
                      When given a filename or file, Iris can determine the
                      file format.

    Kwargs:

        * saver     - Optional. Specifies the save function to use.
                      If omitted, Iris will attempt to determine the format.

                      This keyword can be used to implement a custom save
                      format. Function form must be:
                      ``my_saver(cube, target)`` plus any custom keywords. It
                      is assumed that a saver will accept an ``append`` keyword
                      if it's file format can handle multiple cubes. See also
                      :func:`iris.io.add_saver`.

    All other keywords are passed through to the saver function; see the
    relevant saver documentation for more information on keyword arguments.

    Examples::

        # Save a cube to PP
        iris.save(my_cube, "myfile.pp")

        # Save a cube list to a PP file, appending to the contents of the file
        # if it already exists
        iris.save(my_cube_list, "myfile.pp", append=True)

        # Save a cube to netCDF, defaults to NETCDF4 file format
        iris.save(my_cube, "myfile.nc")

        # Save a cube list to netCDF, using the NETCDF4_CLASSIC storage option
        iris.save(my_cube_list, "myfile.nc", netcdf_format="NETCDF3_CLASSIC")

    """
    # Determine format from filename
    if isinstance(target, basestring) and saver is None:
        saver = find_saver(target)
    elif isinstance(target, types.FileType) and saver is None:
        saver = find_saver(target.name)
    elif isinstance(saver, basestring):
        saver = find_saver(saver)
    if saver is None:
        raise ValueError("Cannot save; no saver")

    # Don't overwrite!
    if ((not('append' in kwargs) or kwargs['append'] is False)
            and os.path.exists(target)):
        msg = "File \'{!s}\' exists and \'append\' has not been set True."
        raise ValueError(msg.format(target))

    # Single cube?
    if isinstance(source, iris.cube.Cube):
        saver(source, target, **kwargs)

    # CubeList or sequence of cubes?
    elif (isinstance(source, iris.cube.CubeList) or
          (isinstance(source, (list, tuple)) and
           all([isinstance(i, iris.cube.Cube) for i in source]))):
        # Only allow cubelist saving for those fileformats that are capable.
        if not 'iris.fileformats.netcdf' in saver.__module__:
            # Make sure the saver accepts an append keyword
            if not "append" in saver.__code__.co_varnames:
                raise ValueError("Cannot append cubes using saver function "
                                 "'%s' in '%s'" %
                                 (saver.__code__.co_name,
                                  saver.__code__.co_filename))
            # Force append=True for the tail cubes. Don't modify the incoming
            # kwargs.
            kwargs = kwargs.copy()
            for i, cube in enumerate(source):
                if i != 0:
                    kwargs['append'] = True
                saver(cube, target, **kwargs)
        # Netcdf saver.
        else:
            saver(source, target, **kwargs)

    else:
        raise ValueError("Cannot save; non Cube found in source")
