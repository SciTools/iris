# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides an interface to manage URI scheme support in iris.

"""

import collections
from collections import OrderedDict
import glob
import os.path
import pathlib
import re

import iris.exceptions


# Saving routines, indexed by file extension.
class _SaversDict(dict):
    """A dictionary that can only have string keys with no overlap."""

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise ValueError("key is not a string")
        if key in self:
            raise ValueError("A saver already exists for", key)
        for k in self.keys():
            if k.endswith(key) or key.endswith(k):
                raise ValueError(
                    "key %s conflicts with existing key %s" % (key, k)
                )
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

    .. note::

        This function maintains laziness when called; it does not realise data.
        See more at :doc:`/userguide/real_and_lazy_data`.

    """
    from iris.cube import Cube

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
        elif not isinstance(result, Cube):
            raise TypeError(
                "Callback function returned an " "unhandled data type."
            )
    return result


def decode_uri(uri, default="file"):
    r"""
    Decodes a single URI into scheme and scheme-specific parts.

    In addition to well-formed URIs, it also supports bare file paths as strings
    or :class:`pathlib.PurePath`. Both Windows and UNIX style paths are
    accepted.

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

    """
    if isinstance(uri, pathlib.PurePath):
        uri = str(uri)
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


def expand_filespecs(file_specs, files_expected=True):
    """
    Find all matching file paths from a list of file-specs.

    Parameters
    ----------
    file_specs : iterable of str
        File paths which may contain ``~`` elements or wildcards.
    files_expected : bool, default=True
        Whether file is expected to exist (i.e. for load).

    Returns
    -------
    list of str
        if files_expected is ``True``:
            A well-ordered list of matching absolute file paths.
            If any of the file-specs match no existing files, an
            exception is raised.
        if files_expected is ``False``:
            A list of expanded file paths.
    """
    # Remove any hostname component - currently unused
    filenames = [
        os.path.abspath(
            os.path.expanduser(fn[2:] if fn.startswith("//") else fn)
        )
        for fn in file_specs
    ]

    if files_expected:
        # Try to expand all filenames as globs
        glob_expanded = OrderedDict(
            [[fn, sorted(glob.glob(fn))] for fn in filenames]
        )

        # If any of the specs expanded to an empty list then raise an error
        all_expanded = glob_expanded.values()
        if not all(all_expanded):
            msg = "One or more of the files specified did not exist:"
            for pattern, expanded in glob_expanded.items():
                if expanded:
                    msg += '\n    - "{}" matched {} file(s)'.format(
                        pattern, len(expanded)
                    )
                else:
                    msg += '\n    * "{}" didn\'t match any files'.format(
                        pattern
                    )
            raise IOError(msg)
        result = [fname for fnames in all_expanded for fname in fnames]
    else:
        result = filenames
    return result


def load_files(filenames, callback, constraints=None):
    """
    Takes a list of filenames which may also be globs, and optionally a
    constraint set and a callback function, and returns a
    generator of Cubes from the given files.

    .. note::

        Typically, this function should not be called directly; instead, the
        intended interface for loading is :func:`iris.load`.

    """
    from iris.fileformats import FORMAT_AGENT

    all_file_paths = expand_filespecs(filenames)

    # Create default dict mapping iris format handler to its associated filenames
    handler_map = collections.defaultdict(list)
    for fn in all_file_paths:
        with open(fn, "rb") as fh:
            handling_format_spec = FORMAT_AGENT.get_spec(
                os.path.basename(fn), fh
            )
            handler_map[handling_format_spec].append(fn)

    # Call each iris format handler with the approriate filenames
    for handling_format_spec in sorted(handler_map):
        fnames = handler_map[handling_format_spec]
        if handling_format_spec.constraint_aware_handler:
            for cube in handling_format_spec.handler(
                fnames, callback, constraints
            ):
                yield cube
        else:
            for cube in handling_format_spec.handler(fnames, callback):
                yield cube


def load_http(urls, callback):
    """
    Takes a list of OPeNDAP URLs and a callback function, and returns a generator
    of Cubes from the given URLs.

    .. note::

        Typically, this function should not be called directly; instead, the
        intended interface for loading is :func:`iris.load`.

    """
    # Create default dict mapping iris format handler to its associated filenames
    from iris.fileformats import FORMAT_AGENT

    handler_map = collections.defaultdict(list)
    for url in urls:
        handling_format_spec = FORMAT_AGENT.get_spec(url, None)
        handler_map[handling_format_spec].append(url)

    # Call each iris format handler with the appropriate filenames
    for handling_format_spec in sorted(handler_map):
        fnames = handler_map[handling_format_spec]
        for cube in handling_format_spec.handler(fnames, callback):
            yield cube


def _dot_save(cube, target):
    # A simple wrapper for `iris.fileformats.dot.save` which allows the
    # saver to be registered without triggering the import of
    # `iris.fileformats.dot`.
    from iris.fileformats.dot import save

    return save(cube, target)


def _dot_save_png(cube, target, **kwargs):
    # A simple wrapper for `iris.fileformats.dot.save_png` which allows the
    # saver to be registered without triggering the import of
    # `iris.fileformats.dot`.
    from iris.fileformats.dot import save_png

    return save_png(cube, target, **kwargs)


def _grib_save(cube, target, append=False, **kwargs):
    # A simple wrapper for the grib save routine, which allows the saver to be
    # registered without having the grib implementation installed.
    try:
        from iris_grib import save_grib2
    except ImportError:
        raise RuntimeError(
            "Unable to save GRIB file - "
            '"iris_grib" package is not installed.'
        )

    save_grib2(cube, target, append, **kwargs)


def _check_init_savers():
    from iris.fileformats import netcdf, pp

    if "pp" not in _savers:
        _savers.update(
            {
                "pp": pp.save,
                "nc": netcdf.save,
                "dot": _dot_save,
                "dotpng": _dot_save_png,
                "grib2": _grib_save,
            }
        )


def add_saver(file_extension, new_saver):
    """
    Add a custom saver to the Iris session.

    Args:

    * file_extension: A string such as "pp" or "my_format".
    * new_saver:      A function of the form ``my_saver(cube, target)``.

    See also :func:`iris.io.save`

    """
    # Make sure it's a func with 2+ args
    if (
        not hasattr(new_saver, "__call__")
        or new_saver.__code__.co_argcount < 2
    ):
        raise ValueError("Saver routines must be callable with 2+ arguments.")

    # Try to add this saver. Invalid keys will be rejected.
    _savers[file_extension] = new_saver


def find_saver(filespec):
    """
    Find the saver function appropriate to the given filename or extension.

    Args:

        * filespec
            A string such as "my_file.pp" or "PP".

    Returns:
        A save function or None.
        Save functions can be passed to :func:`iris.io.save`.

    """
    _check_init_savers()
    matches = [
        ext
        for ext in _savers
        if filespec.lower().endswith("." + ext) or filespec.lower() == ext
    ]
    # Multiple matches could occur if one of the savers included a '.':
    #   e.g. _savers = {'.dot.png': dot_png_saver, '.png': png_saver}
    if len(matches) > 1:
        fmt = "Multiple savers found for %r: %s"
        matches = ", ".join(map(repr, matches))
        raise ValueError(fmt % (filespec, matches))
    return _savers[matches[0]] if matches else None


def save(source, target, saver=None, **kwargs):
    """
    Save one or more Cubes to file (or other writeable).

    Iris currently supports three file formats for saving, which it can
    recognise by filename extension:

        * netCDF - the Unidata network Common Data Format:
            * see :func:`iris.fileformats.netcdf.save`
        * GRIB2 - the WMO GRIdded Binary data format:
            * see :func:`iris_grib.save_grib2`.
        * PP - the Met Office UM Post Processing Format:
            * see :func:`iris.fileformats.pp.save`

    A custom saver can be provided to the function to write to a different
    file format.

    Parameters
    ----------
    source : :class:`iris.cube.Cube` or :class:`iris.cube.CubeList`
    target : str or pathlib.PurePath or io.TextIOWrapper
        When given a filename or file, Iris can determine the
        file format.
    saver : str or function, optional
        Specifies the file format to save.
        If omitted, Iris will attempt to determine the format.
        If a string, this is the recognised filename extension
        (where the actual filename may not have it).

        Otherwise the value is a saver function, of the form:
        ``my_saver(cube, target)`` plus any custom keywords. It
        is assumed that a saver will accept an ``append`` keyword
        if its file format can handle multiple cubes. See also
        :func:`iris.io.add_saver`.
    **kwargs : dict, optional
        All other keywords are passed through to the saver function; see the
        relevant saver documentation for more information on keyword arguments.

    Warnings
    --------
    Saving a cube whose data has been loaded lazily
    (if `cube.has_lazy_data()` returns `True`) to the same file it expects
    to load data from will cause both the data in-memory and the data on
    disk to be lost.

    .. code-block:: python

       cube = iris.load_cube("somefile.nc")
       # The next line causes data loss in 'somefile.nc' and the cube.
       iris.save(cube, "somefile.nc")

    In general, overwriting a file which is the source for any lazily loaded
    data can result in corruption. Users should proceed with caution when
    attempting to overwrite an existing file.

    Examples
    --------
    >>> # Setting up
    >>> import iris
    >>> my_cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> my_cube_list = iris.load(iris.sample_data_path('space_weather.nc'))

    >>> # Save a cube to PP
    >>> iris.save(my_cube, "myfile.pp")

    >>> # Save a cube list to a PP file, appending to the contents of the file
    >>> # if it already exists
    >>> iris.save(my_cube_list, "myfile.pp", append=True)

    >>> # Save a cube to netCDF, defaults to NETCDF4 file format
    >>> iris.save(my_cube, "myfile.nc")

    >>> # Save a cube list to netCDF, using the NETCDF3_CLASSIC storage option
    >>> iris.save(my_cube_list, "myfile.nc", netcdf_format="NETCDF3_CLASSIC")

    Notes
    ------

    This function maintains laziness when called; it does not realise data.
    See more at :doc:`/userguide/real_and_lazy_data`.

    """
    from iris.cube import Cube, CubeList

    # Determine format from filename
    if isinstance(target, pathlib.PurePath):
        target = str(target)
    if isinstance(target, str) and saver is None:
        # Converts tilde or wildcards to absolute path
        (target,) = expand_filespecs([str(target)], False)
        saver = find_saver(target)
    elif hasattr(target, "name") and saver is None:
        saver = find_saver(target.name)
    elif isinstance(saver, str):
        saver = find_saver(saver)
    if saver is None:
        raise ValueError("Cannot save; no saver")

    # Single cube?
    if isinstance(source, Cube):
        saver(source, target, **kwargs)

    # CubeList or sequence of cubes?
    elif isinstance(source, CubeList) or (
        isinstance(source, (list, tuple))
        and all([isinstance(i, Cube) for i in source])
    ):
        # Only allow cubelist saving for those fileformats that are capable.
        if "iris.fileformats.netcdf" not in saver.__module__:
            # Make sure the saver accepts an append keyword
            if "append" not in saver.__code__.co_varnames:
                raise ValueError(
                    "Cannot append cubes using saver function "
                    "'%s' in '%s'"
                    % (saver.__code__.co_name, saver.__code__.co_filename)
                )
            # Force append=True for the tail cubes. Don't modify the incoming
            # kwargs.
            kwargs = kwargs.copy()
            for i, cube in enumerate(source):
                if i != 0:
                    kwargs["append"] = True
                saver(cube, target, **kwargs)
        # Netcdf saver.
        else:
            saver(source, target, **kwargs)

    else:
        raise ValueError("Cannot save; non Cube found in source")
