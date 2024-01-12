# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Support for UM "fieldsfile-like" files.

At present, the only UM file types supported are true FieldsFiles and LBCs.
Other types of UM file may fail to load correctly (or at all).

"""

from iris.fileformats._ff import FF2PP
from iris.fileformats.pp import _load_cubes_variable_loader


def um_to_pp(filename, read_data=False, word_depth=None):
    """Extract individual PPFields from within a UM Fieldsfile-like file.

    Returns an iterator over the fields contained within the FieldsFile,
    returned as :class:`iris.fileformats.pp.PPField` instances.

    Parameters
    ----------
    filename : str
        Specify the name of the FieldsFile.
    read_data : bool, optional, default=read_data
        Specify whether to read the associated PPField data within
        the FieldsFile.  Default value is False.
    word_depth : optional, default=None

    Returns
    -------
    Iteration of :class:`iris.fileformats.pp.PPField`.

    Examples
    --------
    ::

        >>> for field in um.um_to_pp(filename):
        ...     print(field)

    """
    if word_depth is None:
        ff2pp = FF2PP(filename, read_data=read_data)
    else:
        ff2pp = FF2PP(filename, read_data=read_data, word_depth=word_depth)

    # Note: unlike the original wrapped case, we will return an actual
    # iterator, rather than an object that can provide an iterator.
    return iter(ff2pp)


def load_cubes(filenames, callback, constraints=None, _loader_kwargs=None):
    """Loads cubes from filenames of UM fieldsfile-like files.

    Parameters
    ----------
    filenames :
        list of filenames to load
    callback :
        A function which can be passed on to :func:`iris.io.run_callback`
    constraints : optional, default=None
    _loader_kwargs : optional, default=None

    Notes
    -----
    .. note::

        The resultant cubes may not be in the order that they are in the
        file (order is not preserved when there is a field with
        orography references).

    """
    return _load_cubes_variable_loader(
        filenames,
        callback,
        FF2PP,
        constraints=constraints,
        loading_function_kwargs=_loader_kwargs,
    )


def load_cubes_32bit_ieee(filenames, callback, constraints=None):
    """Loads cubes from filenames of 32bit ieee converted UM fieldsfile-like files.

    See Also
    --------
    :func:`load_cubes`
        For keyword details

    """
    return load_cubes(
        filenames,
        callback,
        constraints=constraints,
        _loader_kwargs={"word_depth": 4},
    )
