# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Support loading and saving NetCDF files using CF conventions for metadata interpretation.

.. z_reference:: iris.fileformats.netcdf
   :tags: topic_load_save

   API reference

See : `NetCDF User's Guide <https://docs.unidata.ucar.edu/nug/current/>`_
and `netCDF4 python module <https://github.com/Unidata/netcdf4-python>`_.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""

import logging

from iris._deprecation import warn_deprecated
import iris.config

# Note: *must* be done before importing from submodules, as they also use this !
logger: logging.Logger = iris.config.get_logger(__name__, propagate=True)

# Note: these probably shouldn't be public, but for now they are.
from .._nc_load_rules.helpers import UnknownCellMethodWarning, parse_cell_methods
from ._bytecoding_datasets import (
    DECODE_TO_STRINGS_ON_READ,
    DEFAULT_READ_ENCODING,
    DEFAULT_WRITE_ENCODING,
    SUPPORTED_ENCODINGS,
)
from .loader import DEBUG, NetCDFDataProxy, load_cubes
from .saver import Saver, save

# Export all public elements from the loader and saver submodules.
# NOTE: the separation is purely for neatness and developer convenience; from
# the user point of view, it is still all one module.
__all__ = (
    "CFNameCoordMap",
    "CF_CONVENTIONS_VERSION",
    "DEBUG",
    "DEFAULT_READ_ENCODING",
    "DEFAULT_WRITE_ENCODING",
    "MESH_ELEMENTS",
    "NetCDFDataProxy",
    "SPATIO_TEMPORAL_AXES",
    "SUPPORTED_ENCODINGS",
    "Saver",
    "UnknownCellMethodWarning",
    "load_cubes",
    "logger",
    "parse_cell_methods",
    "save",
)


def __getattr__(name):
    """Forward a relocated name to :mod:`iris.fileformats.cf.saver`.

    .. deprecated:: 3.17
        Four saver names moved to :mod:`iris.fileformats.cf.saver`, because
        they are not specific to netCDF. Reading them here still works, and
        warns.

    Parameters
    ----------
    name : str
        The attribute being looked up.

    Returns
    -------
    Any
        The object of that name - the same object, not a copy.

    Raises
    ------
    AttributeError
        If ``name`` is not one of the relocated names.

    """
    from .saver import _RELOCATED_NAMES

    if name not in _RELOCATED_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from iris.fileformats.cf import saver as cf_saver

    warn_deprecated(
        f"iris.fileformats.netcdf.{name} has moved to "
        f"iris.fileformats.cf.saver.{name}, because it is not specific to "
        "netCDF. The name here will be removed in a future release.",
        stacklevel=3,
    )
    return getattr(cf_saver, name)


def __dir__():
    """List module attributes, including the relocated names.

    Returns
    -------
    list of str
        The real module globals, unioned with the relocated names.

    """
    from .saver import _RELOCATED_NAMES

    return sorted(set(globals()) | _RELOCATED_NAMES)
