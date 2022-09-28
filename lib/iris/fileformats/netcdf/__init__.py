# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module to support the loading and saving of NetCDF files, also using the CF conventions
for metadata interpretation.

See : `NetCDF User's Guide <https://docs.unidata.ucar.edu/nug/current/>`_
and `netCDF4 python module <https://github.com/Unidata/netcdf4-python>`_.

Also : `CF Conventions <https://cfconventions.org/>`_.

"""
import iris.config

# Note: *must* be done before importing from submodules, as they also use this !
logger = iris.config.get_logger(__name__)

from .loader import DEBUG, NetCDFDataProxy, load_cubes
from .saver import (
    CF_CONVENTIONS_VERSION,
    MESH_ELEMENTS,
    SPATIO_TEMPORAL_AXES,
    CFNameCoordMap,
    Saver,
    UnknownCellMethodWarning,
    parse_cell_methods,
    save,
)

# Export all public elements from the loader and saver submodules.
# NOTE: the separation is purely for neatness and developer convenience; from
# the user point of view, it is still all one module.
__all__ = (
    "CFNameCoordMap",
    "CF_CONVENTIONS_VERSION",
    "DEBUG",
    "MESH_ELEMENTS",
    "NetCDFDataProxy",
    "SPATIO_TEMPORAL_AXES",
    "Saver",
    "UnknownCellMethodWarning",
    "load_cubes",
    "logger",
    "parse_cell_methods",
    "save",
)
