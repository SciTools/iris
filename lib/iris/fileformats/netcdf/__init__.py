#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module to support the loading of a NetCDF file into an Iris cube.

See also: `netCDF4 python <https://github.com/Unidata/netcdf4-python>`_

Also refer to document 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

"""
from .loader import DEBUG, NetCDFDataProxy, load_cubes, logger
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
    CF_CONVENTIONS_VERSION,
    CFNameCoordMap,
    DEBUG,
    MESH_ELEMENTS,
    Saver,
    SPATIO_TEMPORAL_AXES,
    NetCDFDataProxy,
    UnknownCellMethodWarning,
    load_cubes,
    logger,
    parse_cell_methods,
    save,
)
