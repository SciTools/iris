# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provide capability to load netCDF files and interpret them.

.. z_reference:: iris.fileformats.cf
   :tags: topic_load_save

   API reference

Provides the capability to load netCDF files and interpret them
according to the 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

This package is a re-export surface and holds no logic of its own. Every
public name below is defined in exactly one private module and imported here
verbatim, so ``class CFReader`` still finds its definition in a single grep.
Import from :mod:`iris.fileformats.cf`, not from the private modules: the file
layout is deliberately free to change, and it will, as the format-agnostic CF
machinery grows a dataset abstraction, a loader and a saver.

References
----------
    [CF]  NetCDF Climate and Forecast (CF) Metadata conventions.
    [NUG] NetCDF User's Guide, https://docs.unidata.ucar.edu/nug/current/

"""

from ._group import CFGroup
from ._reader import CFReader, reference_terms
from ._variables import (
    CFAncillaryDataVariable,
    CFAuxiliaryCoordinateVariable,
    CFBoundaryVariable,
    CFClimatologyVariable,
    CFCoordinateVariable,
    CFDataVariable,
    CFGridMappingVariable,
    CFLabelVariable,
    CFMeasureVariable,
    CFUGridAuxiliaryCoordinateVariable,
    CFUGridConnectivityVariable,
    CFUGridMeshVariable,
    CFVariable,
)

__all__ = [
    "CFAncillaryDataVariable",
    "CFAuxiliaryCoordinateVariable",
    "CFBoundaryVariable",
    "CFClimatologyVariable",
    "CFCoordinateVariable",
    "CFDataVariable",
    "CFGridMappingVariable",
    "CFGroup",
    "CFLabelVariable",
    "CFMeasureVariable",
    "CFReader",
    "CFUGridAuxiliaryCoordinateVariable",
    "CFUGridConnectivityVariable",
    "CFUGridMeshVariable",
    "CFVariable",
    "reference_terms",
]
