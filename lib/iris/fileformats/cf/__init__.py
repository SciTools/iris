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

#: Supported dimensionless vertical coordinate reference surface/phemomenon
#: formula terms. Ref: [CF] Appendix D.
reference_terms = reference_terms
# Re-stating the name here, with the doc comment that travels with it, is what
# puts it on the API reference page. Sphinx autodoc documents module data only
# where the module's own source assigns it; an imported name it silently drops,
# with no warning and no build failure. See the plan, section 8.4.

# Present every re-exported class as belonging to this module, which is where
# it was defined before the package split and where callers are told to reach
# it. This is not cosmetic: __module__ is what repr() prints, what pickle
# records, and what Sphinx uses to decide a class's canonical name. Leaving it
# pointing at the private module would change all three, and would make Sphinx
# register CFGroup twice - once directly and once as the CFReader.CFGroup
# alias - which fails the docs build outright under RTD's fail_on_warning.
for _name in __all__:
    _obj = globals()[_name]
    if isinstance(_obj, type):
        _obj.__module__ = __name__
del _name, _obj
