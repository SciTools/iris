# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Provides iris loading support for UM Fieldsfile-like file types, and PP.

At present, the only UM file types supported are true FieldsFiles and LBCs.
Other types of UM file may fail to load correctly (or at all).

"""

from ._fast_load import FieldCollation, structured_um_loading

# Publish the FF-replacement features here, and include documentation.
from ._ff_replacement import load_cubes, load_cubes_32bit_ieee, um_to_pp

__all__ = [
    "FieldCollation",
    "load_cubes",
    "load_cubes_32bit_ieee",
    "structured_um_loading",
    "um_to_pp",
]
