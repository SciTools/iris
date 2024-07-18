# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Legacy import location for mesh support.

See :mod:`iris.ugrid` for the new, correct import location.

Notes
-----
This import path alios is provided for backwards compatibility, but will be removed
in a future release :  Please re-write code to import from the new module path.

This legacy import module will be removed in a future release.
N.B. it does **not** need to wait for a major release, since the former API was
experimental.

.. deprecated:: 3.10
    All the former :mod:`iris.experimental.ugrid` modules have been relocated to
    :mod:`iris.ugrid` and its submodules.  Please re-write code to import from the new
    module path.
    This import path alios is provided for backwards compatibility, but will be removed
    in a future release : N.B. removing this does **not** need to wait for a major
    release, since the former API was experimental.

"""

from .._deprecation import warn_deprecated
from ..ugrid.load import PARSE_UGRID_ON_LOAD, load_mesh, load_meshes
from ..ugrid.mesh import Connectivity, Mesh, MeshCoord
from ..ugrid.save import save_mesh
from ..ugrid.utils import recombine_submeshes

__all__ = [
    "Connectivity",
    "Mesh",
    "MeshCoord",
    "PARSE_UGRID_ON_LOAD",
    "load_mesh",
    "load_meshes",
    "recombine_submeshes",
    "save_mesh",
]

warn_deprecated(
    "All the former :mod:`iris.experimental.ugrid` modules have been relocated to "
    "module 'iris.ugrid' and its submodules. "
    "Please re-write code to import from the new module path."
)
