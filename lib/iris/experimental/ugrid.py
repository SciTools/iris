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
from ..ugrid.mesh import Connectivity as _Connectivity
from ..ugrid.mesh import MeshCoord as _MeshCoord
from ..ugrid.mesh import MeshXY as _MeshXY
from ..ugrid.save import save_mesh
from ..ugrid.utils import recombine_submeshes


# NOTE: publishing the original Mesh, MeshCoord and Connectivity here causes a Sphinx
# Sphinx warning, E.G.:
#   "WARNING: duplicate object description of iris.ugrid.mesh.Mesh, other instance
#       in generated/api/iris.experimental.ugrid, use :no-index: for one of them"
# For some reason, this only happens for the classes, and not the functions.
#
# This is a fatal problem, i.e. breaks the build since we are building with -W.
# We couldn't fix this with "autodoc_suppress_warnings", so the solution for now is to
# wrap the classes.  Which is really ugly.
# TODO: remove this when we remove iris.experimental.ugrid
class MeshXY(_MeshXY):
    pass


class MeshCoord(_MeshCoord):
    pass


class Connectivity(_Connectivity):
    pass


__all__ = [
    "Connectivity",
    "MeshCoord",
    "MeshXY",
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
