# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Infra-structure for unstructured mesh support.

.. deprecated:: 1.10

    :data:`PARSE_UGRID_ON_LOAD` is due to be removed at next major release.
    Please remove all uses of this, which are no longer needed :
    UGRID loading is now **always** active for files containing a UGRID mesh.

Based on CF UGRID Conventions (v1.0), https://ugrid-conventions.github.io/ugrid-conventions/.

.. note::

    For the docstring of :const:`PARSE_UGRID_ON_LOAD`: see the original
    definition at :const:`iris.ugrid.load.PARSE_UGRID_ON_LOAD`.

"""

from ..config import get_logger
from .load import load_mesh, load_meshes
from .mesh import Connectivity, MeshCoord, MeshXY
from .save import save_mesh
from .utils import recombine_submeshes

__all__ = [
    "Connectivity",
    "MeshCoord",
    "MeshXY",
    "load_mesh",
    "load_meshes",
    "recombine_submeshes",
    "save_mesh",
]

# Configure the logger as a root logger.
logger = get_logger(__name__, fmt="[%(cls)s.%(funcName)s]", level="NOTSET")
