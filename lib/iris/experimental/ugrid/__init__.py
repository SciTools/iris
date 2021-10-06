# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

"""
Infra-structure for unstructured mesh support, based on
CF UGRID Conventions (v1.0), https://ugrid-conventions.github.io/ugrid-conventions/

.. note::

    For the docstring of :const:`PARSE_UGRID_ON_LOAD`: see the original
    definition at :const:`iris.experimental.ugrid.load.PARSE_UGRID_ON_LOAD`.

"""
from ...config import get_logger
from .load import PARSE_UGRID_ON_LOAD, load_mesh, load_meshes
from .mesh import Connectivity, Mesh, MeshCoord
from .save import save_mesh

__all__ = [
    "Connectivity",
    "Mesh",
    "MeshCoord",
    "PARSE_UGRID_ON_LOAD",
    "load_mesh",
    "load_meshes",
    "save_mesh",
]

# Configure the logger as a root logger.
logger = get_logger(__name__, fmt="[%(cls)s.%(funcName)s]", level="NOTSET")
