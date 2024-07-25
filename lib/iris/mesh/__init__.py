# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Infra-structure for unstructured mesh support.

Based on CF UGRID Conventions (v1.0), https://ugrid-conventions.github.io/ugrid-conventions/.
"""

from iris.config import get_logger
from iris.fileformats.netcdf.saver import save_mesh
from iris.fileformats.netcdf.ugrid_load import load_mesh, load_meshes

from .components import Connectivity, MeshCoord, MeshXY
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
