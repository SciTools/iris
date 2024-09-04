# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Legacy import location for mesh support.

See :mod:`iris.mesh` for the new, correct import location.

Notes
-----
This import path alios is provided for backwards compatibility, but will be removed
in a future release :  Please re-write code to import from the new module path.

This legacy import module will be removed in a future release.
N.B. it does **not** need to wait for a major release, since the former API was
experimental.

.. deprecated:: 3.10
    All the former :mod:`iris.experimental.mesh` modules have been relocated to
    :mod:`iris.mesh` and its submodules.  Please re-write code to import from the new
    module path, and replace any 'iris.experimental.ugrid.Mesh' with
    'iris.mesh.MeshXY'.

    This import path alias is provided for backwards compatibility, but will be removed
    in a future release : N.B. removing this does **not** need to wait for a major
    release, since the API is experimental.

"""

from __future__ import annotations

from contextlib import contextmanager
import threading

from .._deprecation import warn_deprecated
from ..mesh import Connectivity as _Connectivity
from ..mesh import MeshCoord as _MeshCoord
from ..mesh import MeshXY as _MeshXY
from ..mesh import load_mesh, load_meshes, recombine_submeshes, save_mesh


# NOTE: publishing the original Mesh, MeshCoord and Connectivity here causes a Sphinx
# Sphinx warning, E.G.:
#   "WARNING: duplicate object description of iris.mesh.Mesh, other instance
#       in generated/api/iris.experimental.mesh, use :no-index: for one of them"
# For some reason, this only happens for the classes, and not the functions.
#
# This is a fatal problem, i.e. breaks the build since we are building with -W.
# We couldn't fix this with "autodoc_suppress_warnings", so the solution for now is to
# wrap the classes.  Which is really ugly.
class Mesh(_MeshXY):
    pass


class MeshCoord(_MeshCoord):
    pass


class Connectivity(_Connectivity):
    pass


class ParseUGridOnLoad(threading.local):
    def __init__(self):
        """Thead-safe state to enable UGRID-aware NetCDF loading.

        A flag for dictating whether to use the experimental UGRID-aware
        version of Iris NetCDF loading. Object is thread-safe.

        Use via the run-time switch
        :const:`~iris.mesh.load.PARSE_UGRID_ON_LOAD`.
        Use :meth:`context` to temporarily activate.

        Notes
        -----
            .. deprecated:: 1.10
        Do not use -- due to be removed at next major release :
        UGRID loading is now **always** active for files containing a UGRID mesh.

        """

    def __bool__(self):
        return True

    @contextmanager
    def context(self):
        """Activate UGRID-aware NetCDF loading.

        Use the standard Iris loading API while within the context manager. If
        the loaded file(s) include any UGRID content, this will be parsed and
        attached to the resultant cube(s) accordingly.

        Use via the run-time switch
        :const:`~iris.mesh.load.PARSE_UGRID_ON_LOAD`.

        For example::

            with PARSE_UGRID_ON_LOAD.context():
                my_cube_list = iris.load([my_file_path, my_file_path2],
                                         constraint=my_constraint,
                                         callback=my_callback)

        Notes
        -----
            .. deprecated:: 1.10
        Do not use -- due to be removed at next major release :
        UGRID loading is now **always** active for files containing a UGRID mesh.

        Examples
        --------
        Replace usage, for example:

        .. code-block:: python

            with iris.experimental.mesh.PARSE_UGRID_ON_LOAD.context():
                mesh_cubes = iris.load(path)

        with:

        .. code-block:: python

            mesh_cubes = iris.load(path)

        """
        wmsg = (
            "iris.experimental.mesh.load.PARSE_UGRID_ON_LOAD has been deprecated "
            "and will be removed. Please remove all uses : these are no longer needed, "
            "as UGRID loading is now applied to any file containing a mesh."
        )
        warn_deprecated(wmsg)
        yield


#: Run-time switch for experimental UGRID-aware NetCDF loading. See :class:`~iris.mesh.load.ParseUGridOnLoad`.
PARSE_UGRID_ON_LOAD = ParseUGridOnLoad()


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
    "All the former :mod:`iris.experimental.mesh` modules have been relocated to "
    "module 'iris.mesh' and its submodules. "
    "Please re-write code to import from the new module path, and replace any "
    "'iris.experimental.ugrid.Mesh' with 'iris.mesh.MeshXY'."
)
