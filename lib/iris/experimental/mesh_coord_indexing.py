# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Experimental module for alternative modes of indexing a :class:`~iris.mesh.MeshCoord`.

:class:`Options` describes the available indexing modes.

Select the desired option using the run-time setting :data:`SETTING`.

Examples
--------
.. testsetup::

    import iris
    my_mesh_cube = iris.load_cube(iris.sample_data_path("mesh_C4_synthetic_float.nc"))

    # Remove non-compliant content.
    my_mesh = my_mesh_cube.mesh
    wanted_roles = ["edge_node_connectivity", "face_node_connectivity"]
    for conn in my_mesh.all_connectivities:
        if conn is not None and conn.cf_role not in wanted_roles:
            my_mesh.remove_connectivity(conn)

Here is a simple :class:`~iris.cube.Cube` with :class:`~iris.mesh.MeshCoord` s:

>>> print(my_mesh)
synthetic / (1)                     (-- : 96)
    Mesh coordinates:
        latitude                        x
        longitude                       x
    Mesh:
        name                        Topology data of 2D unstructured mesh
        location                    face
    Attributes:
        NCO                         'netCDF Operators version 4.7.5 (Homepage = http://nco.sf.net, Code = h ...'
        history                     'Mon Apr 12 01:44:41 2021: ncap2 -s synthetic=float(synthetic) mesh_C4_synthetic.nc ...'
        nco_openmp_thread_number    1
>>> print(my_mesh_cube.aux_coords)
(<MeshCoord: latitude / (degrees)  mesh(Topology data of 2D unstructured mesh) location(face)  [...]+bounds  shape(96,)>, <MeshCoord: longitude / (degrees)  mesh(Topology data of 2D unstructured mesh) location(face)  [...]+bounds  shape(96,)>)

Here is the default indexing behaviour:

>>> from iris.experimental import mesh_coord_indexing
>>> print(mesh_coord_indexing.SETTING.value)
Options.AUX_COORD
>>> indexed_cube = my_mesh_cube[:36]
>>> print(indexed_cube.aux_coords)
(<AuxCoord: latitude / (degrees)  [29.281, 33.301, ..., 33.301, 29.281]+bounds  shape(36,)>, <AuxCoord: longitude / (degrees)  [325.894, 348.621, ..., 191.379, 214.106]+bounds  shape(36,)>)

Set the indexing mode to return a new mesh:

>>> mesh_coord_indexing.SETTING.value = mesh_coord_indexing.Options.NEW_MESH
>>> indexed_cube = my_mesh_cube[:36]
>>> print(indexed_cube.aux_coords)
(<MeshCoord: latitude / (degrees)  mesh(Topology data of 2D unstructured mesh) location(face)  [...]+bounds  shape(36,)>, <MeshCoord: longitude / (degrees)  mesh(Topology data of 2D unstructured mesh) location(face)  [...]+bounds  shape(36,)>)

Or set via a context manager:

>>> with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.AUX_COORD):
...    indexed_cube = my_mesh_cube[:36]
...    print(indexed_cube.aux_coords)
(<AuxCoord: latitude / (degrees)  [29.281, 33.301, ..., 33.301, 29.281]+bounds  shape(36,)>, <AuxCoord: longitude / (degrees)  [325.894, 348.621, ..., 191.379, 214.106]+bounds  shape(36,)>)

"""

from contextlib import contextmanager
import enum
import threading


class Options(enum.Enum):
    """Options for what is returned when a :class:`~iris.mesh.MeshCoord` is indexed.

    See the module docstring for usage instructions:
    :mod:`~iris.experimental.mesh_coord_indexing`.
    """

    AUX_COORD = enum.auto()
    """The default. Convert the ``MeshCoord`` to a
    :class:`~iris.coords.AuxCoord` and index that AuxCoord.
    """

    NEW_MESH = enum.auto()
    """Index the :attr:`~iris.mesh.MeshCoord.mesh` of the ``MeshCoord`` to
    produce a new :class:`~iris.mesh.MeshXY` instance, then return a new
    :class:`~iris.mesh.MeshCoord` instance based on that new mesh.
    """

    MESH_INDEX_SET = enum.auto()
    """**EXPERIMENTAL.** Produce a :class:`iris.mesh.components._MeshIndexSet`
    instance that references the original :class:`~iris.mesh.MeshXY` instance,
    then return a new :class:`~iris.mesh.MeshCoord` instance based on that new
    index set. :class:`~Iris.mesh.components._MeshIndexSet` is a read-only
    indexed 'view' onto its original :class:`~iris.mesh.MeshXY`; behaviour of
    this class may change from release to release while the design is
    finalised.
    """


class _Setting(threading.local):
    """Setting for what is returned when a :class:`~iris.mesh.MeshCoord` is indexed.

    See the module docstring for usage instructions:
    :mod:`~iris.experimental.mesh_coord_indexing`.
    """

    def __init__(self):
        self._value = Options.AUX_COORD

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = Options(value)

    @contextmanager
    def context(self, value):
        new_value = Options(value)
        original_value = self._value
        try:
            self._value = new_value
            yield
        finally:
            self._value = original_value


SETTING = _Setting()
"""
Run-time setting for alternative modes of indexing a
:class:`~iris.mesh.MeshCoord`. See the module docstring for usage
instructions: :mod:`~iris.experimental.mesh_coord_indexing`.
"""
