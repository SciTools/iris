import enum
import threading


class Options(enum.Enum):
    # TODO: docstring

    AUX_COORD = enum.auto()
    """Convert the ``MeshCoord`` to a :class:`~iris.coords.AuxCoord`."""

    NEW_MESH = enum.auto()
    """
    Index the ``MeshCoord`` :attr:`~iris.mesh.MeshCoord.mesh`, returning a new 
    :class:`~iris.mesh.MeshXY` instance.
    """

    MESH_INDEX_SET = enum.auto()
    """
    Index the ``MeshCoord`` :attr:`~iris.mesh.MeshCoord.mesh`, returning a
    :class:`iris.mesh.components._MeshIndexSet` instance. This is a read-only 
    'view' onto the original :class:`~iris.mesh.MeshXY` instance, and is
    currently experimental, subject to future iterative changes.
    """


class Setting(threading.local):
    """Setting for what is returned when a :class:`~iris.mesh.MeshCoord` is indexed.

    Use via the run-time setting
    :data:`iris.experimental.mesh_coord_indexing.SETTING`.

    See :class:`iris.experimental.mesh_coord_indexing.Options` for the valid
    settings.
    """
    # TODO: docstring example
    # TODO: context manager

    def __init__(self):
        self._value = Options.AUX_COORD

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = Options(value)


SETTING = Setting()
"""
Run-time setting for alternative modes of indexing a
:class:`~iris.mesh.MeshCoord`. See
:class:`iris.experimental.mesh_coord_indexing.Setting` for more.
"""
