# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Experimental code can be introduced to Iris through this package.

Changes to experimental code may be more extensive than in the rest of the
codebase. The code is expected to graduate, eventually, to "full status".

"""

# import enum
# import threading
#
#
# class MeshCoordIndexing(threading.local):
#     class Settings(enum.Enum):
#         """Setting for what is returned when a :class:`~iris.mesh.MeshCoord` is indexed.
#
#         Use via the run-time setting :data:`~iris.experimental.MESH_COORD_INDEXING`.
#         """
#         # TODO: thread safety?
#         # TODO: docstring example
#
#         AUX_COORD = enum.auto()
#         """Convert the ``MeshCoord`` to a :class:`~iris.coords.AuxCoord`."""
#
#         NEW_MESH = enum.auto()
#         """
#         Index the ``MeshCoord`` :attr:`~iris.mesh.MeshCoord.mesh`, returning a new
#         :class:`~iris.mesh.MeshXY` instance.
#         """
#
#         MESH_INDEX_SET = enum.auto()
#         """
#         Index the ``MeshCoord`` :attr:`~iris.mesh.MeshCoord.mesh`, returning a
#         :class:`iris.mesh.components._MeshIndexSet` instance. This is a read-only
#         'view' onto the original :class:`~iris.mesh.MeshXY` instance, and is
#         currently experimental, subject to future iterative changes.
#         """
#
#     def __init__(self):
#         self._setting = MeshCoordIndexing.Settings.AUX_COORD
#
#     @property
#     def setting(self):
#         return self._setting
#
#     @setting.setter
#     def setting(self, value):
#         self._setting = value
#
#     # class Setting(threading.local):
#     #     # setting: "MeshCoordIndexing"
#     #
#     #     def __init__(self):
#     #         self.set(MeshCoordIndexing.AUX_COORD)
#     #
#     #     def set(self, value):
#     #         if isinstance(value, MeshCoordIndexing):
#     #             self.setting = value
#     #         else:
#     #             message = (
#     #                 "MeshCoordIndexing.Setting.set() must be called with a "
#     #                 f"MeshCoordIndexing enum value. Instead, got: {value}"
#     #             )
#     #             raise ValueError(message)
#
#
# MESH_COORD_INDEXING = MeshCoordIndexing()
# """
# Run-time setting for alternative modes of indexing a
# :class:`~iris.mesh.MeshCoord`. See
# :class:`~iris.experimental.MeshCoordIndexing` for more.
# """
