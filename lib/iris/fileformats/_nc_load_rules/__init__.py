# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Support for cube-specific CF-to-Iris translation operations.

Interprets CF concepts identified by :mod:`iris.fileformats.cf` to add
components into loaded cubes.

For now : the API which mimics :class:`pyke.knowledge_engine.engine`.
As this is aiming to replace the old Pyke-based logic rules.
TODO: simplify once the parallel operation with Pyke is no  longer required.

"""
