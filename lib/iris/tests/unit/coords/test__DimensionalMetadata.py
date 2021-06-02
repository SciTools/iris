# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.coords._DimensionalMetadata` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.coords import _DimensionalMetadata


class Test___init____abstractmethod(tests.IrisTest):
    def test(self):
        emsg = (
            "Can't instantiate abstract class _DimensionalMetadata with "
            "abstract methods __init__"
        )
        with self.assertRaisesRegex(TypeError, emsg):
            _ = _DimensionalMetadata(0)


if __name__ == "__main__":
    tests.main()
