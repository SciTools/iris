# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :mod:`iris.fileformats.zarr`.

``zarr`` is optional, so every module below this one is skipped when it is
absent. The skip lives here rather than in a ``conftest.py``: the same call
there raises during collection and is reported as an **error**, not a skip.
"""

import pytest

pytest.importorskip("zarr", minversion="3.1.0")
