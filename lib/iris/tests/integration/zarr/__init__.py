# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for native Zarr loading.

Skipped when zarr-python is absent, which is also how the unit tests do it.
"""

import pytest

pytest.importorskip("zarr", minversion="3.1.0")
