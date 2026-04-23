# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.experimental.geovista` module."""

import pytest

# Skip this whole package if geovista (and by extension pyvista) is not available:
pytest.importorskip(
    "geovista", reason="Skipping geovista unit tests as `geovista` is not installed"
)
