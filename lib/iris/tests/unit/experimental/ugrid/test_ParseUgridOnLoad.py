# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.experimental.ugrid.ParseUgridOnLoad` class.

TODO: remove this module when ParseUGridOnLoad itself is removed.

"""

import pytest

from iris._deprecation import IrisDeprecation
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD, ParseUGridOnLoad


def test_creation():
    # I.E. "does not fail".
    _ = ParseUGridOnLoad()


def test_context():
    ugridswitch = ParseUGridOnLoad()
    with pytest.warns(IrisDeprecation, match="PARSE_UGRID_ON_LOAD has been deprecated"):
        with ugridswitch.context():
            pass


def test_constant():
    assert isinstance(PARSE_UGRID_ON_LOAD, ParseUGridOnLoad)
