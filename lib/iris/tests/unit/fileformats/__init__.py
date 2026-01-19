# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.fileformats` package."""

import pytest
from pytest_mock import MockerFixture


class MockerMixin:
    mocker: MockerFixture

    @pytest.fixture(autouse=True)
    def _mocker_mixin_setup(self, mocker):
        self.mocker = mocker
