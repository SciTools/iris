# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Top-level fixture infra-structure.

Before adding to this: consider if :mod:`iris.tests.unit.conftest` or
:mod:`iris.tests.integration.conftest` might be more appropriate.
"""

from collections import defaultdict
from typing import Callable

import pytest

import iris.tests.graphics


@pytest.fixture(scope="session", autouse=True)
def test_call_counter():
    """Provide a session-persistent tracker of the number of calls per test name.

    Used by :func:`_unique_id` to ensure uniqueness if called multiple times
    per test.
    """
    counter = defaultdict(int)
    return counter


@pytest.fixture
def _unique_id(request: pytest.FixtureRequest, test_call_counter) -> Callable:
    """Provide a function returning a unique ID of calling test and call number.

    Example: ``iris.tests.unit.test_cube.TestCube.test_data.my_param.0``

    Used by :func:`iris.tests.graphics.check_graphic_caller` to ensure unique
    image names.
    """
    id_sequence = [request.module.__name__, request.node.originalname]
    if request.cls is not None:
        id_sequence.insert(-1, request.cls.__name__)
    if hasattr(request.node, "callspec"):
        id_sequence.append(request.node.callspec.id)
    test_id = ".".join(id_sequence)

    def generate_id():
        assertion_id = test_call_counter[test_id]
        test_call_counter[test_id] += 1
        return f"{test_id}.{assertion_id}"

    return generate_id


# Share this existing fixture from the expected location.
check_graphic_caller = iris.tests.graphics._check_graphic_caller
