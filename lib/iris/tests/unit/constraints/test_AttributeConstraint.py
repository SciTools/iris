# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris._constraints.AttributeConstraint` class."""

# TODO: migrate AttributeConstraint content from iris/tests/test_constraints.py

import numpy as np
import pytest

from iris._constraints import AttributeConstraint
from iris.tests import stock


@pytest.fixture
def simple_1d():
    return stock.simple_1d()


@pytest.fixture
def cube_w_numpy_attribute(simple_1d):
    # Guarantee the new attribute is the only attribute.
    assert simple_1d.attributes == {}
    attr_name = "numpy_attr"
    attr_value = np.array([1, 2, 3])
    simple_1d.attributes[attr_name] = attr_value
    return simple_1d


def test_numpy_attribute_match(cube_w_numpy_attribute):
    attr = cube_w_numpy_attribute.attributes
    constraint = AttributeConstraint(**attr)
    assert cube_w_numpy_attribute.extract(constraint) == cube_w_numpy_attribute


def test_numpy_attribute_mismatch(cube_w_numpy_attribute):
    attr = cube_w_numpy_attribute.attributes
    attr = {key: value + 1 for key, value in attr.items()}
    constraint = AttributeConstraint(**attr)
    assert cube_w_numpy_attribute.extract(constraint) is None


def test_numpy_attribute_against_str(cube_w_numpy_attribute):
    # Should not error.
    attr = cube_w_numpy_attribute.attributes
    attr = {key: "foo" for key, value in attr.items()}
    constraint = AttributeConstraint(**attr)
    assert cube_w_numpy_attribute.extract(constraint) is None


def test_numpy_attribute_incompatible(cube_w_numpy_attribute):
    attr = cube_w_numpy_attribute.attributes
    attr = {key: value[:-1] for key, value in attr.items()}
    constraint = AttributeConstraint(**attr)
    with pytest.raises(ValueError, match="Error comparing numpy_attr attributes"):
        _ = cube_w_numpy_attribute.extract(constraint)
