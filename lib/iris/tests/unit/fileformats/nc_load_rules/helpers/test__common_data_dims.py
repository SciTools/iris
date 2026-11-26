# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test function :func:`iris.fileformats._nc_load_rules.helpers._common_data_dims`."""

from dataclasses import dataclass

from iris.fileformats._nc_load_rules.helpers import _common_data_dims


@dataclass
class Variable:
    """The two members the dimension rule reads off a CF variable."""

    dimensions: tuple
    shape: tuple


def test_the_one_shared_dimension():
    coord = Variable(("x",), (3,))
    data = Variable(("y", "x"), (2, 3))
    assert _common_data_dims(coord, data) == [1]


def test_several_shared_dimensions_keep_the_coordinates_order():
    coord = Variable(("x", "y"), (3, 2))
    data = Variable(("y", "x"), (2, 3))
    assert _common_data_dims(coord, data) == [1, 0]


def test_a_scalar_variable_shares_nothing():
    # `None`, not `[]`: it is what `Cube.add_aux_coord` takes for "no
    # dimensions", and an empty list is not.
    assert _common_data_dims(Variable((), ()), Variable(("x",), (3,))) is None


def test_a_dimension_the_data_variable_does_not_have():
    assert _common_data_dims(Variable(("t",), (4,)), Variable(("x",), (3,))) is None


def test_the_same_name_at_a_different_length_shares_nothing():
    """A name is not an identity once a Zarr store is in scope.

    `CFVariable.spans` decided this the same way before the coordinate got
    this far, so answering differently here would attach a coordinate that
    was never found to span the data variable.
    """
    assert _common_data_dims(Variable(("x",), (3,)), Variable(("x",), (4,))) is None


def test_a_repeated_dimension_name_is_resolved_by_length():
    """The case a Zarr store can present and no netCDF file can.

    A netCDF dimension name is unique within its group and carries one size;
    a Zarr dimension name is a label with no such guarantee, so one variable
    may name both of its dimensions alike at two lengths. Matched on the bare
    name, the coordinate lands on dimension 0 and is rejected as the wrong
    length.
    """
    coord = Variable(("d",), (3,))
    data = Variable(("d", "d"), (2, 3))
    assert _common_data_dims(coord, data) == [1]


def test_an_anonymous_dimension_matches_nothing():
    """Not even another anonymous dimension: there is nothing to match on.

    A Zarr version 3 store may leave a dimension unnamed, and
    `CFVariable.spans` rejects a coordinate that has one, so this is the same
    answer arrived at twice rather than two rules disagreeing.
    """
    coord = Variable((None,), (3,))
    data = Variable((None,), (3,))
    assert _common_data_dims(coord, data) is None
