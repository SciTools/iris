# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.fileformats.pp.load` function."""

import iris
from iris.fileformats.pp import STASH, _convert_constraints


class Test_convert_constraints:
    def _single_stash(self):
        constraint = iris.AttributeConstraint(STASH="m01s03i236")
        return _convert_constraints(constraint)

    def test_single_stash(self, mocker):
        pp_filter = self._single_stash()
        stcube = mocker.Mock(stash=STASH.from_msi("m01s03i236"))
        assert pp_filter(stcube)

    def test_stash_object(self, mocker):
        constraint = iris.AttributeConstraint(STASH=STASH.from_msi("m01s03i236"))
        pp_filter = _convert_constraints(constraint)
        stcube = mocker.Mock(stash=STASH.from_msi("m01s03i236"))
        assert pp_filter(stcube)

    def test_surface_altitude(self, mocker):
        # Ensure that surface altitude fields are not filtered.
        pp_filter = self._single_stash()
        orography_cube = mocker.Mock(stash=STASH.from_msi("m01s00i033"))
        assert pp_filter(orography_cube)

    def test_surface_pressure(self, mocker):
        # Ensure that surface pressure fields are not filtered.
        pp_filter = self._single_stash()
        pressure_cube = mocker.Mock(stash=STASH.from_msi("m01s00i001"))
        assert pp_filter(pressure_cube)

    def test_double_stash(self, mocker):
        stcube236 = mocker.Mock(stash=STASH.from_msi("m01s03i236"))
        stcube4 = mocker.Mock(stash=STASH.from_msi("m01s00i004"))
        stcube7 = mocker.Mock(stash=STASH.from_msi("m01s00i007"))
        constraints = [
            iris.AttributeConstraint(STASH="m01s03i236"),
            iris.AttributeConstraint(STASH="m01s00i004"),
        ]
        pp_filter = _convert_constraints(constraints)
        assert pp_filter(stcube236)
        assert pp_filter(stcube4)
        assert not pp_filter(stcube7)

    def test_callable_stash(self, mocker):
        stcube236 = mocker.Mock(stash=STASH.from_msi("m01s03i236"))
        stcube4 = mocker.Mock(stash=STASH.from_msi("m01s00i004"))
        stcube7 = mocker.Mock(stash=STASH.from_msi("m01s00i007"))
        con1 = iris.AttributeConstraint(STASH=lambda s: s.endswith("004"))
        con2 = iris.AttributeConstraint(STASH=lambda s: s == "m01s00i007")
        constraints = [con1, con2]
        pp_filter = _convert_constraints(constraints)
        assert not pp_filter(stcube236)
        assert pp_filter(stcube4)
        assert pp_filter(stcube7)

    def test_multiple_with_stash(self):
        constraints = [
            iris.Constraint("air_potential_temperature"),
            iris.AttributeConstraint(STASH="m01s00i004"),
        ]
        pp_filter = _convert_constraints(constraints)
        assert pp_filter is None

    def test_no_stash(self):
        constraints = [
            iris.Constraint("air_potential_temperature"),
            iris.AttributeConstraint(source="asource"),
        ]
        pp_filter = _convert_constraints(constraints)
        assert pp_filter is None

    def test_no_constraint(self):
        constraints = []
        pp_filter = _convert_constraints(constraints)
        assert pp_filter is None
