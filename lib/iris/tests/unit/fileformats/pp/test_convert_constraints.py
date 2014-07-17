# (C) British Crown Copyright 2014, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.fileformats.pp.load` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import mock

import iris
import iris.fileformats.pp as pp


class Test_convert_constraints(tests.IrisTest):
    def test_single_stash(self):
        stcube = mock.Mock(stash=pp.STASH.from_msi('m01s03i236'))
        constraint = iris.AttributeConstraint(STASH='m01s03i236')
        pp_filter = pp._convert_constraints(constraint)
        self.assertTrue(pp_filter(stcube))

    def test_multiple_with_stash(self):
        constraints = [iris.Constraint('air_potential_temperature'),
                       iris.AttributeConstraint(STASH='m01s00i004')]
        pp_filter = pp._convert_constraints(constraints)
        stcube = mock.Mock(stash=pp.STASH.from_msi('m01s00i004'))
        self.assertTrue(pp_filter(stcube))

    def test_no_stash(self):
        constraints = [iris.Constraint('air_potential_temperature'),
                       iris.AttributeConstraint(source='asource')]
        pp_filter = pp._convert_constraints(constraints)
        stcube = mock.Mock(stash=pp.STASH.from_msi('m01s00i004'))
        self.assertTrue(pp_filter(stcube))


if __name__ == "__main__":
    tests.main()
