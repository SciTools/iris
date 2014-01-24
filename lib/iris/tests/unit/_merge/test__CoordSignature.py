# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris._merge._CoordSignature` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris._merge import _CoordSignature
import iris.exceptions


class test_match(tests.IrisTest):

    def test_verbose(self):
        # make sure we can trigger verbose reporting
        dummy_stuff_a = [(1, 2), (3, 4), (5, 6)]
        a = _CoordSignature(scalar_defns=dummy_stuff_a,
                            vector_dim_coords_and_dims=dummy_stuff_a,
                            vector_aux_coords_and_dims=dummy_stuff_a,
                            factory_defns=dummy_stuff_a)

        dummy_stuff_b = [(7, 8), (9, 10), (11, 12)]
        b = _CoordSignature(scalar_defns=dummy_stuff_b,
                            vector_dim_coords_and_dims=dummy_stuff_b,
                            vector_aux_coords_and_dims=dummy_stuff_b,
                            factory_defns=dummy_stuff_b)

        with self.assertRaises(iris.exceptions.MergeError) as arc:
            a.match(b, or_fail=True, fail_verbose=True)

        for dummy_thing in dummy_stuff_a + dummy_stuff_b:
            self.assertIn("scalar coord: {}".format(dummy_thing),
                          arc.exception.message, arc.exception.message)
            self.assertIn("dim coord: {}".format(dummy_thing),
                          arc.exception.message, arc.exception.message)
            self.assertIn("aux coord: {}".format(dummy_thing),
                          arc.exception.message, arc.exception.message)


if __name__ == "__main__":
    tests.main()
