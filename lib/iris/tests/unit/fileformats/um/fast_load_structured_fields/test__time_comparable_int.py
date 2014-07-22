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
"""
Unit tests for the function
:func:`iris.fileformats.um._fast_load_structured_fields._time_comparable_int`.

"""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests


from iris.fileformats.um._fast_load_structured_fields import \
    _time_comparable_int as time_comparable_int


class Test(tests.IrisTest):
    def test(self):
        # Define a list of date-time tuples, which should remain both all
        # distinct and in ascending order when converted...
        test_date_tuples = [
            # Increment each component in turn to check that all are handled.
            (2004, 1, 1, 0, 0, 0),
            (2004, 1, 1, 0, 0, 1),
            (2004, 1, 1, 0, 1, 0),
            (2004, 1, 1, 1, 0, 0),
            (2004, 1, 2, 0, 0, 0),
            (2004, 2, 1, 0, 0, 0),
            # Go across 2004-02-29 leap-day, and on to "Feb 31 .. Mar 1".
            (2004, 2, 27, 0, 0, 0),
            (2004, 2, 28, 0, 0, 0),
            (2004, 2, 29, 0, 0, 0),
            (2004, 2, 30, 0, 0, 0),
            (2004, 2, 31, 0, 0, 0),
            (2004, 3, 1, 0, 0, 0),
            (2005, 1, 1, 0, 0, 0)]
        test_date_ints = [time_comparable_int(*test_tuple)
                          for test_tuple in test_date_tuples]
        # Check all values are distinct.
        self.assertEqual(len(test_date_ints), len(set(test_date_ints), ))
        # Check all values are in order.
        self.assertEqual(test_date_ints, sorted(test_date_ints))


if __name__ == "__main__":
    tests.main()
