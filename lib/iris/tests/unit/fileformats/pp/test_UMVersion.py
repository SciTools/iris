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
"""Unit tests for :class:`iris.fileformats.pp.UMVersion`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

from iris.fileformats.pp import UMVersion


class Test___init__(tests.IrisTest):
    def _check_init(self, umver, *content):
        major, minor, unknown = content
        self.assertEqual(umver.major, major)
        self.assertEqual(umver.minor, minor)
        self.assertEqual(umver.is_unknown(), unknown)

    def test_simple(self):
        umver = UMVersion(2, 5)
        self._check_init(umver, 2, 5, False)

    def test_unknown(self):
        umver = UMVersion(None, None)
        self._check_init(umver, None, None, True)

    def test_unknown__noargs(self):
        umver = UMVersion()
        self._check_init(umver, None, None, True)

    def test_unknown__nomajor(self):
        umver = UMVersion(1)
        self._check_init(umver, 1, None, True)

    def test_unknown__nominor(self):
        umver = UMVersion(None, 1)
        self._check_init(umver, None, 1, True)

    def test_float(self):
        umver = UMVersion(2.0000001, 4.9999999)
        self._check_init(umver, 2, 5, False)

    def test_zeros(self):
        umver = UMVersion(0, 0)
        self._check_init(umver, 0, 0, False)

    def test_fail_float_major(self):
        with self.assertRaises(ValueError) as err_context:
            umver = UMVersion(2.3, 4)
        msg = err_context.exception.message
        self.assertIn('integers', msg)

    def test_fail_float_minor(self):
        with self.assertRaises(ValueError) as err_context:
            umver = UMVersion(2, 4.1)
        msg = err_context.exception.message
        self.assertIn('integers', msg)

    def test_fail_negative_major(self):
        with self.assertRaises(ValueError) as err_context:
            umver = UMVersion(-2, 4)
        msg = err_context.exception.message
        self.assertIn('0 <= major', msg)

    def test_fail_negative_minor(self):
        with self.assertRaises(ValueError) as err_context:
            umver = UMVersion(2, -4)
        msg = err_context.exception.message
        self.assertIn('0 <= minor', msg)

    def test_fail_minor_range(self):
        with self.assertRaises(ValueError) as err_context:
            umver = UMVersion(2, 105)
        msg = err_context.exception.message
        self.assertIn('minor <= 99', msg)


class Test___slots__(tests.IrisTest):
    def check_fails_attr_write(self, attr_name):
        with self.assertRaises(AttributeError) as err_context:
            umver = UMVersion(1, 1)
            attr_value = getattr(umver, attr_name)
            setattr(umver, attr_name, attr_value)
        msg = err_context.exception.message
        self.assertEqual(msg, "can't set attribute")

    def test_major_no_write(self):
        self.check_fails_attr_write('major')

    def test_minor_no_write(self):
        self.check_fails_attr_write('minor')


class Test___str__(tests.IrisTest):
    def test_simple(self):
        self.assertEqual(str(UMVersion(2, 5)), '2.5')

    def test_zeros(self):
        self.assertEqual(str(UMVersion(0, 0)), '0.0')

    def test_unknown(self):
        self.assertEqual(str(UMVersion()), '')


class Test_lbsrce(tests.IrisTest):
    def test_simple(self):
        self.assertEqual(UMVersion(2, 5).lbsrce(), 2051111)

    def test_zeros(self):
        self.assertEqual(UMVersion(0, 0).lbsrce(), 1111)

    def test_unknown(self):
        self.assertEqual(UMVersion().lbsrce(), 0)


class Test_from_lbsrce(tests.IrisTest):
    def _check_lbsrce(self, umver, content, string, lbsrce):
        major, minor, unknown = content
        self.assertEqual(umver.major, major)
        self.assertEqual(umver.minor, minor)
        self.assertEqual(umver.is_unknown(), unknown)
        self.assertEqual(str(umver), string)
        self.assertEqual(umver.lbsrce(), lbsrce)

    def test_simple(self):
        umver = UMVersion.from_lbsrce(2051111)
        self._check_lbsrce(umver,
                           (2, 5, False),
                           '2.5',
                           2051111)

    def test_no_version(self):
        umver = UMVersion.from_lbsrce(1111)
        self._check_lbsrce(umver,
                           (0, 0, False),
                           '0.0',
                           1111)

    def test_unknown(self):
        umver = UMVersion.from_lbsrce(123)
        self._check_lbsrce(umver,
                           (None, None, True),
                           '',
                           0)


class Test__comparisons(tests.IrisTest):
    def test_eq(self):
        self.assertEqual(UMVersion(2, 5), UMVersion(2, 5))

    def test_eq__unknown(self):
        self.assertEqual(UMVersion(), UMVersion())

    def test_ne__minor(self):
        self.assertNotEqual(UMVersion(2, 5), UMVersion(2, 6))

    def test_ne__major(self):
        self.assertNotEqual(UMVersion(1, 5), UMVersion(2, 5))

    def test_ne__both(self):
        self.assertNotEqual(UMVersion(1, 5), UMVersion(2, 5))

    def test_ne__unknown(self):
        self.assertNotEqual(UMVersion(1, 5), UMVersion(None, None))


if __name__ == "__main__":
    tests.main()
