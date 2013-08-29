# (C) British Crown Copyright 2010 - 2013, Met Office
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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import iris
import iris.fileformats.pp
import iris.io
import iris.util
import iris.tests.stock


class TestPPStash(tests.IrisTest):

    @iris.tests.skip_data
    def test_cube_attributes(self):
        cube = tests.stock.simple_pp()
        self.assertEqual('m01s16i203', cube.attributes['STASH'])
        self.assertNotEqual('m01s16i999', cube.attributes['STASH'])
        self.assertEqual(cube.attributes['STASH'], 'm01s16i203')
        self.assertNotEqual(cube.attributes['STASH'], 'm01s16i999')

    @iris.tests.skip_data
    def test_ppfield(self):
        data_path = tests.get_data_path(('PP', 'simple_pp', 'global.pp'))
        pps = iris.fileformats.pp.load(data_path)
        for pp in pps:
            self.assertEqual('m01s16i203', pp.stash)
            self.assertNotEqual('m01s16i999', pp.stash)
            self.assertEqual(pp.stash, 'm01s16i203')
            self.assertNotEqual(pp.stash, 'm01s16i999')

    def test_stash_against_stash(self):
        self.assertEqual(iris.fileformats.pp.STASH(1,2,3), iris.fileformats.pp.STASH(1,2,3))
        self.assertNotEqual(iris.fileformats.pp.STASH(1,2,3), iris.fileformats.pp.STASH(2,3,4))

    def test_stash_against_str(self):
        self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'm01s02i003')
        self.assertEqual('m01s02i003', iris.fileformats.pp.STASH(1,2,3))
        self.assertNotEqual(iris.fileformats.pp.STASH(1,2,3), 'm02s03i004')
        self.assertNotEqual('m02s03i004', iris.fileformats.pp.STASH(1,2,3))

    def test_irregular_stash_str(self):
        self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'm01s02i0000000003')
        self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'm01s02i3')
        self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'm01s2i3')
        self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'm1s2i3')

        self.assertEqual('m01s02i0000000003', iris.fileformats.pp.STASH(1,2,3))
        self.assertEqual('m01s02i3', iris.fileformats.pp.STASH(1,2,3))
        self.assertEqual('m01s2i3', iris.fileformats.pp.STASH(1,2,3))
        self.assertEqual('m1s2i3', iris.fileformats.pp.STASH(1,2,3))

        self.assertNotEqual(iris.fileformats.pp.STASH(2,3,4), 'm01s02i0000000003')
        self.assertNotEqual(iris.fileformats.pp.STASH(2,3,4), 'm01s02i3')
        self.assertNotEqual(iris.fileformats.pp.STASH(2,3,4), 'm01s2i3')
        self.assertNotEqual(iris.fileformats.pp.STASH(2,3,4), 'm1s2i3')

        self.assertNotEqual('m01s02i0000000003', iris.fileformats.pp.STASH(2,3,4))
        self.assertNotEqual('m01s02i3', iris.fileformats.pp.STASH(2,3,4))
        self.assertNotEqual('m01s2i3', iris.fileformats.pp.STASH(2,3,4))
        self.assertNotEqual('m1s2i3', iris.fileformats.pp.STASH(2,3,4))

        self.assertEqual(iris.fileformats.pp.STASH.from_msi('M01s02i003'), 'm01s02i003')
        self.assertEqual('m01s02i003', iris.fileformats.pp.STASH.from_msi('M01s02i003'))

    def test_illegal_stash_str_range(self):
        self.assertEqual(iris.fileformats.pp.STASH(0,2,3), 'm??s02i003')
        self.assertNotEqual(iris.fileformats.pp.STASH(0,2,3), 'm01s02i003')
        self.assertEqual('m??s02i003', iris.fileformats.pp.STASH(0,2,3))
        self.assertNotEqual('m01s02i003', iris.fileformats.pp.STASH(0,2,3))

        self.assertEqual(iris.fileformats.pp.STASH(0,2,3), 'm??s02i003')
        self.assertEqual(iris.fileformats.pp.STASH(0,2,3), 'm00s02i003')
        self.assertEqual('m??s02i003', iris.fileformats.pp.STASH(0,2,3))
        self.assertEqual('m00s02i003', iris.fileformats.pp.STASH(0,2,3))

        self.assertEqual(iris.fileformats.pp.STASH(100,2,3), 'm??s02i003')
        self.assertEqual(iris.fileformats.pp.STASH(100,2,3), 'm100s02i003')
        self.assertEqual('m??s02i003', iris.fileformats.pp.STASH(100,2,3))
        self.assertEqual('m100s02i003', iris.fileformats.pp.STASH(100,2,3))

    def test_illegal_stash_stash_range(self):
        self.assertEqual(iris.fileformats.pp.STASH(0,2,3), iris.fileformats.pp.STASH(0,2,3))
        self.assertEqual(iris.fileformats.pp.STASH(100,2,3), iris.fileformats.pp.STASH(100,2,3))
        self.assertEqual(iris.fileformats.pp.STASH(100,2,3), iris.fileformats.pp.STASH(999,2,3))

    def test_illegal_stash_format(self):
        with self.assertRaises(ValueError):
            self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'abc')
        with self.assertRaises(ValueError):
            self.assertEqual('abc', iris.fileformats.pp.STASH(1,2,3))

        with self.assertRaises(ValueError):
            self.assertEqual(iris.fileformats.pp.STASH(1,2,3), 'm01s02003')
        with self.assertRaises(ValueError):
            self.assertEqual('m01s02003', iris.fileformats.pp.STASH(1,2,3))

    def test_illegal_stash_type(self):
        with self.assertRaises(TypeError):
            self.assertEqual(iris.fileformats.pp.STASH.from_msi(0102003), 'm01s02i003')

        with self.assertRaises(TypeError):
            self.assertEqual('m01s02i003', iris.fileformats.pp.STASH.from_msi(0102003))

        with self.assertRaises(TypeError):
            self.assertEqual(iris.fileformats.pp.STASH.from_msi(['m01s02i003']), 'm01s02i003')

        with self.assertRaises(TypeError):
            self.assertEqual('m01s02i003', iris.fileformats.pp.STASH.from_msi(['m01s02i003']))

    def test_stash_lbuser(self):
        stash = iris.fileformats.pp.STASH(2, 32, 456)
        self.assertEqual(stash.lbuser6(), 2)
        self.assertEqual(stash.lbuser3(), 32456)

if __name__ == "__main__":
    tests.main()
