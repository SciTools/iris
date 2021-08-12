# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

import iris
import iris.fileformats.pp
import iris.io
import iris.tests.stock
import iris.util


class TestPPStash(tests.IrisTest):
    @tests.skip_data
    def test_cube_attributes(self):
        cube = tests.stock.simple_pp()
        self.assertEqual("m01s16i203", cube.attributes["STASH"])
        self.assertNotEqual("m01s16i999", cube.attributes["STASH"])
        self.assertEqual(cube.attributes["STASH"], "m01s16i203")
        self.assertNotEqual(cube.attributes["STASH"], "m01s16i999")

    @tests.skip_data
    def test_ppfield(self):
        data_path = tests.get_data_path(("PP", "simple_pp", "global.pp"))
        pps = iris.fileformats.pp.load(data_path)
        for pp in pps:
            self.assertEqual("m01s16i203", pp.stash)
            self.assertNotEqual("m01s16i999", pp.stash)
            self.assertEqual(pp.stash, "m01s16i203")
            self.assertNotEqual(pp.stash, "m01s16i999")

    def test_stash_against_stash(self):
        self.assertEqual(
            iris.fileformats.pp.STASH(1, 2, 3),
            iris.fileformats.pp.STASH(1, 2, 3),
        )
        self.assertNotEqual(
            iris.fileformats.pp.STASH(1, 2, 3),
            iris.fileformats.pp.STASH(2, 3, 4),
        )

    def test_stash_against_str(self):
        self.assertEqual(iris.fileformats.pp.STASH(1, 2, 3), "m01s02i003")
        self.assertEqual("m01s02i003", iris.fileformats.pp.STASH(1, 2, 3))
        self.assertNotEqual(iris.fileformats.pp.STASH(1, 2, 3), "m02s03i004")
        self.assertNotEqual("m02s03i004", iris.fileformats.pp.STASH(1, 2, 3))

    def test_irregular_stash_str(self):
        self.assertEqual(
            iris.fileformats.pp.STASH(1, 2, 3), "m01s02i0000000003"
        )
        self.assertEqual(iris.fileformats.pp.STASH(1, 2, 3), "m01s02i3")
        self.assertEqual(iris.fileformats.pp.STASH(1, 2, 3), "m01s2i3")
        self.assertEqual(iris.fileformats.pp.STASH(1, 2, 3), "m1s2i3")

        self.assertEqual(
            "m01s02i0000000003", iris.fileformats.pp.STASH(1, 2, 3)
        )
        self.assertEqual("m01s02i3", iris.fileformats.pp.STASH(1, 2, 3))
        self.assertEqual("m01s2i3", iris.fileformats.pp.STASH(1, 2, 3))
        self.assertEqual("m1s2i3", iris.fileformats.pp.STASH(1, 2, 3))

        self.assertNotEqual(
            iris.fileformats.pp.STASH(2, 3, 4), "m01s02i0000000003"
        )
        self.assertNotEqual(iris.fileformats.pp.STASH(2, 3, 4), "m01s02i3")
        self.assertNotEqual(iris.fileformats.pp.STASH(2, 3, 4), "m01s2i3")
        self.assertNotEqual(iris.fileformats.pp.STASH(2, 3, 4), "m1s2i3")

        self.assertNotEqual(
            "m01s02i0000000003", iris.fileformats.pp.STASH(2, 3, 4)
        )
        self.assertNotEqual("m01s02i3", iris.fileformats.pp.STASH(2, 3, 4))
        self.assertNotEqual("m01s2i3", iris.fileformats.pp.STASH(2, 3, 4))
        self.assertNotEqual("m1s2i3", iris.fileformats.pp.STASH(2, 3, 4))

        self.assertEqual(
            iris.fileformats.pp.STASH.from_msi("M01s02i003"), "m01s02i003"
        )
        self.assertEqual(
            "m01s02i003", iris.fileformats.pp.STASH.from_msi("M01s02i003")
        )

    def test_illegal_stash_str_range(self):

        self.assertEqual(iris.fileformats.pp.STASH(0, 2, 3), "m??s02i003")
        self.assertNotEqual(iris.fileformats.pp.STASH(0, 2, 3), "m01s02i003")

        self.assertEqual("m??s02i003", iris.fileformats.pp.STASH(0, 2, 3))
        self.assertNotEqual("m01s02i003", iris.fileformats.pp.STASH(0, 2, 3))

        self.assertEqual(iris.fileformats.pp.STASH(0, 2, 3), "m??s02i003")
        self.assertEqual(iris.fileformats.pp.STASH(0, 2, 3), "m00s02i003")
        self.assertEqual("m??s02i003", iris.fileformats.pp.STASH(0, 2, 3))
        self.assertEqual("m00s02i003", iris.fileformats.pp.STASH(0, 2, 3))

        self.assertEqual(iris.fileformats.pp.STASH(100, 2, 3), "m??s02i003")
        self.assertEqual(iris.fileformats.pp.STASH(100, 2, 3), "m100s02i003")
        self.assertEqual("m??s02i003", iris.fileformats.pp.STASH(100, 2, 3))
        self.assertEqual("m100s02i003", iris.fileformats.pp.STASH(100, 2, 3))

    def test_illegal_stash_stash_range(self):
        self.assertEqual(
            iris.fileformats.pp.STASH(0, 2, 3),
            iris.fileformats.pp.STASH(0, 2, 3),
        )
        self.assertEqual(
            iris.fileformats.pp.STASH(100, 2, 3),
            iris.fileformats.pp.STASH(100, 2, 3),
        )
        self.assertEqual(
            iris.fileformats.pp.STASH(100, 2, 3),
            iris.fileformats.pp.STASH(999, 2, 3),
        )

    def test_illegal_stash_format(self):
        test_values = (
            ("abc", (1, 2, 3)),
            ("mlotstmin", (1, 2, 3)),
            ("m01s02003", (1, 2, 3)),
        )

        for (test_value, reference) in test_values:
            msg = "Expected STASH code .* {!r}".format(test_value)
            with self.assertRaisesRegex(ValueError, msg):
                test_value == iris.fileformats.pp.STASH(*reference)
            with self.assertRaisesRegex(ValueError, msg):
                iris.fileformats.pp.STASH(*reference) == test_value

    def test_illegal_stash_type(self):
        test_values = (
            (102003, "m01s02i003"),
            (["m01s02i003"], "m01s02i003"),
        )

        for (test_value, reference) in test_values:
            msg = "Expected STASH code .* {!r}".format(test_value)
            with self.assertRaisesRegex(TypeError, msg):
                iris.fileformats.pp.STASH.from_msi(test_value) == reference
            with self.assertRaisesRegex(TypeError, msg):
                reference == iris.fileformats.pp.STASH.from_msi(test_value)

    def test_stash_lbuser(self):
        stash = iris.fileformats.pp.STASH(2, 32, 456)
        self.assertEqual(stash.lbuser6(), 2)
        self.assertEqual(stash.lbuser3(), 32456)


if __name__ == "__main__":
    tests.main()
