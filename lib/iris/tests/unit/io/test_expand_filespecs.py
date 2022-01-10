# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.io.expand_filespecs` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import os
import shutil
import tempfile
import textwrap

import iris.io as iio


class TestExpandFilespecs(tests.IrisTest):
    def setUp(self):
        tests.IrisTest.setUp(self)
        self.tmpdir = os.path.realpath(tempfile.mkdtemp())
        self.fnames = ["a.foo", "b.txt"]
        for fname in self.fnames:
            with open(os.path.join(self.tmpdir, fname), "w") as fh:
                fh.write("anything")

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_absolute_path(self):
        result = iio.expand_filespecs([os.path.join(self.tmpdir, "*")])
        expected = [os.path.join(self.tmpdir, fname) for fname in self.fnames]
        self.assertEqual(result, expected)

    def test_double_slash(self):
        product = iio.expand_filespecs(["//" + os.path.join(self.tmpdir, "*")])
        predicted = [os.path.join(self.tmpdir, fname) for fname in self.fnames]
        self.assertEqual(product, predicted)

    def test_relative_path(self):
        cwd = os.getcwd()
        try:
            os.chdir(self.tmpdir)
            item_out = iio.expand_filespecs(["*"])
            item_in = [
                os.path.join(self.tmpdir, fname) for fname in self.fnames
            ]
            self.assertEqual(item_out, item_in)
        finally:
            os.chdir(cwd)

    def test_return_order(self):
        # It is really quite important what order we return the
        # files. They should be in the order that was provided,
        # so that we can control the order of load (for instance,
        # this can be used with PP files to ensure that there is
        # a surface reference).
        patterns = [
            os.path.join(self.tmpdir, "a.*"),
            os.path.join(self.tmpdir, "b.*"),
        ]
        expected = [
            os.path.join(self.tmpdir, fname) for fname in ["a.foo", "b.txt"]
        ]
        result = iio.expand_filespecs(patterns)
        self.assertEqual(result, expected)
        result = iio.expand_filespecs(patterns[::-1])
        self.assertEqual(result, expected[::-1])

    def test_no_files_found(self):
        msg = r"\/no_exist.txt\" didn\'t match any files"
        with self.assertRaisesRegex(IOError, msg):
            iio.expand_filespecs([os.path.join(self.tmpdir, "no_exist.txt")])

    def test_files_and_none(self):
        with self.assertRaises(IOError) as err:
            iio.expand_filespecs(
                [
                    os.path.join(self.tmpdir, "does_not_exist.txt"),
                    os.path.join(self.tmpdir, "*"),
                ]
            )
        expected = (
            textwrap.dedent(
                """
            One or more of the files specified did not exist:
                * "{0}/does_not_exist.txt" didn\'t match any files
                - "{0}/*" matched 2 file(s)
            """
            )
            .strip()
            .format(self.tmpdir)
        )

        self.assertStringEqual(str(err.exception), expected)


if __name__ == "__main__":
    tests.main()
