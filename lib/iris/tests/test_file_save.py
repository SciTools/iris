# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Test the file saving mechanism.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests  # isort:skip

from io import StringIO
import os
import pathlib
import unittest

import iris
import iris.cube
import iris.fileformats.dot as dot
import iris.fileformats.pp as pp
import iris.util

# Make a test skip decorator, for when DOT not available
skip_dotpng = unittest.skipIf(
    not dot.DOT_AVAILABLE,
    'Test(s) require the "dot" executable, which was not found. '
    "Check the dot_path setting in site.cfg.",
)

CHKSUM_ERR = "Mismatch between checksum of iris.save and {}.save."


def save_by_filename(filename1, filename2, cube, saver_fn, iosaver=None):
    """Saves a cube to two different filenames using iris.save and the save method of the object representing the file type directly"""
    # Save from object direct
    saver_fn(cube, filename1)

    # Call save on iris
    iris.save(
        cube, filename2, iosaver
    )  # Optional iris.io.find_saver passed in from test


def save_by_filehandle(
    filehandle1, filehandle2, cube, fn_saver, binary_mode=True
):
    """Saves a cube to two different filehandles using iris.save and the save method of the object representing the file type directly"""
    mode = "wb" if binary_mode else "w"

    # Save from object direct
    with open(filehandle1, mode) as outfile:
        fn_saver(cube, outfile)

    # Call save on iris
    with open(filehandle2, mode) as outfile:
        iris.save(cube, outfile)


@tests.skip_data
class TestSaveMethods(tests.IrisTest):
    """Base class for file saving tests. Loads data and creates/deletes tempfiles"""

    def setUp(self):
        self.cube1 = iris.load_cube(
            tests.get_data_path(("PP", "aPPglob1", "global.pp"))
        )
        self.cube2 = iris.load_cube(
            tests.get_data_path(("PP", "aPPglob1", "global_t_forecast.pp"))
        )
        self.temp_filename1 = iris.util.create_temp_filename(self.ext)
        self.temp_filename2 = iris.util.create_temp_filename(self.ext)

    def tearDown(self):
        for tempfile in (self.temp_filename1, self.temp_filename2):
            try:
                os.remove(tempfile)
            except Exception:
                pass


class TestSavePP(TestSaveMethods):
    """Test saving cubes to PP format"""

    ext = ".pp"

    def test_filename(self):
        # Save using iris.save and pp.save
        save_by_filename(
            self.temp_filename1, self.temp_filename2, self.cube1, pp.save
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

    def test_filename_path_object(self):
        # Save using iris.save and pp.save, passing filename for
        # iris.save as pathlib.Path
        save_by_filename(
            self.temp_filename1,
            pathlib.Path(self.temp_filename2),
            self.cube1,
            pp.save,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

    def test_filehandle(self):
        # Save using iris.save and pp.save
        save_by_filehandle(
            self.temp_filename1,
            self.temp_filename2,
            self.cube1,
            pp.save,
            binary_mode=True,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

        # Check we can't save when file handle is not binary
        with self.assertRaises(ValueError):
            save_by_filehandle(
                self.temp_filename1,
                self.temp_filename2,
                self.cube1,
                pp.save,
                binary_mode=False,
            )


class TestSaveDot(TestSaveMethods):
    """Test saving cubes to DOT format"""

    ext = ".dot"

    def test_filename(self):
        # Save using iris.save and dot.save
        save_by_filename(
            self.temp_filename1, self.temp_filename2, self.cube1, dot.save
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

    def test_filehandle(self):
        # Save using iris.save and dot.save
        save_by_filehandle(
            self.temp_filename1,
            self.temp_filename2,
            self.cube1,
            dot.save,
            binary_mode=False,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

        # Check we can't save when file handle is binary
        with self.assertRaises(ValueError):
            save_by_filehandle(
                self.temp_filename1,
                self.temp_filename2,
                self.cube1,
                dot.save,
                binary_mode=True,
            )

    def test_bytesio(self):
        sio = StringIO()

        # Save from dot direct
        dot.save(self.cube1, self.temp_filename1)

        # Call save on iris
        iris.save(self.cube1, sio, iris.io.find_saver(self.ext))

        with open(self.temp_filename1) as infile:
            data = infile.read()

        # Compare files
        self.assertEqual(
            data,
            sio.getvalue(),
            "Mismatch in data when comparing iris bytesio save "
            "and dot.save.",
        )


@skip_dotpng
class TestSavePng(TestSaveMethods):
    """Test saving cubes to png"""

    ext = ".dotpng"

    def test_filename(self):
        # Save using iris.save and dot.save_png
        save_by_filename(
            self.temp_filename1, self.temp_filename2, self.cube1, dot.save_png
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

    def test_filehandle(self):
        # Save using iris.save and dot.save_png
        save_by_filehandle(
            self.temp_filename1,
            self.temp_filename2,
            self.cube1,
            dot.save_png,
            binary_mode=True,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

        # Check we can't save when file handle is not binary
        with self.assertRaises(ValueError):
            save_by_filehandle(
                self.temp_filename1,
                self.temp_filename2,
                self.cube1,
                dot.save_png,
                binary_mode=False,
            )


class TestSaver(TestSaveMethods):
    """Test saving to Iris when we define the saver type to use"""

    ext = ".spam"

    def test_pp(self):
        # Make our own saver
        pp_saver = iris.io.find_saver("PP")
        save_by_filename(
            self.temp_filename1,
            self.temp_filename2,
            self.cube1,
            pp.save,
            pp_saver,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

    def test_dot(self):
        # Make our own saver
        dot_saver = iris.io.find_saver("DOT")
        save_by_filename(
            self.temp_filename1,
            self.temp_filename2,
            self.cube1,
            dot.save,
            dot_saver,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )

    @skip_dotpng
    def test_png(self):
        # Make our own saver
        png_saver = iris.io.find_saver("DOTPNG")
        save_by_filename(
            self.temp_filename1,
            self.temp_filename2,
            self.cube1,
            dot.save_png,
            png_saver,
        )

        # Compare files
        self.assertEqual(
            self.file_checksum(self.temp_filename2),
            self.file_checksum(self.temp_filename1),
            CHKSUM_ERR.format(self.ext),
        )


class TestSaveInvalid(TestSaveMethods):
    """Test iris cannot automatically save to file extensions it does not know about"""

    ext = ".invalid"

    def test_filename(self):
        # Check we can't save a file with an unhandled extension
        with self.assertRaises(ValueError):
            iris.save(self.cube1, self.temp_filename2)


if __name__ == "__main__":
    tests.main()
