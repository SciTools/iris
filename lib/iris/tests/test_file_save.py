# (C) British Crown Copyright 2010 - 2012, Met Office
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
Test the file saving mechanism.

"""

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import cStringIO
import os

import iris
import iris.cube
import iris.util
import iris.fileformats.pp as pp
import iris.fileformats.dot as dot

CHKSUM_ERR = "Mismatch between checksum of iris.save and {}.save."

def save_by_filename(filename1, filename2, cube, saver_fn, iosaver=None):
    """ Saves a cube to two different filenames using iris.save and the save method of the object representing the file type directly"""
    # Save from object direct
    saver_fn(cube, filename1)
    
    # Call save on iris
    iris.save(cube, filename2, iosaver) # Optional iris.io.find_saver passed in from test

def save_by_filehandle(filehandle1, filehandle2, cube, fn_saver, binary_mode = True):
    """ Saves a cube to two different filehandles using iris.save and the save method of the object representing the file type directly"""
    mode = "wb" if binary_mode else "w"
    
    # Save from object direct
    with open(filehandle1, mode) as outfile:
        fn_saver(cube, outfile)
    
    # Call save on iris
    with open(filehandle2, mode) as outfile:
        iris.save(cube, outfile)


@iris.tests.skip_data
class TestSaveMethods(tests.IrisTest):
    """ Base class for file saving tests. Loads data and creates/deletes tempfiles"""
    def setUp(self):
        self.cube1 = iris.load_cube(tests.get_data_path(('PP', 'aPPglob1', 'global.pp')))
        self.cube2 = iris.load_cube(tests.get_data_path(('PP', 'aPPglob1', 'global_t_forecast.pp')))
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
        save_by_filename(self.temp_filename1, self.temp_filename2, self.cube1, pp.save)

        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

    def test_filehandle(self):
        # Save using iris.save and pp.save 
        save_by_filehandle(self.temp_filename1, self.temp_filename2, self.cube1, pp.save, binary_mode = True)
        
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

        # Check we can't save when file handle is not binary        
        with self.assertRaises(ValueError):
            save_by_filehandle(self.temp_filename1, self.temp_filename2, self.cube1, pp.save, binary_mode = False)

class TestSaveDot(TestSaveMethods):
    """Test saving cubes to DOT format"""
    ext = ".dot"
    
    def test_filename(self):
        # Save using iris.save and dot.save 
        save_by_filename(self.temp_filename1, self.temp_filename2, self.cube1, dot.save)
        
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

    def test_filehandle(self):  
        # Save using iris.save and dot.save
        save_by_filehandle(self.temp_filename1, self.temp_filename2, self.cube1, dot.save, binary_mode = False)
    
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

        # Check we can't save when file handle is binary        
        with self.assertRaises(ValueError):
            save_by_filehandle(self.temp_filename1, self.temp_filename2, self.cube1, dot.save, binary_mode = True)

    def test_cstringio(self):
        string_io = cStringIO.StringIO()
    
        # Save from dot direct
        dot.save(self.cube1, self.temp_filename1)
    
        # Call save on iris
        iris.save(self.cube1, string_io, iris.io.find_saver(self.ext))

        with open(self.temp_filename1) as infile:
            data = infile.read()

        # Compare files
        self.assertEquals(data, string_io.getvalue(), "Mismatch in data when comparing iris cstringio save and dot.save.")

class TestSavePng(TestSaveMethods):
    """Test saving cubes to png"""
    ext = ".dotpng"

    def test_filename(self):
        # Save using iris.save and dot.save_png
        save_by_filename(self.temp_filename1, self.temp_filename2, self.cube1, dot.save_png)
        
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

    def test_filehandle(self):
        # Save using iris.save and dot.save_png
        save_by_filehandle(self.temp_filename1, self.temp_filename2, self.cube1, dot.save_png, binary_mode = True)
    
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))
        
        # Check we can't save when file handle is not binary        
        with self.assertRaises(ValueError):
            save_by_filehandle(self.temp_filename1, self.temp_filename2, self.cube1, dot.save_png, binary_mode = False)

class TestSaver(TestSaveMethods):
    """Test saving to Iris when we define the saver type to use"""
    ext = ".spam"
    
    def test_pp(self):
        # Make our own saver
        pp_saver = iris.io.find_saver("PP")
        save_by_filename(self.temp_filename1, self.temp_filename2, self.cube1, pp.save, pp_saver)
        
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

    def test_dot(self):
        # Make our own saver
        dot_saver = iris.io.find_saver("DOT")
        save_by_filename(self.temp_filename1, self.temp_filename2, self.cube1, dot.save, dot_saver)
        
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

    def test_png(self):
        # Make our own saver
        png_saver = iris.io.find_saver("DOTPNG")
        save_by_filename(self.temp_filename1, self.temp_filename2, self.cube1, dot.save_png, png_saver)
        
        # Compare files
        self.assertEquals(self.file_checksum(self.temp_filename2), self.file_checksum(self.temp_filename1), CHKSUM_ERR.format(self.ext))

class TestSaveInvalid(TestSaveMethods):
    """Test iris cannot automatically save to file extensions it does not know about"""
    ext = ".invalid"

    def test_filename(self):
        # Check we can't save a file with an unhandled extension
        with self.assertRaises(ValueError):
            iris.save(self.cube1, self.temp_filename2)

if __name__ == "__main__":
    tests.main()
    
