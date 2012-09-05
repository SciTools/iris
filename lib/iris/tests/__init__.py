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
Provides iris specific testing capabilities and customisations.

The primary class for this module is :class:`IrisTest`.

When importing this module, sys.argv is inspected to identify the flags ``-d`` and ``-sf`` which toggle displaying and saving image tests respectively.

.. note:: The ``-d`` option sets the matplotlib backend to either agg or tkagg. For this reason ``iris.tests`` **must** be imported before ``matplotlib.pyplot``


"""
import collections
import contextlib
import difflib
import filecmp
import logging
import os
import os.path
import platform
import re
import shutil
import StringIO
import subprocess
import sys
import unittest
import warnings
import xml.dom.minidom
import zlib

import matplotlib
# NB pyplot is imported after main() so that a backend can be defined.
# import matplotlib.pyplot as plt
import numpy

import iris.cube
import iris.config
import iris.io
import iris.util


_RESULT_PATH = os.path.join(iris.config.ROOT_PATH, 'tests', 'results')
"""Basepath for test results."""


if '--data-files-used' in sys.argv:
    sys.argv.remove('--data-files-used')
    fname = '/var/tmp/all_iris_test_resource_paths.txt'
    print 'saving list of files used by tests to %s' % fname
    _EXPORT_DATAPATHS_FILE = open(fname, 'w')
else:
    _EXPORT_DATAPATHS_FILE = None
    
# A shared logger for use by unit tests
logger = logging.getLogger('tests')


# Whether to display matplotlib output to the screen.
_DISPLAY_FIGURES = False

# Whether to save matplotlib output to files.
_SAVE_FIGURES = False

if '-d' in sys.argv:
    sys.argv.remove('-d')
    matplotlib.use('tkagg')
    _DISPLAY_FIGURES = True
else:
    matplotlib.use('agg')

# Imported now so that matplotlib.use can work 
import matplotlib.pyplot as plt

if '-sf' in sys.argv or os.environ.get('IRIS_TEST_SAVE_FIGURES', '') == '1':
    if '-sf' in sys.argv: sys.argv.remove('-sf')
    _SAVE_FIGURES = True


_PLATFORM = '%s_%s' % (''.join(platform.dist()[:2]), platform.architecture()[0])


def main():
    """A wrapper for unittest.main() which adds iris.test specific options to the help (-h) output."""
    if '-h' in sys.argv or '--help' in sys.argv:
        stdout = sys.stdout
        buff = StringIO.StringIO()
        # NB. unittest.main() raises an exception after it's shown the help text
        try:
            sys.stdout = buff
            unittest.main()
        finally:
            sys.stdout = stdout
            lines = buff.getvalue().split('\n')
            lines.insert(9, 'Iris-specific options:')
            lines.insert(10, '  -d                   Display matplotlib figures (uses tkagg)')
            lines.insert(11, '  -sf                  Save matplotlib figures to subfolder "image_results"')
            lines.insert(12, '                       Note: Compare branches with iris_tests/idiff.py')
            lines.insert(13, '  --data-files-used    Save a list of files used to a temporary file')
            print '\n'.join(lines)
    else:
        unittest.main()


def get_data_path(relative_path):
    """Returns the absolute path to a data file when given the relative path
    as a string, or sequence of strings."""

    if _EXPORT_DATAPATHS_FILE is not None:
        if isinstance(relative_path, str):
            relative_path = tuple(relative_path)
        _EXPORT_DATAPATHS_FILE.write(os.path.join(*relative_path) + '\n')
        
    return iris.io.select_data_path('tests', relative_path)


def get_result_path(relative_path):
    """Returns the absolute path to a result file when given the relative path
    as a string, or sequence of strings."""
    if not isinstance(relative_path, basestring):
        relative_path = os.path.join(*relative_path)
    return os.path.abspath(os.path.join(_RESULT_PATH, relative_path))


class IrisTest(unittest.TestCase):
    """A subclass of unittest.TestCase which provides Iris specific testing functionality."""

    _assertion_counts = collections.defaultdict(int)

    def _assert_str_same(self, reference_str, test_str, reference_filename, type_comparison_name='Strings'):
        if reference_str != test_str:
            diff = ''.join(difflib.unified_diff(reference_str.splitlines(1), test_str.splitlines(1),
                                                 'Reference', 'Test result', '', '', 0))
            self.fail("%s do not match: %s\n%s" % (type_comparison_name, reference_filename, diff))
            
    def _assert_cml(self, cube_xml, reference_xml, reference_filename):
        self._assert_str_same(reference_xml, cube_xml, reference_filename, 'CML')
    
    def assertCMLApproxData(self, cubes, reference_filename, *args, **kwargs):
        # passes args and kwargs on to approx equal
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]
        for i, cube in enumerate(cubes):
            fname = list(reference_filename)
            # don't want the ".cml" for the numpy data file
            if fname[-1].endswith(".cml"):
                fname[-1] = fname[-1][:-4]
            fname[-1] += '.data.%d.npy' % i
            self.assertCubeDataAlmostEqual(cube, fname, *args, **kwargs)
            
        self.assertCML(cubes, reference_filename, checksum=False)
    
    def assertCDL(self, netcdf_filename, reference_filename):
        """
        Converts the given CF-netCDF file to CDL for comparison with
        the reference CDL file, or creates the reference file if it
        doesn't exist.

        """
        # Convert the netCDF file to CDL file format.
        cdl_filename = iris.util.create_temp_filename(suffix='.cdl')

        with open(cdl_filename, 'w') as cdl_file:
            subprocess.check_call(['ncdump', '-h', netcdf_filename], stderr=cdl_file, stdout=cdl_file)

        # Ingest the CDL for comparison, excluding first line.
        with open(cdl_filename, 'r') as cdl_file:
           cdl = ''.join(cdl_file.readlines()[1:])

        os.remove(cdl_filename)
        reference_path = get_result_path(reference_filename)
        self._check_same(cdl, reference_path, reference_filename, type_comparison_name='CDL')

    def assertCML(self, cubes, reference_filename, checksum=True):
        """
        Checks the given cubes match the reference file, or creates the
        reference file if it doesn't exist.

        """
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]
        
        if isinstance(cubes, (list, tuple)):
            xml = iris.cube.CubeList(cubes).xml(checksum=checksum)
        else:
            xml = cubes.xml(checksum=checksum)
        reference_path = get_result_path(reference_filename)
        self._check_same(xml, reference_path, reference_filename)
        
    def assertTextFile(self, source_filename, reference_filename, desc="text file"):
        """Check if two text files are the same, printing any diffs."""
        with open(source_filename) as source_file:
            source_text = source_file.readlines()
        with open(reference_filename) as reference_file:
            reference_text = reference_file.readlines()
        if reference_text != source_text:
            diff = ''.join(difflib.unified_diff(reference_text, source_text, 'Reference', 'Test result', '', '', 0))
            self.fail("%s does not match reference file: %s\n%s" % (desc, reference_filename, diff))

    def assertCubeDataAlmostEqual(self, cube, reference_filename, *args, **kwargs):
        reference_path = get_result_path(reference_filename)
        if os.path.isfile(reference_path):
            kwargs.setdefault('err_msg', 'Reference file %s' % reference_path)
            
            result = numpy.load(reference_path)
            if isinstance(result, numpy.lib.npyio.NpzFile):
                self.assertIsInstance(cube.data, numpy.ma.MaskedArray, 'Cube data was not a masked array.')
                mask = result['mask']
                # clear the cube's data where it is masked to avoid any non-initialised array data
                cube.data.data[cube.data.mask] = cube.data.fill_value
                numpy.testing.assert_array_almost_equal(cube.data.data, result['data'], *args, **kwargs)
                numpy.testing.assert_array_equal(cube.data.mask, mask, *args, **kwargs)
            else:
                numpy.testing.assert_array_almost_equal(cube.data, result, *args, **kwargs)
        else:
            self._ensure_folder(reference_path)
            logger.warning('Creating result file: %s', reference_path)
            if isinstance(cube.data, numpy.ma.MaskedArray):
                # clear the cube's data where it is masked to avoid any non-initialised array data
                data = cube.data.data[cube.data.mask] = cube.data.fill_value
                numpy.savez(file(reference_path, 'wb'), data=data, mask=cube.data.mask)
            else:
                numpy.save(file(reference_path, 'wb'), cube.data)
    
    def assertFilesEqual(self, test_filename, reference_filename):
        reference_path = get_result_path(reference_filename)
        if os.path.isfile(reference_path):
            self.assertTrue(filecmp.cmp(test_filename, reference_path))
        else:
            self._ensure_folder(reference_path)
            logger.warning('Creating result file: %s', reference_path)
            shutil.copy(test_filename, reference_path)
    
    def assertString(self, string, reference_filename):
        reference_path = get_result_path(reference_filename)
        self._check_same(string, reference_path, reference_filename, type_comparison_name='Strings')
    
    def assertRepr(self, obj, reference_filename):
        self.assertString(repr(obj), reference_filename)
    
    def _check_same(self, item, reference_path, reference_filename, type_comparison_name='CML'):
        if os.path.isfile(reference_path):
            reference = ''.join(open(reference_path, 'r').readlines())
            self._assert_str_same(reference, item, reference_filename, type_comparison_name)
        else:
            self._ensure_folder(reference_path)
            logger.warning('Creating result file: %s', reference_path)
            open(reference_path, 'w').writelines(item)

    def assertXMLElement(self, obj, reference_filename):
        """
        Calls the xml_element method given obj and asserts the result is the same as the test file.
        
        """
        doc = xml.dom.minidom.Document()
        doc.appendChild(obj.xml_element(doc))
        pretty_xml = doc.toprettyxml(indent="  ")
        reference_path = get_result_path(reference_filename)
        self._check_same(pretty_xml, reference_path, reference_filename, type_comparison_name='XML')
        
    def assertArrayEqual(self, a, b):
        numpy.testing.assert_array_equal(a, b)

    def assertAttributesEqual(self, attr1, attr2):
        """
        Asserts two mappings (dictionaries) are equal after
        stripping out all timestamps of the form 'dd/mm/yy hh:mm:ss Iris: ' 
        from values associated with a key of 'history'. This allows 
        tests that compare the attributes property of cubes to be 
        independent of timestamp.

        """
        def attr_filter(attr):
            result = {}
            for key, value in attr.iteritems():
                if key == 'history':
                    value = re.sub("[\d\/]{8} [\d\:]{8} Iris\: ", '', str(value))
                else:
                    value = str(value)
            return result
        return self.assertEqual(attr_filter(attr1), attr_filter(attr2)) 

    @contextlib.contextmanager
    def temp_filename(self, suffix=''):
        filename = iris.util.create_temp_filename(suffix)
        yield filename
        os.remove(filename)

    def file_checksum(self, file_path):
        """
        Generate checksum from file.
        """
        in_file = open(file_path, "rb")
        return zlib.crc32(in_file.read())
    
    def _unique_id(self):
        """
        Returns the unique ID for the current assertion.

        The ID is composed of two parts: a unique ID for the current test
        (which is itself composed of the module, class, and test names), and
        a sequential counter (specific to the current test) that is incremented
        on each call.

        For example, calls from a "test_tx" routine followed by a "test_ty"
        routine might result in::
            test_plot.TestContourf.test_tx.0
            test_plot.TestContourf.test_tx.1
            test_plot.TestContourf.test_tx.2
            test_plot.TestContourf.test_ty.0

        """
        # Obtain a consistent ID for the current test.
        
        # NB. unittest.TestCase.id() returns different values depending on
        # whether the test has been run explicitly, or via test discovery.
        # For example:
        #   python tests/test_plot.py => '__main__.TestContourf.test_tx'
        #   ird -t => 'iris.tests.test_plot.TestContourf.test_tx'
        bits = self.id().split('.')[-3:]
        if bits[0] == '__main__':
            file_name = os.path.basename(sys.modules['__main__'].__file__)
            bits[0] = os.path.splitext(file_name)[0]
        test_id = '.'.join(bits)

        # Derive the sequential assertion ID within the test
        assertion_id = self._assertion_counts[test_id]
        self._assertion_counts[test_id] += 1
        
        return test_id + '.' + str(assertion_id)
    
    def _ensure_folder(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            logger.warning('Creating folder: %s', dir_path)
            os.makedirs(dir_path)
    
    def _get_image_checksum(self, unique_id, resultant_checksum):
        checksum_result_path = get_result_path(('image_checksums', _PLATFORM, unique_id + '.txt'))
        if os.path.isfile(checksum_result_path):
            with open(checksum_result_path, 'r') as checksum_file:
                checksum = int(checksum_file.readline().strip())
        else:
            self._ensure_folder(checksum_result_path)
            logger.warning('Creating image checksum result file: %s', checksum_result_path)
            checksum = resultant_checksum
            open(checksum_result_path, 'w').writelines(str(checksum))
        return checksum
    
    def check_graphic(self):
        """Checks the CRC matches for the current matplotlib.pyplot figure, and closes the figure."""

        unique_id = self._unique_id()
        
        figure = plt.gcf()
        
        try:
            suffix = '.png'
            if _SAVE_FIGURES:
                file_path = os.path.join('image_results', unique_id) + suffix
                dir_path = os.path.dirname(file_path)
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)
            else:
                file_path = iris.util.create_temp_filename(suffix)

            figure.savefig(file_path)
            resultant_checksum = self.file_checksum(file_path)
            
            if not _SAVE_FIGURES:
                os.remove(file_path)

            checksum = self._get_image_checksum(unique_id, resultant_checksum)

            if _DISPLAY_FIGURES:
                if resultant_checksum != checksum:
                    print 'Test would have failed (new checksum: %s ; old checksum: %s)' % (resultant_checksum, checksum)
                plt.show()
            else:
                self.assertEqual(resultant_checksum, checksum, 'Image checksums not equal for %s' % unique_id)
        finally:
            plt.close()


class GraphicsTest(IrisTest):
    
    def tearDown(self):
        # If a plotting test bombs out it can leave the current figure in an odd state, so we make
        # sure it's been disposed of.
        plt.close()
  

def skip_data(fn):
    """
    Decorator to decide if to run the test or not based on the
    availability of external data.

    """
    valid_data = (iris.config.DATA_REPOSITORY and
                  os.path.isdir(iris.config.DATA_REPOSITORY))
    if valid_data and not os.environ.get('IRIS_TEST_NO_DATA'):
        return fn
    else:
        skip = unittest.skip("These/this test(s) requires external data.")
        return skip(fn)
