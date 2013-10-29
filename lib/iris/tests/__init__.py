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
"""
Provides testing capabilities and customisations specific to Iris.

.. note:: This module needs to control the matplotlib backend, so it
          **must** be imported before ``matplotlib.pyplot``.

The primary class for this module is :class:`IrisTest`.

By default, this module sets the matplotlib backend to "agg". But when
this module is imported it checks ``sys.argv`` for the flag "-d". If
found, it is removed from ``sys.argv`` and the matplotlib backend is
switched to "tkagg" to allow the interactive visual inspection of
graphical test results.

"""
import collections
import contextlib
import difflib
import filecmp
import gzip
import logging
import os
import os.path
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
import matplotlib.testing.compare as mcompare
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import iris.cube
import iris.config
import iris.util


#: Basepath for test results.
_RESULT_PATH = os.path.join(os.path.dirname(__file__), 'results')

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

if '-d' in sys.argv:
    sys.argv.remove('-d')
    plt.switch_backend('tkagg')
    _DISPLAY_FIGURES = True
else:
    plt.switch_backend('agg')

_DEFAULT_IMAGE_TOLERANCE = 0.001


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
            lines.insert(10, '  -d                   Display matplotlib figures (uses tkagg).')
            lines.insert(11, '                       NOTE: To compare results of failing tests, ')
            lines.insert(12, '                             use idiff.py instead')
            lines.insert(13, '  --data-files-used    Save a list of files used to a temporary file')
            print '\n'.join(lines)
    else:
        unittest.main()


def get_data_path(relative_path):
    """
    Return the absolute path to a data file when given the relative path
    as a string, or sequence of strings.

    """
    if not isinstance(relative_path, basestring):
        relative_path = os.path.join(*relative_path)
    data_path = os.path.join(iris.config.TEST_DATA_DIR, relative_path)

    if _EXPORT_DATAPATHS_FILE is not None:
        _EXPORT_DATAPATHS_FILE.write(data_path + '\n')

    if isinstance(data_path, basestring) and not os.path.exists(data_path):
        # if the file is gzipped, ungzip it and return the path of the ungzipped
        # file.
        gzipped_fname = data_path + '.gz'
        if os.path.exists(gzipped_fname):
            with gzip.open(gzipped_fname, 'rb') as gz_fh:
                with open(data_path, 'wb') as fh:
                    fh.writelines(gz_fh)

    return data_path


def get_result_path(relative_path):
    """Returns the absolute path to a result file when given the relative path
    as a string, or sequence of strings."""
    if not isinstance(relative_path, basestring):
        relative_path = os.path.join(*relative_path)
    return os.path.abspath(os.path.join(_RESULT_PATH, relative_path))


class IrisTest(unittest.TestCase):
    """A subclass of unittest.TestCase which provides Iris specific testing functionality."""

    _assertion_counts = collections.defaultdict(int)

    @classmethod
    def setUpClass(cls):
        # Ensure that the CF profile if turned-off for testing.
        iris.site_configuration['cf_profile'] = None

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

    def assertCDL(self, netcdf_filename, reference_filename, flags='-h'):
        """
        Converts the given CF-netCDF file to CDL for comparison with
        the reference CDL file, or creates the reference file if it
        doesn't exist.

        """
        # Convert the netCDF file to CDL file format.
        cdl_filename = iris.util.create_temp_filename(suffix='.cdl')

        if flags is None:
            flags = []
        elif isinstance(flags, basestring):
            flags = flags.split()
        else:
            flags = map(str, flags)

        with open(cdl_filename, 'w') as cdl_file:
            subprocess.check_call(['ncdump'] + flags + [netcdf_filename],
                                  stderr=cdl_file, stdout=cdl_file)

        # Ingest the CDL for comparison, excluding first line.
        with open(cdl_filename, 'r') as cdl_file:
            lines = cdl_file.readlines()[1:]

        # Sort the dimensions (except for the first, which can be unlimited).
        # This gives consistent CDL across different platforms.
        sort_key = lambda line: ('UNLIMITED' not in line, line)
        dimension_lines = slice(lines.index('dimensions:\n') + 1,
                                lines.index('variables:\n'))
        lines[dimension_lines] = sorted(lines[dimension_lines], key=sort_key)
        cdl = ''.join(lines)

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

            result = np.load(reference_path)
            if isinstance(result, np.lib.npyio.NpzFile):
                self.assertIsInstance(cube.data, ma.MaskedArray, 'Cube data was not a masked array.')
                # Avoid comparing any non-initialised array data.
                data = cube.data.filled()
                np.testing.assert_array_almost_equal(data, result['data'],
                                                     *args, **kwargs)
                np.testing.assert_array_equal(cube.data.mask, result['mask'])
            else:
                np.testing.assert_array_almost_equal(cube.data, result, *args, **kwargs)
        else:
            self._ensure_folder(reference_path)
            logger.warning('Creating result file: %s', reference_path)
            if isinstance(cube.data, ma.MaskedArray):
                # Avoid recording any non-initialised array data.
                data = cube.data.filled()
                np.savez(file(reference_path, 'wb'), data=data, mask=cube.data.mask)
            else:
                np.save(file(reference_path, 'wb'), cube.data)

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
        # If the test string is a unicode string, encode as
        # utf-8 before comparison to the reference string.
        if isinstance(string, unicode):
            string = string.encode('utf-8')
        self._check_same(string, reference_path, reference_filename,
                         type_comparison_name='Strings')

    def assertRepr(self, obj, reference_filename):
        self.assertString(repr(obj), reference_filename)

    def _check_same(self, item, reference_path, reference_filename, type_comparison_name='CML'):
        if os.path.isfile(reference_path):
            reference = ''.join(open(reference_path, 'r').readlines())
            self._assert_str_same(reference, item, reference_filename, type_comparison_name)
        else:
            self._ensure_folder(reference_path)
            logger.warning('Creating result file: %s', reference_path)
            open(reference_path, 'w').writelines(
                part.encode('utf-8') if isinstance(part, unicode) else part
                for part in item)

    def assertXMLElement(self, obj, reference_filename):
        """
        Calls the xml_element method given obj and asserts the result is the same as the test file.

        """
        doc = xml.dom.minidom.Document()
        doc.appendChild(obj.xml_element(doc))
        pretty_xml = doc.toprettyxml(indent="  ")
        reference_path = get_result_path(reference_filename)
        self._check_same(pretty_xml, reference_path, reference_filename, type_comparison_name='XML')
        
    def assertArrayEqual(self, a, b, err_msg=''):
        np.testing.assert_array_equal(a, b, err_msg=err_msg)

    def assertMaskedArrayEqual(self, a, b):
        """
        Check that masked arrays are equal. This requires the
        unmasked values and masks to be identical.

        """
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_array_equal(
            a[~a.mask].data, b[~b.mask].data)

    def assertArrayAlmostEqual(self, a, b):
        np.testing.assert_array_almost_equal(a, b)

    def assertMaskedArrayAlmostEqual(self, a, b):
        """
        Check that masked arrays are almost equal. This requires the
        masks to be identical, and the unmasked values to be almost
        equal.

        """
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_array_almost_equal(
            a[~a.mask].data, b[~b.mask].data)

    def assertArrayAllClose(self, a, b, rtol=1.0e-7, atol=0.0, **kwargs):
        """
        Check arrays are equal, within given relative + absolute tolerances.

        Args:

        * a, b (array-like):
            Two arrays to compare.

        Kwargs:

        * rtol, atol (float):
            Relative and absolute tolerances to apply.

        Any additional kwargs are passed to numpy.testing.assert_allclose.

        Performs pointwise toleranced comparison, and raises an assertion if
        the two are not equal 'near enough'.
        For full details see underlying routine numpy.testing.assert_allclose.

        """
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, **kwargs)

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

    def check_graphic(self, tol=_DEFAULT_IMAGE_TOLERANCE):
        """Checks the CRC matches for the current matplotlib.pyplot figure, and closes the figure."""

        unique_id = self._unique_id()

        figure = plt.gcf()

        try:
            expected_fname = os.path.join(os.path.dirname(__file__),
                                          'results', 'visual_tests',
                                          unique_id + '.png')

            if not os.path.isdir(os.path.dirname(expected_fname)):
                os.makedirs(os.path.dirname(expected_fname))

            result_fname = os.path.join(os.path.dirname(__file__),
                                        'result_image_comparison',
                                        'result-' + unique_id + '.png')

            if not os.path.isdir(os.path.dirname(result_fname)):
                # Handle race-condition where the directories are
                # created sometime between the check above and the
                # creation attempt below.
                try:
                    os.makedirs(os.path.dirname(result_fname))
                except OSError as err:
                    # Don't care about "File exists"
                    if err.errno != 17:
                        raise

            figure.savefig(result_fname)

            if not os.path.exists(expected_fname):
                warnings.warn('Created image for test %s' % unique_id)
                shutil.copy2(result_fname, expected_fname)

            err = mcompare.compare_images(expected_fname, result_fname, tol=tol)

            if _DISPLAY_FIGURES:
                if err:
                    print 'Image comparison would have failed. Message: %s' % err
                plt.show()
            else:
                assert not err, 'Image comparison failed. Message: %s' % err

        finally:
            plt.close()


class GraphicsTest(IrisTest):

    def tearDown(self):
        # If a plotting test bombs out it can leave the current figure
        # in an odd state, so we make sure it's been disposed of.
        plt.close()


def skip_data(fn):
    """
    Decorator to choose whether to run tests, based on the availability of
    external data.

    Example usage:
        @skip_data
        class MyDataTests(tests.IrisTest):
            ...

    """
    no_data = (not iris.config.TEST_DATA_DIR
               or not os.path.isdir(iris.config.TEST_DATA_DIR)
               or os.environ.get('IRIS_TEST_NO_DATA'))

    skip = unittest.skipIf(
        condition=no_data,
        reason='Test(s) require external data.')

    return skip(fn)
