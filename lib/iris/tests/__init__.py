# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
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

import codecs
import collections
from collections.abc import Mapping
import contextlib
import datetime
import difflib
import filecmp
import functools
import gzip
import inspect
import io
import json
import math
import os
import os.path
import re
import shutil
import subprocess
import sys
import threading
from typing import Dict, List
import unittest
from unittest import mock
import warnings
import xml.dom.minidom
import zlib

import filelock
import numpy as np
import numpy.ma as ma
import requests

import iris.config
import iris.cube
import iris.util

# Test for availability of matplotlib.
# (And remove matplotlib as an iris.tests dependency.)
try:
    import matplotlib

    # Override any user settings e.g. from matplotlibrc file.
    matplotlib.rcdefaults()
    # Set backend *after* rcdefaults, as we don't want that overridden (#3846).
    matplotlib.use("agg")
    # Standardise the figure size across matplotlib versions.
    # This permits matplotlib png image comparison.
    matplotlib.rcParams["figure.figsize"] = [8.0, 6.0]
    import matplotlib.pyplot as plt
except ImportError:
    MPL_AVAILABLE = False
else:
    MPL_AVAILABLE = True

try:
    from osgeo import gdal  # noqa
except ImportError:
    GDAL_AVAILABLE = False
else:
    GDAL_AVAILABLE = True

try:
    import iris_sample_data  # noqa
except ImportError:
    SAMPLE_DATA_AVAILABLE = False
else:
    SAMPLE_DATA_AVAILABLE = True

try:
    import nc_time_axis  # noqa

    NC_TIME_AXIS_AVAILABLE = True
except ImportError:
    NC_TIME_AXIS_AVAILABLE = False

try:
    # Added a timeout to stop the call to requests.get hanging when running
    # on a platform which has restricted/no internet access.
    requests.get("https://github.com/SciTools/iris", timeout=10.0)
    INET_AVAILABLE = True
except requests.exceptions.ConnectionError:
    INET_AVAILABLE = False

try:
    import stratify  # noqa

    STRATIFY_AVAILABLE = True
except ImportError:
    STRATIFY_AVAILABLE = False

#: Basepath for test results.
_RESULT_PATH = os.path.join(os.path.dirname(__file__), "results")
#: Default perceptual hash size.
_HASH_SIZE = 16
#: Default maximum perceptual hash hamming distance.
_HAMMING_DISTANCE = 2

if "--data-files-used" in sys.argv:
    sys.argv.remove("--data-files-used")
    fname = "/var/tmp/all_iris_test_resource_paths.txt"
    print("saving list of files used by tests to %s" % fname)
    _EXPORT_DATAPATHS_FILE = open(fname, "w")
else:
    _EXPORT_DATAPATHS_FILE = None


if "--create-missing" in sys.argv:
    sys.argv.remove("--create-missing")
    print("Allowing creation of missing test results.")
    os.environ["IRIS_TEST_CREATE_MISSING"] = "true"


# Whether to display matplotlib output to the screen.
_DISPLAY_FIGURES = False

if MPL_AVAILABLE and "-d" in sys.argv:
    sys.argv.remove("-d")
    plt.switch_backend("tkagg")
    _DISPLAY_FIGURES = True

# Threading non re-entrant blocking lock to ensure thread-safe plotting.
_lock = threading.Lock()


def main():
    """A wrapper for unittest.main() which adds iris.test specific options to the help (-h) output."""
    if "-h" in sys.argv or "--help" in sys.argv:
        stdout = sys.stdout
        buff = io.StringIO()
        # NB. unittest.main() raises an exception after it's shown the help text
        try:
            sys.stdout = buff
            unittest.main()
        finally:
            sys.stdout = stdout
            lines = buff.getvalue().split("\n")
            lines.insert(9, "Iris-specific options:")
            lines.insert(
                10,
                "  -d                   Display matplotlib figures (uses tkagg).",
            )
            lines.insert(
                11,
                "                       NOTE: To compare results of failing tests, ",
            )
            lines.insert(
                12, "                             use idiff.py instead"
            )
            lines.insert(
                13,
                "  --data-files-used    Save a list of files used to a temporary file",
            )
            lines.insert(
                14, "  -m                   Create missing test results"
            )
            print("\n".join(lines))
    else:
        unittest.main()


def get_data_path(relative_path):
    """
    Return the absolute path to a data file when given the relative path
    as a string, or sequence of strings.

    """
    if not isinstance(relative_path, str):
        relative_path = os.path.join(*relative_path)
    test_data_dir = iris.config.TEST_DATA_DIR
    if test_data_dir is None:
        test_data_dir = ""
    data_path = os.path.join(test_data_dir, relative_path)

    if _EXPORT_DATAPATHS_FILE is not None:
        _EXPORT_DATAPATHS_FILE.write(data_path + "\n")

    if isinstance(data_path, str) and not os.path.exists(data_path):
        # if the file is gzipped, ungzip it and return the path of the ungzipped
        # file.
        gzipped_fname = data_path + ".gz"
        if os.path.exists(gzipped_fname):
            with gzip.open(gzipped_fname, "rb") as gz_fh:
                try:
                    with open(data_path, "wb") as fh:
                        fh.writelines(gz_fh)
                except IOError:
                    # Put ungzipped data file in a temporary path, since we
                    # can't write to the original path (maybe it is owned by
                    # the system.)
                    _, ext = os.path.splitext(data_path)
                    data_path = iris.util.create_temp_filename(suffix=ext)
                    with open(data_path, "wb") as fh:
                        fh.writelines(gz_fh)

    return data_path


class IrisTest_nometa(unittest.TestCase):
    """A subclass of unittest.TestCase which provides Iris specific testing functionality."""

    _assertion_counts = collections.defaultdict(int)

    @classmethod
    def setUpClass(cls):
        # Ensure that the CF profile if turned-off for testing.
        iris.site_configuration["cf_profile"] = None

    def _assert_str_same(
        self,
        reference_str,
        test_str,
        reference_filename,
        type_comparison_name="Strings",
    ):
        if reference_str != test_str:
            diff = "".join(
                difflib.unified_diff(
                    reference_str.splitlines(1),
                    test_str.splitlines(1),
                    "Reference",
                    "Test result",
                    "",
                    "",
                    0,
                )
            )
            self.fail(
                "%s do not match: %s\n%s"
                % (type_comparison_name, reference_filename, diff)
            )

    @staticmethod
    def get_result_path(relative_path):
        """
        Returns the absolute path to a result file when given the relative path
        as a string, or sequence of strings.

        """
        if not isinstance(relative_path, str):
            relative_path = os.path.join(*relative_path)
        return os.path.abspath(os.path.join(_RESULT_PATH, relative_path))

    def assertStringEqual(
        self, reference_str, test_str, type_comparison_name="strings"
    ):
        if reference_str != test_str:
            diff = "\n".join(
                difflib.unified_diff(
                    reference_str.splitlines(),
                    test_str.splitlines(),
                    "Reference",
                    "Test result",
                    "",
                    "",
                    0,
                )
            )
            self.fail(
                "{} do not match:\n{}".format(type_comparison_name, diff)
            )

    def result_path(self, basename=None, ext=""):
        """
        Return the full path to a test result, generated from the \
        calling file, class and, optionally, method.

        Optional kwargs :

            * basename    - File basename. If omitted, this is \
                            generated from the calling method.
            * ext         - Appended file extension.

        """
        if ext and not ext.startswith("."):
            ext = "." + ext

        # Generate the folder name from the calling file name.
        path = os.path.abspath(inspect.getfile(self.__class__))
        path = os.path.splitext(path)[0]
        sub_path = path.rsplit("iris", 1)[1].split("tests", 1)[1][1:]

        # Generate the file name from the calling function name?
        if basename is None:
            stack = inspect.stack()
            for frame in stack[1:]:
                if "test_" in frame[3]:
                    basename = frame[3].replace("test_", "")
                    break
        filename = basename + ext

        result = os.path.join(
            self.get_result_path(""),
            sub_path.replace("test_", ""),
            self.__class__.__name__.replace("Test_", ""),
            filename,
        )
        return result

    def assertCMLApproxData(self, cubes, reference_filename=None, **kwargs):
        # passes args and kwargs on to approx equal
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]
        if reference_filename is None:
            reference_filename = self.result_path(None, "cml")
            reference_filename = [self.get_result_path(reference_filename)]
        for i, cube in enumerate(cubes):
            fname = list(reference_filename)
            # don't want the ".cml" for the json stats file
            if fname[-1].endswith(".cml"):
                fname[-1] = fname[-1][:-4]
            fname[-1] += ".data.%d.json" % i
            self.assertDataAlmostEqual(cube.data, fname, **kwargs)
        self.assertCML(cubes, reference_filename, checksum=False)

    def assertCDL(self, netcdf_filename, reference_filename=None, flags="-h"):
        """
        Test that the CDL for the given netCDF file matches the contents
        of the reference file.

        If the environment variable IRIS_TEST_CREATE_MISSING is
        non-empty, the reference file is created if it doesn't exist.

        Args:

        * netcdf_filename:
            The path to the netCDF file.

        Kwargs:

        * reference_filename:
            The relative path (relative to the test results directory).
            If omitted, the result is generated from the calling
            method's name, class, and module using
            :meth:`iris.tests.IrisTest.result_path`.

        * flags:
            Command-line flags for `ncdump`, as either a whitespace
            separated string or an iterable. Defaults to '-h'.

        """
        if reference_filename is None:
            reference_path = self.result_path(None, "cdl")
        else:
            reference_path = self.get_result_path(reference_filename)

        # Convert the netCDF file to CDL file format.
        if flags is None:
            flags = []
        elif isinstance(flags, str):
            flags = flags.split()
        else:
            flags = list(map(str, flags))

        try:
            # Python3 only: use subprocess.run()
            args = ["ncdump"] + flags + [netcdf_filename]
            cdl = subprocess.check_output(args, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            print(exc.output)
            raise

        # Ingest the CDL for comparison, excluding first line.
        lines = cdl.decode("ascii").splitlines()
        lines = lines[1:]

        # Ignore any lines of the general form "... :_NCProperties = ..."
        # (an extra global attribute, displayed by older versions of ncdump).
        re_ncprop = re.compile(r"^\s*:_NCProperties *=")
        lines = [line for line in lines if not re_ncprop.match(line)]

        # Sort the dimensions (except for the first, which can be unlimited).
        # This gives consistent CDL across different platforms.
        def sort_key(line):
            return ("UNLIMITED" not in line, line)

        dimension_lines = slice(
            lines.index("dimensions:") + 1, lines.index("variables:")
        )
        lines[dimension_lines] = sorted(lines[dimension_lines], key=sort_key)
        cdl = "\n".join(lines) + "\n"

        self._check_same(cdl, reference_path, type_comparison_name="CDL")

    def assertCML(self, cubes, reference_filename=None, checksum=True):
        """
        Test that the CML for the given cubes matches the contents of
        the reference file.

        If the environment variable IRIS_TEST_CREATE_MISSING is
        non-empty, the reference file is created if it doesn't exist.

        Args:

        * cubes:
            Either a Cube or a sequence of Cubes.

        Kwargs:

        * reference_filename:
            The relative path (relative to the test results directory).
            If omitted, the result is generated from the calling
            method's name, class, and module using
            :meth:`iris.tests.IrisTest.result_path`.

        * checksum:
            When True, causes the CML to include a checksum for each
            Cube's data. Defaults to True.

        """
        if isinstance(cubes, iris.cube.Cube):
            cubes = [cubes]
        if reference_filename is None:
            reference_filename = self.result_path(None, "cml")

        if isinstance(cubes, (list, tuple)):
            xml = iris.cube.CubeList(cubes).xml(
                checksum=checksum, order=False, byteorder=False
            )
        else:
            xml = cubes.xml(checksum=checksum, order=False, byteorder=False)
        reference_path = self.get_result_path(reference_filename)
        self._check_same(xml, reference_path)

    def assertTextFile(
        self, source_filename, reference_filename, desc="text file"
    ):
        """Check if two text files are the same, printing any diffs."""
        with open(source_filename) as source_file:
            source_text = source_file.readlines()
        with open(reference_filename) as reference_file:
            reference_text = reference_file.readlines()
        if reference_text != source_text:
            diff = "".join(
                difflib.unified_diff(
                    reference_text,
                    source_text,
                    "Reference",
                    "Test result",
                    "",
                    "",
                    0,
                )
            )
            self.fail(
                "%s does not match reference file: %s\n%s"
                % (desc, reference_filename, diff)
            )

    def assertDataAlmostEqual(self, data, reference_filename, **kwargs):
        reference_path = self.get_result_path(reference_filename)
        if self._check_reference_file(reference_path):
            kwargs.setdefault("err_msg", "Reference file %s" % reference_path)
            with open(reference_path, "r") as reference_file:
                stats = json.load(reference_file)
                self.assertEqual(stats.get("shape", []), list(data.shape))
                self.assertEqual(
                    stats.get("masked", False), ma.is_masked(data)
                )
                nstats = np.array(
                    (
                        stats.get("mean", 0.0),
                        stats.get("std", 0.0),
                        stats.get("max", 0.0),
                        stats.get("min", 0.0),
                    ),
                    dtype=np.float64,
                )
                if math.isnan(stats.get("mean", 0.0)):
                    self.assertTrue(math.isnan(data.mean()))
                else:
                    data_stats = np.array(
                        (data.mean(), data.std(), data.max(), data.min()),
                        dtype=np.float64,
                    )
                    self.assertArrayAllClose(nstats, data_stats, **kwargs)
        else:
            self._ensure_folder(reference_path)
            stats = collections.OrderedDict(
                [
                    ("std", np.float64(data.std())),
                    ("min", np.float64(data.min())),
                    ("max", np.float64(data.max())),
                    ("shape", data.shape),
                    ("masked", ma.is_masked(data)),
                    ("mean", np.float64(data.mean())),
                ]
            )
            with open(reference_path, "w") as reference_file:
                reference_file.write(json.dumps(stats))

    def assertFilesEqual(self, test_filename, reference_filename):
        reference_path = self.get_result_path(reference_filename)
        if self._check_reference_file(reference_path):
            fmt = "test file {!r} does not match reference {!r}."
            self.assertTrue(
                filecmp.cmp(test_filename, reference_path),
                fmt.format(test_filename, reference_path),
            )
        else:
            self._ensure_folder(reference_path)
            shutil.copy(test_filename, reference_path)

    def assertString(self, string, reference_filename=None):
        """
        Test that `string` matches the contents of the reference file.

        If the environment variable IRIS_TEST_CREATE_MISSING is
        non-empty, the reference file is created if it doesn't exist.

        Args:

        * string:
            The string to check.

        Kwargs:

        * reference_filename:
            The relative path (relative to the test results directory).
            If omitted, the result is generated from the calling
            method's name, class, and module using
            :meth:`iris.tests.IrisTest.result_path`.

        """
        if reference_filename is None:
            reference_path = self.result_path(None, "txt")
        else:
            reference_path = self.get_result_path(reference_filename)
        self._check_same(
            string, reference_path, type_comparison_name="Strings"
        )

    def assertRepr(self, obj, reference_filename):
        self.assertString(repr(obj), reference_filename)

    def _check_same(self, item, reference_path, type_comparison_name="CML"):
        if self._check_reference_file(reference_path):
            with open(reference_path, "rb") as reference_fh:
                reference = "".join(
                    part.decode("utf-8") for part in reference_fh.readlines()
                )
            self._assert_str_same(
                reference, item, reference_path, type_comparison_name
            )
        else:
            self._ensure_folder(reference_path)
            with open(reference_path, "wb") as reference_fh:
                reference_fh.writelines(part.encode("utf-8") for part in item)

    def assertXMLElement(self, obj, reference_filename):
        """
        Calls the xml_element method given obj and asserts the result is the same as the test file.

        """
        doc = xml.dom.minidom.Document()
        doc.appendChild(obj.xml_element(doc))
        # sort the attributes on xml elements before testing against known good state.
        # this is to be compatible with stored test output where xml attrs are stored in alphabetical order,
        # (which was default behaviour in python <3.8, but changed to insert order in >3.8)
        doc = iris.cube.Cube._sort_xml_attrs(doc)
        pretty_xml = doc.toprettyxml(indent="  ")
        reference_path = self.get_result_path(reference_filename)
        self._check_same(
            pretty_xml, reference_path, type_comparison_name="XML"
        )

    def assertArrayEqual(self, a, b, err_msg=""):
        np.testing.assert_array_equal(a, b, err_msg=err_msg)

    @contextlib.contextmanager
    def _recordWarningMatches(self, expected_regexp=""):
        # Record warnings raised matching a given expression.
        matches = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield matches
        messages = [str(warning.message) for warning in w]
        expr = re.compile(expected_regexp)
        matches.extend(message for message in messages if expr.search(message))

    @contextlib.contextmanager
    def assertWarnsRegexp(self, expected_regexp=""):
        # Check that a warning is raised matching a given expression.
        with self._recordWarningMatches(expected_regexp) as matches:
            yield

        msg = "Warning matching '{}' not raised."
        msg = msg.format(expected_regexp)
        self.assertTrue(matches, msg)

    @contextlib.contextmanager
    def assertLogs(self, logger=None, level=None, msg_regex=None):
        """
        An extended version of the usual :meth:`unittest.TestCase.assertLogs`,
        which also exercises the logger's message formatting.

        Also adds the ``msg_regex`` kwarg:
        If used, check that the result is a single message of the specified
        level, and that it matches this regex.

        The inherited version of this method temporarily *replaces* the logger
        in order to capture log records generated within the context.
        However, in doing so it prevents any messages from being formatted
        by the original logger.
        This version first calls the original method, but then *also* exercises
        the message formatters of all the logger's handlers, just to check that
        there are no formatting errors.

        """
        # Invoke the standard assertLogs behaviour.
        assertlogging_context = super().assertLogs(logger, level)
        with assertlogging_context as watcher:
            # Run the caller context, as per original method.
            yield watcher
        # Check for any formatting errors by running all the formatters.
        for record in watcher.records:
            for handler in assertlogging_context.logger.handlers:
                handler.format(record)

        # Check message, if requested.
        if msg_regex:
            self.assertEqual(len(watcher.records), 1)
            rec = watcher.records[0]
            self.assertEqual(level, rec.levelname)
            self.assertRegex(rec.msg, msg_regex)

    @contextlib.contextmanager
    def assertNoWarningsRegexp(self, expected_regexp=""):
        # Check that no warning matching the given expression is raised.
        with self._recordWarningMatches(expected_regexp) as matches:
            yield

        msg = "Unexpected warning(s) raised, matching '{}' : {!r}."
        msg = msg.format(expected_regexp, matches)
        self.assertFalse(matches, msg)

    def _assertMaskedArray(self, assertion, a, b, strict, **kwargs):
        # Define helper function to extract unmasked values as a 1d
        # array.
        def unmasked_data_as_1d_array(array):
            array = ma.asarray(array)
            if array.ndim == 0:
                if array.mask:
                    data = np.array([])
                else:
                    data = np.array([array.data])
            else:
                data = array.data[~ma.getmaskarray(array)]
            return data

        # Compare masks. This will also check that the array shapes
        # match, which is not tested when comparing unmasked values if
        # strict is False.
        a_mask, b_mask = ma.getmaskarray(a), ma.getmaskarray(b)
        np.testing.assert_array_equal(a_mask, b_mask)

        if strict:
            assertion(a.data, b.data, **kwargs)
        else:
            assertion(
                unmasked_data_as_1d_array(a),
                unmasked_data_as_1d_array(b),
                **kwargs,
            )

    def assertMaskedArrayEqual(self, a, b, strict=False):
        """
        Check that masked arrays are equal. This requires the
        unmasked values and masks to be identical.

        Args:

        * a, b (array-like):
            Two arrays to compare.

        Kwargs:

        * strict (bool):
            If True, perform a complete mask and data array equality check.
            If False (default), the data array equality considers only unmasked
            elements.

        """
        self._assertMaskedArray(np.testing.assert_array_equal, a, b, strict)

    def assertArrayAlmostEqual(self, a, b, decimal=6):
        np.testing.assert_array_almost_equal(a, b, decimal=decimal)

    def assertMaskedArrayAlmostEqual(self, a, b, decimal=6, strict=False):
        """
        Check that masked arrays are almost equal. This requires the
        masks to be identical, and the unmasked values to be almost
        equal.

        Args:

        * a, b (array-like):
            Two arrays to compare.

        Kwargs:

        * strict (bool):
            If True, perform a complete mask and data array equality check.
            If False (default), the data array equality considers only unmasked
            elements.

        * decimal (int):
            Equality tolerance level for
            :meth:`numpy.testing.assert_array_almost_equal`, with the meaning
            'abs(desired-actual) < 0.5 * 10**(-decimal)'

        """
        self._assertMaskedArray(
            np.testing.assert_array_almost_equal, a, b, strict, decimal=decimal
        )

    def assertArrayAllClose(self, a, b, rtol=1.0e-7, atol=1.0e-8, **kwargs):
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
        For full details see underlying routine numpy.allclose.

        """
        # Handle the 'err_msg' kwarg, which is the only API difference
        # between np.allclose and np.testing_assert_allclose.
        msg = kwargs.pop("err_msg", None)
        ok = np.allclose(a, b, rtol=rtol, atol=atol, **kwargs)
        if not ok:
            # Calculate errors above a pointwise tolerance : The method is
            # taken from "numpy.core.numeric.isclose".
            a, b = np.broadcast_arrays(a, b)
            errors = np.abs(a - b) - atol + rtol * np.abs(b)
            worst_inds = np.unravel_index(np.argmax(errors.flat), errors.shape)

            if msg is None:
                # Build a more useful message than np.testing.assert_allclose.
                msg = (
                    '\nARRAY CHECK FAILED "assertArrayAllClose" :'
                    "\n  with shapes={} {}, atol={}, rtol={}"
                    "\n  worst at element {} :  a={}  b={}"
                    "\n  absolute error ~{:.3g}, equivalent to rtol ~{:.3e}"
                )
                aval, bval = a[worst_inds], b[worst_inds]
                absdiff = np.abs(aval - bval)
                equiv_rtol = absdiff / bval
                msg = msg.format(
                    a.shape,
                    b.shape,
                    atol,
                    rtol,
                    worst_inds,
                    aval,
                    bval,
                    absdiff,
                    equiv_rtol,
                )

            raise AssertionError(msg)

    @contextlib.contextmanager
    def temp_filename(self, suffix=""):
        filename = iris.util.create_temp_filename(suffix)
        try:
            yield filename
        finally:
            os.remove(filename)

    def file_checksum(self, file_path):
        """
        Generate checksum from file.
        """
        with open(file_path, "rb") as in_file:
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
        bits = self.id().split(".")
        if bits[0] == "__main__":
            floc = sys.modules["__main__"].__file__
            path, file_name = os.path.split(os.path.abspath(floc))
            bits[0] = os.path.splitext(file_name)[0]
            folder, location = os.path.split(path)
            bits = [location] + bits
            while location not in ["iris", "gallery_tests"]:
                folder, location = os.path.split(folder)
                bits = [location] + bits
        test_id = ".".join(bits)

        # Derive the sequential assertion ID within the test
        assertion_id = self._assertion_counts[test_id]
        self._assertion_counts[test_id] += 1

        return test_id + "." + str(assertion_id)

    def _check_reference_file(self, reference_path):
        reference_exists = os.path.isfile(reference_path)
        if not (
            reference_exists or os.environ.get("IRIS_TEST_CREATE_MISSING")
        ):
            msg = "Missing test result: {}".format(reference_path)
            raise AssertionError(msg)
        return reference_exists

    def _ensure_folder(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def check_graphic(self):
        """
        Check the hash of the current matplotlib figure matches the expected
        image hash for the current graphic test.

        To create missing image test results, set the IRIS_TEST_CREATE_MISSING
        environment variable before running the tests. This will result in new
        and appropriately "<hash>.png" image files being generated in the image
        output directory, and the imagerepo.json file being updated.

        """
        from PIL import Image
        import imagehash

        dev_mode = os.environ.get("IRIS_TEST_CREATE_MISSING")
        unique_id = self._unique_id()
        repo_fname = os.path.join(_RESULT_PATH, "imagerepo.json")
        with open(repo_fname, "rb") as fi:
            repo: Dict[str, List[str]] = json.load(
                codecs.getreader("utf-8")(fi)
            )

        try:
            #: The path where the images generated by the tests should go.
            image_output_directory = os.path.join(
                os.path.dirname(__file__), "result_image_comparison"
            )
            if not os.access(image_output_directory, os.W_OK):
                if not os.access(os.getcwd(), os.W_OK):
                    raise IOError(
                        "Write access to a local disk is required "
                        "to run image tests.  Run the tests from a "
                        "current working directory you have write "
                        "access to to avoid this issue."
                    )
                else:
                    image_output_directory = os.path.join(
                        os.getcwd(), "iris_image_test_output"
                    )
            result_fname = os.path.join(
                image_output_directory, "result-" + unique_id + ".png"
            )

            if not os.path.isdir(image_output_directory):
                # Handle race-condition where the directories are
                # created sometime between the check above and the
                # creation attempt below.
                try:
                    os.makedirs(image_output_directory)
                except OSError as err:
                    # Don't care about "File exists"
                    if err.errno != 17:
                        raise

            def _create_missing():
                fname = "{}.png".format(phash)
                base_uri = (
                    "https://scitools.github.io/test-iris-imagehash/"
                    "images/v4/{}"
                )
                uri = base_uri.format(fname)
                hash_fname = os.path.join(image_output_directory, fname)
                uris = repo.setdefault(unique_id, [])
                uris.append(uri)
                print("Creating image file: {}".format(hash_fname))
                figure.savefig(hash_fname)
                msg = "Creating imagerepo entry: {} -> {}"
                print(msg.format(unique_id, uri))
                lock = filelock.FileLock(
                    os.path.join(_RESULT_PATH, "imagerepo.lock")
                )
                # The imagerepo.json file is a critical resource, so ensure
                # thread safe read/write behaviour via platform independent
                # file locking.
                with lock.acquire(timeout=600):
                    with open(repo_fname, "wb") as fo:
                        json.dump(
                            repo,
                            codecs.getwriter("utf-8")(fo),
                            indent=4,
                            sort_keys=True,
                        )

            # Calculate the test result perceptual image hash.
            buffer = io.BytesIO()
            figure = plt.gcf()
            figure.savefig(buffer, format="png")
            buffer.seek(0)
            phash = imagehash.phash(Image.open(buffer), hash_size=_HASH_SIZE)

            if unique_id not in repo:
                # The unique id might not be fully qualified, e.g.
                # expects iris.tests.test_quickplot.TestLabels.test_contour.0,
                # but got test_quickplot.TestLabels.test_contour.0
                # if we find single partial match from end of the key
                # then use that, else fall back to the unknown id state.
                matches = [key for key in repo if key.endswith(unique_id)]
                if len(matches) == 1:
                    unique_id = matches[0]

            if unique_id in repo:
                uris = repo[unique_id]
                # Extract the hex basename strings from the uris.
                hexes = [
                    os.path.splitext(os.path.basename(uri))[0] for uri in uris
                ]
                # Create the expected perceptual image hashes from the uris.
                to_hash = imagehash.hex_to_hash
                expected = [to_hash(uri_hex) for uri_hex in hexes]

                # Calculate hamming distance vector for the result hash.
                distances = [e - phash for e in expected]

                if np.all([hd > _HAMMING_DISTANCE for hd in distances]):
                    if dev_mode:
                        _create_missing()
                    else:
                        figure.savefig(result_fname)
                        msg = (
                            "Bad phash {} with hamming distance {} "
                            "for test {}."
                        )
                        msg = msg.format(phash, distances, unique_id)
                        if _DISPLAY_FIGURES:
                            emsg = "Image comparison would have failed: {}"
                            print(emsg.format(msg))
                        else:
                            emsg = "Image comparison failed: {}"
                            raise AssertionError(emsg.format(msg))
            else:
                if dev_mode:
                    _create_missing()
                else:
                    figure.savefig(result_fname)
                    emsg = "Missing image test result: {}."
                    raise AssertionError(emsg.format(unique_id))

            if _DISPLAY_FIGURES:
                plt.show()

        finally:
            plt.close()

    def _remove_testcase_patches(self):
        """Helper to remove per-testcase patches installed by :meth:`patch`."""
        # Remove all patches made, ignoring errors.
        for p in self.testcase_patches:
            p.stop()
        # Reset per-test patch control variable.
        self.testcase_patches.clear()

    def patch(self, *args, **kwargs):
        """
        Install a mock.patch, to be removed after the current test.

        The patch is created with mock.patch(*args, **kwargs).

        Returns:
            The substitute object returned by patch.start().

        For example::

            mock_call = self.patch('module.Class.call', return_value=1)
            module_Class_instance.call(3, 4)
            self.assertEqual(mock_call.call_args_list, [mock.call(3, 4)])

        """
        # Make the new patch and start it.
        patch = mock.patch(*args, **kwargs)
        start_result = patch.start()

        # Create the per-testcases control variable if it does not exist.
        # NOTE: this mimics a setUp method, but continues to work when a
        # subclass defines its own setUp.
        if not hasattr(self, "testcase_patches"):
            self.testcase_patches = {}

        # When installing the first patch, schedule remove-all at cleanup.
        if not self.testcase_patches:
            self.addCleanup(self._remove_testcase_patches)

        # Record the new patch and start object for reference.
        self.testcase_patches[patch] = start_result

        # Return patch replacement object.
        return start_result

    def assertArrayShapeStats(self, result, shape, mean, std_dev, rtol=1e-6):
        """
        Assert that the result, a cube, has the provided shape and that the
        mean and standard deviation of the data array are also as provided.
        Thus build confidence that a cube processing operation, such as a
        cube.regrid, has maintained its behaviour.

        """
        self.assertEqual(result.shape, shape)
        self.assertArrayAllClose(result.data.mean(), mean, rtol=rtol)
        self.assertArrayAllClose(result.data.std(), std_dev, rtol=rtol)

    def assertDictEqual(self, lhs, rhs, msg=None):
        """
        This method overrides unittest.TestCase.assertDictEqual (new in Python3.1)
        in order to cope with dictionary comparison where the value of a key may
        be a numpy array.

        """
        if not isinstance(lhs, Mapping):
            emsg = (
                f"Provided LHS argument is not a 'Mapping', got {type(lhs)}."
            )
            self.fail(emsg)

        if not isinstance(rhs, Mapping):
            emsg = (
                f"Provided RHS argument is not a 'Mapping', got {type(rhs)}."
            )
            self.fail(emsg)

        if set(lhs.keys()) != set(rhs.keys()):
            emsg = f"{lhs!r} != {rhs!r}."
            self.fail(emsg)

        for key in lhs.keys():
            lvalue, rvalue = lhs[key], rhs[key]

            if ma.isMaskedArray(lvalue) or ma.isMaskedArray(rvalue):
                if not ma.isMaskedArray(lvalue):
                    emsg = (
                        f"Dictionary key {key!r} values are not equal, "
                        f"the LHS value has type {type(lvalue)} and "
                        f"the RHS value has type {ma.core.MaskedArray}."
                    )
                    raise AssertionError(emsg)

                if not ma.isMaskedArray(rvalue):
                    emsg = (
                        f"Dictionary key {key!r} values are not equal, "
                        f"the LHS value has type {ma.core.MaskedArray} and "
                        f"the RHS value has type {type(lvalue)}."
                    )
                    raise AssertionError(emsg)

                self.assertMaskedArrayEqual(lvalue, rvalue)
            elif isinstance(lvalue, np.ndarray) or isinstance(
                rvalue, np.ndarray
            ):
                if not isinstance(lvalue, np.ndarray):
                    emsg = (
                        f"Dictionary key {key!r} values are not equal, "
                        f"the LHS value has type {type(lvalue)} and "
                        f"the RHS value has type {np.ndarray}."
                    )
                    raise AssertionError(emsg)

                if not isinstance(rvalue, np.ndarray):
                    emsg = (
                        f"Dictionary key {key!r} values are not equal, "
                        f"the LHS value has type {np.ndarray} and "
                        f"the RHS value has type {type(rvalue)}."
                    )
                    raise AssertionError(emsg)

                self.assertArrayEqual(lvalue, rvalue)
            else:
                if lvalue != rvalue:
                    emsg = (
                        f"Dictionary key {key!r} values are not equal, "
                        f"{lvalue!r} != {rvalue!r}."
                    )
                    raise AssertionError(emsg)

    def assertEqualAndKind(self, value, expected):
        # Check a value, and also its type 'kind' = float/integer/string.
        self.assertEqual(value, expected)
        self.assertEqual(
            np.array(value).dtype.kind, np.array(expected).dtype.kind
        )


# An environment variable controls whether test timings are output.
#
# NOTE: to run tests with timing output, nosetests cannot be used.
# At present, that includes not using "python setup.py test"
# The typically best way is like this :
#    $ export IRIS_TEST_TIMINGS=1
#    $ python -m unittest discover -s iris.tests
# and commonly adding ...
#    | grep "TIMING TEST" >iris_test_output.txt
#
_PRINT_TEST_TIMINGS = bool(int(os.environ.get("IRIS_TEST_TIMINGS", 0)))


def _method_path(meth, cls):
    return ".".join([cls.__module__, cls.__name__, meth.__name__])


def _testfunction_timing_decorator(fn, cls):
    # Function decorator for making a testcase print its execution time.
    @functools.wraps(fn)
    def inner(*args, **kwargs):
        start_time = datetime.datetime.now()
        try:
            result = fn(*args, **kwargs)
        finally:
            end_time = datetime.datetime.now()
            elapsed_time = (end_time - start_time).total_seconds()
            msg = '\n  TEST TIMING -- "{}" took : {:12.6f} sec.'
            name = _method_path(fn, cls)
            print(msg.format(name, elapsed_time))
        return result

    return inner


def iristest_timing_decorator(cls):
    # Class decorator to make all "test_.." functions print execution timings.
    if _PRINT_TEST_TIMINGS:
        # NOTE: 'dir' scans *all* class properties, including inherited ones.
        attr_names = dir(cls)
        for attr_name in attr_names:
            attr = getattr(cls, attr_name)
            if callable(attr) and attr_name.startswith("test"):
                attr = _testfunction_timing_decorator(attr, cls)
                setattr(cls, attr_name, attr)
    return cls


class _TestTimingsMetaclass(type):
    # An alternative metaclass for IrisTest subclasses, which makes
    # them print execution timings for all the testcases.
    # This is equivalent to applying the @iristest_timing_decorator to
    # every test class that inherits from IrisTest.
    # NOTE: however, it means you *cannot* specify a different metaclass for
    # your test class inheriting from IrisTest.
    # See below for how to solve that where needed.
    def __new__(cls, clsname, base_classes, attrs):
        result = type.__new__(cls, clsname, base_classes, attrs)
        if _PRINT_TEST_TIMINGS:
            result = iristest_timing_decorator(result)
        return result


class IrisTest(IrisTest_nometa, metaclass=_TestTimingsMetaclass):
    # Derive the 'ordinary' IrisTest from IrisTest_nometa, but add the
    # metaclass that enables test timings output.
    # This means that all subclasses also get the timing behaviour.
    # However, if a different metaclass is *wanted* for an IrisTest subclass,
    # this would cause a metaclass conflict.
    # Instead, you can inherit from IrisTest_nometa and apply the
    # @iristest_timing_decorator explicitly to your new testclass.
    pass


get_result_path = IrisTest.get_result_path


class GraphicsTestMixin:

    # nose directive: dispatch tests concurrently.
    _multiprocess_can_split_ = True

    def setUp(self):
        # Acquire threading non re-entrant blocking lock to ensure
        # thread-safe plotting.
        _lock.acquire()
        # Make sure we have no unclosed plots from previous tests before
        # generating this one.
        if MPL_AVAILABLE:
            plt.close("all")

    def tearDown(self):
        # If a plotting test bombs out it can leave the current figure
        # in an odd state, so we make sure it's been disposed of.
        if MPL_AVAILABLE:
            plt.close("all")
        # Release the non re-entrant blocking lock.
        _lock.release()


class GraphicsTest(GraphicsTestMixin, IrisTest):
    pass


class GraphicsTest_nometa(GraphicsTestMixin, IrisTest_nometa):
    # Graphicstest without the metaclass providing test timings.
    pass


def skip_data(fn):
    """
    Decorator to choose whether to run tests, based on the availability of
    external data.

    Example usage:
        @skip_data
        class MyDataTests(tests.IrisTest):
            ...

    """
    no_data = (
        not iris.config.TEST_DATA_DIR
        or not os.path.isdir(iris.config.TEST_DATA_DIR)
        or os.environ.get("IRIS_TEST_NO_DATA")
    )

    skip = unittest.skipIf(
        condition=no_data, reason="Test(s) require external data."
    )

    return skip(fn)


def skip_gdal(fn):
    """
    Decorator to choose whether to run tests, based on the availability of the
    GDAL library.

    Example usage:
        @skip_gdal
        class MyGeoTiffTests(test.IrisTest):
            ...

    """
    skip = unittest.skipIf(
        condition=not GDAL_AVAILABLE, reason="Test requires 'gdal'."
    )
    return skip(fn)


def skip_plot(fn):
    """
    Decorator to choose whether to run tests, based on the availability of the
    matplotlib library.

    Example usage:
        @skip_plot
        class MyPlotTests(test.GraphicsTest):
            ...

    """
    skip = unittest.skipIf(
        condition=not MPL_AVAILABLE,
        reason="Graphics tests require the matplotlib library.",
    )

    return skip(fn)


skip_sample_data = unittest.skipIf(
    not SAMPLE_DATA_AVAILABLE,
    ('Test(s) require "iris-sample-data", ' "which is not available."),
)


skip_nc_time_axis = unittest.skipIf(
    not NC_TIME_AXIS_AVAILABLE,
    'Test(s) require "nc_time_axis", which is not available.',
)


skip_inet = unittest.skipIf(
    not INET_AVAILABLE,
    ('Test(s) require an "internet connection", ' "which is not available."),
)


skip_stratify = unittest.skipIf(
    not STRATIFY_AVAILABLE,
    'Test(s) require "python-stratify", which is not available.',
)


def no_warnings(func):
    """
    Provides a decorator to ensure that there are no warnings raised
    within the test, otherwise the test will fail.

    """

    @functools.wraps(func)
    def wrapped(self, *args, **kwargs):
        with mock.patch("warnings.warn") as warn:
            result = func(self, *args, **kwargs)
        self.assertEqual(
            0,
            warn.call_count,
            ("Got unexpected warnings." " \n{}".format(warn.call_args_list)),
        )
        return result

    return wrapped
