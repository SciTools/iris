# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Provides testing capabilities and customisations specific to Iris."""

import collections
from collections.abc import Mapping
import contextlib
import difflib
import filecmp
import functools
import gzip
import json
import math
import os
import os.path
from pathlib import Path
import re
import shutil
import subprocess
from typing import Optional
import warnings
import xml.dom.minidom
import zlib

import numpy as np
import numpy.ma as ma
import pytest
import requests

import iris.config
import iris.cube
import iris.fileformats
import iris.tests
import iris.tests.graphics as graphics
import iris.util

MPL_AVAILABLE = graphics.MPL_AVAILABLE


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


def _assert_masked_array(assertion, a, b, strict, **kwargs):
    # Compare masks.
    a_mask, b_mask = ma.getmaskarray(a), ma.getmaskarray(b)
    np.testing.assert_array_equal(a_mask, b_mask)  # pytest already?

    if strict:
        # Compare all data values.
        assertion(a.data, b.data, **kwargs)
    else:
        # Compare only unmasked data values.
        assertion(
            ma.compressed(a),
            ma.compressed(b),
            **kwargs,
        )


def assert_masked_array_equal(a, b, strict=False):
    """Check that masked arrays are equal. This requires the
    unmasked values and masks to be identical.

    Parameters
    ----------
    a, b : array-like
        Two arrays to compare.
    strict : bool, optional
        If True, perform a complete mask and data array equality check.
        If False (default), the data array equality considers only unmasked
        elements.

    """
    _assert_masked_array(np.testing.assert_array_equal, a, b, strict)


def assert_masked_array_almost_equal(a, b, decimal=6, strict=False):
    """Check that masked arrays are almost equal. This requires the
    masks to be identical, and the unmasked values to be almost
    equal.

    Parameters
    ----------
    a, b : array-like
        Two arrays to compare.
    strict : bool, optional
        If True, perform a complete mask and data array equality check.
        If False (default), the data array equality considers only unmasked
        elements.
    decimal : int, optional, default=6
        Equality tolerance level for
        :meth:`numpy.testing.assert_array_almost_equal`, with the meaning
        'abs(desired-actual) < 0.5 * 10**(-decimal)'

    """
    _assert_masked_array(
        np.testing.assert_array_almost_equal, a, b, strict, decimal=decimal
    )


def _assert_str_same(
    reference_str,
    test_str,
    reference_filename,
    type_comparison_name="Strings",
):
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
    fail_string = f"{type_comparison_name} do not match: {reference_filename}\n{diff}"
    assert reference_str == test_str, fail_string


def get_data_path(relative_path):
    """Return the absolute path to a data file when given the relative path
    as a string, or sequence of strings.

    """
    if not isinstance(relative_path, str):
        relative_path = os.path.join(*relative_path)
    test_data_dir = iris.config.TEST_DATA_DIR
    if test_data_dir is None:
        test_data_dir = ""
    data_path = os.path.join(test_data_dir, relative_path)

    if iris.tests._EXPORT_DATAPATHS_FILE is not None:
        iris.tests._EXPORT_DATAPATHS_FILE.write(data_path + "\n")

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


def get_result_path(relative_path):
    """Returns the absolute path to a result file when given the relative path
    as a string, or sequence of strings.

    """
    if not isinstance(relative_path, str):
        relative_path = os.path.join(*relative_path)
    return os.path.abspath(os.path.join(_RESULT_PATH, relative_path))


def _check_for_request_fixture(request, func_name: str):
    """Raise an error if the first argument is not a pytest.FixtureRequest.

    Written to provide the clearest possible message for devs refactoring from
    the deprecated IrisTest style tests.
    """
    if not hasattr(request, "fixturenames"):
        message = (
            f"{func_name}() expected: pytest.FixtureRequest instance, got: {request}"
        )
        raise ValueError(message)


def result_path(request: pytest.FixtureRequest, basename=None, ext=""):
    """Generate the path to a test result; from the calling file, class, method.

    Parameters
    ----------
    request : pytest.FixtureRequest
        A pytest ``request`` fixture passed down from the calling test. Is
        interpreted for the automatic generation of a result path. See Examples
        for how to access the ``request`` fixture.
    basename : optional, default=None
        File basename. If omitted, this is generated from the calling method.
    ext : str, optional, default=""
        Appended file extension.

    Examples
    --------
    The PyTest ``request`` fixture is always available as a test argument:

    >>> def test_one(request):
    ...     path_one = (result_path(request))

    """
    _check_for_request_fixture(request, "result_path")

    if __package__ != "iris.tests":
        # Relying on this being the location so that we can derive the full
        #  path of the tests root.
        # Would normally use assert, but this means something to PyTest.
        message = "result_path() must be in the iris.tests root to function."
        raise RuntimeError(message)
    tests_root = Path(__file__).parent

    if ext and not ext.startswith("."):
        ext = f".{ext}"

    def remove_test(string: str):
        result = string
        result = re.sub(r"(?i)test_", "", result)
        result = re.sub(r"(?i)test", "", result)
        return result

    # Generate the directory name from the calling file name.
    output_path = get_result_path("") / request.path.relative_to(tests_root)
    output_path = output_path.with_suffix("")
    output_path = output_path.with_name(remove_test(output_path.name))

    # Optionally add a class subdirectory if called from a class.
    if request.cls is not None:
        output_class = remove_test(request.cls.__name__)
        output_path = output_path / output_class

    # Generate the file name from the calling function name.
    node_name = request.node.originalname
    if basename is not None:
        output_func = basename
    elif node_name == "<module>":
        output_func = ""
    else:
        output_func = remove_test(node_name)
    output_path = output_path / output_func

    # Optionally use parameter values as the file name if parameterised.
    #  (The function becomes a subdirectory in this case).
    if hasattr(request.node, "callspec"):
        output_path = output_path / request.node.callspec.id

    output_path = output_path.with_suffix(ext)

    return str(output_path)


def assert_CML_approx_data(
    request: pytest.FixtureRequest, cubes, reference_filename=None, **kwargs
):
    # passes args and kwargs on to approx equal
    # See result_path() Examples for how to access the ``request`` fixture.

    _check_for_request_fixture(request, "assert_CML_approx_data")

    if isinstance(cubes, iris.cube.Cube):
        cubes = [cubes]
    if reference_filename is None:
        reference_filename = result_path(request, None, "cml")
        reference_filename = [get_result_path(reference_filename)]
    for i, cube in enumerate(cubes):
        fname = list(reference_filename)
        # don't want the ".cml" for the json stats file
        fname[-1] = fname[-1].removesuffix(".cml")
        fname[-1] += ".data.%d.json" % i
        assert_data_almost_equal(cube.data, fname, **kwargs)
    assert_CML(request, cubes, reference_filename, checksum=False)


def assert_CDL(
    request: pytest.FixtureRequest, netcdf_filename, reference_filename=None, flags="-h"
):
    """Test that the CDL for the given netCDF file matches the contents
    of the reference file.

    If the environment variable IRIS_TEST_CREATE_MISSING is
    non-empty, the reference file is created if it doesn't exist.

    Parameters
    ----------
    request : pytest.FixtureRequest
        A pytest ``request`` fixture passed down from the calling test. Is
        required by :func:`result_path`. See :func:`result_path` Examples
        for how to access the ``request`` fixture.
    netcdf_filename :
        The path to the netCDF file.
    reference_filename : optional, default=None
        The relative path (relative to the test results directory).
        If omitted, the result is generated from the calling
        method's name, class, and module using
        :meth:`iris.tests.IrisTest.result_path`.
    flags : str, optional
        Command-line flags for `ncdump`, as either a whitespace
        separated string or an iterable. Defaults to '-h'.

    """
    _check_for_request_fixture(request, "assert_CDL")

    if reference_filename is None:
        reference_path = result_path(request, None, "cdl")
    else:
        reference_path = get_result_path(reference_filename)

    # Convert the netCDF file to CDL file format.
    if flags is None:
        flags = []
    elif isinstance(flags, str):
        flags = flags.split()
    else:
        flags = list(map(str, flags))

    try:
        exe_path = env_bin_path("ncdump")
        args = [exe_path] + flags + [netcdf_filename]
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

    dimension_lines = slice(lines.index("dimensions:") + 1, lines.index("variables:"))
    lines[dimension_lines] = sorted(lines[dimension_lines], key=sort_key)
    cdl = "\n".join(lines) + "\n"  # type: ignore[assignment]

    _check_same(cdl, reference_path, type_comparison_name="CDL")


def assert_CML(
    request: pytest.FixtureRequest, cubes, reference_filename=None, checksum=True
):
    """Test that the CML for the given cubes matches the contents of
    the reference file.

    If the environment variable IRIS_TEST_CREATE_MISSING is
    non-empty, the reference file is created if it doesn't exist.

    Parameters
    ----------
    request : pytest.FixtureRequest
        A pytest ``request`` fixture passed down from the calling test. Is
        required by :func:`result_path`. See :func:`result_path` Examples
        for how to access the ``request`` fixture.
    cubes :
        Either a Cube or a sequence of Cubes.
    reference_filename : optional, default=None
        The relative path (relative to the test results directory).
        If omitted, the result is generated from the calling
        method's name, class, and module using
        :meth:`iris.tests.IrisTest.result_path`.
    checksum : bool, optional
        When True, causes the CML to include a checksum for each
        Cube's data. Defaults to True.

    """
    _check_for_request_fixture(request, "assert_CML")

    if isinstance(cubes, iris.cube.Cube):
        cubes = [cubes]
    if reference_filename is None:
        reference_filename = result_path(request, None, "cml")

    if isinstance(cubes, (list, tuple)):
        xml = iris.cube.CubeList(cubes).xml(
            checksum=checksum, order=False, byteorder=False
        )
    else:
        xml = cubes.xml(checksum=checksum, order=False, byteorder=False)
    reference_path = get_result_path(reference_filename)
    _check_same(xml, reference_path)


def assert_text_file(source_filename, reference_filename, desc="text file"):
    """Check if two text files are the same, printing any diffs."""
    with open(source_filename) as source_file:
        source_text = source_file.readlines()
    with open(reference_filename) as reference_file:
        reference_text = reference_file.readlines()

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
    fail_string = (
        f"{desc} does not match: reference file {reference_filename} \n {diff}"
    )
    assert reference_text == source_text, fail_string


def assert_data_almost_equal(data, reference_filename, **kwargs):
    reference_path = get_result_path(reference_filename)

    def fixed_std(data):
        # When data is constant, std() is too sensitive.
        if data.max() == data.min():
            data_std = 0
        else:
            data_std = data.std()
        return data_std

    if _check_reference_file(reference_path):
        kwargs.setdefault("err_msg", "Reference file %s" % reference_path)
        with open(reference_path, "r") as reference_file:
            stats = json.load(reference_file)
            assert stats.get("shape", []) == list(data.shape)
            assert stats.get("masked", False) == ma.is_masked(data)
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
                assert math.isnan(data.mean())
            else:
                data_stats = np.array(
                    (data.mean(), fixed_std(data), data.max(), data.min()),
                    dtype=np.float64,
                )
                assert_array_all_close(nstats, data_stats, **kwargs)
    else:
        _ensure_folder(reference_path)
        stats = collections.OrderedDict(
            [
                ("std", np.float64(fixed_std(data))),
                ("min", np.float64(data.min())),
                ("max", np.float64(data.max())),
                ("shape", data.shape),
                ("masked", ma.is_masked(data)),
                ("mean", np.float64(data.mean())),
            ]
        )
        with open(reference_path, "w") as reference_file:
            reference_file.write(json.dumps(stats))


def assert_files_equal(test_filename, reference_filename):
    reference_path = get_result_path(reference_filename)
    if _check_reference_file(reference_path):
        fmt = "test file {!r} does not match reference {!r}."
        assert filecmp.cmp(test_filename, reference_path) and fmt.format(
            test_filename, reference_path
        )
    else:
        _ensure_folder(reference_path)
        shutil.copy(test_filename, reference_path)


def assert_string(request: pytest.FixtureRequest, string, reference_filename=None):
    """Test that `string` matches the contents of the reference file.

    If the environment variable IRIS_TEST_CREATE_MISSING is
    non-empty, the reference file is created if it doesn't exist.

    Parameters
    ----------
    request: pytest.FixtureRequest
        A pytest ``request`` fixture passed down from the calling test. Is
        required by :func:`result_path`. See :func:`result_path` Examples
        for how to access the ``request`` fixture.
    string : str
        The string to check.
    reference_filename : optional, default=None
        The relative path (relative to the test results directory).
        If omitted, the result is generated from the calling
        method's name, class, and module using
        :meth:`iris.tests.IrisTest.result_path`.

    """
    _check_for_request_fixture(request, "assert_string")

    if reference_filename is None:
        reference_path = result_path(request, None, "txt")
    else:
        reference_path = get_result_path(reference_filename)
    _check_same(string, reference_path, type_comparison_name="Strings")


def assert_repr(request: pytest.FixtureRequest, obj, reference_filename):
    assert_string(request, repr(obj), reference_filename)


def _check_same(item, reference_path, type_comparison_name="CML"):
    if _check_reference_file(reference_path):
        with open(reference_path, "rb") as reference_fh:
            reference = "".join(part.decode("utf-8") for part in reference_fh)
        _assert_str_same(reference, item, reference_path, type_comparison_name)
    else:
        _ensure_folder(reference_path)
        with open(reference_path, "wb") as reference_fh:
            reference_fh.writelines(part.encode("utf-8") for part in item)


def assert_XML_element(obj, reference_filename):
    """Calls the xml_element method given obj and asserts the result is the same as the test file."""
    doc = xml.dom.minidom.Document()
    doc.appendChild(obj.xml_element(doc))
    # sort the attributes on xml elements before testing against known good state.
    # this is to be compatible with stored test output where xml attrs are stored in alphabetical order,
    # (which was default behaviour in python <3.8, but changed to insert order in >3.8)
    doc = iris.cube.Cube._sort_xml_attrs(doc)
    pretty_xml = iris.util._print_xml(doc)
    reference_path = get_result_path(reference_filename)
    _check_same(pretty_xml, reference_path, type_comparison_name="XML")


def assert_array_equal(a, b, err_msg=""):
    np.testing.assert_array_equal(a, b, err_msg=err_msg)


@contextlib.contextmanager
def _record_warning_matches(expected_regexp=""):
    # Record warnings raised matching a given expression.
    matches = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        yield matches
    messages = [str(warning.message) for warning in w]
    expr = re.compile(expected_regexp)
    matches.extend(message for message in messages if expr.search(message))


@contextlib.contextmanager
def assert_logs(caplog, logger=None, level=None, msg_regex=None):
    """If msg_regex is used, checks that the result is a single message of the specified
    level, and that it matches this regex.

    Checks that there is at least one message logged at the given parameters,
    but then *also* exercises the message formatters of all the logger's handlers,
    just to check that there are no formatting errors.

    """
    with caplog.at_level(level, logger.name):
        assert len(caplog.records) != 0
        # Check for any formatting errors by running all the formatters.
        for record in caplog.records:
            for handler in caplog.logger.handlers:
                handler.format(record)

        # Check message, if requested.
        if msg_regex:
            assert len(caplog.records) == 1
            rec = caplog.records[0]
            assert level == rec.levelname
            assert re.match(msg_regex, rec.msg)


@contextlib.contextmanager
def assert_no_warnings_regexp(expected_regexp=""):
    # Check that no warning matching the given expression is raised.
    with _record_warning_matches(expected_regexp) as matches:
        yield

    msg = "Unexpected warning(s) raised, matching '{}' : {!r}."
    msg = msg.format(expected_regexp, matches)
    assert not matches, msg


def assert_array_almost_equal(a, b, decimal=6):
    np.testing.assert_array_almost_equal(a, b, decimal=decimal)


def assert_array_all_close(a, b, rtol=1.0e-7, atol=1.0e-8, **kwargs):
    """Check arrays are equal, within given relative + absolute tolerances.

    Parameters
    ----------
    a, b : array-like
        Two arrays to compare.
    rtol, atol : float, optional
        Relative and absolute tolerances to apply.

    Other Parameters
    ----------------
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
                '\nARRAY CHECK FAILED "assert_array_all_close" :'
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


def file_checksum(file_path):
    """Generate checksum from file."""
    with open(file_path, "rb") as in_file:
        return zlib.crc32(in_file.read())


def _check_reference_file(reference_path):
    reference_exists = os.path.isfile(reference_path)
    if not (reference_exists or os.environ.get("IRIS_TEST_CREATE_MISSING")):
        msg = "Missing test result: {}".format(reference_path)
        raise AssertionError(msg)
    return reference_exists


def _ensure_folder(path):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# todo: relied on unitest functionality, need to find a pytest alternative
def patch(*args, **kwargs):
    """Install a mock.patch, to be removed after the current test.

    The patch is created with mock.patch(*args, **kwargs).

    Returns
    -------
    The substitute object returned by patch.start().

    Examples
    --------
    ::

        mock_call = self.patch('module.Class.call', return_value=1)
        module_Class_instance.call(3, 4)
        self.assertEqual(mock_call.call_args_list, [mock.call(3, 4)])

    """
    raise NotImplementedError()


def assert_array_shape_stats(result, shape, mean, std_dev, rtol=1e-6):
    """Assert that the result, a cube, has the provided shape and that the
    mean and standard deviation of the data array are also as provided.
    Thus build confidence that a cube processing operation, such as a
    cube.regrid, has maintained its behaviour.

    """
    assert result.shape == shape
    assert_array_all_close(result.data.mean(), mean, rtol=rtol)
    assert_array_all_close(result.data.std(), std_dev, rtol=rtol)


def assert_dict_equal(lhs, rhs):
    """Dictionary Comparison.

    This allows us to cope with dictionary comparison where the value of a key
    may be a numpy array.
    """
    emsg = f"Provided LHS argument is not a 'Mapping', got {type(lhs)}."
    assert isinstance(lhs, Mapping), emsg

    emsg = f"Provided RHS argument is not a 'Mapping', got {type(rhs)}."
    assert isinstance(rhs, Mapping), emsg

    emsg = f"{lhs!r} != {rhs!r}."
    assert set(lhs.keys()) == set(rhs.keys()), emsg

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

            assert_masked_array_equal(lvalue, rvalue)
        elif isinstance(lvalue, np.ndarray) or isinstance(rvalue, np.ndarray):
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

            assert_array_equal(lvalue, rvalue)
        else:
            if lvalue != rvalue:
                emsg = (
                    f"Dictionary key {key!r} values are not equal, "
                    f"{lvalue!r} != {rvalue!r}."
                )
                raise AssertionError(emsg)


def assert_equal_and_kind(value, expected):
    # Check a value, and also its type 'kind' = float/integer/string.
    assert value == expected
    assert np.array(value).dtype.kind == np.array(expected).dtype.kind


@contextlib.contextmanager
def pp_cube_save_test(
    reference_txt_path,
    reference_cubes=None,
    reference_pp_path=None,
    **kwargs,
):
    """A context manager for testing the saving of Cubes to PP files.

    Args:

    * reference_txt_path:
        The path of the file containing the textual PP reference data.

    Kwargs:

    * reference_cubes:
        The cube(s) from which the textual PP reference can be re-built if necessary.
    * reference_pp_path:
        The location of a PP file from which the textual PP reference can be re-built if necessary.
        NB. The "reference_cubes" argument takes precedence over this argument.

    The return value from the context manager is the name of a temporary file
    into which the PP data to be tested should be saved.

    Example::
        with self.pp_cube_save_test(reference_txt_path, reference_cubes=cubes) as temp_pp_path:
            iris.save(cubes, temp_pp_path)

    """

    def _create_reference_txt(txt_path, pp_path):
        # Load the reference data
        pp_fields = list(iris.fileformats.pp.load(pp_path))
        for pp_field in pp_fields:
            pp_field.data

        # Clear any header words we don't use
        unused = ("lbexp", "lbegin", "lbnrec", "lbproj", "lbtyp")
        for pp_field in pp_fields:
            for word_name in unused:
                setattr(pp_field, word_name, 0)

        # Save the textual representation of the PP fields
        with open(txt_path, "w") as txt_file:
            txt_file.writelines(str(pp_fields))

    # Watch out for a missing reference text file
    if not os.path.isfile(reference_txt_path):
        if reference_cubes:
            temp_pp_path = iris.util.create_temp_filename(".pp")
            try:
                iris.save(reference_cubes, temp_pp_path, **kwargs)
                _create_reference_txt(reference_txt_path, temp_pp_path)
            finally:
                os.remove(temp_pp_path)
        elif reference_pp_path:
            _create_reference_txt(reference_txt_path, reference_pp_path)
        else:
            raise ValueError("Missing all of reference txt file, cubes, and PP path.")

    temp_pp_path = iris.util.create_temp_filename(".pp")
    try:
        # This value is returned to the target of the "with" statement's "as" clause.
        yield temp_pp_path

        # Load deferred data for all of the fields (but don't do anything with it)
        pp_fields = list(iris.fileformats.pp.load(temp_pp_path))
        for pp_field in pp_fields:
            pp_field.data
        with open(reference_txt_path, "r") as reference_fh:
            reference = "".join(reference_fh)
        _assert_str_same(
            reference + "\n",
            str(pp_fields) + "\n",
            reference_txt_path,
            type_comparison_name="PP files",
        )
    finally:
        os.remove(temp_pp_path)


def skip_data(fn):
    """Decorator to choose whether to run tests, based on the availability of
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

    skip = pytest.mark.skipif(
        condition=no_data, reason="Test(s) require external data."
    )

    return skip(fn)


def skip_gdal(fn):
    """Decorator to choose whether to run tests, based on the availability of the
    GDAL library.

    Example usage:
        @skip_gdal
        class MyGeoTiffTests(test.IrisTest):
            ...

    """
    skip = pytest.mark.skipif(
        condition=not GDAL_AVAILABLE, reason="Test requires 'gdal'."
    )
    return skip(fn)


skip_plot = graphics.skip_plot

skip_sample_data = pytest.mark.skipif(
    not SAMPLE_DATA_AVAILABLE,
    reason=('Test(s) require "iris-sample-data", which is not available.'),
)


skip_nc_time_axis = pytest.mark.skipif(
    not NC_TIME_AXIS_AVAILABLE,
    reason='Test(s) require "nc_time_axis", which is not available.',
)


skip_inet = pytest.mark.skipif(
    not INET_AVAILABLE,
    reason=('Test(s) require an "internet connection", which is not available.'),
)


skip_stratify = pytest.mark.skipif(
    not STRATIFY_AVAILABLE,
    reason='Test(s) require "python-stratify", which is not available.',
)


def no_warnings(func):
    """Provides a decorator to ensure that there are no warnings raised
    within the test, otherwise the test will fail.

    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with pytest.mock.patch("warnings.warn") as warn:
            result = func(*args, **kwargs)
        assert 0 == warn.call_count, "Got unexpected warnings.\n{}".format(
            warn.call_args_list
        )
        return result

    return wrapped


def env_bin_path(exe_name: Optional[str] = None):
    """Return a Path object for (an executable in) the environment bin directory.

    Parameters
    ----------
    exe_name : str
        If set, the name of an executable to append to the path.

    Returns
    -------
    exe_path : Path
        A path to the bin directory, or an executable file within it.

    Notes
    -----
    For use in tests which spawn commands which should call executables within
    the Python environment, since many IDEs (Eclipse, PyCharm) don't
    automatically include this location in $PATH (as opposed to $PYTHONPATH).
    """
    exe_path = Path(os.__file__)
    exe_path = (exe_path / "../../../bin").resolve()
    if exe_name is not None:
        exe_path = exe_path / exe_name
    return exe_path


class GraphicsTest:
    """All inheriting classes automatically have access to ``self.check_graphic()``."""

    @pytest.fixture(autouse=True)
    def _get_check_graphics(self, check_graphic_caller):
        self.check_graphic = check_graphic_caller
