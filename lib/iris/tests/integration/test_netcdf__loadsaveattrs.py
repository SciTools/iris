# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Integration tests for loading and saving netcdf file attributes.

Notes:
(1) attributes in netCDF files can be either "global attributes", or variable
("local") type.

(2) in CF terms, this testcode classifies specific attributes (names) as either
"global" = names recognised by convention as normally stored in a file-global
setting; "local"  = recognised names specifying details of variable data
encoding, which only make sense as a "local" attribute (i.e. on a variable),
and "user" = any additional attributes *not* recognised in conventions, which
might be recorded either globally or locally.

"""
import inspect
import re
from typing import Iterable, List, Optional, Union
import warnings

import numpy as np
import pytest

import iris
import iris.coord_systems
from iris.coords import DimCoord
from iris.cube import Cube
import iris.fileformats.netcdf
import iris.fileformats.netcdf._thread_safe_nc as threadsafe_nc4

# First define the known controlled attribute names defined by netCDf and CF conventions
#
# Note: certain attributes are "normally" global (e.g. "Conventions"), whilst others
# will only usually appear on a data-variable (e.g. "scale_factor"", "coordinates").
# I'm calling these 'global-style' and 'local-style'.
# Any attributes either belongs to one of these 2 groups, or neither.  Those 3 distinct
# types may then have different behaviour in Iris load + save.

# A list of "global-style" attribute names : those which should be global attributes by
# default (i.e. file- or group-level, *not* attached to a variable).

_GLOBAL_TEST_ATTRS = set(iris.fileformats.netcdf.saver._CF_GLOBAL_ATTRS)
# Remove this one, which has peculiar behaviour + is tested separately
# N.B. this is not the same as 'Conventions', but is caught in the crossfire when that
# one is processed.
_GLOBAL_TEST_ATTRS -= set(["conventions"])


# Define a fixture to parametrise tests over the 'global-style' test attributes.
# This just provides a more concise way of writing parametrised tests.
@pytest.fixture(params=sorted(_GLOBAL_TEST_ATTRS))
def global_attr(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


# A list of "local-style" attribute names : those which should be variable attributes
# by default (aka "local", "variable" or "data" attributes) .
_LOCAL_TEST_ATTRS = (
    iris.fileformats.netcdf.saver._CF_DATA_ATTRS
    + iris.fileformats.netcdf.saver._UKMO_DATA_ATTRS
)


# Define a fixture to parametrise over the 'local-style' test attributes.
# This just provides a more concise way of writing parametrised tests.
@pytest.fixture(params=sorted(_LOCAL_TEST_ATTRS))
def local_attr(request):
    # N.B. "request" is a standard PyTest fixture
    return request.param  # Return the name of the attribute to test.


def check_captured_warnings(
    expected_keys: List[str], captured_warnings: List[warnings.WarningMessage]
):
    """
    Compare captured warning messages with a list of regexp-matches.

    We allow them to occur in any order, and replace each actual result in the list
    with its matching regexp, if any, as this makes failure results much easier to
    comprehend.

    """
    if expected_keys is None:
        expected_keys = []
    elif hasattr(expected_keys, "upper"):
        # Handle a single string
        expected_keys = [expected_keys]
    expected_keys = [re.compile(key) for key in expected_keys]
    found_results = [str(warning.message) for warning in captured_warnings]
    remaining_keys = expected_keys.copy()
    for i_message, message in enumerate(found_results.copy()):
        for key in remaining_keys:
            if key.search(message):
                # Hit : replace one message in the list with its matching "key"
                found_results[i_message] = key
                # remove the matching key
                remaining_keys.remove(key)
                # skip on to next message
                break

    assert set(found_results) == set(expected_keys)


class MixinAttrsTesting:
    @staticmethod
    def _calling_testname():
        """
        Search up the callstack for a function named "test_*", and return the name for
        use as a test identifier.

        Idea borrowed from :meth:`iris.tests.IrisTest.result_path`.

        Returns
        -------
        test_name : str
            Returns a string, with the initial "test_" removed.
        """
        test_name = None
        stack = inspect.stack()
        for frame in stack[1:]:
            full_name = frame[3]
            if full_name.startswith("test_"):
                # Return the name with the initial "test_" removed.
                test_name = full_name.replace("test_", "")
                break
        # Search should not fail, unless we were called from an inappropriate place?
        assert test_name is not None
        return test_name

    @pytest.fixture(autouse=True)
    def make_tempdir(self, tmp_path_factory):
        """
        Automatically-run fixture to activate the 'tmp_path_factory' fixture on *every*
        test: Make a directory for temporary files, and record it on the test instance.

        N.B. "tmp_path_factory" is a standard PyTest fixture, which provides a dirpath
        *shared* by all tests.  This is a bit quicker and more debuggable than having a
        directory per-testcase.
        """
        # Store the temporary directory path on the test instance
        self.tmpdir = str(tmp_path_factory.getbasetemp())

    def _testfile_path(self, basename: str) -> str:
        # Make a filepath in the temporary directory, based on the name of the calling
        # test method, and the "self.attrname" it sets up.
        testname = self._calling_testname()
        # Turn that into a suitable temporary filename
        ext_name = getattr(self, "testname_extension", "")
        if ext_name:
            basename = basename + "_" + ext_name
        path_str = f"{self.tmpdir}/{self.__class__.__name__}__test_{testname}-{self.attrname}__{basename}.nc"
        return path_str

    @staticmethod
    def _default_vars_and_attrvalues(vars_and_attrvalues):
        # Simple default strategy : turn a simple value into {'var': value}
        if not isinstance(vars_and_attrvalues, dict):
            # Treat single non-dict argument as a value for a single variable
            vars_and_attrvalues = {"var": vars_and_attrvalues}
        return vars_and_attrvalues

    def create_testcase_files_or_cubes(
        self,
        attr_name: str,
        global_value_file1: Optional[str] = None,
        var_values_file1: Union[None, str, dict] = None,
        global_value_file2: Optional[str] = None,
        var_values_file2: Union[None, str, dict] = None,
        cubes: bool = False,
    ):
        """
        Create temporary input netcdf files, or cubes, with specific content.

        Creates a temporary netcdf test file (or two) with the given global and
        variable-local attributes.  Or build cubes, similarly.
        If ``cubes`` is ``True``, save cubes in ``self.input_cubes``.
        Else save filepaths in ``self.input_filepaths``.

        Note: 'var_values_file<X>' args are dictionaries.  The named variables are
        created, with an attribute = the dictionary value, *except* that a dictionary
        value of None means that a local attribute is _not_ created on the variable.
        """
        # save attribute on the instance
        self.attrname = attr_name

        if not cubes:
            # Make some input file paths.
            filepath1 = self._testfile_path("testfile")
            filepath2 = self._testfile_path("testfile2")

        def make_file(
            filepath: str, global_value=None, var_values=None
        ) -> str:
            ds = threadsafe_nc4.DatasetWrapper(filepath, "w")
            if global_value is not None:
                ds.setncattr(attr_name, global_value)
            ds.createDimension("x", 3)
            # Rationalise the per-variable requirements
            # N.B. this *always* makes at least one variable, as otherwise we would
            # load no cubes.
            var_values = self._default_vars_and_attrvalues(var_values)
            for var_name, value in var_values.items():
                v = ds.createVariable(var_name, int, ("x",))
                if value is not None:
                    v.setncattr(attr_name, value)
            ds.close()
            return filepath

        def make_cubes(var_name, global_value=None, var_values=None):
            cubes = []
            var_values = self._default_vars_and_attrvalues(var_values)
            for varname, local_value in var_values.items():
                cube = Cube(np.arange(3.0), var_name=var_name)
                cubes.append(cube)
                dimco = DimCoord(np.arange(3.0), var_name="x")
                cube.add_dim_coord(dimco, 0)
                if global_value is not None:
                    cube.attributes.globals[attr_name] = global_value
                if local_value is not None:
                    cube.attributes.locals[attr_name] = local_value
            return cubes

        if cubes:
            results = make_cubes("v1", global_value_file1, var_values_file1)
            if global_value_file2 is not None or var_values_file2 is not None:
                results.extend(
                    make_cubes("v2", global_value_file2, var_values_file2)
                )
        else:
            results = [
                make_file(filepath1, global_value_file1, var_values_file1)
            ]
            if global_value_file2 is not None or var_values_file2 is not None:
                # Make a second testfile and add it to files-to-be-loaded.
                results.append(
                    make_file(filepath2, global_value_file2, var_values_file2)
                )

        # Save results on the instance
        if cubes:
            self.input_cubes = results
        else:
            self.input_filepaths = results
        return results

    def run_testcase(
        self,
        attr_name: str,
        values: Union[List, List[List]],
        create_cubes_or_files: str = "files",
    ) -> None:
        """
        Create testcase inputs (files or cubes) with specified attributes.

        Parameters
        ----------
        attr_name : str
            name for all attributes created in this testcase.
            Also saved as ``self.attrname``, as used by ``fetch_results``.
        values : list
            list, or lists, of values for created attributes, each containing one global
            and one-or-more local attribute values as [global, local1, local2...]
        create_cubes_or_files : str, default "files"
            create either cubes or testfiles.

        If ``create_cubes_or_files`` == "files", create one temporary netCDF file per
        values-list, and record in ``self.input_filepaths``.
        Else if ``create_cubes_or_files`` == "cubes", create sets of cubes with common
        global values and store all of them to ``self.input_cubes``.

        """
        # Save common attribute-name on the instance
        self.attrname = attr_name

        # Standardise input to a list-of-lists, each inner list = [global, *locals]
        assert isinstance(values, list)
        if not isinstance(values[0], list):
            values = [values]
        assert len(values) in (1, 2)
        assert len(values[0]) > 1

        # Decode into global1, *locals1, and optionally global2, *locals2
        global1 = values[0][0]
        vars1 = {}
        i_var = 0
        for value in values[0][1:]:
            vars1[f"var_{i_var}"] = value
            i_var += 1
        if len(values) == 1:
            global2 = None
            vars2 = None
        else:
            assert len(values) == 2
            global2 = values[1][0]
            vars2 = {}
            for value in values[1][1:]:
                vars2[f"var_{i_var}"] = value
                i_var += 1

        # Create test files or cubes (and store data on the instance)
        assert create_cubes_or_files in ("cubes", "files")
        make_cubes = create_cubes_or_files == "cubes"
        self.create_testcase_files_or_cubes(
            attr_name=attr_name,
            global_value_file1=global1,
            var_values_file1=vars1,
            global_value_file2=global2,
            var_values_file2=vars2,
            cubes=make_cubes,
        )

    def fetch_results(
        self,
        filepath: str = None,
        cubes: Iterable[Cube] = None,
        oldstyle_combined: bool = False,
    ):
        """
        Return testcase results from an output file or cubes in a standardised form.

        Unpick the global+local values of the attribute ``self.attrname``, resulting
        from a test operation.
        A file result is always [global_value, *local_values]
        A cubes result is [*[global_value, *local_values]] (over different global vals)

        When ``oldstyle_combined`` is ``True``, simulate the "legacy" style results,
        that is when each cube had a single combined attribute dictionary.
        This enables us to check against former behaviour, by combining results into a
        single dictionary.  N.B. per-cube single results are then returned in the form:
        [None, cube1, cube2...].
        N.B. if results are from a *file*, this key has **no effect**.

        """
        attr_name = self.attrname
        if filepath is not None:
            # Fetch global and local values from a file
            try:
                ds = threadsafe_nc4.DatasetWrapper(filepath)
                global_result = (
                    ds.getncattr(attr_name)
                    if attr_name in ds.ncattrs()
                    else None
                )
                # Fetch local attr value from all data variables :  In our testcases,
                # that is all *except* dimcoords (ones named after dimensions).
                local_vars_results = [
                    (
                        var.name,
                        (
                            var.getncattr(attr_name)
                            if attr_name in var.ncattrs()
                            else None
                        ),
                    )
                    for var in ds.variables.values()
                    if var.name not in ds.dimensions
                ]
            finally:
                ds.close()
            # This version always returns a single result set [global, local1[, local2]]
            # Return global, plus locals sorted by varname
            local_vars_results = sorted(local_vars_results, key=lambda x: x[0])
            results = [global_result] + [val for _, val in local_vars_results]
        else:
            assert cubes is not None
            # Sort result cubes according to a standard ordering.
            cubes = sorted(cubes, key=lambda cube: cube.name())
            # Fetch globals and locals from cubes.
            if oldstyle_combined:
                # Replace cubes attributes with all-combined dictionaries
                cubes = [cube.copy() for cube in cubes]
                for cube in cubes:
                    combined = dict(cube.attributes)
                    cube.attributes.clear()
                    cube.attributes.locals = combined
            global_values = set(
                cube.attributes.globals.get(attr_name, None) for cube in cubes
            )
            # This way returns *multiple* result 'sets', one for each global value
            results = [
                [globalval]
                + [
                    cube.attributes.locals.get(attr_name, None)
                    for cube in cubes
                    if cube.attributes.globals.get(attr_name, None)
                    == globalval
                ]
                for globalval in sorted(global_values)
            ]
        return results


class TestRoundtrip(MixinAttrsTesting):
    """
    Test handling of attributes in roundtrip netcdf-iris-netcdf.

    This behaviour should be (almost) unchanged by the adoption of
    split-attribute handling.

    NOTE: the tested combinations in the 'TestLoad' test all match tests here, but not
    *all* of the tests here are useful there.  To avoid confusion (!) the ones which are
    paralleled in TestLoad there have the identical test-names.  However, as the tests
    are all numbered that means there are missing numbers there.
    The tests are numbered only so it is easier to review the discovered test list
    (which is sorted).

    """

    # Parametrise all tests over split/unsplit saving.
    @pytest.fixture(
        params=[False, True], ids=["nosplit", "split"], autouse=True
    )
    def do_split(self, request):
        do_split = request.param
        self.save_split_attrs = do_split
        return do_split

    def run_roundtrip_testcase(self, attr_name, values):
        """
        Initialise the testcase from the passed-in controls, configure the input
        files and run a save-load roundtrip to produce the output file.

        The name of the attribute, and the input and output temporary filepaths are
        stored on the instance, where "self.check_roundtrip_results()" can get them.

        """
        self.run_testcase(
            attr_name=attr_name, values=values, create_cubes_or_files="files"
        )
        self.result_filepath = self._testfile_path("result")

        with warnings.catch_warnings(record=True) as captured_warnings:
            # Do a load+save to produce a testable output result in a new file.
            cubes = iris.load(self.input_filepaths)
            # Ensure stable result order.
            cubes = sorted(cubes, key=lambda cube: cube.name())
            do_split = getattr(self, "save_split_attrs", False)
            with iris.FUTURE.context(save_split_attrs=do_split):
                iris.save(cubes, self.result_filepath)

        self.captured_warnings = captured_warnings

    def check_roundtrip_results(self, expected, expected_warnings=None):
        """
        Run checks on the generated output file.

        The counterpart to :meth:`run_roundtrip_testcase`, with similar arguments.
        Check existence (or not) of a global attribute, and a number of local
        (variable) attributes.
        Values of 'None' mean to check that the relevant global/local attribute does
        *not* exist.

        Also check the warnings captured during the testcase run.
        """
        # N.B. there is only ever one result-file, but it can contain various variables
        # which came from different input files.
        results = self.fetch_results(filepath=self.result_filepath)
        assert results == expected
        check_captured_warnings(expected_warnings, self.captured_warnings)

    #######################################################
    # Tests on "user-style" attributes.
    # This means any arbitrary attribute which a user might have added -- i.e. one with
    # a name which is *not* recognised in the netCDF or CF conventions.
    #

    def test_01_userstyle_single_global(self):
        self.run_roundtrip_testcase(
            attr_name="myname", values=["single-value", None]
        )
        # Default behaviour for a general global user-attribute.
        # It simply remains global.
        self.check_roundtrip_results(["single-value", None])

    def test_02_userstyle_single_local(self, do_split):
        # Default behaviour for a general local user-attribute.
        # It results in a "promoted" global attribute.
        self.run_roundtrip_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            values=[None, "single-value"],
        )
        if do_split:
            expected = [None, "single-value"]
        else:
            expected = ["single-value", None]
        self.check_roundtrip_results(expected)

    def test_03_userstyle_multiple_different(self, do_split):
        # Default behaviour for general user-attributes.
        # The global attribute is lost because there are local ones.
        self.run_roundtrip_testcase(
            attr_name="random",  # A generic "user" attribute with no special handling
            values=[
                ["common_global", "f1v1", "f1v2"],
                ["common_global", "x1", "x2"],
            ],
        )
        expected_result = ["common_global", "f1v1", "f1v2", "x1", "x2"]
        if not do_split:
            # in legacy mode, global is lost
            expected_result[0] = None
        # just check they are all there and distinct
        self.check_roundtrip_results(expected_result)

    def test_04_userstyle_matching_promoted(self, do_split):
        # matching local user-attributes are "promoted" to a global one.
        # (but not when saving split attributes)
        input_values = ["global_file1", "same-value", "same-value"]
        self.run_roundtrip_testcase(
            attr_name="random",
            values=input_values,
        )
        if do_split:
            expected = input_values
        else:
            expected = ["same-value", None, None]
        self.check_roundtrip_results(expected)

    def test_05_userstyle_matching_crossfile_promoted(self, do_split):
        # matching user-attributes are promoted, even across input files.
        # (but not when saving split attributes)
        self.run_roundtrip_testcase(
            attr_name="random",
            values=[
                ["global_file1", "same-value", "same-value"],
                [None, "same-value", "same-value"],
            ],
        )
        if do_split:
            # newstyle saves: locals are preserved, mismathced global is *lost*
            expected_result = [
                None,
                "same-value",
                "same-value",
                "same-value",
                "same-value",
            ]
            # warnings about the clash
            expected_warnings = [
                "Saving.* global attributes.* as local",
                'attributes.* of cube "var_0" were not saved',
                'attributes.* of cube "var_1" were not saved',
            ]
        else:
            # oldstyle saves: matching locals promoted, override original global
            expected_result = ["same-value", None, None, None, None]
            expected_warnings = None

        self.check_roundtrip_results(expected_result, expected_warnings)

    def test_06_userstyle_nonmatching_remainlocal(self, do_split):
        # Non-matching user attributes remain 'local' to the individual variables.
        input_values = ["global_file1", "value-1", "value-2"]
        if do_split:
            # originals are preserved
            expected_result = input_values
        else:
            # global is lost
            expected_result = [None, "value-1", "value-2"]
        self.run_roundtrip_testcase(attr_name="random", values=input_values)
        self.check_roundtrip_results(expected_result)

    #######################################################
    # Tests on "Conventions" attribute.
    # Note: the usual 'Conventions' behaviour is already tested elsewhere
    # - see :class:`TestConventionsAttributes` above
    #
    # TODO: the name 'conventions' (lower-case) is also listed in _CF_GLOBAL_ATTRS, but
    # we have excluded it from the global-attrs testing here.  We probably still need to
    # test what that does, though it's inclusion might simply be a mistake.
    #

    def test_07_conventions_var_local(self):
        # What happens if 'Conventions' appears as a variable-local attribute.
        # N.B. this is not good CF, but we'll see what happens anyway.
        self.run_roundtrip_testcase(
            attr_name="Conventions",
            values=[None, "user_set"],
        )
        self.check_roundtrip_results(["CF-1.7", None])

    def test_08_conventions_var_both(self):
        # What happens if 'Conventions' appears as both global + local attribute.
        self.run_roundtrip_testcase(
            attr_name="Conventions",
            values=["global-setting", "local-setting"],
        )
        # standard content from Iris save
        self.check_roundtrip_results(["CF-1.7", None])

    #######################################################
    # Tests on "global" style attributes
    #  = those specific ones which 'ought' only to be global (except on collisions)
    #
    def test_09_globalstyle__global(self, global_attr):
        attr_content = f"Global tracked {global_attr}"
        self.run_roundtrip_testcase(
            attr_name=global_attr,
            values=[attr_content, None],
        )
        self.check_roundtrip_results([attr_content, None])

    def test_10_globalstyle__local(self, global_attr, do_split):
        # Strictly, not correct CF, but let's see what it does with it.
        attr_content = f"Local tracked {global_attr}"
        input_values = [None, attr_content]
        self.run_roundtrip_testcase(
            attr_name=global_attr,
            values=input_values,
        )
        if do_split:
            # remains local as supplied, but there is a warning
            expected_result = input_values
            expected_warning = f"'{global_attr}'.* should only be a CF global"
        else:
            # promoted to global
            expected_result = [attr_content, None]
            expected_warning = None
        self.check_roundtrip_results(expected_result, expected_warning)

    def test_11_globalstyle__both(self, global_attr, do_split):
        attr_global = f"Global-{global_attr}"
        attr_local = f"Local-{global_attr}"
        input_values = [attr_global, attr_local]
        self.run_roundtrip_testcase(
            attr_name=global_attr,
            values=input_values,
        )
        if do_split:
            # remains local as supplied, but there is a warning
            expected_result = input_values
            expected_warning = "should only be a CF global"
        else:
            # promoted to global, no local value, original global lost
            expected_result = [attr_local, None]
            expected_warning = None
        self.check_roundtrip_results(expected_result, expected_warning)

    def test_12_globalstyle__multivar_different(self, global_attr):
        # Multiple *different* local settings are retained, not promoted
        attr_1 = f"Local-{global_attr}-1"
        attr_2 = f"Local-{global_attr}-2"
        expect_warning = "should only be a CF global attribute"
        # A warning should be raised when writing the result.
        self.run_roundtrip_testcase(
            attr_name=global_attr,
            values=[None, attr_1, attr_2],
        )
        self.check_roundtrip_results([None, attr_1, attr_2], expect_warning)

    def test_13_globalstyle__multivar_same(self, global_attr, do_split):
        # Multiple *same* local settings are promoted to a common global one
        attrval = f"Locally-defined-{global_attr}"
        input_values = [None, attrval, attrval]
        self.run_roundtrip_testcase(
            attr_name=global_attr,
            values=input_values,
        )
        if do_split:
            # remains local, but with a warning
            expected_warning = "should only be a CF global"
            expected_result = input_values
        else:
            # promoted to global
            expected_warning = None
            expected_result = [attrval, None, None]
        self.check_roundtrip_results(expected_result, expected_warning)

    def test_14_globalstyle__multifile_different(self, global_attr, do_split):
        # Different global attributes from multiple files are retained as local ones
        attr_1 = f"Global-{global_attr}-1"
        attr_2 = f"Global-{global_attr}-2"
        self.run_roundtrip_testcase(
            attr_name=global_attr,
            values=[[attr_1, None], [attr_2, None]],
        )
        # A warning should be raised when writing the result.
        expected_warnings = ["should only be a CF global attribute"]
        if do_split:
            # An extra warning, only when saving with split-attributes.
            expected_warnings = ["Saving.* as local"] + expected_warnings
        self.check_roundtrip_results([None, attr_1, attr_2], expected_warnings)

    def test_15_globalstyle__multifile_same(self, global_attr):
        # Matching global-type attributes in multiple files are retained as global
        attrval = f"Global-{global_attr}"
        self.run_roundtrip_testcase(
            attr_name=global_attr, values=[[attrval, None], [attrval, None]]
        )
        self.check_roundtrip_results([attrval, None, None])

    #######################################################
    # Tests on "local" style attributes
    #  = those specific ones which 'ought' to appear attached to a variable, rather than
    #  being global
    #

    @pytest.mark.parametrize("origin_style", ["input_global", "input_local"])
    def test_16_localstyle(self, local_attr, origin_style, do_split):
        # local-style attributes should *not* get 'promoted' to global ones
        # Set the name extension to avoid tests with different 'style' params having
        # collisions over identical testfile names
        self.testname_extension = origin_style

        attrval = f"Attr-setting-{local_attr}"
        if local_attr == "missing_value":
            # Special-cases : 'missing_value' type must be compatible with the variable
            attrval = 303
        elif local_attr == "ukmo__process_flags":
            # What this does when a GLOBAL attr seems to be weird + unintended.
            # 'this' --> 't h i s'
            attrval = "process"
            # NOTE: it's also supposed to handle vector values - which we are not
            # testing.

        # NOTE: results *should* be the same whether the original attribute is written
        # as global or a variable attribute
        if origin_style == "input_global":
            # Record in source as a global attribute
            values = [attrval, None]
        else:
            assert origin_style == "input_local"
            # Record in source as a variable-local attribute
            values = [None, attrval]
        self.run_roundtrip_testcase(attr_name=local_attr, values=values)

        if (
            local_attr in ("missing_value", "standard_error_multiplier")
            and origin_style == "input_local"
        ):
            # These ones are actually discarded by roundtrip.
            # Not clear why, but for now this captures the facts.
            expect_global = None
            expect_var = None
        else:
            expect_global = None
            if (
                local_attr == "ukmo__process_flags"
                and origin_style == "input_global"
                and not do_split
            ):
                # This is very odd behaviour + surely unintended.
                # It's supposed to handle vector values (which we are not checking).
                # But the weird behaviour only applies to the 'global' test, which is
                # obviously not normal usage anyway.
                attrval = "p r o c e s s"
            expect_var = attrval

        if local_attr == "STASH" and (
            origin_style == "input_local" or not do_split
        ):
            # A special case, output translates this to a different attribute name.
            self.attrname = "um_stash_source"

        expected_result = [expect_global, expect_var]
        if do_split and origin_style == "input_global":
            # The result is simply the "other way around"
            expected_result = expected_result[::-1]
        self.check_roundtrip_results(expected_result)


class TestLoad(MixinAttrsTesting):
    """
    Test loading of file attributes into Iris cube attribute dictionaries.

    Tests loading of various combinations to cube dictionaries, treated as a
    single combined result (i.e. not split).  This behaviour should be (almost)
    conserved with the adoption of  split attributes **except possibly for key
    orderings** -- i.e. we test only up to dictionary equality.

    NOTE: the tested combinations are identical to the roundtrip test.  Test numbering
    is kept the same, so some (which are inapplicable for this) are missing.

    """

    def run_load_testcase(self, attr_name, values):
        self.run_testcase(
            attr_name=attr_name, values=values, create_cubes_or_files="files"
        )

    def check_load_results(self, expected, oldstyle_combined=False):
        result_cubes = iris.load(self.input_filepaths)
        results = self.fetch_results(
            cubes=result_cubes, oldstyle_combined=oldstyle_combined
        )
        # Standardise expected form to list(lists).
        assert isinstance(expected, list)
        if not isinstance(expected[0], list):
            expected = [expected]
        assert results == expected

    #######################################################
    # Tests on "user-style" attributes.
    # This means any arbitrary attribute which a user might have added -- i.e. one with
    # a name which is *not* recognised in the netCDF or CF conventions.
    #

    def test_01_userstyle_single_global(self):
        self.run_load_testcase(
            attr_name="myname", values=["single_value", None, None]
        )
        # Legacy-equivalent result check (single attributes dict per cube)
        self.check_load_results(
            [None, "single_value", "single_value"],
            oldstyle_combined=True,
        )
        # Full new-style results check
        self.check_load_results(["single_value", None, None])

    def test_02_userstyle_single_local(self):
        # Default behaviour for a general local user-attribute.
        # It is attached to only the specific cube.
        self.run_load_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            values=[None, "single-value", None],
        )
        self.check_load_results(
            [None, "single-value", None], oldstyle_combined=True
        )
        self.check_load_results([None, "single-value", None])

    def test_03_userstyle_multiple_different(self):
        # Default behaviour for differing local user-attributes.
        # The global attribute is simply lost, because there are local ones.
        self.run_load_testcase(
            attr_name="random",  # A generic "user" attribute with no special handling
            values=[
                ["global_file1", "f1v1", "f1v2"],
                ["global_file2", "x1", "x2"],
            ],
        )
        self.check_load_results(
            [None, "f1v1", "f1v2", "x1", "x2"],
            oldstyle_combined=True,
        )
        self.check_load_results(
            [["global_file1", "f1v1", "f1v2"], ["global_file2", "x1", "x2"]]
        )

    def test_04_userstyle_multiple_same(self):
        # Nothing special to note in this case
        # TODO: ??remove??
        self.run_load_testcase(
            attr_name="random",
            values=["global_file1", "same-value", "same-value"],
        )
        self.check_load_results(
            oldstyle_combined=True, expected=[None, "same-value", "same-value"]
        )
        self.check_load_results(["global_file1", "same-value", "same-value"])

    #######################################################
    # Tests on "Conventions" attribute.
    # Note: the usual 'Conventions' behaviour is already tested elsewhere
    # - see :class:`TestConventionsAttributes` above
    #
    # TODO: the name 'conventions' (lower-case) is also listed in _CF_GLOBAL_ATTRS, but
    # we have excluded it from the global-attrs testing here.  We probably still need to
    # test what that does, though it's inclusion might simply be a mistake.
    #

    def test_07_conventions_var_local(self):
        # What happens if 'Conventions' appears as a variable-local attribute.
        # N.B. this is not good CF, but we'll see what happens anyway.
        self.run_load_testcase(
            attr_name="Conventions",
            values=[None, "user_set"],
        )
        # Legacy result
        self.check_load_results([None, "user_set"], oldstyle_combined=True)
        # Newstyle result
        self.check_load_results([None, "user_set"])

    def test_08_conventions_var_both(self):
        # What happens if 'Conventions' appears as both global + local attribute.
        self.run_load_testcase(
            attr_name="Conventions",
            values=["global-setting", "local-setting"],
        )
        # (#1): legacy result : the global version gets lost.
        self.check_load_results(
            [None, "local-setting"], oldstyle_combined=True
        )
        # (#2): newstyle results : retain both.
        self.check_load_results(["global-setting", "local-setting"])

    #######################################################
    # Tests on "global" style attributes
    #  = those specific ones which 'ought' only to be global (except on collisions)
    #

    def test_09_globalstyle__global(self, global_attr):
        attr_content = f"Global tracked {global_attr}"
        self.run_load_testcase(
            attr_name=global_attr, values=[attr_content, None]
        )
        # (#1) legacy
        self.check_load_results([None, attr_content], oldstyle_combined=True)
        # (#2) newstyle : global status preserved.
        self.check_load_results([attr_content, None])

    def test_10_globalstyle__local(self, global_attr):
        # Strictly, not correct CF, but let's see what it does with it.
        attr_content = f"Local tracked {global_attr}"
        self.run_load_testcase(
            attr_name=global_attr,
            values=[None, attr_content],
        )
        # (#1): legacy result = treated the same as a global setting
        self.check_load_results([None, attr_content], oldstyle_combined=True)
        # (#2): newstyle result : remains local
        self.check_load_results(
            [None, attr_content],
        )

    def test_11_globalstyle__both(self, global_attr):
        attr_global = f"Global-{global_attr}"
        attr_local = f"Local-{global_attr}"
        self.run_load_testcase(
            attr_name=global_attr,
            values=[attr_global, attr_local],
        )
        # (#1) legacy result : promoted local setting "wins"
        self.check_load_results([None, attr_local], oldstyle_combined=True)
        # (#2) newstyle result : both retained
        self.check_load_results([attr_global, attr_local])

    def test_12_globalstyle__multivar_different(self, global_attr):
        # Multiple *different* local settings are retained
        attr_1 = f"Local-{global_attr}-1"
        attr_2 = f"Local-{global_attr}-2"
        self.run_load_testcase(
            attr_name=global_attr,
            values=[None, attr_1, attr_2],
        )
        # (#1): legacy values, for cube.attributes viewed as a single dict
        self.check_load_results([None, attr_1, attr_2], oldstyle_combined=True)
        # (#2): exact results, with newstyle "split" cube attrs
        self.check_load_results([None, attr_1, attr_2])

    def test_14_globalstyle__multifile_different(self, global_attr):
        # Different global attributes from multiple files
        attr_1 = f"Global-{global_attr}-1"
        attr_2 = f"Global-{global_attr}-2"
        self.run_load_testcase(
            attr_name=global_attr,
            values=[[attr_1, None, None], [attr_2, None, None]],
        )
        # (#1) legacy : multiple globals retained as local ones
        self.check_load_results(
            [None, attr_1, attr_1, attr_2, attr_2], oldstyle_combined=True
        )
        # (#1) newstyle : result same as input
        self.check_load_results([[attr_1, None, None], [attr_2, None, None]])

    #######################################################
    # Tests on "local" style attributes
    #  = those specific ones which 'ought' to appear attached to a variable, rather than
    #  being global
    #

    @pytest.mark.parametrize("origin_style", ["input_global", "input_local"])
    def test_16_localstyle(self, local_attr, origin_style):
        # local-style attributes should *not* get 'promoted' to global ones
        # Set the name extension to avoid tests with different 'style' params having
        # collisions over identical testfile names
        self.testname_extension = origin_style

        attrval = f"Attr-setting-{local_attr}"
        if local_attr == "missing_value":
            # Special-case : 'missing_value' type must be compatible with the variable
            attrval = 303
        elif local_attr == "ukmo__process_flags":
            # Another special case : the handling of this one is "unusual".
            attrval = "process"

        # Create testfiles and load them, which should always produce a single cube.
        if origin_style == "input_global":
            # Record in source as a global attribute
            values = [attrval, None]
        else:
            assert origin_style == "input_local"
            # Record in source as a variable-local attribute
            values = [None, attrval]

        self.run_load_testcase(attr_name=local_attr, values=values)

        # Work out the expected result.
        result_value = attrval
        # ... there are some special cases
        if origin_style == "input_local":
            if local_attr == "ukmo__process_flags":
                # Some odd special behaviour here.
                result_value = (result_value,)
            elif local_attr in ("standard_error_multiplier", "missing_value"):
                # For some reason, these ones never appear on the cube
                result_value = None

        # NOTE: **legacy** result is the same, whether the original attribute was
        # provided as a global or local attribute ...
        expected_result_legacy = [None, result_value]

        # While 'newstyle' results preserve the input type local/global.
        if origin_style == "input_local":
            expected_result_newstyle = [None, result_value]
        else:
            expected_result_newstyle = [result_value, None]

        # (#1): legacy values, for cube.attributes viewed as a single dict
        self.check_load_results(expected_result_legacy, oldstyle_combined=True)
        # (#2): exact results, with newstyle "split" cube attrs
        self.check_load_results(expected_result_newstyle)


class TestSave(MixinAttrsTesting):
    """
    Test saving from cube attributes dictionary (various categories) into files.

    """

    # Parametrise all tests over split/unsplit saving.
    @pytest.fixture(
        params=[False, True], ids=["nosplit", "split"], autouse=True
    )
    def do_split(self, request):
        do_split = request.param
        self.save_split_attrs = do_split
        return do_split

    def run_save_testcase(self, attr_name: str, values: list):
        # Create input cubes.
        self.run_testcase(
            attr_name=attr_name,
            values=values,
            create_cubes_or_files="cubes",
        )

        # Save input cubes to a temporary result file.
        with warnings.catch_warnings(record=True) as captured_warnings:
            self.result_filepath = self._testfile_path("result")
            do_split = getattr(self, "save_split_attrs", False)
            with iris.FUTURE.context(save_split_attrs=do_split):
                iris.save(self.input_cubes, self.result_filepath)

        self.captured_warnings = captured_warnings

    def run_save_testcase_legacytype(self, attr_name: str, values: list):
        """
        Legacy-type means : before cubes had split attributes.

        This just means we have only one "set" of cubes, with ***no*** distinct global
        attribute.
        """
        if not isinstance(values, list):
            # Translate single input value to list-of-1
            values = [values]

        self.run_save_testcase(attr_name, [None] + values)

    def check_save_results(
        self, expected: list, expected_warnings: List[str] = None
    ):
        results = self.fetch_results(filepath=self.result_filepath)
        assert results == expected
        check_captured_warnings(expected_warnings, self.captured_warnings)

    def test_userstyle__single(self, do_split):
        self.run_save_testcase_legacytype("random", "value-x")
        if do_split:
            # result as input values
            expected_result = [None, "value-x"]
        else:
            # in legacy mode, promoted = stored as a *global* by default.
            expected_result = ["value-x", None]
        self.check_save_results(expected_result)

    def test_userstyle__multiple_same(self, do_split):
        self.run_save_testcase_legacytype("random", ["value-x", "value-x"])
        if do_split:
            # result as input values
            expected_result = [None, "value-x", "value-x"]
        else:
            # in legacy mode, promoted = stored as a *global* by default.
            expected_result = ["value-x", None, None]
        self.check_save_results(expected_result)

    def test_userstyle__multiple_different(self):
        # Clashing values are stored as locals on the individual variables.
        self.run_save_testcase_legacytype("random", ["value-A", "value-B"])
        self.check_save_results([None, "value-A", "value-B"])

    def test_userstyle__multiple_onemissing(self, global_attr):
        # Multiple user-type, with one missing, behave like different values.
        self.run_save_testcase_legacytype(
            global_attr,
            ["value", None],
        )
        # Stored as locals when there are differing values.
        self.check_save_results(
            [None, "value", None],
            expected_warnings="should only be a CF global attribute",
        )

    def test_Conventions__single(self):
        self.run_save_testcase_legacytype("Conventions", "x")
        # Always discarded + replaced by a single global setting.
        self.check_save_results(["CF-1.7", None])

    def test_Conventions__multiple_same(self):
        self.run_save_testcase_legacytype(
            "Conventions", ["same-value", "same-value"]
        )
        # Always discarded + replaced by a single global setting.
        self.check_save_results(["CF-1.7", None, None])

    def test_Conventions__multiple_different(self):
        self.run_save_testcase_legacytype(
            "Conventions", ["value-A", "value-B"]
        )
        # Always discarded + replaced by a single global setting.
        self.check_save_results(["CF-1.7", None, None])

    def test_globalstyle__single(self, global_attr, do_split):
        self.run_save_testcase_legacytype(global_attr, ["value"])
        if do_split:
            # result as input values
            expected_warning = "should only be a CF global"
            expected_result = [None, "value"]
        else:
            # in legacy mode, promoted
            expected_warning = None
            expected_result = ["value", None]
        self.check_save_results(expected_result, expected_warning)

    def test_globalstyle__multiple_same(self, global_attr, do_split):
        # Multiple global-type with same values are made global.
        self.run_save_testcase_legacytype(
            global_attr,
            ["value-same", "value-same"],
        )
        if do_split:
            # result as input values
            expected_result = [None, "value-same", "value-same"]
            expected_warning = "should only be a CF global attribute"
        else:
            # in legacy mode, promoted
            expected_result = ["value-same", None, None]
            expected_warning = None
        self.check_save_results(expected_result, expected_warning)

    def test_globalstyle__multiple_different(self, global_attr):
        # Multiple global-type with different values become local, with warning.
        self.run_save_testcase_legacytype(global_attr, ["value-A", "value-B"])
        # *Only* stored as locals when there are differing values.
        msg_regexp = (
            f"'{global_attr}' is being added as CF data variable attribute,"
            f".* should only be a CF global attribute."
        )
        self.check_save_results(
            [None, "value-A", "value-B"], expected_warnings=msg_regexp
        )

    def test_globalstyle__multiple_onemissing(self, global_attr):
        # Multiple global-type, with one missing, behave like different values.
        self.run_save_testcase_legacytype(
            global_attr, ["value", "value", None]
        )
        # Stored as locals when there are differing values.
        msg_regexp = (
            f"'{global_attr}' is being added as CF data variable attribute,"
            f".* should only be a CF global attribute."
        )
        self.check_save_results(
            [None, "value", "value", None], expected_warnings=msg_regexp
        )

    def test_localstyle__single(self, local_attr):
        self.run_save_testcase_legacytype(local_attr, ["value"])

        # Defaults to local
        expected_results = [None, "value"]
        # .. but a couple of special cases
        if local_attr == "ukmo__process_flags":
            # A particular, really weird case
            expected_results = [None, "v a l u e"]
        elif local_attr == "STASH":
            # A special case : the stored name is different
            self.attrname = "um_stash_source"

        self.check_save_results(expected_results)

    def test_localstyle__multiple_same(self, local_attr):
        self.run_save_testcase_legacytype(
            local_attr, ["value-same", "value-same"]
        )

        # They remain separate + local
        expected_results = [None, "value-same", "value-same"]
        if local_attr == "ukmo__process_flags":
            # A particular, really weird case
            expected_results = [
                None,
                "v a l u e - s a m e",
                "v a l u e - s a m e",
            ]
        elif local_attr == "STASH":
            # A special case : the stored name is different
            self.attrname = "um_stash_source"

        self.check_save_results(expected_results)

    def test_localstyle__multiple_different(self, local_attr):
        self.run_save_testcase_legacytype(local_attr, ["value-A", "value-B"])
        # Different values are treated just the same as matching ones.
        expected_results = [None, "value-A", "value-B"]
        if local_attr == "ukmo__process_flags":
            # A particular, really weird case
            expected_results = [
                None,
                "v a l u e - A",
                "v a l u e - B",
            ]
        elif local_attr == "STASH":
            # A special case : the stored name is different
            self.attrname = "um_stash_source"
        self.check_save_results(expected_results)

    #
    # Test handling of newstyle independent global+local cube attributes.
    #
    def test_globallocal_clashing(self, do_split):
        # A cube has clashing local + global attrs.
        original_values = ["valueA", "valueB"]
        self.run_save_testcase("userattr", original_values)
        expected_result = original_values.copy()
        if not do_split:
            # in legacy mode, "promote" = lose the local one
            expected_result[0] = expected_result[1]
            expected_result[1] = None
        self.check_save_results(expected_result)

    def test_globallocal_oneeach_same(self, do_split):
        # One cube with global attr, another with identical local one.
        self.run_save_testcase(
            "userattr", values=[[None, "value"], ["value", None]]
        )
        if do_split:
            expected = [None, "value", "value"]
            expected_warning = (
                "Saving the cube global attributes \\['userattr'\\] as local"
            )
        else:
            # N.B. legacy code sees only two equal values (and promotes).
            expected = ["value", None, None]
            expected_warning = None

        self.check_save_results(expected, expected_warning)

    def test_globallocal_oneeach_different(self, do_split):
        # One cube with global attr, another with a *different* local one.
        self.run_save_testcase(
            "userattr", [[None, "valueA"], ["valueB", None]]
        )
        if do_split:
            warning = (
                "Saving the cube global attributes \\['userattr'\\] as local"
            )
        else:
            # N.B. legacy code does not warn of global-to-local "demotion".
            warning = None
        self.check_save_results([None, "valueA", "valueB"], warning)

    def test_globallocal_one_other_clashingglobals(self, do_split):
        # Two cubes with both, second cube has a clashing global attribute.
        self.run_save_testcase(
            "userattr",
            values=[["valueA", "valueB"], ["valueXXX", "valueB"]],
        )
        if do_split:
            expected = [None, "valueB", "valueB"]
            expected_warnings = [
                "Saving.* global attributes.* as local",
                'attributes.* of cube "v1" were not saved',
                'attributes.* of cube "v2" were not saved',
            ]
        else:
            # N.B. legacy code sees only the locals, and promotes them.
            expected = ["valueB", None, None]
            expected_warnings = None
        self.check_save_results(expected, expected_warnings)

    def test_globallocal_one_other_clashinglocals(self, do_split):
        # Two cubes with both, second cube has a clashing local attribute.
        inputs = [["valueA", "valueB"], ["valueA", "valueXXX"]]
        if do_split:
            expected = ["valueA", "valueB", "valueXXX"]
        else:
            # N.B. legacy code sees only the locals.
            expected = [None, "valueB", "valueXXX"]
        self.run_save_testcase("userattr", values=inputs)
        self.check_save_results(expected)
