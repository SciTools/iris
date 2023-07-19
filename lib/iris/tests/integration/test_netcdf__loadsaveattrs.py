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
from typing import Iterable, Optional, Union

import pytest

import iris
import iris.coord_systems
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


class MixinAttrsTesting:
    @staticmethod
    def _calling_testname():
        """
        Search up the callstack for a function named "test_*", and return the name for
        use as a test identifier.

        Idea borrowed from :meth:`iris.tests.IrisTest_nometa.result_path`.

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

    def create_testcase_files(
        self,
        attr_name: str,
        global_value_file1: Optional[str] = None,
        var_values_file1: Union[None, str, dict] = None,
        global_value_file2: Optional[str] = None,
        var_values_file2: Union[None, str, dict] = None,
    ):
        """
        Create temporary input netcdf files with specific content.

        Creates a temporary netcdf test file (or two) with the given global and
        variable-local attributes.
        The file(s) are used to test the behaviour of the attribute.

        Note: 'var_values_file<X>' args are dictionaries.  The named variables are
        created, with an attribute = the dictionary value, *except* that a dictionary
        value of None means that a local attribute is _not_ created on the variable.
        """
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

        # Create one input file (always).
        filepaths = [
            make_file(
                filepath1,
                global_value=global_value_file1,
                var_values=var_values_file1,
            )
        ]
        if global_value_file2 is not None or var_values_file2 is not None:
            # Make a second testfile and add it to files-to-be-loaded.
            filepaths.append(
                make_file(
                    filepath2,
                    global_value=global_value_file2,
                    var_values=var_values_file2,
                ),
            )
        return filepaths


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

    def _roundtrip_load_and_save(
        self, input_filepaths: Union[str, Iterable[str]], output_filepath: str
    ) -> None:
        """
        Load netcdf input file(s) and re-write all to a given output file.
        """
        # Do a load+save to produce a testable output result in a new file.
        cubes = iris.load(input_filepaths)
        iris.save(cubes, output_filepath)

    def create_roundtrip_testcase(
        self,
        attr_name,
        global_value_file1=None,
        vars_values_file1=None,
        global_value_file2=None,
        vars_values_file2=None,
    ):
        """
        Initialise the testcase from the passed-in controls, configure the input
        files and run a save-load roundtrip to produce the output file.

        The name of the attribute, and the input and output temporary filepaths are
        stored on the instance, where "self.check_roundtrip_results()" can get them.

        """
        self.attrname = attr_name
        self.input_filepaths = self.create_testcase_files(
            attr_name=attr_name,
            global_value_file1=global_value_file1,
            var_values_file1=vars_values_file1,
            global_value_file2=global_value_file2,
            var_values_file2=vars_values_file2,
        )
        self.result_filepath = self._testfile_path("result")
        self._roundtrip_load_and_save(
            self.input_filepaths, self.result_filepath
        )

    def check_roundtrip_results(
        self, global_attr_value=None, var_attr_vals=None
    ):
        """
        Run checks on the generated output file.

        The counterpart to create_testcase, with similar control arguments.
        Check existence (or not) of : a global attribute, named variables, and their
        local attributes.  Values of 'None' mean to check that the relevant global/local
        attribute does *not* exist.
        """
        # N.B. there is only ever one result-file, but it can contain various variables
        # which came from different input files.
        ds = threadsafe_nc4.DatasetWrapper(self.result_filepath)
        if global_attr_value is None:
            assert self.attrname not in ds.ncattrs()
        else:
            assert self.attrname in ds.ncattrs()
            assert ds.getncattr(self.attrname) == global_attr_value
        if var_attr_vals:
            var_attr_vals = self._default_vars_and_attrvalues(var_attr_vals)
            for var_name, value in var_attr_vals.items():
                assert var_name in ds.variables
                v = ds.variables[var_name]
                if value is None:
                    assert self.attrname not in v.ncattrs()
                else:
                    assert self.attrname in v.ncattrs()
                    assert v.getncattr(self.attrname) == value

    #######################################################
    # Tests on "user-style" attributes.
    # This means any arbitrary attribute which a user might have added -- i.e. one with
    # a name which is *not* recognised in the netCDF or CF conventions.
    #

    def test_01_userstyle_single_global(self):
        self.create_roundtrip_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            global_value_file1="single-value",
            vars_values_file1={
                "myvar": None
            },  # the variable has no such attribute
        )
        # Default behaviour for a general global user-attribute.
        # It simply remains global.
        self.check_roundtrip_results(
            global_attr_value="single-value",  # local values eclipse the global ones
            var_attr_vals={
                "myvar": None
            },  # the variable has no such attribute
        )

    def test_02_userstyle_single_local(self):
        # Default behaviour for a general local user-attribute.
        # It results in a "promoted" global attribute.
        self.create_roundtrip_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            vars_values_file1={"myvar": "single-value"},
        )
        self.check_roundtrip_results(
            global_attr_value="single-value",  # local values eclipse the global ones
            # N.B. the output var has NO such attribute
        )

    def test_03_userstyle_multiple_different(self):
        # Default behaviour for general user-attributes.
        # The global attribute is lost because there are local ones.
        vars1 = {"f1_v1": "f1v1", "f1_v2": "f2v2"}
        vars2 = {"f2_v1": "x1", "f2_v2": "x2"}
        self.create_roundtrip_testcase(
            attr_name="random",  # A generic "user" attribute with no special handling
            global_value_file1="global_file1",
            vars_values_file1=vars1,
            global_value_file2="global_file2",
            vars_values_file2=vars2,
        )
        # combine all 4 vars in one dict
        all_vars_and_attrs = vars1.copy()
        all_vars_and_attrs.update(vars2)
        # TODO: replace with "|", when we drop Python 3.8
        # see: https://peps.python.org/pep-0584/
        # just check they are all there and distinct
        assert len(all_vars_and_attrs) == len(vars1) + len(vars2)
        self.check_roundtrip_results(
            global_attr_value=None,  # local values eclipse the global ones
            var_attr_vals=all_vars_and_attrs,
        )

    def test_04_userstyle_matching_promoted(self):
        # matching local user-attributes are "promoted" to a global one.
        self.create_roundtrip_testcase(
            attr_name="random",
            global_value_file1="global_file1",
            vars_values_file1={"v1": "same-value", "v2": "same-value"},
        )
        self.check_roundtrip_results(
            global_attr_value="same-value",
            var_attr_vals={"v1": None, "v2": None},
        )

    def test_05_userstyle_matching_crossfile_promoted(self):
        # matching user-attributes are promoted, even across input files.
        self.create_roundtrip_testcase(
            attr_name="random",
            global_value_file1="global_file1",
            vars_values_file1={"v1": "same-value", "v2": "same-value"},
            vars_values_file2={"f2_v1": "same-value", "f2_v2": "same-value"},
        )
        self.check_roundtrip_results(
            global_attr_value="same-value",
            var_attr_vals={x: None for x in ("v1", "v2", "f2_v1", "f2_v2")},
        )

    def test_06_userstyle_nonmatching_remainlocal(self):
        # Non-matching user attributes remain 'local' to the individual variables.
        self.create_roundtrip_testcase(
            attr_name="random",
            global_value_file1="global_file1",
            vars_values_file1={"v1": "value-1", "v2": "value-2"},
        )
        self.check_roundtrip_results(
            global_attr_value=None,  # NB it still destroys the global one !!
            var_attr_vals={"v1": "value-1", "v2": "value-2"},
        )

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
        self.create_roundtrip_testcase(
            attr_name="Conventions",
            global_value_file1=None,
            vars_values_file1="user_set",
        )
        self.check_roundtrip_results(
            global_attr_value="CF-1.7",  # standard content from Iris save
            var_attr_vals=None,
        )

    def test_08_conventions_var_both(self):
        # What happens if 'Conventions' appears as both global + local attribute.
        self.create_roundtrip_testcase(
            attr_name="Conventions",
            global_value_file1="global-setting",
            vars_values_file1="local-setting",
        )
        self.check_roundtrip_results(
            global_attr_value="CF-1.7",  # standard content from Iris save
            var_attr_vals=None,
        )

    #######################################################
    # Tests on "global" style attributes
    #  = those specific ones which 'ought' only to be global (except on collisions)
    #

    def test_09_globalstyle__global(self, global_attr):
        attr_content = f"Global tracked {global_attr}"
        self.create_roundtrip_testcase(
            attr_name=global_attr,
            global_value_file1=attr_content,
        )
        self.check_roundtrip_results(global_attr_value=attr_content)

    def test_10_globalstyle__local(self, global_attr):
        # Strictly, not correct CF, but let's see what it does with it.
        attr_content = f"Local tracked {global_attr}"
        self.create_roundtrip_testcase(
            attr_name=global_attr,
            vars_values_file1=attr_content,
        )
        self.check_roundtrip_results(
            global_attr_value=attr_content
        )  # "promoted"

    def test_11_globalstyle__both(self, global_attr):
        attr_global = f"Global-{global_attr}"
        attr_local = f"Local-{global_attr}"
        self.create_roundtrip_testcase(
            attr_name=global_attr,
            global_value_file1=attr_global,
            vars_values_file1=attr_local,
        )
        self.check_roundtrip_results(
            global_attr_value=attr_local  # promoted local setting "wins"
        )

    def test_12_globalstyle__multivar_different(self, global_attr):
        # Multiple *different* local settings are retained, not promoted
        attr_1 = f"Local-{global_attr}-1"
        attr_2 = f"Local-{global_attr}-2"
        with pytest.warns(
            UserWarning, match="should only be a CF global attribute"
        ):
            # A warning should be raised when writing the result.
            self.create_roundtrip_testcase(
                attr_name=global_attr,
                vars_values_file1={"v1": attr_1, "v2": attr_2},
            )
        self.check_roundtrip_results(
            global_attr_value=None,
            var_attr_vals={"v1": attr_1, "v2": attr_2},
        )

    def test_13_globalstyle__multivar_same(self, global_attr):
        # Multiple *same* local settings are promoted to a common global one
        attrval = f"Locally-defined-{global_attr}"
        self.create_roundtrip_testcase(
            attr_name=global_attr,
            vars_values_file1={"v1": attrval, "v2": attrval},
        )
        self.check_roundtrip_results(
            global_attr_value=attrval,
            var_attr_vals={"v1": None, "v2": None},
        )

    def test_14_globalstyle__multifile_different(self, global_attr):
        # Different global attributes from multiple files are retained as local ones
        attr_1 = f"Global-{global_attr}-1"
        attr_2 = f"Global-{global_attr}-2"
        with pytest.warns(
            UserWarning, match="should only be a CF global attribute"
        ):
            # A warning should be raised when writing the result.
            self.create_roundtrip_testcase(
                attr_name=global_attr,
                global_value_file1=attr_1,
                vars_values_file1={"v1": None},
                global_value_file2=attr_2,
                vars_values_file2={"v2": None},
            )
        self.check_roundtrip_results(
            # Combining them "demotes" the common global attributes to local ones
            var_attr_vals={"v1": attr_1, "v2": attr_2}
        )

    def test_15_globalstyle__multifile_same(self, global_attr):
        # Matching global-type attributes in multiple files are retained as global
        attrval = f"Global-{global_attr}"
        self.create_roundtrip_testcase(
            attr_name=global_attr,
            global_value_file1=attrval,
            vars_values_file1={"v1": None},
            global_value_file2=attrval,
            vars_values_file2={"v2": None},
        )
        self.check_roundtrip_results(
            # The attribute remains as a common global setting
            global_attr_value=attrval,
            # The individual variables do *not* have an attribute of this name
            var_attr_vals={"v1": None, "v2": None},
        )

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
            self.create_roundtrip_testcase(
                attr_name=local_attr, global_value_file1=attrval
            )
        else:
            assert origin_style == "input_local"
            # Record in source as a variable-local attribute
            self.create_roundtrip_testcase(
                attr_name=local_attr, vars_values_file1=attrval
            )

        if local_attr in iris.fileformats.netcdf.saver._CF_DATA_ATTRS:
            # These ones are simply discarded on loading.
            # By experiment, this overlap between _CF_ATTRS and _CF_DATA_ATTRS
            # currently contains only 'missing_value' and 'standard_error_multiplier'.
            expect_global = None
            expect_var = None
        else:
            expect_global = None
            if (
                local_attr == "ukmo__process_flags"
                and origin_style == "input_global"
            ):
                # This is very odd behaviour + surely unintended.
                # It's supposed to handle vector values (which we are not checking).
                # But the weird behaviour only applies to the 'global' test, which is
                # obviously not normal usage anyway.
                attrval = "p r o c e s s"
            expect_var = attrval

        if local_attr == "STASH":
            # A special case, output translates this to a different attribute name.
            self.attrname = "um_stash_source"

        self.check_roundtrip_results(
            global_attr_value=expect_global,
            var_attr_vals=expect_var,
        )


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

    def create_load_testcase(
        self,
        attr_name,
        global_value_file1=None,
        vars_values_file1=None,
        global_value_file2=None,
        vars_values_file2=None,
    ) -> iris.cube.CubeList:
        """
        Initialise the testcase from the passed-in controls, configure the input
        files and run a save-load roundtrip to produce the output file.

        The name of the tested attribute and all the temporary filepaths are stored
        on the instance, from where "self.check_load_results()" can get them.

        """
        self.attrname = attr_name
        self.input_filepaths = self.create_testcase_files(
            attr_name=attr_name,
            global_value_file1=global_value_file1,
            var_values_file1=vars_values_file1,
            global_value_file2=global_value_file2,
            var_values_file2=vars_values_file2,
        )
        result_cubes = iris.load(self.input_filepaths)
        result_cubes = sorted(result_cubes, key=lambda cube: cube.name())
        return result_cubes

    #######################################################
    # Tests on "user-style" attributes.
    # This means any arbitrary attribute which a user might have added -- i.e. one with
    # a name which is *not* recognised in the netCDF or CF conventions.
    #

    def test_01_userstyle_single_global(self):
        cube1, cube2 = self.create_load_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            global_value_file1="single-value",
            vars_values_file1={
                "myvar": None,
                "myvar2": None,
            },  # the variable has no such attribute
        )
        # Default behaviour for a general global user-attribute.
        # It is attached to all loaded cubes.
        assert cube1.attributes == {"myname": "single-value"}
        assert cube2.attributes == {"myname": "single-value"}

    def test_02_userstyle_single_local(self):
        # Default behaviour for a general local user-attribute.
        # It is attached to only the specific cube.
        cube1, cube2 = self.create_load_testcase(
            attr_name="myname",  # A generic "user" attribute with no special handling
            vars_values_file1={"myvar1": "single-value", "myvar2": None},
        )
        assert cube1.attributes == {"myname": "single-value"}
        assert cube2.attributes == {}

    def test_03_userstyle_multiple_different(self):
        # Default behaviour for differing local user-attributes.
        # The global attribute is simply lost, because there are local ones.
        vars1 = {"f1_v1": "f1v1", "f1_v2": "f1v2"}
        vars2 = {"f2_v1": "x1", "f2_v2": "x2"}
        cube1, cube2, cube3, cube4 = self.create_load_testcase(
            attr_name="random",  # A generic "user" attribute with no special handling
            global_value_file1="global_file1",
            vars_values_file1=vars1,
            global_value_file2="global_file2",
            vars_values_file2=vars2,
        )
        assert cube1.attributes == {"random": "f1v1"}
        assert cube2.attributes == {"random": "f1v2"}
        assert cube3.attributes == {"random": "x1"}
        assert cube4.attributes == {"random": "x2"}

    def test_04_userstyle_multiple_same(self):
        # Nothing special to note in this case
        # TODO: ??remove??
        cube1, cube2 = self.create_load_testcase(
            attr_name="random",
            global_value_file1="global_file1",
            vars_values_file1={"v1": "same-value", "v2": "same-value"},
        )
        assert cube1.attributes == {"random": "same-value"}
        assert cube2.attributes == {"random": "same-value"}

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
        (cube,) = self.create_load_testcase(
            attr_name="Conventions",
            global_value_file1=None,
            vars_values_file1="user_set",
        )
        assert cube.attributes == {"Conventions": "user_set"}

    def test_08_conventions_var_both(self):
        # What happens if 'Conventions' appears as both global + local attribute.
        # = the global version gets lost.
        (cube,) = self.create_load_testcase(
            attr_name="Conventions",
            global_value_file1="global-setting",
            vars_values_file1="local-setting",
        )
        assert cube.attributes == {"Conventions": "local-setting"}

    #######################################################
    # Tests on "global" style attributes
    #  = those specific ones which 'ought' only to be global (except on collisions)
    #

    def test_09_globalstyle__global(self, global_attr):
        attr_content = f"Global tracked {global_attr}"
        (cube,) = self.create_load_testcase(
            attr_name=global_attr,
            global_value_file1=attr_content,
        )
        assert cube.attributes == {global_attr: attr_content}

    def test_10_globalstyle__local(self, global_attr):
        # Strictly, not correct CF, but let's see what it does with it.
        # = treated the same as a global setting
        attr_content = f"Local tracked {global_attr}"
        (cube,) = self.create_load_testcase(
            attr_name=global_attr,
            vars_values_file1=attr_content,
        )
        assert cube.attributes == {global_attr: attr_content}

    def test_11_globalstyle__both(self, global_attr):
        attr_global = f"Global-{global_attr}"
        attr_local = f"Local-{global_attr}"
        (cube,) = self.create_load_testcase(
            attr_name=global_attr,
            global_value_file1=attr_global,
            vars_values_file1=attr_local,
        )
        # promoted local setting "wins"
        assert cube.attributes == {global_attr: attr_local}

    def test_12_globalstyle__multivar_different(self, global_attr):
        # Multiple *different* local settings are retained
        attr_1 = f"Local-{global_attr}-1"
        attr_2 = f"Local-{global_attr}-2"
        cube1, cube2 = self.create_load_testcase(
            attr_name=global_attr,
            vars_values_file1={"v1": attr_1, "v2": attr_2},
        )
        assert cube1.attributes == {global_attr: attr_1}
        assert cube2.attributes == {global_attr: attr_2}

    def test_14_globalstyle__multifile_different(self, global_attr):
        # Different global attributes from multiple files are retained as local ones
        attr_1 = f"Global-{global_attr}-1"
        attr_2 = f"Global-{global_attr}-2"
        cube1, cube2, cube3, cube4 = self.create_load_testcase(
            attr_name=global_attr,
            global_value_file1=attr_1,
            vars_values_file1={"f1v1": None, "f1v2": None},
            global_value_file2=attr_2,
            vars_values_file2={"f2v1": None, "f2v2": None},
        )
        assert cube1.attributes == {global_attr: attr_1}
        assert cube2.attributes == {global_attr: attr_1}
        assert cube3.attributes == {global_attr: attr_2}
        assert cube4.attributes == {global_attr: attr_2}

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
            (cube,) = self.create_load_testcase(
                attr_name=local_attr, global_value_file1=attrval
            )
        else:
            assert origin_style == "input_local"
            # Record in source as a variable-local attribute
            (cube,) = self.create_load_testcase(
                attr_name=local_attr, vars_values_file1=attrval
            )

        # Work out the expected result.
        # NOTE: generally, result will be the same whether the original attribute is
        # provided as a global or variable attribute ...
        expected_result = {local_attr: attrval}
        # ... but there are some special cases
        if origin_style == "input_local":
            if local_attr == "ukmo__process_flags":
                # Some odd special behaviour here.
                expected_result = {local_attr: ("process",)}
            elif local_attr in ("standard_error_multiplier", "missing_value"):
                # For some reason, these ones never appear on the cube
                expected_result = {}

        assert cube.attributes == expected_result


class TestSave(MixinAttrsTesting):
    """
    Test saving from cube attributes dictionary (various categories) into files.

    """

    def create_save_testcase(self, attr_name, value1, value2=None):
        """
        Test attribute saving for cube(s) with given value(s).

        Create cubes(s) and save to temporary file, then return the global and all
        variable-local attributes of that name (or None-s) from the file.
        """
        self.attrname = (
            attr_name  # Required for common testfile-naming function.
        )
        if value2 is None:
            n_cubes = 1
            values = [value1]
        else:
            n_cubes = 2
            values = [value1, value2]
        cube_names = [f"cube_{i_cube}" for i_cube in range(n_cubes)]
        cubes = [
            Cube([0], long_name=cube_name, attributes={attr_name: attr_value})
            for cube_name, attr_value in zip(cube_names, values)
        ]
        self.result_filepath = self._testfile_path("result")
        iris.save(cubes, self.result_filepath)
        # Get the global+local attribute values directly from the file with netCDF4
        if attr_name == "STASH":
            # A special case : the stored name is different
            attr_name = "um_stash_source"
        try:
            ds = threadsafe_nc4.DatasetWrapper(self.result_filepath)
            global_result = (
                ds.getncattr(attr_name) if attr_name in ds.ncattrs() else None
            )
            local_results = [
                (
                    var.getncattr(attr_name)
                    if attr_name in var.ncattrs()
                    else None
                )
                for var in ds.variables.values()
            ]
        finally:
            ds.close()
        return [global_result] + local_results

    def test_01_userstyle__single(self):
        results = self.create_save_testcase("random", "value-x")
        # It is stored as a *global* by default.
        assert results == ["value-x", None]

    def test_02_userstyle__multiple_same(self):
        results = self.create_save_testcase("random", "value-x", "value-x")
        # As above.
        assert results == ["value-x", None, None]

    def test_03_userstyle__multiple_different(self):
        results = self.create_save_testcase("random", "value-A", "value-B")
        # Clashing values are stored as locals on the individual variables.
        assert results == [None, "value-A", "value-B"]

    def test_04_Conventions__single(self):
        results = self.create_save_testcase("Conventions", "x")
        # Always discarded + replaced by a single global setting.
        assert results == ["CF-1.7", None]

    def test_05_Conventions__multiple_same(self):
        results = self.create_save_testcase(
            "Conventions", "same-value", "same-value"
        )
        # Always discarded + replaced by a single global setting.
        assert results == ["CF-1.7", None, None]

    def test_06_Conventions__multiple_different(self):
        results = self.create_save_testcase(
            "Conventions", "value-A", "value-B"
        )
        # Always discarded + replaced by a single global setting.
        assert results == ["CF-1.7", None, None]

    def test_07_globalstyle__single(self, global_attr):
        results = self.create_save_testcase(global_attr, "value")
        # Defaults to global
        assert results == ["value", None]

    def test_08_globalstyle__multiple_same(self, global_attr):
        results = self.create_save_testcase(
            global_attr, "value-same", "value-same"
        )
        assert results == ["value-same", None, None]

    def test_09_globalstyle__multiple_different(self, global_attr):
        msg_regexp = (
            f"'{global_attr}' is being added as CF data variable attribute,"
            f".* should only be a CF global attribute."
        )
        with pytest.warns(UserWarning, match=msg_regexp):
            results = self.create_save_testcase(
                global_attr, "value-A", "value-B"
            )
        # *Only* stored as locals when there are differing values.
        assert results == [None, "value-A", "value-B"]

    def test_10_localstyle__single(self, local_attr):
        results = self.create_save_testcase(local_attr, "value")
        # Defaults to local
        expected_results = [None, "value"]
        if local_attr == "ukmo__process_flags":
            # A particular, really weird case
            expected_results = [None, "v a l u e"]
        assert results == expected_results

    def test_11_localstyle__multiple_same(self, local_attr):
        results = self.create_save_testcase(
            local_attr, "value-same", "value-same"
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
        assert results == expected_results

    def test_12_localstyle__multiple_different(self, local_attr):
        results = self.create_save_testcase(local_attr, "value-A", "value-B")
        # Different values are treated just the same as matching ones.
        expected_results = [None, "value-A", "value-B"]
        if local_attr == "ukmo__process_flags":
            # A particular, really weird case
            expected_results = [
                None,
                "v a l u e - A",
                "v a l u e - B",
            ]
        assert results == expected_results
