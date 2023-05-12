# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Integration tests for fast-loading FF and PP files."""

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests  # isort:skip

from collections.abc import Iterable
import shutil
import tempfile

import numpy as np

import iris
from iris.coord_systems import GeogCS
import iris.coords
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.cube import Cube, CubeList
from iris.exceptions import IgnoreCubeException
from iris.fileformats.pp import EARTH_RADIUS, STASH
from iris.fileformats.um._fast_load import STRUCTURED_LOAD_CONTROLS


class Mixin_FieldTest:
    # A mixin providing common facilities for fast-load testing :
    #   * create 'raw' cubes to produce the desired PP fields in a test file.
    #   * save 'raw' cubes to temporary PP files that get deleted afterwards.
    #   * control whether tests run with 'normal' or 'fast' loading.

    def setUp(self):
        # Create a private temporary directory.
        self.temp_dir_path = tempfile.mkdtemp()
        # Initialise temporary filename generation.
        self.tempfile_count = 0
        self.tempfile_path_fmt = (
            "{dir_path}/tempfile_{prefix}_{file_number:06d}{suffix}"
        )
        # Enable fast loading, if the inheritor enables it.
        # N.B. *requires* the user to define "self.do_fast_loads" (no default).
        if self.do_fast_loads:
            # Enter a 'structured load' context.
            self.load_context = STRUCTURED_LOAD_CONTROLS.context(
                loads_use_structured=True
            )
            # N.B. we can't use a 'with', so issue separate 'enter' and 'exit'
            # calls instead.
            self.load_context.__enter__()

    def tearDown(self):
        # Delete temporary directory.
        shutil.rmtree(self.temp_dir_path)
        if self.do_fast_loads:
            # End the 'fast loading' context.
            self.load_context.__exit__(None, None, None)

    def _temp_filepath(self, user_name="", suffix=".pp"):
        # Return the filepath for a new temporary file.
        self.tempfile_count += 1
        file_path = self.tempfile_path_fmt.format(
            dir_path=self.temp_dir_path,
            prefix=user_name,
            file_number=self.tempfile_count,
            suffix=suffix,
        )
        return file_path

    def save_fieldcubes(self, cubes, basename=""):
        # Save cubes to a temporary file, and return its filepath.
        file_path = self._temp_filepath(user_name=basename, suffix=".pp")
        iris.save(cubes, file_path)
        return file_path

    def fields(
        self,
        c_t=None,
        cft=None,
        ctp=None,
        c_h=None,
        c_p=None,
        phn=0,
        mmm=None,
        pse=None,
    ):
        # Return a list of 2d cubes representing raw PPFields, from args
        # specifying sequences of (scalar) coordinate values.
        # TODO? : add bounds somehow ?
        #
        # Arguments 'c<xx>' are either a single int value, making a scalar
        # coord, or a string of characters : '0'-'9' (index) or '-' (missing).
        # The indexes select point values from fixed list of possibles.
        #
        # Argument 'c_h' and 'c_p' represent height or pressure values, so
        # ought to be mutually exclusive -- these control LBVC.
        #
        # Argument 'phn' indexes phenomenon types.
        #
        # Argument 'mmm' denotes existence (or not) of a cell method of type
        # 'average' or 'min' or 'max' (values '012' respectively), applying to
        # the time values -- ultimately, this controls LBTIM.
        #
        # Argument 'pse' denotes pseudo-level numbers.
        # These translate into 'LBUSER5' values.

        # Get the number of result cubes, defined by the 'longest' arg.
        def arglen(arg):
            # Get the 'length' of a control argument.
            if arg is None:
                result = 0
            elif isinstance(arg, str):
                result = len(arg)
            else:
                result = 1
            return result

        n_flds = max(arglen(x) for x in (c_t, cft, ctp, c_h, c_p, mmm))

        # Make basic anonymous test cubes.
        ny, nx = 3, 5
        data = np.arange(n_flds * ny * nx, dtype=np.float32)
        data = data.reshape((n_flds, ny, nx))
        cubes = [Cube(data[i]) for i in range(n_flds)]

        # Define test point values for making coordinates.
        time_unit = "hours since 1970-01-01"
        period_unit = "hours"
        height_unit = "m"
        pressure_unit = "hPa"
        time_values = 24.0 * np.arange(10)
        height_values = 100.0 * np.arange(1, 11)
        pressure_values = [
            100.0,
            150.0,
            200.0,
            250.0,
            300.0,
            500.0,
            850.0,
            1000.0,
        ]
        pseudolevel_values = range(1, 11)  # A valid value is >= 1.

        # Test phenomenon details.
        # NOTE: in order to write/readback as identical, these also contain a
        # canonical unit and matching STASH attribute.
        # Those could in principle be looked up, but it's a bit awkward.
        phenomenon_values = [
            ("air_temperature", "K", "m01s01i004"),
            ("x_wind", "m s-1", "m01s00i002"),
            ("y_wind", "m s-1", "m01s00i003"),
            ("specific_humidity", "kg kg-1", "m01s00i010"),
        ]

        # Test cell-methods.
        # NOTE: if you add an *interval* to any of these cell-methods, it is
        # not saved into the PP file (?? or maybe not loaded back again ??).
        # This could be a PP save/load bug, or maybe just because no bounds ?
        cell_method_values = [
            CellMethod("mean", "time"),
            CellMethod("maximum", "time"),
            CellMethod("minimum", "time"),
        ]

        # Define helper to decode an argument as a list of test values.
        def arg_vals(arg, vals):
            # Decode an argument to a list of 'n_flds' coordinate point values.
            # (or 'None' where missing)

            # First get a list of value indices from the argument.
            # Can be: a single index value; a list of indices; or a string.
            if isinstance(arg, Iterable) and not isinstance(arg, str):
                # Can also just pass a simple iterable of values.
                inds = [int(val) for val in arg]
            else:
                n_vals = arglen(arg)
                if n_vals == 0:
                    inds = [None] * n_flds
                elif n_vals == 1:
                    inds = [int(arg)] * n_flds
                else:
                    assert isinstance(arg, str)
                    inds = [None if char == "-" else int(char) for char in arg]

            # Convert indices to selected point values.
            values = [None if ind is None else vals[int(ind)] for ind in inds]

            return values

        # Apply phenomenon_values definitions.
        phenomena = arg_vals(phn, phenomenon_values)
        for cube, (name, units, stash) in zip(cubes, phenomena):
            cube.rename(name)
            # NOTE: in order to get a cube that will write+readback the same,
            # the units must be the canonical one.
            cube.units = units
            # NOTE: in order to get a cube that will write+readback the same,
            # we must include a STASH attribute.
            cube.attributes["STASH"] = STASH.from_msi(stash)
            cube.fill_value = np.float32(-1e30)

        # Add x and y coords.
        cs = GeogCS(EARTH_RADIUS)
        xvals = np.linspace(0.0, 180.0, nx)
        co_x = DimCoord(
            np.array(xvals, dtype=np.float32),
            standard_name="longitude",
            units="degrees",
            coord_system=cs,
        )
        yvals = np.linspace(-45.0, 45.0, ny)
        co_y = DimCoord(
            np.array(yvals, dtype=np.float32),
            standard_name="latitude",
            units="degrees",
            coord_system=cs,
        )
        for cube in cubes:
            cube.add_dim_coord(co_y, 0)
            cube.add_dim_coord(co_x, 1)

        # Add multiple scalar coordinates as defined by the arguments.
        def arg_coords(arg, name, unit, vals=None):
            # Decode an argument to a list of scalar coordinates.
            if vals is None:
                vals = np.arange(n_flds + 2)  # Note allowance
            vals = arg_vals(arg, vals)
            coords = [
                None if val is None else DimCoord([val], units=unit)
                for val in vals
            ]
            # Apply names separately, as 'pressure' is not a standard name.
            for coord in coords:
                if coord:
                    coord.rename(name)
                    # Also fix heights to match what comes from a PP file.
                    if name == "height":
                        coord.attributes["positive"] = "up"
            return coords

        def add_arg_coords(arg, name, unit, vals=None):
            # Add scalar coordinates to each cube, for one argument.
            coords = arg_coords(arg, name, unit, vals)
            for cube, coord in zip(cubes, coords):
                if coord:
                    cube.add_aux_coord(coord)

        add_arg_coords(c_t, "time", time_unit, time_values)
        add_arg_coords(cft, "forecast_reference_time", time_unit)
        add_arg_coords(ctp, "forecast_period", period_unit, time_values)
        add_arg_coords(c_h, "height", height_unit, height_values)
        add_arg_coords(c_p, "pressure", pressure_unit, pressure_values)
        add_arg_coords(pse, "pseudo_level", "1", pseudolevel_values)

        # Add cell methods as required.
        methods = arg_vals(mmm, cell_method_values)
        for cube, method in zip(cubes, methods):
            if method:
                cube.add_cell_method(method)

        return cubes


class MixinBasic:
    # A mixin of tests that can be applied to *either* standard or fast load.
    # The "real" test classes must inherit this, and Mixin_FieldTest,
    # and define 'self.do_fast_loads' as True or False.
    #
    # Basic functional tests.

    def test_basic(self):
        # Show that basic load merging works.
        flds = self.fields(c_t="123", cft="000", ctp="123", c_p=0)
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_phenomena(self):
        # Show that different phenomena are merged into distinct cubes.
        flds = self.fields(c_t="1122", phn="0101")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_cross_file_concatenate(self):
        # Combine vector dimensions (i.e. concatenate) across multiple files.
        fldset_1 = self.fields(c_t="12")
        fldset_2 = self.fields(c_t="34")
        file_1 = self.save_fieldcubes(fldset_1)
        file_2 = self.save_fieldcubes(fldset_2)
        results = iris.load((file_1, file_2))
        expected = CubeList(fldset_1 + fldset_2).merge()
        self.assertEqual(results, expected)

    def test_cell_method(self):
        # Check that cell methods (i.e. LBPROC values) produce distinct
        # phenomena.
        flds = self.fields(c_t="000111222", mmm="-01-01-01")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(
            CubeList(flds[i_start::3]).merge_cube() for i_start in range(3)
        )
        self.assertEqual(results, expected)


class MixinCallDetails:
    # A mixin of tests that can be applied to *either* standard or fast load.
    # The "real" test classes must inherit this, and Mixin_FieldTest,
    # and define 'self.do_fast_loads' as True or False.
    #
    # Tests for different load calls and load-call arguments.

    def test_stash_constraint(self):
        # Check that an attribute constraint functions correctly.
        # Note: this is a special case in "fileformats.pp".
        flds = self.fields(c_t="1122", phn="0101")
        file = self.save_fieldcubes(flds)
        airtemp_flds = [fld for fld in flds if fld.name() == "air_temperature"]
        stash_attribute = airtemp_flds[0].attributes["STASH"]
        results = iris.load(
            file, iris.AttributeConstraint(STASH=stash_attribute)
        )
        expected = CubeList(airtemp_flds).merge()
        self.assertEqual(results, expected)

    def test_ordinary_constraint(self):
        # Check that a 'normal' constraint functions correctly.
        # Note: *should* be independent of structured loading.
        flds = self.fields(c_h="0123")
        file = self.save_fieldcubes(flds)
        height_constraint = iris.Constraint(height=lambda h: 150.0 < h < 350.0)
        results = iris.load(file, height_constraint)
        expected = CubeList(flds[1:3]).merge()
        self.assertEqual(results, expected)

    def test_callback(self):
        # Use 2 timesteps each of (air-temp on height) and (rh on pressure).
        flds = self.fields(c_t="0011", phn="0303", c_h="0-1-", c_p="-2-3")
        file = self.save_fieldcubes(flds)

        if not self.do_fast_loads:

            def callback(cube, field, filename):
                self.assertEqual(filename, file)
                lbvc = field.lbvc
                if lbvc == 1:
                    # reject the height level data (accept only pressure).
                    raise IgnoreCubeException()
                else:
                    # Record the LBVC value.
                    cube.attributes["LBVC"] = lbvc

        else:

            def callback(cube, collation, filename):
                self.assertEqual(filename, file)
                lbvcs = [fld.lbvc for fld in collation.fields]
                lbvc0 = lbvcs[0]
                if not np.all(lbvcs == lbvc0):
                    msg = "Fields have different LBVCs : {}"
                    raise ValueError(msg.format(set(lbvcs)))
                if lbvc0 == 1:
                    # reject the height level data (accept only pressure).
                    raise IgnoreCubeException()
                else:
                    # Record the LBVC values.
                    cube.attributes["A_LBVC"] = lbvcs

        results = iris.load(file, callback=callback)

        # Make an 'expected' from selected fields, with the expected attribute.
        expected = CubeList([flds[1], flds[3]]).merge()
        if not self.do_fast_loads:
            # This is actually a NumPy int32, so honour that here.
            expected[0].attributes["LBVC"] = np.int32(8)
        else:
            expected[0].attributes["A_LBVC"] = [8, 8]

        self.assertEqual(results, expected)

    def test_load_cube(self):
        flds = self.fields(c_t="123", cft="000", ctp="123", c_p=0)
        file = self.save_fieldcubes(flds)
        results = iris.load_cube(file)
        expected = CubeList(flds).merge_cube()
        self.assertEqual(results, expected)

    def test_load_cubes(self):
        flds = self.fields(c_h="0123")
        file = self.save_fieldcubes(flds)
        height_constraints = [
            iris.Constraint(height=300.0),
            iris.Constraint(height=lambda h: 150.0 < h < 350.0),
            iris.Constraint("air_temperature"),
        ]
        results = iris.load_cubes(file, height_constraints)
        expected = CubeList(
            [
                flds[2],
                CubeList(flds[1:3]).merge_cube(),
                CubeList(flds).merge_cube(),
            ]
        )
        self.assertEqual(results, expected)

    def test_load_raw(self):
        fldset_1 = self.fields(c_t="015", phn="001")
        fldset_2 = self.fields(c_t="234")
        file_1 = self.save_fieldcubes(fldset_1)
        file_2 = self.save_fieldcubes(fldset_2)
        results = iris.load_raw((file_1, file_2))
        if not self.do_fast_loads:
            # Each 'raw' cube is just one field.
            expected = CubeList(fldset_1 + fldset_2)
        else:
            # 'Raw' cubes have combined (vector) times within each file.
            # The 'other' phenomenon appears separately.
            expected = CubeList(
                [
                    CubeList(fldset_1[:2]).merge_cube(),
                    CubeList(fldset_2).merge_cube(),
                    fldset_1[2],
                ]
            )

        # Again here, the order of these results is not stable :
        # It varies with random characters in the temporary filepath.
        #
        # *****************************************************************
        # *** Here, this is clearly ALSO the case for "standard" loads. ***
        # *****************************************************************
        #
        # E.G. run "test_fast_load.py -v TestCallDetails__Iris.test_load_raw" :
        # If you remove the sort operations, this fails "sometimes".
        #
        # To fix this, sort both expected and results by (first) timepoint
        # - for which purpose we made all the time values different.

        def timeorder(cube):
            return cube.coord("time").points[0]

        expected = sorted(expected, key=timeorder)
        results = sorted(results, key=timeorder)

        self.assertEqual(results, expected)


class MixinDimsAndOrdering:
    # A mixin of tests that can be applied to *either* standard or fast load.
    # The "real" test classes must inherit this, and Mixin_FieldTest,
    # and define 'self.do_fast_loads' as True or False.
    #
    # Tests for multidimensional results and dimension orderings.

    def test_multidim(self):
        # Check that a full 2-phenom * 2d structure all works properly.
        flds = self.fields(c_t="00001111", c_h="00110011", phn="01010101")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_odd_order(self):
        # Show that an erratic interleaving of phenomena fields still works.
        # N.B. field sequences *within* each phenomenon are properly ordered.
        flds = self.fields(c_t="00010111", c_h="00101101", phn="01001011")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_v_t_order(self):
        # With height varying faster than time, first dimension is time,
        # which matches the 'normal' load behaviour.
        flds = self.fields(c_t="000111", c_h="012012")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        # Order is (t, h, y, x), which is "standard".
        self.assertEqual(expected[0].coord_dims("time"), (0,))
        self.assertEqual(expected[0].coord_dims("height"), (1,))
        self.assertEqual(results, expected)

    def test_t_v_order(self):
        # With time varying faster than height, first dimension is height,
        # which does not match the 'normal' load.
        flds = self.fields(c_t="010101", c_h="001122")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        if not self.do_fast_loads:
            # Order is (t, h, y, x), which is "standard".
            self.assertEqual(results[0].coord_dims("time"), (0,))
            self.assertEqual(results[0].coord_dims("height"), (1,))
        else:
            # Order is (h, t, y, x), which is *not* "standard".
            self.assertEqual(results[0].coord_dims("time"), (1,))
            self.assertEqual(results[0].coord_dims("height"), (0,))
            expected[0].transpose((1, 0, 2, 3))
        self.assertEqual(results, expected)

    def test_missing_combination(self):
        # A case where one field is 'missing' to make a 2d result.
        flds = self.fields(c_t="00011", c_h="01202")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(expected[0].coord_dims("time"), (0,))
        self.assertEqual(expected[0].coord_dims("height"), (0,))
        if self.do_fast_loads:
            # Something a bit weird happens to the 'height' coordinate in this
            # case (and not for standard load).
            for cube in expected:
                cube.coord("height").points = np.array(
                    cube.coord("height").points, dtype=np.float32
                )
                cube.coord("height").attributes = {}
        self.assertEqual(results, expected)


class MixinProblemCases:
    def test_FAIL_scalar_vector_concatenate(self):
        # Structured load can produce a scalar coordinate from one file, and a
        # matching vector one from another file, but these won't "combine".
        # We'd really like to fix this one...
        (single_timepoint_fld,) = self.fields(c_t="1")
        multi_timepoint_flds = self.fields(c_t="23")
        file_single = self.save_fieldcubes(
            [single_timepoint_fld], basename="single"
        )
        file_multi = self.save_fieldcubes(
            multi_timepoint_flds, basename="multi"
        )

        results = iris.load((file_single, file_multi))
        if not self.do_fast_loads:
            # This is what we'd LIKE to get (what iris.load gives).
            expected = CubeList(
                multi_timepoint_flds + [single_timepoint_fld]
            ).merge()
        else:
            # This is what we ACTUALLY get at present.
            # It can't combine the scalar and vector time coords.
            expected = CubeList(
                [
                    CubeList(multi_timepoint_flds).merge_cube(),
                    single_timepoint_fld,
                ]
            )
            # NOTE: in this case, we need to sort the results to ensure a
            # repeatable ordering, because ??somehow?? the random temporary
            # directory name affects the ordering of the cubes in the result !
            results = CubeList(sorted(results, key=lambda cube: cube.shape))
        self.assertEqual(results, expected)

    def test_FAIL_phenomena_nostash(self):
        # If we remove the 'STASH' attributes, certain phenomena can still be
        # successfully encoded+decoded by standard load using LBFC values.
        # Structured loading gets this wrong, because it does not use LBFC in
        # characterising phenomena.
        flds = self.fields(c_t="1122", phn="0101")
        for fld in flds:
            del fld.attributes["STASH"]
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        if not self.do_fast_loads:
            # This is what we'd LIKE to get (what iris.load gives).
            expected = CubeList(flds).merge()
        else:
            # At present, we get a cube incorrectly combined together over all
            # 4 timepoints, with the same phenomenon for all (!wrong!).
            # It's a bit tricky to arrange the existing data like that.
            # Do it by hacking the time values to allow merge, and then fixing
            # up the time
            old_t1, old_t2 = (
                fld.coord("time").points[0] for fld in (flds[0], flds[2])
            )
            for i_fld, fld in enumerate(flds):
                # Hack the phenomena to all look like the first one.
                fld.rename("air_temperature")
                fld.units = "K"
                # Hack the time points so the 4 cube can merge into one.
                fld.coord("time").points = [old_t1 + i_fld]
            one_cube = CubeList(flds).merge_cube()
            # Replace time dim with an anonymous dim.
            co_t_fake = one_cube.coord("time")
            one_cube.remove_coord(co_t_fake)
            # Reconstruct + add back the expected auxiliary time coord.
            co_t_new = AuxCoord(
                [old_t1, old_t1, old_t2, old_t2],
                standard_name="time",
                units=co_t_fake.units,
            )
            one_cube.add_aux_coord(co_t_new, 0)
            expected = [one_cube]
        self.assertEqual(results, expected)

    def test_FAIL_pseudo_levels(self):
        # Show how pseudo levels are handled.
        flds = self.fields(c_t="000111222", pse="123123123")
        file = self.save_fieldcubes(flds)
        results = iris.load(file)
        expected = CubeList(flds).merge()

        # NOTE: this problem is now fixed : Structured load gives the same answer.
        #
        #        if not self.do_fast_loads:
        #            expected = CubeList(flds).merge()
        #        else:
        #            # Structured loading doesn't understand pseudo-level.
        #            # The result is rather horrible...
        #
        #            # First get a cube over 9 timepoints.
        #            flds = self.fields(c_t='012345678',
        #                               pse=1)  # result gets level==2, not clear why.
        #
        #            # Replace the time coord with an AUX coord.
        #            nine_timepoints_cube = CubeList(flds).merge_cube()
        #            co_time = nine_timepoints_cube.coord('time')
        #            nine_timepoints_cube.remove_coord(co_time)
        #            nine_timepoints_cube.add_aux_coord(AuxCoord.from_coord(co_time),
        #                                               0)
        #            # Set the expected timepoints equivalent to '000111222'.
        #            nine_timepoints_cube.coord('time').points = \
        #                np.array([0.0, 0.0, 0.0, 24.0, 24.0, 24.0, 48.0, 48.0, 48.0])
        #            # Make a cubelist with this single cube.
        #            expected = CubeList([nine_timepoints_cube])

        self.assertEqual(results, expected)


class TestBasic__Iris(Mixin_FieldTest, MixinBasic, tests.IrisTest):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'basic' tests with *normal* loading.
    do_fast_loads = False


class TestBasic__Fast(Mixin_FieldTest, MixinBasic, tests.IrisTest):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'basic' tests with *FAST* loading.
    do_fast_loads = True


class TestCallDetails__Iris(Mixin_FieldTest, MixinCallDetails, tests.IrisTest):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'call details' tests with *normal* loading.
    do_fast_loads = False


class TestCallDetails__Fast(Mixin_FieldTest, MixinCallDetails, tests.IrisTest):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'call details' tests with *FAST* loading.
    do_fast_loads = True


class TestDimsAndOrdering__Iris(
    Mixin_FieldTest, MixinDimsAndOrdering, tests.IrisTest
):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'dimensions and ordering' tests with *normal* loading.
    do_fast_loads = False


class TestDimsAndOrdering__Fast(
    Mixin_FieldTest, MixinDimsAndOrdering, tests.IrisTest
):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'dimensions and ordering' tests with *FAST* loading.
    do_fast_loads = True


class TestProblems__Iris(Mixin_FieldTest, MixinProblemCases, tests.IrisTest):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'failure cases' tests with *normal* loading.
    do_fast_loads = False


class TestProblems__Fast(Mixin_FieldTest, MixinProblemCases, tests.IrisTest):
    # Finally, an actual test-class (unittest.TestCase) :
    # run the 'failure cases' tests with *FAST* loading.
    do_fast_loads = True


if __name__ == "__main__":
    tests.main()
