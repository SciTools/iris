# (C) British Crown Copyright 2014 - 2016, Met Office
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
"""Integration tests for fast-loading FF and PP files."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from collections import Iterable

import os.path
import tempfile
import shutil
import os

from cf_units import Unit
import numpy as np
import six

import iris.coords
from iris.coords import DimCoord, AuxCoord
from iris.coord_systems import GeogCS
from iris.cube import Cube, CubeList
from iris.fileformats.pp import EARTH_RADIUS, STASH

from iris import load as iris_load
from iris.fileformats.um import structured_um_loading


class Mixin_FieldTest(object):
    # A mixin for tests making temporary PP files for fast-load testing.

    def setUp(self):
        # Create a private temporary directory.
        self.temp_dir_path = tempfile.mkdtemp()
        # Initialise temporary filename generation.
        self.tempfile_count = 0
        self.tempfile_path_fmt = \
            '{dir_path}/tempfile_{prefix}_{file_number:06d}{suffix}'

    def tearDown(self):
        # Delete temporary directory.
        shutil.rmtree(self.temp_dir_path)

    def temp_filepath(self, user_name='', suffix='.pp'):
        # Return the filepath for a new temporary file.
        self.tempfile_count += 1
        file_path = self.tempfile_path_fmt.format(
            dir_path=self.temp_dir_path,
            prefix=user_name,
            file_number=self.tempfile_count,
            suffix=suffix)
        return file_path

    def save_fieldcubes(self, cubes, basename=''):
        # Save cubes to a temporary file, and return its filepath.
        file_path = self.temp_filepath(user_name=basename, suffix='.pp')
        iris.save(cubes, file_path)
        return file_path

    def load_function(self, *args, **kwargs):
        # Return data from "iris.load", using either 'normal' or 'fast' method
        # as selected by the test class.
        if self.load_type == 'iris':
            return iris_load(*args, **kwargs)
        elif self.load_type == 'fast':
            with structured_um_loading():
                return iris_load(*args, **kwargs)

    # Reference values for making coordinate contents.
    time_unit = 'hours since 1970-01-01'
    period_unit = 'hours'
    time_values = 24.0 * np.arange(5)
    height_values = [100.0, 200.0, 300.0, 400.0]
    pressure_values = [300.0, 500.0, 850.0, 1000.0]
    # NOTE: in order to write/readback as identical, these test phenomena
    # settings also provide the canonical unit and a matching STASH attribute.
    # These could in principle be looked up, but it's a bit awkward.
    phenomena = [
        ('air_temperature', 'K', 'm01s01i004'),
        ('x_wind', 'm s-1', 'm01s00i002'),
        ('y_wind', 'm s-1', 'm01s00i003'),
        ('specific_humidity', 'kg kg-1', 'm01s00i010'),
        ]

    def fields(self, c_t=None, cft=None, ctp=None,
               c_h=None, c_p=None, phn=0, mmm=None):
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

        # Get the number of result cubes, defined by the 'longest' arg.
        def arglen(arg):
            # Get the 'length' of a control argument.
            if arg is None:
                result = 0
            elif isinstance(arg, six.string_types):
                result = len(arg)
            else:
                result = 1
            return result

        n_flds = max(arglen(x)
                     for x in (c_t, cft, ctp, c_h, c_p, mmm))

        # Make basic anonymous test cubes.
        ny, nx = 3, 5
        data = np.arange(n_flds * ny * nx, dtype=np.float32)
        data = data.reshape((n_flds, ny, nx))
        cubes = [Cube(data[i]) for i in range(n_flds)]

        # Apply phenomena definitions.
        def arg_vals(arg, vals):
            # Decode an argument to a list of 'n_flds' coordinate point values.
            # (or 'None' where missing)

            # First get a list of value indices from the argument.
            # Can be: a single index value; a list of indices; or a string.
            if (isinstance(arg, Iterable) and
                    not isinstance(arg, six.string_types)):
                # Can also just pass a simple iterable of values.
                inds = [int(val) for val in arg]
            else:
                n_vals = arglen(arg)
                if n_vals == 0:
                    inds = [None] * n_flds
                elif n_vals == 1:
                    inds = [int(arg)] * n_flds
                else:
                    assert isinstance(arg, six.string_types)
                    inds = [None if char == '-' else int(char)
                            for char in arg]

            # Convert indices to selected point values.
            values = [None if ind is None else vals[int(ind)]
                      for ind in inds]

            return values

        phenomena = arg_vals(phn, self.phenomena)
        for cube, (name, units, stash) in zip(cubes, phenomena):
            cube.rename(name)
            # NOTE: in order to get a cube that will write+readback the same,
            # the units must be the canonical one.
            cube.units = units
            # NOTE: in order to get a cube that will write+readback the same,
            # we must include a STASH attribute.
            cube.attributes['STASH'] = STASH.from_msi(stash)

        # Add x and y coords.
        cs = GeogCS(EARTH_RADIUS)
        xvals = np.linspace(0.0, 180.0, nx)
        co_x = DimCoord(np.array(xvals, dtype=np.float32),
                        standard_name='longitude', units='degrees',
                        coord_system=cs)
        yvals = np.linspace(-45.0, 45.0, ny)
        co_y = DimCoord(np.array(yvals, dtype=np.float32),
                        standard_name='latitude', units='degrees',
                        coord_system=cs)
        for cube in cubes:
            cube.add_dim_coord(co_y, 0)
            cube.add_dim_coord(co_x, 1)

        # Add multiple scalar coordinates as defined by the arguments.
        def arg_coords(arg, name, unit, vals=None):
            # Decode an argument to a list of scalar coordinates.
            if vals is None:
                vals = np.arange(n_flds + 2)  # Note allowance
            vals = arg_vals(arg, vals)
            coords = [None if val is None else DimCoord([val], units=unit)
                      for val in vals]
            # Apply names separately, as 'pressure' is not a standard name.
            for coord in coords:
                if coord:
                    coord.rename(name)
            return coords

        def add_arg_coords(arg, name, unit, vals=None):
            # Add scalar coordinates to each cube, for one argument.
            coords = arg_coords(arg, name, unit, vals)
            for cube, coord in zip(cubes, coords):
                if coord:
                    cube.add_aux_coord(coord)

        add_arg_coords(c_t, 'time', self.time_unit, self.time_values)
        add_arg_coords(cft, 'forecast_reference_time', self.time_unit)
        add_arg_coords(ctp, 'forecast_period', 'hours', self.time_values)
        add_arg_coords(c_h, 'height', 'm', self.height_values)
        add_arg_coords(c_p, 'pressure', 'hPa', self.pressure_values)

        return cubes


class MixinBasic(Mixin_FieldTest):
    # A set of tests that can be applied to *either* standard iris load
    # functions, for confirmation of test results, or to fast-load.
    # "Real" tests for each interface inherit this.

    def _debug(self, expected, results):
        def pcubes(name, cubes):
            print('\n\n{}:\n'.format(name), cubes)
            for i, cube in enumerate(cubes):
                print('@{}'.format(i))
                print(cube)
        pcubes('expected', expected)
        pcubes('results', results)

    def test_basic(self):
        flds = self.fields(c_t='123', cft='000', ctp='123', c_p=0)
        file = self.save_fieldcubes(flds)
        results = self.load_function(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_phenomena(self):
        flds = self.fields(c_t='1122', phn='0101')
        file = self.save_fieldcubes(flds)
        results = self.load_function(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_FAIL_phenomena_nostash(self):
        # If we remove the 'STASH' attributes, certain phenomena can still be
        # successfully encoded+decoded by standard load using LBFC values.
        # Structured loading gets this wrong, because it does not use LBFC in
        # characterising phenomena.
        flds = self.fields(c_t='1122', phn='0101')
        for fld in flds:
            del fld.attributes['STASH']
        file = self.save_fieldcubes(flds)
        results = self.load_function(file)
        if self.load_type == 'iris':
            # This is what we'd LIKE to get (what iris.load gives).
            expected = CubeList(flds).merge()
        else:
            # At present, we get a cube incorrectly combined together over all
            # 4 timepoints, with the same phenomenon for all (!wrong!).
            # It's a bit tricky to arrange the existing data like that.
            # Do it by hacking the time values to allow merge, and then fixing
            # up the time
            old_t1, old_t2 = (fld.coord('time').points[0]
                              for fld in (flds[0], flds[2]))
            for i_fld, fld in enumerate(flds):
                # Hack the phenomena to all look like the first one.
                fld.rename('air_temperature')
                fld.units = 'K'
                # Hack the time points so the 4 cube can merge into one.
                fld.coord('time').points = [old_t1 + i_fld]
            one_cube = CubeList(flds).merge_cube()
            # Replace time dim with an anonymous dim.
            co_t_fake = one_cube.coord('time')
            one_cube.remove_coord(co_t_fake)
            # Reconstruct + add back the expected auxiliary time coord.
            co_t_new = AuxCoord([old_t1, old_t1, old_t2, old_t2],
                                standard_name='time', units=co_t_fake.units)
            one_cube.add_aux_coord(co_t_new, 0)
            expected = [one_cube]
        self.assertEqual(results, expected)

    def test_cross_file_concatenate(self):
        per_file_cubes = [self.fields(c_t=times)
                          for times in ('12', '34')]
        files = [self.save_fieldcubes(flds)
                 for flds in per_file_cubes]
        results = self.load_function(files)
        expected = CubeList(fld_cube
                            for cubes in per_file_cubes
                            for fld_cube in cubes).merge()
        self.assertEqual(results, expected)

    def test_FAIL_scalar_vector_concatenate(self):
        # We'd really like to fix this one...
        single_timepoint_fld, = self.fields(c_t='1')
        multi_timepoint_flds = self.fields(c_t='23')
        file_single = self.save_fieldcubes([single_timepoint_fld],
                                           basename='single')
        file_multi = self.save_fieldcubes(multi_timepoint_flds,
                                          basename='multi')

        results = self.load_function((file_single, file_multi))
        if self.load_type == 'iris':
            # This is what we'd LIKE to get (what iris.load gives).
            expected = CubeList(multi_timepoint_flds +
                                [single_timepoint_fld]).merge()
        else:
            # NOTE: in this case, we need to sort the results to ensure a
            # repeatable ordering, because ??somehow?? the random temporary
            # directory name affects the ordering of the cubes in the result !
            results = CubeList(sorted(results,
                                      key=lambda cube: cube.shape))
            # This is what we ACTUALLY get at present.
            # It can't combine the scalar and vector time coords.
            expected = CubeList([CubeList(multi_timepoint_flds).merge_cube(),
                                 single_timepoint_fld])

        self.assertEqual(results, expected)


class TestBasicIris(MixinBasic, tests.IrisTest):
    load_type = 'iris'


class TestBasicFast(MixinBasic, tests.IrisTest):
    load_type = 'fast'


if __name__ == '__main__':
    tests.main()
