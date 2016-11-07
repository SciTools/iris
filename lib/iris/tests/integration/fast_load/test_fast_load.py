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
from iris.coords import DimCoord
from iris.coord_systems import GeogCS
from iris.cube import Cube, CubeList
from iris.fileformats.pp import EARTH_RADIUS, STASH

# from iris import load as fast_load
from iris.fileformats.um import fast_load


class Mixin_FieldTest(object):
    # A mixin for tests making temporary PP files for fast-load testing.

    def setUp(self):
        # Create a private temporary directory.
        self.temp_dir_path = tempfile.mkdtemp()
        # Note: these are used to keep the files in a definite order,
        # otherwise random filenames --> random load results !!
        self.tempfile_count = 0
        self.tempfile_prefix_fmt = '_{:06d}_'

    def tearDown(self):
        # Delete temporary directory.
        shutil.rmtree(self.temp_dir_path)

    def temp_filepath(self, suffix='', prefix='A'):
        self.tempfile_count += 1
        standard_prefix = self.tempfile_prefix_fmt.format(self.tempfile_count)
        prefix = prefix + standard_prefix
        file_path = tempfile.mktemp(suffix=suffix, prefix=prefix,
                                    dir=self.temp_dir_path)
        return file_path

    # Reference values for making coordinate contents.
    time_unit = 'hours since 1970-01-01'
    period_unit = 'hours'
    time_values = 24.0 * np.arange(5)
    height_values = [100.0, 200.0, 300.0, 400.0]
    pressure_values = [300.0, 500.0, 850.0, 1000.0]
    # NOTE: in order to write/readback as identical, these test phenomena
    # settings also provide the canonical unit and a matching STASH attribute.
    # These could in principle be looked up, but it's a bit awkward.
    phenomena = [('air_temperature', 'K'),
                 ('air_density', 'kg m-3'),
                 ('air_pressure', 'm s-1'),
                 ('wind_speed', 'm s-1'),
                 ]
    phenomena = [
        ('air_potential_temperature', 'K', 'm01s00i004'),
        ('x_wind', 'm s-1', 'm01s00i002'),
        ('y_wind', 'm s-1', 'm01s00i003'),
        ('specific_humidity', 'kg kg-1', 'm01s00i010'),
        ]

    def fields(self, c_t=None, cft=None, ctp=None,
               c_h=None, c_p=None, mmm=None, phn=0):
        # Return a list of 2d cubes representing raw PPFields, from args
        # specifying sequences of (scalar) coordinate values.
        # TODO? : add bounds somehow ?
        #
        # Arguments 'c<xx>' are either a single int value, making a scalar
        # coord, or a string of characters 0-9 (value) or '-' (missing).
        #
        # Argument 'mmm' denotes existence (or not) of a cell method of type
        # 'average' or 'min' or 'max' (values '012' respectively), applying to
        # the time values -- ultimately, this controls LBTIM.
        #
        # Argument 'c_h' and 'c_p' represent height or pressure values, so
        # ought to be mutually exclusive -- these control LBVC.
        #
        # Argument 'phn' indexes phenomenon types.
        def arglen(arg):
            if arg is None:
                result = 0
            elif isinstance(arg, six.string_types):
                result = len(arg)
            else:
                result = 1
            return result

        n_flds = max(arglen(x)
                     for x in (c_t, cft, ctp, c_h, c_p, mmm))

        def arg_inds(arg):
            # Return an argument decoded as an array of n_flds integers.
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
            return inds

        def arg_vals(arg, vals):
            return [None if ind is None else vals[int(ind)]
                    for ind in arg_inds(arg)]

        def arg_coords(arg, name, unit, vals=None):
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

        ny, nx = 3, 5
        data = np.arange(n_flds * ny * nx, dtype=np.float32)
        data = data.reshape((n_flds, ny, nx))

        # Make basic anonymous test cubes.
        cubes = [Cube(data[i]) for i in range(n_flds)]

        # Apply phenomena definitions.
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

        # Add multiple scalar coordinates as requested.
        def add_arg_coords(arg, name, unit, vals=None):
            coords = arg_coords(arg, name, unit, vals)
            for cube, coord in zip(cubes, coords):
                if coord:
                    cube.add_aux_coord(coord)

# ? DON'T have a model_level_number coord ?
#        add_arg_coords(np.arange(1, n_flds+1), 'model_level_number', '1')

        add_arg_coords(c_t, 'time', self.time_unit, self.time_values)
        add_arg_coords(cft, 'forecast_reference_time', self.time_unit)
        add_arg_coords(ctp, 'forecast_period', 'hours', self.time_values)
        add_arg_coords(c_h, 'height', 'm', self.height_values)
        add_arg_coords(c_p, 'pressure', 'hPa', self.pressure_values)
        return cubes

    def save_fieldcubes(self, cubes, basename='a'):
        file_path = self.temp_filepath(suffix='.pp', prefix=basename)
        iris.save(cubes, file_path)
        return file_path


class TestBasic(Mixin_FieldTest, tests.IrisTest):
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
        results = fast_load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_phenomena(self):
        flds = self.fields(c_t='1122', phn='0101')
        file = self.save_fieldcubes(flds)
        results = fast_load(file)
        expected = CubeList(flds).merge()
        self.assertEqual(results, expected)

    def test_cross_file_concatenate(self):
        per_file_cubes = [self.fields(c_t=times)
                          for times in ('12', '34')]
        files = [self.save_fieldcubes(flds)
                 for flds in per_file_cubes]
        results = iris.load(files)
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
#        print('FILENAMES:', file_single, file_multi,
#              sorted([file_single, file_multi]))
        print('TEMPDIR LISTING: ')
        for filename in os.listdir(self.temp_dir_path):
            print('  ', filename)
        results = fast_load((file_single, file_multi))
        print('RESULT SHAPES: ', [cube.shape for cube in results])
        # This is what we'd LIKE to get (which is what iris.load gives)
        expected = CubeList(multi_timepoint_flds +
                            [single_timepoint_fld]).merge()
        # This is what we ACTUALLY get at present.
        # It can't combine the scalar and vector time coords.
        expected = CubeList([CubeList(multi_timepoint_flds).merge_cube(),
                             single_timepoint_fld])

        self.assertEqual(results, expected)


if __name__ == '__main__':
    tests.main()
