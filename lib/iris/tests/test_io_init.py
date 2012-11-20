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
Test the io/__init__.py module.

"""
# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import os
import unittest

import iris.fileformats as iff
import iris.io


class TestDecodeUri(unittest.TestCase):
    def test_decode_uri(self):
        tests = {
            '/data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp': (
                'file', '/data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp'
            ),
            'C:\data\local\someDir\PP\COLPEX\COLPEX_16a_pj001.pp': (
                'file', 'C:\data\local\someDir\PP\COLPEX\COLPEX_16a_pj001.pp'
            ),
            'file:///data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp': (
                'file', '///data/local/someDir/PP/COLPEX/COLPEX_16a_pj001.pp'
            ),
            'http://www.somehost.com:8080/resource/thing.grib': (
                'http', '//www.somehost.com:8080/resource/thing.grib'
            ),
        }
        for uri, pair in tests.items():
            self.assertEqual(pair, iris.io.decode_uri(uri))


@iris.tests.skip_data
class TestFileFormatPicker(tests.IrisTest):
    def test_known_formats(self):
        a = str(iff.FORMAT_AGENT)
        self.assertString(a, tests.get_result_path(('file_load', 'known_loaders.txt')))


    def test_format_picker(self):
        # ways to test the format picker = list of (format-name, file-spec)
        test_specs = [
            ('NetCDF',
                ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc']),
            ('NetCDF 64 bit offset format',
                ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc.k2']),
            ('NetCDF_v4',
                ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc4.k3']),
            ('NetCDF_v4',
                ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc4.k4']),
            ('UM Fields file (FF) pre v3.1',
                ['FF', 'n48_multi_field']),
            ('GRIB',
                ['GRIB', 'grib1_second_order_packing', 'GRIB_00008_FRANX01']),
            ('GRIB',
                ['GRIB', 'jpeg2000', 'file.grib2']),
            ('UM Post Processing file (PP)',
                ['PP', 'simple_pp', 'global.pp']),
#            ('BUFR',
#                ['BUFR', 'mss', 'BUFR_Samples', 
#                 'JUPV78_EGRR_121200_00002501']),
            ('NIMROD',
                ['NIMROD', 'uk2km', 'WO0000000003452',
                 '201007020900_u1096_ng_ey00_visibility0180_screen_2km']),
#            ('NAME',
#                ['NAME', '20100509_18Z_variablesource_12Z_VAAC', 
#                 'Fields_grid1_201005110000.txt']),
        ]
        
        # test that each filespec is identified as the expected format
        for (expected_format_name, file_spec) in test_specs:
            test_path = tests.get_data_path(file_spec)
            with open(test_path, 'r') as test_file:
                a = iff.FORMAT_AGENT.get_spec(test_path, test_file)
                self.assertEqual(a.name, expected_format_name)


@iris.tests.skip_data
class TestFileExceptions(tests.IrisTest):
    def test_pp_little_endian(self):
        filename = tests.get_data_path(('PP', 'aPPglob1', 'global_little_endian.pp'))
        self.assertRaises(ValueError, iris.load_cube, filename)


if __name__ == '__main__':
    tests.main()
