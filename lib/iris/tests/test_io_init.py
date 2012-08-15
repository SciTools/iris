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

import iris.fileformats
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
        a = str(iris.fileformats.FORMAT_AGENT)
        self.assertString(a, tests.get_result_path(('file_load', 'known_loaders.txt')))


    def test_format_picker(self):
        fspecs = [
                  ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc'], # NetCDF
                  ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc.k2'], # NetCDF 64-bit offset
                  ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc4.k3'], # NetCDF - 4 
                  ['NetCDF', 'global', 'xyt', 'SMALL_total_column_co2.nc4.k4'], # NetCDF - 4 "classic model"
                  ['ssps', 'qtgl.ssps_006'], # UM FF
                  ['GRIB', 'grib1_second_order_packing', 'GRIB_00008_FRANX01'], # GRIB1
                  ['GRIB', 'jpeg2000', 'file.grib2'], # GRIB2
                  ['PP', 'uk4', 'uk4par09.pp'], # PP
#                  ['BUFR', 'mss', 'BUFR_Samples', 'JUPV78_EGRR_121200_00002501'], # BUFFR
                  ['NIMROD', 'uk2km', 'WO0000000003452', '201007020900_u1096_ng_ey00_visibility0180_screen_2km'], # nimrod 
#                  ['NAME', '20100509_18Z_variablesource_12Z_VAAC', 'Fields_grid1_201005110000.txt'], # NAME
              ]
        
        result = []
        for spec in fspecs:
            relpath = os.path.join(*spec)
            actpath = tests.get_data_path(spec)
            a = iris.fileformats.FORMAT_AGENT.get_spec(actpath, open(actpath, 'r'))
            result.append('%s - %s' % (a.name, relpath))
            
        self.assertString('\n'.join(result), tests.get_result_path(('file_load', 'format_associations.txt')))


@iris.tests.skip_data
class TestFileExceptions(tests.IrisTest):
    def test_pp_little_endian(self):
        filename = tests.get_data_path(('PP', 'aPPglob1', 'global_little_endian.pp'))
        self.assertRaises(ValueError, iris.load_strict, filename)


if __name__ == '__main__':
    tests.main()
