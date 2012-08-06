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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import datetime

import numpy

import iris
import iris.cube
import iris.coords as coords
import iris.coord_systems as coord_systems
import iris.tests.stock
import iris.unit


def old_style_coords_TEST_COMPAT(c, *coord_names):
    for coord_name in coord_names:
        c.coord(coord_name)._TEST_COMPAT_override_axis = coord_name
        if coord_name == "pressure":
            c.coord(coord_name)._TEST_COMPAT_override_axis = "z"
        if coord_name in ["source", "pressure", "time", "forecast_period"]:
            c.coord(coord_name)._TEST_COMPAT_definitive = False


class TestXML(tests.IrisTest):

    @iris.tests.skip_data
    def test_pp(self):
        # Test xml output of a cube loaded from pp.
        cubes = iris.cube.CubeList([iris.tests.stock.simple_pp()])
        old_style_coords_TEST_COMPAT(cubes[0], "forecast_period", "source")
        cubes[0].coord("forecast_period")._TEST_COMPAT_definitive = True

        self.assertCML(cubes, ('xml', 'pp.cml'))

    def test_handmade(self):
        # Test xml output of a handmade cube.        
        data = numpy.array( [ [1, 2, 3, 4, 5], 
                              [2, 3, 4, 5, 6],
                              [3, 4, 5, 6, 7],
                              [4, 5, 6, 7, 8],
                              [5, 6, 7, 8, 9] ], dtype=numpy.int32)
        cubes = []

        # Different types of test
        for ll_dtype in [numpy.float32, numpy.int32]:
            for rotated in [False, True]:
                for forecast_or_time_mean in ["forecast", "time_mean"]:
                    for TEST_COMPAT_i in xrange(2): # TODO: remove with TEST_COMPAT purge - 
                                                    # adds two copies of each cube to cube list
                                                    # in line with redundant data first option
                        cube = iris.cube.Cube(data)

                        cube.attributes['my_attribute'] = 'foobar'
                        
                        if rotated == False:
                            pole_pos = coord_systems.GeoPosition(90, 0)
                        else:
                            pole_pos = coord_systems.GeoPosition(30, 150)

                        lonlat_cs = coord_systems.LatLonCS("datum?", "prime_meridian?", pole_pos, "reference_longitude?")
                        cube.add_dim_coord(coords.DimCoord(numpy.array([-180, -90, 0, 90, 180], dtype=ll_dtype), 
                                           'longitude', units='degrees', coord_system=lonlat_cs), 1)
                        cube.add_dim_coord(coords.DimCoord(numpy.array([-90, -45, 0, 45, 90], dtype=ll_dtype), 
                                           'latitude', units='degrees', coord_system=lonlat_cs), 0)
                        
                        if ll_dtype == numpy.int32:
                            cube.coord("latitude")._TEST_COMPAT_force_explicit = True
                            cube.coord("longitude")._TEST_COMPAT_force_explicit = True
                        
                        # height
                        cube.add_aux_coord(coords.AuxCoord(numpy.array([1000], dtype=numpy.int32), 
                                                           long_name='pressure', units='Pa'))
                        
                        # phenom
                        cube.rename("temperature")
                        cube.units = "K"
                        
                        # source
                        cube.add_aux_coord(coords.AuxCoord(points=["itbb"], long_name='source', units="no_unit"))
                        
                        # forecast dates
                        if forecast_or_time_mean == "forecast":
                            unit = iris.unit.Unit('hours since epoch', calendar=iris.unit.CALENDAR_GREGORIAN)
                            dt = datetime.datetime(2010, 12, 31, 12, 0)
                            cube.add_aux_coord(coords.AuxCoord(numpy.array([6], dtype=numpy.int32), 
                                                               standard_name='forecast_period', units='hours'))
                            cube.add_aux_coord(coords.AuxCoord(numpy.array([unit.date2num(dt)], dtype=numpy.float64), 
                                                               standard_name='time', units=unit))
                            
                        # time mean dates
                        if forecast_or_time_mean == "time_mean":
                            unit = iris.unit.Unit('hours since epoch', calendar=iris.unit.CALENDAR_GREGORIAN)
                            dt1 = datetime.datetime(2010, 12, 31, 6, 0)
                            dt2 = datetime.datetime(2010, 12, 31, 12, 0)
                            dt_mid = datetime.datetime(2010, 12, 31, 9, 0)
                            cube.add_aux_coord(coords.AuxCoord(numpy.array([6], dtype=numpy.int32), 
                                                               standard_name='forecast_period', units='hours'))
                            cube.add_aux_coord(coords.AuxCoord(numpy.array(unit.date2num(dt_mid), dtype=numpy.float64),
                                                               standard_name='time', units=unit, 
                                                               bounds=numpy.array([unit.date2num(dt1), unit.date2num(dt2)], 
                                                                                  dtype=numpy.float64)))
                            cube.add_cell_method(coords.CellMethod('mean', cube.coord('forecast_period')))
                        
                        old_style_coords_TEST_COMPAT(cube, "forecast_period", "source", "time", "pressure")
                        
                        cube.coord("latitude")._TEST_COMPAT_definitive = False
                        cube.coord("longitude")._TEST_COMPAT_definitive = False
                        
                        cubes.append(cube)
        
        # Now we've made all sorts of cube, check the xml...
        self.assertCML(cubes, ('xml', 'handmade.cml'))


if __name__ == "__main__":
    tests.main()
