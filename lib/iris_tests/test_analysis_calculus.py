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

import unittest

import numpy

import iris
import iris.analysis.calculus
import iris.cube
import iris.coord_systems
import iris.coords
import iris.tests.stock

from iris.coords import DimCoord


class TestCubeDelta(tests.IrisTest):
    def test_invalid(self):
        cube = iris.tests.stock.realistic_4d()
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            t = iris.analysis.calculus.cube_delta(cube, 'surface_altitude')
        with self.assertRaises(iris.exceptions.CoordinateMultiDimError):
            t = iris.analysis.calculus.cube_delta(cube, 'altitude')
        with self.assertRaises(ValueError):
            t = iris.analysis.calculus.cube_delta(cube, 'forecast_period')
            

class TestDeltaAndMidpoint(tests.IrisTest):
    def _simple_filename(self, suffix):
        return tests.get_result_path(('analysis', 'delta_and_midpoint', 'simple%s.cml' % suffix))
    
    def test_simple1_delta_midpoint(self):
        a = iris.coords.DimCoord((numpy.arange(4, dtype=numpy.float32) * 90) - 180, long_name='foo', 
                                 units='degrees', circular=True)
        a._TEST_COMPAT_override_axis = 'foo'
        a._TEST_COMPAT_definitive = False
        self.assertXMLElement(a, self._simple_filename('1'))

        delta = iris.analysis.calculus._construct_delta_coord(a)
        delta._TEST_COMPAT_override_axis = 'foo'
        self.assertXMLElement(delta, self._simple_filename('1_delta'))
        
        midpoint = iris.analysis.calculus._construct_midpoint_coord(a)
        midpoint._TEST_COMPAT_override_axis = 'foo'
        midpoint._TEST_COMPAT_definitive = False
        self.assertXMLElement(midpoint, self._simple_filename('1_midpoint'))
        
    def test_simple2_delta_midpoint(self):
        a = iris.coords.DimCoord((numpy.arange(4, dtype=numpy.float32) * -90) + 180, long_name='foo', 
                                 units='degrees', circular=True)
        a._TEST_COMPAT_override_axis = 'foo'
        a._TEST_COMPAT_definitive = False
        self.assertXMLElement(a, self._simple_filename('2'))

        delta = iris.analysis.calculus._construct_delta_coord(a)
        delta._TEST_COMPAT_override_axis = 'foo'
        self.assertXMLElement(delta, self._simple_filename('2_delta'))
        
        midpoint = iris.analysis.calculus._construct_midpoint_coord(a)
        midpoint._TEST_COMPAT_override_axis = 'foo'
        midpoint._TEST_COMPAT_definitive = False
        self.assertXMLElement(midpoint, self._simple_filename('2_midpoint'))

    def test_simple3_delta_midpoint(self):
        a = iris.coords.DimCoord((numpy.arange(4, dtype=numpy.float32) * 90) - 180, long_name='foo', 
                                 units='degrees', circular=True)
        a.guess_bounds(0.5)    
        a._TEST_COMPAT_override_axis = 'foo'
        a._TEST_COMPAT_definitive = False
        self.assertXMLElement(a, self._simple_filename('3'))
        
        delta = iris.analysis.calculus._construct_delta_coord(a)
        delta._TEST_COMPAT_override_axis = 'foo'
        self.assertXMLElement(delta, self._simple_filename('3_delta'))
        
        midpoint = iris.analysis.calculus._construct_midpoint_coord(a)
        midpoint._TEST_COMPAT_override_axis = 'foo'
        midpoint._TEST_COMPAT_definitive = False
        self.assertXMLElement(midpoint, self._simple_filename('3_midpoint'))
        
    def test_simple4_delta_midpoint(self):
        a = iris.coords.AuxCoord(numpy.arange(4, dtype=numpy.float32) * 90 - 180, long_name='foo', units='degrees')
        a.guess_bounds()
        a._TEST_COMPAT_definitive = False
        a._TEST_COMPAT_override_axis = 'foo'
        b = a.copy()
        self.assertXMLElement(b, self._simple_filename('4'))
        
        delta = iris.analysis.calculus._construct_delta_coord(b)
        delta._TEST_COMPAT_override_axis = 'foo'
        self.assertXMLElement(delta, self._simple_filename('4_delta'))
        
        midpoint = iris.analysis.calculus._construct_midpoint_coord(b)
        midpoint._TEST_COMPAT_override_axis = 'foo'
        self.assertXMLElement(midpoint, self._simple_filename('4_midpoint'))
        
    def test_simple5_not_degrees_delta_midpoint(self):
        # Not sure it makes sense to have a circular coordinate which does not have a modulus but test it anyway.
        a = iris.coords.DimCoord(numpy.arange(4, dtype=numpy.float32) * 90 - 180, 
                                 long_name='foo', units='meter', circular=True)
        a._TEST_COMPAT_override_axis = 'foo'
        self.assertXMLElement(a, self._simple_filename('5'))
        
        delta = iris.analysis.calculus._construct_delta_coord(a)
        delta._TEST_COMPAT_override_axis = 'foo'
        delta._TEST_COMPAT_definitive = True
        self.assertXMLElement(delta, self._simple_filename('5_delta'))
        
        midpoints = iris.analysis.calculus._construct_midpoint_coord(a)
        midpoints._TEST_COMPAT_override_axis = 'foo'
        midpoints._TEST_COMPAT_definitive = True
        self.assertXMLElement(midpoints, self._simple_filename('5_midpoint'))
        
    def test_simple6_delta_midpoint(self):
        a = iris.coords.DimCoord(numpy.arange(5, dtype=numpy.float32), long_name='foo', 
                                 units='count', circular=True)
        midpoints = iris.analysis.calculus._construct_midpoint_coord(a)
        midpoints._TEST_COMPAT_override_axis = 'foo'        
        self.assertXMLElement(midpoints, self._simple_filename('6'))
    
    def test_singular_delta(self):
        # Test single valued coordinate mid-points when circular
        lon = iris.coords.DimCoord(numpy.float32(-180.), 'latitude', units='degrees', circular=True)
        
        r_expl = iris.analysis.calculus._construct_delta_coord(lon)
        r_expl._TEST_COMPAT_force_explicit = True
        r_expl._TEST_COMPAT_override_axis = 'x'
        self.assertXMLElement(r_expl, ('analysis', 'delta_and_midpoint', 'delta_one_element_explicit.xml'))
        
        # Test single valued coordinate mid-points when not circular
        lon.circular = False
        with self.assertRaises(ValueError):
            iris.analysis.calculus._construct_delta_coord(lon)  
        
    def test_singular_midpoint(self):
        # Test single valued coordinate mid-points when circular
        lon = iris.coords.DimCoord(numpy.float32(-180.), 'latitude',  units='degrees', circular=True)
        
        r_expl = iris.analysis.calculus._construct_midpoint_coord(lon)
        r_expl._TEST_COMPAT_override_axis = 'x'
        self.assertXMLElement(r_expl, ('analysis', 'delta_and_midpoint', 'midpoint_one_element_explicit.xml'))
        
        # Test single valued coordinate mid-points when not circular
        lon.circular = False
        with self.assertRaises(ValueError):
            iris.analysis.calculus._construct_midpoint_coord(lon)


class TestCalculusSimple3(tests.IrisTest):
    
    def setUp(self):
        data = numpy.arange(2500, dtype=numpy.float32).reshape(50, 50)
        cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")
        
        self.lonlat_cs = iris.coord_systems.LatLonCS(iris.coord_systems.SpheroidDatum(), iris.coord_systems.PrimeMeridian(), iris.coord_systems.GeoPosition(90, 0), "reference_longitude?")
        cube.add_dim_coord(DimCoord(numpy.arange(50, dtype=numpy.float32) * 4.5 -180, 'longitude', units='degrees', coord_system=self.lonlat_cs), 0)
        cube.add_dim_coord(DimCoord(numpy.arange(50, dtype=numpy.float32) * 4.5 -90,  'latitude', units='degrees', coord_system=self.lonlat_cs), 1)
    
        self.cube = cube  
        
    def test_diff_wrt_lon(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'longitude')
        
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade2_wrt_lon.cml'))
        
    def test_diff_wrt_lat(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'latitude')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade2_wrt_lat.cml'))
        

class TestCalculusSimple2(tests.IrisTest):
    
    def setUp(self):
        data = numpy.array( [[1, 2, 3, 4, 5],
                             [2, 3, 4, 5, 6],
                             [3, 4, 5, 6, 7],
                             [4, 5, 6, 7, 9]], dtype=numpy.float32)
        cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")
        
        self.lonlat_cs = iris.coord_systems.LatLonCS(iris.coord_systems.SpheroidDatum(), iris.coord_systems.PrimeMeridian(), iris.coord_systems.GeoPosition(90, 0), "reference_longitude?")
        cube.add_dim_coord(DimCoord(numpy.arange(4, dtype=numpy.float32) * 90 -180, 'longitude', units='degrees', circular=True, coord_system=self.lonlat_cs), 0)
        cube.add_dim_coord(DimCoord(numpy.arange(5, dtype=numpy.float32) * 45 -90, 'latitude', units='degrees', coord_system=self.lonlat_cs), 1)
    
        cube.add_aux_coord(DimCoord(numpy.arange(4, dtype=numpy.float32), long_name='x', units='count', circular=True), 0)
        cube.add_aux_coord(DimCoord(numpy.arange(5, dtype=numpy.float32), long_name='y', units='count'), 1)
        
        self.cube = cube  
        
    def test_diff_wrt_x(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'x')
        t.coord("x")._TEST_COMPAT_definitive = True
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade_wrt_x.cml'))
        
    def test_diff_wrt_y(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'y')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade_wrt_y.cml'))
        
    def test_diff_wrt_lon(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'longitude')
        t.coord("x")._TEST_COMPAT_definitive = True
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade_wrt_lon.cml'))
        
    def test_diff_wrt_lat(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'latitude')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade_wrt_lat.cml'))
                
    def test_delta_wrt_x(self):
        t = iris.analysis.calculus.cube_delta(self.cube, 'x')
        t.coord("x")._TEST_COMPAT_definitive = True
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'delta_handmade_wrt_x.cml'))
        
    def test_delta_wrt_y(self):
        t = iris.analysis.calculus.cube_delta(self.cube, 'y')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'delta_handmade_wrt_y.cml'))
        
    def test_delta_wrt_lon(self):
        t = iris.analysis.calculus.cube_delta(self.cube, 'longitude')
        t.coord("x")._TEST_COMPAT_definitive = True
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'delta_handmade_wrt_lon.cml'))
        
    def test_delta_wrt_lat(self):
        t = iris.analysis.calculus.cube_delta(self.cube, 'latitude')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'delta_handmade_wrt_lat.cml'))


class TestCalculusSimple1(tests.IrisTest):
    
    def setUp(self):
        data = numpy.array( [ [1, 2, 3, 4, 5],
                                   [2, 3, 4, 5, 6],
                                   [3, 4, 5, 6, 7],
                                   [4, 5, 6, 7, 8],
                                   [5, 6, 7, 8, 10] ], dtype=numpy.float32)
        cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")
        
        cube.add_dim_coord(DimCoord(numpy.arange(5, dtype=numpy.float32), long_name='x', units='count'), 0)
        cube.add_dim_coord(DimCoord(numpy.arange(5, dtype=numpy.float32), long_name='y', units='count'), 1)
    
        self.cube = cube  
        
    def test_diff_wrt_x(self):
        t = iris.analysis.calculus.differentiate(self.cube, 'x')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'handmade_simple_wrt_x.cml'))
        
    def test_delta_wrt_x(self):
        t = iris.analysis.calculus.cube_delta(self.cube, 'x')
        self.assertCMLApproxData(t, ('analysis', 'calculus', 'delta_handmade_simple_wrt_x.cml'))
        

def build_cube(data, spherical=False):
    """
    Create a cube suitable for testing.
    
    """
    cube = iris.cube.Cube(data, standard_name="x_wind", units="km/h")

    nx = data.shape[-1]
    ny = data.shape[-2]
    nz = data.shape[-3] if data.ndim > 2 else None

    dimx = data.ndim - 1     
    dimy = data.ndim - 2     
    dimz = data.ndim - 3  if data.ndim > 2 else None
   
    if spherical:
        hcs = iris.coord_systems.LatLonCS( iris.coord_systems.SpheroidDatum(label="Tiny Earth", semi_major_axis=6321, flattening=0.0, units="m"),
                                        iris.coord_systems.PrimeMeridian(), iris.coord_systems.GeoPosition(90, 0), "reference_longitude?")
        cube.add_dim_coord(DimCoord(numpy.arange(-180, 180, 360./nx, dtype=numpy.float32), 'longitude', units='degrees', coord_system=hcs, circular=True), dimx) 
        cube.add_dim_coord(DimCoord(numpy.arange(-90, 90, 180./ny, dtype=numpy.float32), 'latitude', units='degrees',coord_system=hcs), dimy)

    else:
        hcs = iris.coord_systems.HorizontalCS("Cartesian Datum?")
        cube.add_dim_coord(DimCoord(numpy.arange(nx, dtype=numpy.float32) * 2.21 + 2, long_name='x', units='meters', coord_system=hcs), dimx) 
        cube.add_dim_coord(DimCoord(numpy.arange(ny, dtype=numpy.float32) * 25 -50, long_name='y', units='meters', coord_system=hcs), dimy)

    if nz is None:
        cube.add_aux_coord(DimCoord(numpy.array([10], dtype=numpy.float32), long_name='z', units='meters'))
    else:
        cube.add_dim_coord(DimCoord(numpy.arange(nz, dtype=numpy.float32) * 2, long_name='z', units='meters'), dimz)
    
    return cube    


class TestCalculusWKnownSolutions(tests.IrisTest):
    
    def get_coord_pts(self, cube):
        """return (x_pts, x_ones, y_pts, y_ones, z_pts, z_ones) for the given cube."""
        x = cube.coord(axis='X')
        y = cube.coord(axis='Y')
        z = cube.coord(axis='Z')

        if z and z.shape[0] > 1:
            x_shp = (1, 1, x.shape[0])
            y_shp = (1, y.shape[0], 1)
            z_shp = (z.shape[0], 1, 1)
        else:
            x_shp = (1, x.shape[0])
            y_shp = (y.shape[0], 1)
            z_shp = None
        
        x_pts = x.points.reshape(x_shp)
        y_pts = y.points.reshape(y_shp)
        
        x_ones = numpy.ones(x_shp)
        y_ones = numpy.ones(y_shp)
        
        if z_shp:
            z_pts = z.points.reshape(z_shp)
            z_ones = numpy.ones(z_shp)
        else:
            z_pts = None
            z_ones = None
            
        return (x_pts, x_ones, y_pts, y_ones, z_pts, z_ones)  
    
    def test_contrived_differential1(self):
        # testing :
        # F = ( cos(lat) cos(lon) )
        # dF/dLon = - sin(lon) cos(lat)     (and to simplify /cos(lat) )
        cube = build_cube(numpy.empty((30, 60)), spherical=True)

        x = cube.coord('longitude')
        y = cube.coord('latitude')
        y_dim = cube.coord_dims(y)[0]

        cos_x_pts = x.cos().points.reshape(1, x.shape[0])
        cos_y_pts = y.cos().points.reshape(y.shape[0], 1)
    
        cube.data = cos_y_pts * cos_x_pts
    
        lon_coord = x.unit_converted('radians')
        cos_lat_coord = y.cos()
        
        temp = iris.analysis.calculus.differentiate(cube, lon_coord)
        df_dlon = iris.analysis.maths.divide(temp, cos_lat_coord, y_dim)

        x = df_dlon.coord('longitude')
        y = df_dlon.coord('latitude')
        
        sin_x_pts = x.sin().points.reshape(1, x.shape[0])
        y_ones = numpy.ones((y.shape[0] , 1))
        
        data = - sin_x_pts * y_ones
        result = df_dlon.copy(data=data)
        
        numpy.testing.assert_array_almost_equal(result.data, df_dlon.data, decimal=3)

    def test_contrived_differential2(self):        
        # testing :
        # w = y^2
        # dw_dy = 2*y
        cube = build_cube(numpy.empty((10, 30, 60)), spherical=False)
        
        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(cube)
        
        w = cube.copy(data=z_ones * x_ones * pow(y_pts, 2.))
    
        r = iris.analysis.calculus.differentiate(w, 'y')

        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(r)
        result = r.copy(data = y_pts * 2. * x_ones * z_ones)
        
        numpy.testing.assert_array_almost_equal(result.data, r.data, decimal=6)

    def test_contrived_non_sphrical_curl1(self):
        # testing :
        # F(x, y, z) = (y, 0, 0)
        # curl( F(x, y, z) ) = (0, 0, -1)
        
        cube = build_cube(numpy.empty((25, 50)), spherical=False)
        
        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(cube)
        
        u = cube.copy(data=x_ones * y_pts)
        u.rename("u_wind")
        v = cube.copy(data=u.data * 0)
        v.rename("v_wind")
        
        r = iris.analysis.calculus.curl(u, v)
    
        # Curl returns None when there is no components of Curl
        self.assertEqual(r[0], None)
        self.assertEqual(r[1], None)
        self.assertCML(r[2], ('analysis', 'calculus', 'grad_contrived_non_spherical1.cml'))
        
    def test_contrived_non_sphrical_curl2(self):
        # testing :
        # F(x, y, z) = (z^3, x+2, y^2)
        # curl( F(x, y, z) ) = (2y, 3z^2, 1)

        cube = build_cube(numpy.empty((10, 25, 50)), spherical=False)
        
        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(cube)
        
        u = cube.copy(data=pow(z_pts, 3) * x_ones * y_ones)
        v = cube.copy(data=z_ones * (x_pts + 2.) * y_ones)
        w = cube.copy(data=z_ones * x_ones * pow(y_pts, 2.))
        u.rename('u_wind')
        v.rename('v_wind')
        w.rename('w_wind')
        
        r = iris.analysis.calculus.curl(u, v, w)

        
        # TODO #235 When regridding is not nearest neighbour: the commented out code could be made to work
        # r[0].data should now be tending towards result.data as the resolution of the grid gets higher. 
#        result = r[0].copy(data=True)
#        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(result)
#        result.data = y_pts * 2. * x_ones * z_ones
#        print repr(r[0].data[0:1, 0:5, 0:25:5])
#        print repr(result.data[0:1, 0:5, 0:25:5])
#        numpy.testing.assert_array_almost_equal(result.data, r[0].data, decimal=2)
#        
#        result = r[1].copy(data=True)
#        x_pts, x_ones, y_pts, y_ones, z_pts, z_ones = self.get_coord_pts(result)
#        result.data = pow(z_pts, 2) * x_ones * y_ones
#        numpy.testing.assert_array_almost_equal(result.data, r[1].data, decimal=6)

        result = r[2].copy()
        result.data = result.data * 0  + 1
        numpy.testing.assert_array_almost_equal(result.data, r[2].data, decimal=4)
 
        self.assertCML(r, ('analysis', 'calculus', 'curl_contrived_cartesian2.cml'), checksum=False)

    def test_contrived_sphrical_curl1(self):
        # testing:
        # F(lon, lat, r) = (- r sin(lon), -r cos(lon) sin(lat), 0)
        # curl( F(x, y, z) ) = (0, 0, 0)
        cube = build_cube(numpy.empty((30, 60)), spherical=True)
        radius = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

        x = cube.coord('longitude')
        y = cube.coord('latitude')

        cos_x_pts = x.cos().points.reshape(1, x.shape[0])
        sin_x_pts = x.sin().points.reshape(1, x.shape[0])
        cos_y_pts = y.cos().points.reshape(y.shape[0], 1)
        sin_y_pts = y.sin().points.reshape(y.shape[0], 1)
        y_ones = numpy.ones((cube.shape[0], 1))
    
        u = cube.copy(data=-sin_x_pts * y_ones * radius)
        v = cube.copy(data=-cos_x_pts * sin_y_pts * radius)
        u.rename('u_wind')
        v.rename('v_wind')
        
#        lon_coord = x.unit_converted('radians')
#        cos_lat_coord = y.cos()
        
        r = iris.analysis.calculus.curl(u, v)[2]
    
        result = r.copy(data=r.data * 0)
        
        numpy.testing.assert_array_almost_equal(result.data[5:-5], r.data[5:-5], decimal=1)
        
        r.coord("latitude")._TEST_COMPAT_force_explicit = True
        r.coord("longitude")._TEST_COMPAT_force_explicit = True
        
        self.assertCML(r, ('analysis', 'calculus', 'grad_contrived1.cml'), checksum=False)

    def test_contrived_sphrical_curl2(self):
        # testing:
        # F(lon, lat, r) = (r sin(lat) cos(lon), -r sin(lon), 0)
        # curl( F(x, y, z) ) = (0, 0, -2 cos(lon) cos(lat) )
        cube = build_cube(numpy.empty((70, 150)), spherical=True)
        radius = iris.analysis.cartography.DEFAULT_SPHERICAL_EARTH_RADIUS

        x = cube.coord('longitude')
        y = cube.coord('latitude')

        cos_x_pts = x.cos().points.reshape(1, x.shape[0])
        sin_x_pts = x.sin().points.reshape(1, x.shape[0])
        cos_y_pts = y.cos().points.reshape(y.shape[0], 1)
        sin_y_pts = y.sin().points.reshape(y.shape[0], 1)
        y_ones = numpy.ones((cube.shape[0] , 1))
    
        u = cube.copy(data=sin_y_pts * cos_x_pts * radius)
        v = cube.copy(data=-sin_x_pts * y_ones * radius)
        u.rename('u_wind')
        v.rename('v_wind')
    
        lon_coord = x.unit_converted('radians')
        cos_lat_coord = y.cos()
        
        r = iris.analysis.calculus.curl(u, v)[2]
    
        x = r.coord('longitude')
        y = r.coord('latitude')
        
        cos_x_pts = x.cos().points.reshape(1, x.shape[0])
        cos_y_pts = y.cos().points.reshape(y.shape[0], 1)
        
        result = r.copy(data=2*cos_x_pts*cos_y_pts)
        
        numpy.testing.assert_array_almost_equal(result.data[30:-30, :], r.data[30:-30, :], decimal=1)

        r.coord("latitude")._TEST_COMPAT_force_explicit = True
        r.coord("longitude")._TEST_COMPAT_force_explicit = True
        
        self.assertCML(r, ('analysis', 'calculus', 'grad_contrived2.cml'), checksum=False)


class TestCurlInterface(tests.IrisTest):
    def test_non_conformed(self):
        u = build_cube(numpy.empty((50, 20)), spherical=True)
        
        v = u.copy()
        y = v.coord('latitude')
        y.points += 5
        self.assertRaises(ValueError, iris.analysis.calculus.curl, u, v)
        
    def test_standard_name(self):
        nx = 20; ny = 50; nz = None;
        u = build_cube(numpy.empty((50, 20)), spherical=True)
        v = u.copy()
        w = u.copy()
        u.rename('u_wind')
        v.rename('v_wind')
        w.rename('w_wind')
        
        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v)
        self.assertEqual(r, (('u', 'v', 'w'), 'wind'))

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w)
        self.assertEqual(r, (('u', 'v', 'w'), 'wind'))
  
        self.assertRaises(ValueError, iris.analysis.calculus.spatial_vectors_with_phenom_name, u, None, w)
        self.assertRaises(ValueError, iris.analysis.calculus.spatial_vectors_with_phenom_name, None, None, w)
        self.assertRaises(ValueError, iris.analysis.calculus.spatial_vectors_with_phenom_name, None, None, None)
        
        u.rename("x foobar wibble") 
        v.rename("y foobar wibble") 
        w.rename("z foobar wibble") 
        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v)
        self.assertEqual(r, (('x', 'y', 'z'), 'foobar wibble'))

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w)
        self.assertEqual(r, (('x', 'y', 'z'), 'foobar wibble'))
        
        u.rename("wibble foobar") 
        v.rename("wobble foobar") 
        w.rename("tipple foobar") 
#        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w) #should raise a Value Error... 
        self.assertRaises(ValueError, iris.analysis.calculus.spatial_vectors_with_phenom_name, u, v)
        self.assertRaises(ValueError, iris.analysis.calculus.spatial_vectors_with_phenom_name, u, v, w)

        u.rename("eastward_foobar") 
        v.rename("northward_foobar") 
        w.rename("upward_foobar") 
        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v)
        self.assertEqual(r, (('eastward', 'northward', 'upward'), 'foobar'))

        r = iris.analysis.calculus.spatial_vectors_with_phenom_name(u, v, w)
        self.assertEqual(r, (('eastward', 'northward', 'upward'), 'foobar'))

        # Change it to have an inconsistent phenomenon
        v.rename('northward_foobar2') 
        self.assertRaises(ValueError, iris.analysis.calculus.spatial_vectors_with_phenom_name, u, v)


if __name__ == "__main__":
    unittest.main()
