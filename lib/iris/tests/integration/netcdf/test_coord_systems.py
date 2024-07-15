# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for coord-system-related loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from os.path import join as path_join
import shutil
import tempfile
import warnings

import pytest

import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import stock as stock
from iris.tests.stock.netcdf import ncgen_from_cdl
from iris.tests.unit.fileformats.netcdf.loader import test_load_cubes as tlc


@tests.skip_data
class TestCoordSystem(tests.IrisTest):
    def setUp(self):
        tlc.setUpModule()

    def tearDown(self):
        tlc.tearDownModule()

    def test_load_laea_grid(self):
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "lambert_azimuthal_equal_area", "euro_air_temp.nc")
            )
        )
        self.assertCML(cube, ("netcdf", "netcdf_laea.cml"))

    datum_cf_var_cdl = """
        netcdf output {
        dimensions:
            y = 4 ;
            x = 3 ;
        variables:
            float data(y, x) ;
                data :standard_name = "toa_brightness_temperature" ;
                data :units = "K" ;
                data :grid_mapping = "mercator" ;
            int mercator ;
                mercator:grid_mapping_name = "mercator" ;
                mercator:longitude_of_prime_meridian = 0. ;
                mercator:earth_radius = 6378169. ;
                mercator:horizontal_datum_name = "OSGB36" ;
            float y(y) ;
                y:axis = "Y" ;
                y:units = "m" ;
                y:standard_name = "projection_y_coordinate" ;
            float x(x) ;
                x:axis = "X" ;
                x:units = "m" ;
                x:standard_name = "projection_x_coordinate" ;

        // global attributes:
                :Conventions = "CF-1.7" ;
                :standard_name_vocabulary = "CF Standard Name Table v27" ;

        data:

        data =
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11 ;

        mercator = _ ;

        y = 1, 2, 3, 5 ;

        x = -6, -4, -2 ;

        }
    """

    datum_wkt_cdl = """
netcdf output5 {
dimensions:
    y = 4 ;
    x = 3 ;
variables:
    float data(y, x) ;
        data :standard_name = "toa_brightness_temperature" ;
        data :units = "K" ;
        data :grid_mapping = "mercator" ;
    int mercator ;
        mercator:grid_mapping_name = "mercator" ;
        mercator:longitude_of_prime_meridian = 0. ;
        mercator:earth_radius = 6378169. ;
        mercator:longitude_of_projection_origin = 0. ;
        mercator:false_easting = 0. ;
        mercator:false_northing = 0. ;
        mercator:scale_factor_at_projection_origin = 1. ;
        mercator:crs_wkt = "PROJCRS[\\"unknown\\",BASEGEOGCRS[\\"unknown\\",DATUM[\\"OSGB36\\",ELLIPSOID[\\"unknown\\",6378169,0,LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]],PRIMEM[\\"Greenwich\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8901]]],CONVERSION[\\"unknown\\",METHOD[\\"Mercator (variant B)\\",ID[\\"EPSG\\",9805]],PARAMETER[\\"Latitude of 1st standard parallel\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8823]],PARAMETER[\\"Longitude of natural origin\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8802]],PARAMETER[\\"False easting\\",0,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8806]],PARAMETER[\\"False northing\\",0,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8807]]],CS[Cartesian,2],AXIS[\\"(E)\\",east,ORDER[1],LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]],AXIS[\\"(N)\\",north,ORDER[2],LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]]" ;
    float y(y) ;
        y:axis = "Y" ;
        y:units = "m" ;
        y:standard_name = "projection_y_coordinate" ;
    float x(x) ;
        x:axis = "X" ;
        x:units = "m" ;
        x:standard_name = "projection_x_coordinate" ;

// global attributes:
        :standard_name_vocabulary = "CF Standard Name Table v27" ;
        :Conventions = "CF-1.7" ;
data:

 data =
  0, 1, 2,
  3, 4, 5,
  6, 7, 8,
  9, 10, 11 ;

 mercator = _ ;

 y = 1, 2, 3, 5 ;

 x = -6, -4, -2 ;
}
    """

    def test_load_datum_wkt(self):
        expected = "OSGB 1936"
        nc_path = tlc.cdl_to_nc(self.datum_wkt_cdl)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == expected

    def test_no_load_datum_wkt(self):
        nc_path = tlc.cdl_to_nc(self.datum_wkt_cdl)
        with pytest.warns(FutureWarning, match="iris.FUTURE.datum_support"):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == "unknown"

    def test_no_datum_no_warn(self):
        new_cdl = self.datum_wkt_cdl.splitlines()
        new_cdl = [line for line in new_cdl if "DATUM" not in line]
        new_cdl = "\n".join(new_cdl)
        nc_path = tlc.cdl_to_nc(new_cdl)
        with warnings.catch_warnings():
            # pytest's recommended way to assert for no warnings.
            warnings.simplefilter("error", FutureWarning)
            _ = iris.load_cube(nc_path)

    def test_load_datum_cf_var(self):
        expected = "OSGB 1936"
        nc_path = tlc.cdl_to_nc(self.datum_cf_var_cdl)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == expected

    def test_no_load_datum_cf_var(self):
        nc_path = tlc.cdl_to_nc(self.datum_cf_var_cdl)
        with pytest.warns(FutureWarning, match="iris.FUTURE.datum_support"):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == "unknown"

    def test_save_datum(self):
        expected = "OSGB 1936"
        saved_crs = iris.coord_systems.Mercator(
            ellipsoid=iris.coord_systems.GeogCS.from_datum("OSGB36")
        )

        base_cube = stock.realistic_3d()
        base_lat_coord = base_cube.coord("grid_latitude")
        test_lat_coord = DimCoord(
            base_lat_coord.points,
            standard_name="projection_y_coordinate",
            coord_system=saved_crs,
        )
        base_lon_coord = base_cube.coord("grid_longitude")
        test_lon_coord = DimCoord(
            base_lon_coord.points,
            standard_name="projection_x_coordinate",
            coord_system=saved_crs,
        )
        test_cube = Cube(
            base_cube.data,
            standard_name=base_cube.standard_name,
            units=base_cube.units,
            dim_coords_and_dims=(
                (base_cube.coord("time"), 0),
                (test_lat_coord, 1),
                (test_lon_coord, 2),
            ),
        )

        with self.temp_filename(suffix=".nc") as filename:
            iris.save(test_cube, filename)
            with iris.FUTURE.context(datum_support=True):
                cube = iris.load_cube(filename)

        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == expected


class TestLoadMinimalGeostationary(tests.IrisTest):
    """Check we can load data with a geostationary grid-mapping, even when the
    'false-easting' and 'false_northing' properties are missing.

    """

    _geostationary_problem_cdl = """
netcdf geostationary_problem_case {
dimensions:
    y = 2 ;
    x = 3 ;
variables:
    short radiance(y, x) ;
        radiance:standard_name = "toa_outgoing_radiance_per_unit_wavelength" ;
        radiance:units = "W m-2 sr-1 um-1" ;
        radiance:coordinates = "y x" ;
        radiance:grid_mapping = "imager_grid_mapping" ;
    short y(y) ;
        y:units = "rad" ;
        y:axis = "Y" ;
        y:long_name = "fixed grid projection y-coordinate" ;
        y:standard_name = "projection_y_coordinate" ;
    short x(x) ;
        x:units = "rad" ;
        x:axis = "X" ;
        x:long_name = "fixed grid projection x-coordinate" ;
        x:standard_name = "projection_x_coordinate" ;
    int imager_grid_mapping ;
        imager_grid_mapping:grid_mapping_name = "geostationary" ;
        imager_grid_mapping:perspective_point_height = 35786023. ;
        imager_grid_mapping:semi_major_axis = 6378137. ;
        imager_grid_mapping:semi_minor_axis = 6356752.31414 ;
        imager_grid_mapping:latitude_of_projection_origin = 0. ;
        imager_grid_mapping:longitude_of_projection_origin = -75. ;
        imager_grid_mapping:sweep_angle_axis = "x" ;

data:

 // coord values, just so these can be dim-coords
 y = 0, 1 ;
 x = 0, 1, 2 ;

}
"""

    @classmethod
    def setUpClass(cls):
        # Create a temp directory for transient test files.
        cls.temp_dir = tempfile.mkdtemp()
        cls.path_test_cdl = path_join(cls.temp_dir, "geos_problem.cdl")
        cls.path_test_nc = path_join(cls.temp_dir, "geos_problem.nc")
        # Create reference CDL and netcdf files from the CDL text.
        ncgen_from_cdl(
            cdl_str=cls._geostationary_problem_cdl,
            cdl_path=cls.path_test_cdl,
            nc_path=cls.path_test_nc,
        )

    @classmethod
    def tearDownClass(cls):
        # Destroy the temp directory.
        shutil.rmtree(cls.temp_dir)

    def test_geostationary_no_false_offsets(self):
        # Check we can load the test data and coordinate system properties are correct.
        cube = iris.load_cube(self.path_test_nc)
        # Check the coordinate system properties has the correct default properties.
        cs = cube.coord_system()
        assert isinstance(cs, iris.coord_systems.Geostationary)
        assert cs.false_easting == 0.0
        assert cs.false_northing == 0.0


if __name__ == "__main__":
    tests.main()
