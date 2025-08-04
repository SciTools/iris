# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for coord-system-related loading and saving netcdf files."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

import warnings

import numpy as np
import pytest

import iris
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import stock as stock
from iris.tests._shared_utils import assert_CML
from iris.tests.stock.netcdf import ncgen_from_cdl
from iris.tests.unit.fileformats.netcdf.loader import test_load_cubes as tlc


@pytest.fixture
def datum_cf_var_cdl():
    return """
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


@pytest.fixture
def datum_wkt_cdl():
    return """
netcdf output5 {
dimensions:
    y = 4 ;
    x = 3 ;
variables:
    float data(y, x) ;
        data :standard_name = "toa_brightness_temperature" ;
        data :units = "K" ;
        data :grid_mapping = "mercator: x y" ;
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


@pytest.fixture
def multi_cs_osgb_wkt():
    return """
netcdf osgb {
dimensions:
    y = 5 ;
    x = 4 ;
variables:
    double x(x) ;
        x:standard_name = "projection_x_coordinate" ;
        x:long_name = "Easting" ;
        x:units = "m" ;
    double y(y) ;
        y:standard_name = "projection_y_coordinate" ;
        y:long_name = "Northing" ;
        y:units = "m" ;
    double lat(y, x) ;
        lat:standard_name = "latitude" ;
        lat:units = "degrees_north" ;
    double lon(y, x) ;
        lon:standard_name = "longitude" ;
        lon:units = "degrees_east" ;
    float temp(y, x) ;
        temp:standard_name = "air_temperature" ;
        temp:units = "K" ;
        temp:coordinates = "lat lon" ;
        temp:grid_mapping = "crsOSGB: x y crsWGS84: lat lon" ;
    int crsOSGB ;
        crsOSGB:grid_mapping_name = "transverse_mercator" ;
        crsOSGB:semi_major_axis = 6377563.396 ;
        crsOSGB:inverse_flattening = 299.3249646 ;
        crsOSGB:longitude_of_prime_meridian = 0. ;
        crsOSGB:latitude_of_projection_origin = 49. ;
        crsOSGB:longitude_of_central_meridian = -2. ;
        crsOSGB:scale_factor_at_central_meridian = 0.9996012717 ;
        crsOSGB:false_easting = 400000. ;
        crsOSGB:false_northing = -100000. ;
        crsOSGB:unit = "metre" ;
        crsOSGB:crs_wkt = "PROJCRS[\\"unknown\\",BASEGEOGCRS[\\"unknown\\",DATUM[\\"Unknown based on Airy 1830 ellipsoid\\",ELLIPSOID[\\"Airy 1830\\",6377563.396,299.324964600004,LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]],PRIMEM[\\"Greenwich\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8901]]],CONVERSION[\\"unknown\\",METHOD[\\"Transverse Mercator\\",ID[\\"EPSG\\",9807]],PARAMETER[\\"Latitude of natural origin\\",49,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8801]],PARAMETER[\\"Longitude of natural origin\\",-2,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8802]],PARAMETER[\\"Scale factor at natural origin\\",0.9996012717,SCALEUNIT[\\"unity\\",1],ID[\\"EPSG\\",8805]],PARAMETER[\\"False easting\\",400000,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8806]],PARAMETER[\\"False northing\\",-100000,LENGTHUNIT[\\"metre\\",1],ID[\\"EPSG\\",8807]]],CS[Cartesian,2],AXIS[\\"(E)\\",east,ORDER[1],LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]],AXIS[\\"(N)\\",north,ORDER[2],LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]]" ;
    int crsWGS84 ;
        crsWGS84:grid_mapping_name = "latitude_longitude" ;
        crsWGS84:longitude_of_prime_meridian = 0. ;
        crsWGS84:semi_major_axis = 6378137. ;
        crsWGS84:inverse_flattening = 298.257223563 ;
        crsWGS84: crs_wkt = "GEOGCRS[\\"unknown\\",DATUM[\\"Unknown based on WGS 84 ellipsoid\\",ELLIPSOID[\\"WGS 84\\",6378137,298.257223562997,LENGTHUNIT[\\"metre\\",1,ID[\\"EPSG\\",9001]]]],PRIMEM[\\"Greenwich\\",0,ANGLEUNIT[\\"degree\\",0.0174532925199433],ID[\\"EPSG\\",8901]],CS[ellipsoidal,2],AXIS[\\"longitude\\",east,ORDER[1],ANGLEUNIT[\\"degree\\",0.0174532925199433,ID[\\"EPSG\\",9122]]],AXIS[\\"latitude\\",north,ORDER[2],ANGLEUNIT[\\"degree\\",0.0174532925199433,ID[\\"EPSG\\",9122]]]]" ;
data:
    x = 1,2,3,4 ;
    y = 1,2,3,4,5 ;
}
    """


@tests.skip_data
class TestCoordSystem:
    @pytest.fixture(autouse=True)
    def _setup(self):
        tlc.setUpModule()
        yield
        tlc.tearDownModule()

    def test_load_laea_grid(self, request):
        cube = iris.load_cube(
            tests.get_data_path(
                ("NetCDF", "lambert_azimuthal_equal_area", "euro_air_temp.nc")
            )
        )
        assert_CML(request, cube, ("netcdf", "netcdf_laea.cml"))

    def test_load_datum_wkt(self, datum_wkt_cdl):
        expected = "OSGB 1936"
        nc_path = tlc.cdl_to_nc(datum_wkt_cdl)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == expected

    def test_no_load_datum_wkt(self, datum_wkt_cdl):
        nc_path = tlc.cdl_to_nc(datum_wkt_cdl)
        with pytest.warns(FutureWarning, match="iris.FUTURE.datum_support"):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == "unknown"

    def test_no_datum_no_warn(self, datum_wkt_cdl):
        new_cdl = datum_wkt_cdl.splitlines()
        new_cdl = [line for line in new_cdl if "DATUM" not in line]
        new_cdl = "\n".join(new_cdl)
        nc_path = tlc.cdl_to_nc(new_cdl)
        with warnings.catch_warnings():
            # pytest's recommended way to assert for no warnings.
            warnings.simplefilter("error", FutureWarning)
            _ = iris.load_cube(nc_path)

    def test_load_datum_cf_var(self, datum_cf_var_cdl):
        expected = "OSGB 1936"
        nc_path = tlc.cdl_to_nc(datum_cf_var_cdl)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == expected

    def test_no_load_datum_cf_var(self, datum_cf_var_cdl):
        nc_path = tlc.cdl_to_nc(datum_cf_var_cdl)
        with pytest.warns(FutureWarning, match="iris.FUTURE.datum_support"):
            cube = iris.load_cube(nc_path)
        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == "unknown"

    def test_load_multi_cs_wkt(self, multi_cs_osgb_wkt):
        nc_path = tlc.cdl_to_nc(multi_cs_osgb_wkt)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(nc_path)

        assert len(cube.coord_systems()) == 2
        for name in ["projection_y_coordinate", "projection_y_coordinate"]:
            assert (
                cube.coord(name).coord_system.grid_mapping_name == "transverse_mercator"
            )
        for name in ["latitude", "longitude"]:
            assert (
                cube.coord(name).coord_system.grid_mapping_name == "latitude_longitude"
            )
        assert cube.extended_grid_mapping is True

    def test_save_datum(self, tmp_path):
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
        filename = tmp_path / "output.nc"
        iris.save(test_cube, filename)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(filename)

        test_crs = cube.coord("projection_y_coordinate").coord_system
        actual = str(test_crs.as_cartopy_crs().datum)
        assert actual == expected

    def test_save_multi_cs_wkt(self, tmp_path):
        crsOSGB = iris.coord_systems.OSGB()
        crsLatLon = iris.coord_systems.GeogCS(6e6)

        dimx_coord = iris.coords.DimCoord(
            np.arange(4), "projection_x_coordinate", coord_system=crsOSGB
        )
        dimy_coord = iris.coords.DimCoord(
            np.arange(5), "projection_y_coordinate", coord_system=crsOSGB
        )

        auxlon_coord = iris.coords.AuxCoord(
            np.arange(20).reshape((5, 4)),
            standard_name="longitude",
            coord_system=crsLatLon,
        )
        auxlat_coord = iris.coords.AuxCoord(
            np.arange(20).reshape((5, 4)),
            standard_name="latitude",
            coord_system=crsLatLon,
        )

        test_cube = Cube(
            np.ones(20).reshape((5, 4)),
            standard_name="air_pressure",
            units="Pa",
            dim_coords_and_dims=(
                (dimy_coord, 0),
                (dimx_coord, 1),
            ),
            aux_coords_and_dims=(
                (auxlat_coord, (0, 1)),
                (auxlon_coord, (0, 1)),
            ),
        )

        test_cube.extended_grid_mapping = True

        filename = tmp_path / "output.nc"
        iris.save(test_cube, filename)
        with iris.FUTURE.context(datum_support=True):
            cube = iris.load_cube(filename)

        assert len(cube.coord_systems()) == 2
        for name in ["projection_y_coordinate", "projection_y_coordinate"]:
            assert (
                cube.coord(name).coord_system.grid_mapping_name == "transverse_mercator"
            )
        for name in ["latitude", "longitude"]:
            assert (
                cube.coord(name).coord_system.grid_mapping_name == "latitude_longitude"
            )
        assert cube.extended_grid_mapping is True


@pytest.fixture(scope="module")
def geostationary_problem_cdl():
    return """
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


class TestLoadMinimalGeostationary:
    """Check we can load data with a geostationary grid-mapping, even when the
    'false-easting' and 'false_northing' properties are missing.

    """

    @pytest.fixture(scope="class")
    def geostationary_problem_ncfile(self, tmp_path_factory, geostationary_problem_cdl):
        tmp_path = tmp_path_factory.mktemp("geos")
        cdl_path = tmp_path / "geos_problem.cdl"
        nc_path = tmp_path / "geos_problem.nc"
        ncgen_from_cdl(
            cdl_str=geostationary_problem_cdl,
            cdl_path=cdl_path,
            nc_path=nc_path,
        )
        return nc_path

    def test_geostationary_no_false_offsets(
        self, tmp_path, geostationary_problem_ncfile
    ):
        # Check we can load the test data and coordinate system properties are correct.
        cube = iris.load_cube(geostationary_problem_ncfile)
        # Check the coordinate system properties has the correct default properties.
        cs = cube.coord_system()
        assert isinstance(cs, iris.coord_systems.Geostationary)
        assert cs.false_easting == 0.0
        assert cs.false_northing == 0.0
