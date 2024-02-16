import numpy as np

from iris import load_cube
from iris.experimental.geovista import cube_faces_to_polydata
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
from iris.tests import get_data_path


def test_integration_2d():
    pass


def test_integration_1d():
    file_path = get_data_path(
        [
            "NetCDF",
            "unstructured_grid",
            "lfric_surface_mean.nc",
        ]
    )
    with PARSE_UGRID_ON_LOAD.context():
        global_cube = load_cube(file_path, "tstar_sea")

    polydata = cube_faces_to_polydata(global_cube)

    assert polydata.GetNumberOfCells() == 13824
    assert polydata.GetNumberOfPoints() == 13826
    np.testing.assert_array_equal(polydata.active_scalars, global_cube.data)


def test_integration_mesh():
    file_path = get_data_path(
        [
            "NetCDF",
            "unstructured_grid",
            "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
        ]
    )

    with PARSE_UGRID_ON_LOAD.context():
        global_cube = load_cube(file_path, "conv_rain")

    polydata = cube_faces_to_polydata(global_cube[0, :])

    assert polydata.GetNumberOfCells() == 864
    assert polydata.GetNumberOfPoints() == 866
    np.testing.assert_array_equal(polydata.active_scalars, global_cube[0, :].data)
