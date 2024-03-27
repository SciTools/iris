# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for the `iris.experimental.geovista.extract_unstructured_region` function."""

from geovista.geodesic import BBox

from iris import load_cube
from iris.experimental.geovista import cube_to_polydata, extract_unstructured_region
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD
from iris.tests import get_data_path


def test_face_region_extraction():
    file_path = get_data_path(
        [
            "NetCDF",
            "unstructured_grid",
            "lfric_ngvat_2D_72t_face_half_levels_main_conv_rain.nc",
        ]
    )

    with PARSE_UGRID_ON_LOAD.context():
        global_cube = load_cube(file_path, "conv_rain")
    polydata = cube_to_polydata(global_cube[0, :])
    region = BBox(lons=[0, 70, 70, 0], lats=[-25, -25, 45, 45])

    extracted_cube = extract_unstructured_region(
        global_cube, polydata, region, preference="center"
    )

    assert extracted_cube.ndim == 2
    assert extracted_cube.shape == (72, 101)
    assert global_cube.shape == (72, 864)
    assert global_cube.ndim == 2
