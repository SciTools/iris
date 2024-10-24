# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Code to save and re-load hybrid vertical coordinates with variable reference fields.

Tests all combinations of:
   * file format: PP, GRIB and NetCDF
   * reference fields: static (for legacy reference) and time-dependent
   * hybrid coordinate fields:
        * hybrid-height levels with orography, and
        * hybrid-pressure levels with surface-pressure
"""

import numpy as np
import pytest

import iris
from iris import LOAD_POLICY
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.pp import EARTH_RADIUS, STASH
from iris.util import new_axis

try:
    import iris_grib
except ImportError:
    iris_grib = None

# General test dimensions = (timepoints, levels, lats, lons)
NT, NZ, NY, NX = (3, 4, 5, 6)


def make_hybrid_z_testdata(
    hybrid_zcoord_type="height",
    make_reference_time_dependent=True,
    include_reference_as_cube=False,
):
    """Construct a realistic synthetic data cube with a hybrid vertical coordinate.

    Parameters
    ----------
    hybrid_zcoord_type : string, default "height"
        either "height" or "pressure"
    make_reference_time_dependent : bool, default True
        if True, the reference coord has dims (t, y, x), otherwise just (y, x)
    include_reference_as_cube : bool, default False
        if True, the result includes a separate cube of the reference values.
        (Because this must be separately passed to save for the fields-based formats.)

    Returns
    -------
        cubes
            A list containing a cube with (t, z, y, x) dims and the appropriate
            aux-factory.
            Optionally, if "include_reference_as_cube" is True, an extra cube
            containing the reference data is aldo returned.

    """
    crs = GeogCS(EARTH_RADIUS)
    z_dim, t_dim, y_dim, x_dim = 0, 1, 2, 3
    co_t = DimCoord(
        np.arange(NT, dtype=np.float32),
        standard_name="time",
        units="days since 2000-01-01",
    )
    co_z = DimCoord(
        np.arange(1, NZ + 1, dtype=np.int32),
        standard_name="model_level_number",
        units=1,
    )
    co_y = DimCoord(
        np.linspace(0, 120.0, NY, dtype=np.float32),
        standard_name="latitude",
        units="degrees",
        coord_system=crs,
    )
    co_x = DimCoord(
        np.linspace(-30.0, 50.0, NX, dtype=np.float32),
        standard_name="longitude",
        units="degrees",
        coord_system=crs,
    )
    cube = Cube(
        np.zeros((NZ, NT, NY, NX), dtype=np.float32),
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=zip((co_t, co_z, co_y, co_x), (t_dim, z_dim, y_dim, x_dim)),
    )

    delta_vals = np.linspace(200.0, 600, NZ, dtype=np.float32)
    if hybrid_zcoord_type == "pressure":
        co_delta = DimCoord(delta_vals, long_name="delta", units="hPa")
    elif hybrid_zcoord_type == "height":
        co_delta = DimCoord(delta_vals, long_name="level_height", units="m")
    else:
        raise ValueError(f"Unknown hybrid coordinate type: {hybrid_zcoord_type}")

    sigma_vals = np.linspace(0.2, 0.8, NZ, dtype=np.float32)
    co_sigma = DimCoord(sigma_vals, long_name="sigma", units=1)

    # Note: will not save as HH to PP without bounds on delta+sigma
    for coord in (co_delta, co_sigma):
        coord.guess_bounds()
    cube.add_aux_coord(co_delta, z_dim)
    cube.add_aux_coord(co_sigma, z_dim)

    refdata = np.arange(NT * NY * NX, dtype=np.float32)
    refdata = 1000.0 + refdata.reshape(NT, NY, NX)
    if hybrid_zcoord_type == "pressure":
        co_ref = AuxCoord(
            refdata,
            standard_name="surface_air_pressure",
            units="hPa",
            attributes={"STASH": STASH(model=1, section=0, item=409)},
        )
    elif hybrid_zcoord_type == "height":
        co_ref = AuxCoord(
            refdata,
            standard_name="surface_altitude",
            units="m",
            attributes={"STASH": STASH(model=1, section=0, item=33)},
        )
    else:
        raise ValueError(f"Unknown hybrid type: {hybrid_zcoord_type}")

    ref_dims = (t_dim, y_dim, x_dim)
    if not make_reference_time_dependent:
        co_ref = co_ref[0]
        ref_dims = ref_dims[1:]

    cube.add_aux_coord(co_ref, ref_dims)
    if hybrid_zcoord_type == "pressure":
        factory = HybridPressureFactory(
            sigma=co_sigma, delta=co_delta, surface_air_pressure=co_ref
        )
    elif hybrid_zcoord_type == "height":
        factory = HybridHeightFactory(sigma=co_sigma, delta=co_delta, orography=co_ref)
    else:
        raise ValueError(f"Unknown hybrid type: {hybrid_zcoord_type}")

    cube.add_aux_factory(factory)

    cubes = CubeList([cube])

    if include_reference_as_cube:
        ref_dimcoords = [
            cube.coord(dim_coords=True, dimensions=cube_refdim)
            for cube_refdim in cube.coord_dims(co_ref)
        ]
        reference_cube = Cube(
            co_ref.points,
            standard_name=co_ref.standard_name,
            units=co_ref.units,
            dim_coords_and_dims=[
                (ref_dimcoord, i_refdim)
                for i_refdim, ref_dimcoord in enumerate(ref_dimcoords)
            ],
            attributes=co_ref.attributes,
        )
        if not reference_cube.coords("time"):
            # Add a dummy time coordinate to non-time-dependent reference cube
            # - mostly because otherwise it cannot be saved to GRIB format
            # NOTE: we give this a different nominal time to any of the data : when
            # there is only one reference field, it's recorded time value should be
            # **ignored** by the loader
            reference_cube.add_aux_coord(
                DimCoord(
                    np.array(0, dtype=np.float32),
                    standard_name="time",
                    units="days since 1900-01-01",
                )
            )
        cubes.append(reference_cube)

    return cubes


def check_expected(result_cubes, file_extension, time_dependence, zcoord_type):
    assert len(result_cubes) == 2
    result_phenom = result_cubes.extract_cube("air_temperature")

    if zcoord_type == "pressure":
        ref_coord_name = ref_cube_name = "surface_air_pressure"
        if file_extension == "grib2":
            ref_cube_name = "air_pressure"
    elif zcoord_type == "height":
        ref_coord_name = ref_cube_name = "surface_altitude"
    else:
        raise ValueError(f"Unknown hybrid coordinate type: {zcoord_type}")

    result_ref_cube = result_cubes.extract_cube(ref_cube_name)
    result_ref_coord = result_phenom.coord(ref_coord_name)

    # Check that the reference cube and the coord are equivalent
    assert result_ref_coord.shape == result_ref_cube.shape
    assert np.array_equal(result_ref_cube.data, result_ref_coord.points)
    assert not result_ref_coord.bounds  # bounds are unused in our testcases

    # Check the expected phenomenon shape
    if time_dependence == "static" and file_extension in ("pp", "grib2"):
        phenom_shape = (NT, NZ, NY, NX)
    else:
        phenom_shape = (NZ, NT, NY, NX)
    assert result_phenom.shape == phenom_shape

    # Check expected reference values against calculated values.
    # This shows that the reference was correctly divided into 2d fields and
    # reconstructed on load to match the original (for fields-based formats).
    if time_dependence == "static":
        ref_shape = (NY, NX)
    else:
        ref_shape = (NT, NY, NX)
    ref_data = 1000.0 + np.arange(np.prod(ref_shape)).reshape(ref_shape)
    if zcoord_type == "pressure" and file_extension == "grib2":
        # values come back in Pa not hPa
        ref_data *= 100.0
    assert np.array_equal(ref_data, result_ref_cube.data)


_file_formats = ["pp", "nc"]
if iris_grib:
    _file_formats += ["grib2"]


@pytest.fixture(params=_file_formats)
def file_extension(request):
    return request.param


@pytest.fixture(params=["static", "time_varying"])
def time_dependence(request):
    return request.param


@pytest.fixture(params=["height", "pressure"])
def zcoord_type(request):
    return request.param


@pytest.fixture(params=[f"{name}_policy" for name in LOAD_POLICY.SETTINGS])
def load_policy(request):
    return request.param


def test_roundtrip(file_extension, time_dependence, zcoord_type, load_policy, tmp_path):
    if (
        load_policy == "legacy_policy"
        and time_dependence == "time_varying"
        and file_extension in ("pp", "grib2")
    ):
        pytest.skip("Testcase not supported in 'legacy' mode.")

    filepath = tmp_path / f"tmp.{file_extension}"
    include_ref = file_extension in ("grib2", "pp")
    is_time_dependent = time_dependence == "time_varying"
    data = make_hybrid_z_testdata(
        hybrid_zcoord_type=zcoord_type,
        include_reference_as_cube=include_ref,
        make_reference_time_dependent=is_time_dependent,
    )

    iris.save(data, filepath)

    policy_name = load_policy.split("_")[0]
    with LOAD_POLICY.context(policy_name):
        # NOTE: this is default, but "legacy" mode would fail
        readback = iris.load(filepath)

    check_expected(
        readback,
        file_extension=file_extension,
        time_dependence=time_dependence,
        zcoord_type=zcoord_type,
    )


def test_split_netcdf_roundtrip(zcoord_type, load_policy, tmp_path):
    # NetCDF special test : split the data into 2D slices (like "fields"),
    # and save each to a different file.
    policy_name = load_policy.split("_")[0]
    reference_surface_name = {
        "pressure": "surface_air_pressure",
        "height": "surface_altitude",
    }[zcoord_type]

    data = make_hybrid_z_testdata(
        hybrid_zcoord_type=zcoord_type,
        include_reference_as_cube=False,
        make_reference_time_dependent=True,
    )

    # There is just 1 cube
    (data,) = data  # just 1 cube for netcdf, no separate reference cube
    # split it into 2D YX "field" cubes
    field_cubes = list(data.slices(("latitude", "longitude")))
    # Reinstate a length-1 "time" dimension in each cube.
    field_cubes = [
        new_axis(field_cube, "time", expand_extras=[reference_surface_name])
        for field_cube in field_cubes
    ]
    # Save to 1 file per 'field_cube'
    result_paths = [
        tmp_path / f"field_{i_field:02d}.nc" for i_field in range(len(field_cubes))
    ]
    for field_cube, path in zip(field_cubes, result_paths):
        iris.save(field_cube, path)

    # load back with the chosen policy.
    with LOAD_POLICY.context(policy_name):
        readback = iris.load(result_paths)

    n_cubes = len(readback)
    n_datacubes = len(readback.extract("air_temperature"))
    if policy_name == "legacy":
        assert (n_cubes, n_datacubes) == (15, 3)
    elif policy_name == "default":
        assert (n_cubes, n_datacubes) == (15, 3)
    elif policy_name == "recommended":
        assert (n_cubes, n_datacubes) == (5, 1)
    elif policy_name == "comprehensive":
        assert (n_cubes, n_datacubes) == (5, 1)
    else:
        raise ValueError(f"unknown policy {policy_name!r}")

    if n_datacubes == 1:
        check_expected(
            CubeList(
                [
                    readback.extract_cube("air_temperature"),
                    # include only 1 of N (identical) reference cubes
                    # (all this would be easier if we could rely on load-cube ordering!)
                    readback.extract(reference_surface_name)[0],
                ]
            ),
            file_extension=file_extension,
            time_dependence=time_dependence,
            zcoord_type=zcoord_type,
        )
