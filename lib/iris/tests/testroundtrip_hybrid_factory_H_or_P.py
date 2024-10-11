import numpy as np

import iris
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coord_systems import GeogCS
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.pp import EARTH_RADIUS, STASH


def make_hybrid_z_testdata(
    nt=2,
    nz=3,
    ny=4,
    nx=3,
    hybrid_zcoord_type="height",
    make_reference_time_dependent=True,
    include_reference_as_cube=False,
):
    crs = GeogCS(EARTH_RADIUS)
    t_dim, z_dim, y_dim, x_dim = 0, 1, 2, 3
    co_t = DimCoord(
        np.arange(nt, dtype=np.float32),
        standard_name="time",
        units="days since 2000-01-01",
    )
    co_z = DimCoord(
        np.arange(1, nz + 1, dtype=np.int32),
        standard_name="model_level_number",
        units=1,
    )
    co_y = DimCoord(
        np.linspace(0, 120.0, ny, dtype=np.float32),
        standard_name="latitude",
        units="degrees",
        coord_system=crs,
    )
    co_x = DimCoord(
        np.linspace(-30.0, 50.0, nx, dtype=np.float32),
        standard_name="longitude",
        units="degrees",
        coord_system=crs,
    )
    cube = Cube(
        np.zeros((nt, nz, ny, nx), dtype=np.float32),
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=zip((co_t, co_z, co_y, co_x), (t_dim, z_dim, y_dim, x_dim)),
    )

    delta_vals = np.linspace(200.0, 600, nz, dtype=np.float32)
    if hybrid_zcoord_type == "pressure":
        co_delta = DimCoord(delta_vals, long_name="delta", units="hPa")
    elif hybrid_zcoord_type == "height":
        co_delta = DimCoord(delta_vals, long_name="level_height", units="m")
    else:
        raise ValueError(f"Unknown hybrid type: {hybrid_zcoord_type}")

    sigma_vals = np.linspace(0.2, 0.8, nz, dtype=np.float32)
    co_sigma = DimCoord(sigma_vals, long_name="sigma", units=1)

    # Note: will not save as HH to PP without bounds on delta+sigma
    for coord in (co_delta, co_sigma):
        coord.guess_bounds()
    cube.add_aux_coord(co_delta, z_dim)
    cube.add_aux_coord(co_sigma, z_dim)

    refdata = np.arange(nt * ny * nx, dtype=np.float32)
    refdata = 1000.0 + refdata.reshape(nt, ny, nx)
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


# _HYBRID_ZCOORD_TYPE = "height"
_HYBRID_ZCOORD_TYPE = "pressure"

# _FILENAME = "tmp.nc"  # Naturally, this "just works"
_FILENAME = "tmp.pp"
# _FILENAME = "tmp.grib2"

_TEST_TIME_DEPENDENT = True
# _TEST_TIME_DEPENDENT = False


def check_create():
    global _FILENAME, _HYBRID_ZCOORD_TYPE, _TEST_TIME_DEPENDENT
    file_ext = _FILENAME.split(".")[-1]
    include_ref = file_ext in ("grib2", "pp")

    data = make_hybrid_z_testdata(
        hybrid_zcoord_type=_HYBRID_ZCOORD_TYPE,
        include_reference_as_cube=include_ref,
        make_reference_time_dependent=_TEST_TIME_DEPENDENT,
    )

    print()
    print(f"Cubes saving to {_FILENAME}:")
    print(data)
    for cube in data:
        print(cube)

    _EXTRA_COORDS_DEBUG = False
    if _EXTRA_COORDS_DEBUG:
        (datacube,) = [cube for cube in data if "surface" not in cube.name()]
        for name in ("level_height", "sigma", "surface_altitude"):
            print(f"Coord({name}):")
            print(datacube.coord(name))
        print("Ref cube:")
        print(data.extract_cube("surface_altitude"))

    iris.save(data, _FILENAME)
    readback = iris.load(_FILENAME)
    # Apply extra concat : as "raw" cubes with a time dimension won't merge
    readback = readback.concatenate()
    print()
    print("Readback cubes:")
    print(readback)
    for cube in readback:
        print(cube)


def test_roundtrip():
    print("Check with Iris from : ", iris.__file__)
    from iris import (
        LOAD_POLICY,
        LOAD_POLICY_RECOMMENDED,
        # LOAD_POLICY_LEGACY,
        # LOAD_POLICY_COMPREHENSIVE
    )

    with LOAD_POLICY.context(LOAD_POLICY_RECOMMENDED):
        check_create()
