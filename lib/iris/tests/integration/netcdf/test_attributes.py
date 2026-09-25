# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for attribute-related loading and saving netcdf files."""

from contextlib import contextmanager

from cf_units import Unit
import numpy as np
import pytest

import iris
import iris.coord_systems
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from iris.fileformats.netcdf import CF_CONVENTIONS_VERSION
from iris.fileformats.netcdf import _thread_safe_nc as threadsafe_nc
from iris.fileformats.netcdf._dataset import NetCDFDatasetVariable
from iris.tests import _shared_utils


class TestUmVersionAttribute:
    def test_single_saves_as_global(self, tmp_path, request):
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        nc_path = tmp_path / "test.nc"
        iris.save(cube, nc_path)
        _shared_utils.assert_CDL(request, nc_path)

    def test_multiple_same_saves_as_global(self, tmp_path, request):
        cube_a = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        cube_b = Cube(
            [1.0],
            standard_name="air_pressure",
            units="hPa",
            attributes={"um_version": "4.3"},
        )
        nc_path = tmp_path / "test.nc"
        iris.save(CubeList([cube_a, cube_b]), nc_path)
        _shared_utils.assert_CDL(request, nc_path)

    def test_multiple_different_saves_on_variables(self, tmp_path, request):
        cube_a = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"um_version": "4.3"},
        )
        cube_b = Cube(
            [1.0],
            standard_name="air_pressure",
            units="hPa",
            attributes={"um_version": "4.4"},
        )
        nc_path = tmp_path / "test.nc"
        iris.save(CubeList([cube_a, cube_b]), nc_path)
        _shared_utils.assert_CDL(request, nc_path)


@contextmanager
def _patch_site_configuration(mocker):
    def cf_patch_conventions(conventions):
        return ", ".join([conventions, "convention1, convention2"])

    def update(config):
        config["cf_profile"] = mocker.Mock(name="cf_profile")
        config["cf_patch"] = mocker.Mock(name="cf_patch")
        config["cf_patch_conventions"] = cf_patch_conventions

    orig_site_config = iris.site_configuration.copy()
    update(iris.site_configuration)
    yield
    iris.site_configuration = orig_site_config


class TestConventionsAttributes:
    def test_patching_conventions_attribute(self, tmp_path, mocker):
        # Ensure that user defined conventions are wiped and those which are
        # saved patched through site_config can be loaded without an exception
        # being raised.
        cube = Cube(
            [1.0],
            standard_name="air_temperature",
            units="K",
            attributes={"Conventions": "some user defined conventions"},
        )

        # Patch the site configuration dictionary.
        nc_path = tmp_path / "test.nc"
        with _patch_site_configuration(mocker):
            iris.save(cube, nc_path)
            res = iris.load_cube(nc_path)

        assert res.attributes["Conventions"] == "{}, {}, {}".format(
            CF_CONVENTIONS_VERSION, "convention1", "convention2"
        )


class TestCfPatch:
    """What the ``cf_patch`` hook is handed is public API.

    ``cf_patch`` is a reserved :data:`iris.site_configuration` key, documented
    since Iris 1.3 as receiving netCDF4 objects. Handing it a
    :class:`~iris.fileformats.netcdf._dataset.NetCDFDatasetVariable` instead
    breaks it two ways: ``setncattr`` raises, and a plain
    ``variable.name = value`` is discarded in silence, since a CF variable
    defines no ``__setattr__``.

    """

    def test_cf_patch_receives_a_netcdf4_variable(self, tmp_path):
        received = []

        def cf_profile(cube):
            return "a profile"

        def cf_patch(profile, dataset, variable):
            received.append(variable)
            # Write through the netCDF4 API the hook is promised.
            variable.setncattr("patched_by_cf_patch", "cf_patch was here")

        # A real function, not a Mock: a Mock accepts setncattr() whatever it
        # is given, and records a call that never reached a file.
        orig_site_config = iris.site_configuration.copy()
        iris.site_configuration["cf_profile"] = cf_profile
        iris.site_configuration["cf_patch"] = cf_patch
        nc_path = tmp_path / "cf_patch.nc"
        try:
            cube = Cube([1.0], standard_name="air_temperature", units="K")
            iris.save(cube, nc_path)
        finally:
            iris.site_configuration = orig_site_config

        (variable,) = received
        assert not isinstance(variable, NetCDFDatasetVariable)

        # The read-back is the point. Asserting that the hook ran, or that it
        # did not raise, would miss an attribute going quietly nowhere.
        result = iris.load_cube(nc_path)
        assert result.attributes["patched_by_cf_patch"] == "cf_patch was here"


class TestAttributesNamedLikeNetcdf4Members:
    """A coordinate attribute may be named after a netCDF4 Python member.

    ``shape``, ``size`` and the rest below are members of a netCDF4 Variable
    object as well as plausible attribute names. The saver's "don't clobber"
    check used to ask ``hasattr``, so every one of them was silently dropped
    on the way to the file. It asks the CF attribute mapping now, which knows
    the difference between an attribute of the data and a member of the
    object holding it, and the six reach the file like any other.

    """

    NAMES = ("shape", "size", "name", "dtype", "dimensions", "mask")

    def test_attributes_named_after_netcdf4_members_are_saved(self, tmp_path):
        coord = DimCoord(
            np.arange(3.0),
            standard_name="longitude",
            units="degrees",
            attributes={name: f"value of {name}" for name in self.NAMES},
        )
        cube = Cube(np.arange(3.0), standard_name="air_temperature", units="K")
        cube.add_dim_coord(coord, 0)

        nc_path = tmp_path / "netcdf4_member_names.nc"
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(cube, nc_path)

        # Reading back is the only way to see this: the dropped attributes
        # were dropped in silence, with no warning and no error.
        result = iris.load_cube(nc_path).coord("longitude")
        assert {name: result.attributes.get(name) for name in self.NAMES} == {
            name: f"value of {name}" for name in self.NAMES
        }


class TestGridMappingAttributes:
    """Grid-mapping parameters are written by plain attribute assignment.

    Every other attribute the saver writes goes through the CF attribute
    mapping. These do not - they are set straight onto the netCDF4 variable,
    because doing otherwise would change their type in the file (finding F8).
    That makes a mistake in one of them silent: assigning to the wrong object,
    or misspelling the object, leaves a stray Python attribute and no
    attribute in the file at all.

    These two projections are here because they carry the eight parameter
    assignments no other test reads back.

    """

    @staticmethod
    def saved_grid_mapping(coord_system, tmp_path):
        """Save a cube on this coord system; return the grid-mapping attributes."""
        cube = Cube(
            np.zeros((2, 3), dtype=np.float32),
            standard_name="air_temperature",
            units="K",
        )
        for index, axis in enumerate("yx"):
            cube.add_dim_coord(
                DimCoord(
                    np.arange(cube.shape[index], dtype=np.float64),
                    standard_name=f"projection_{axis}_coordinate",
                    units="m",
                    coord_system=coord_system,
                ),
                index,
            )
        nc_path = tmp_path / f"{coord_system.grid_mapping_name}.nc"
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(cube, nc_path)

        # Read the file itself rather than a loaded cube: what is being
        # pinned is that these values reach the variable, not what the
        # loader is able to reconstruct from them.
        dataset = threadsafe_nc.DatasetWrapper(nc_path)
        try:
            variable = dataset.variables[coord_system.grid_mapping_name]
            return {name: variable.getncattr(name) for name in variable.ncattrs()}
        finally:
            dataset.close()

    def test_mercator_scale_factor(self, tmp_path):
        # Mercator takes a scale factor *or* a standard parallel. Only the
        # standard-parallel branch is read back anywhere else.
        coord_system = iris.coord_systems.Mercator(
            longitude_of_projection_origin=90.0,
            scale_factor_at_projection_origin=0.9,
        )
        attributes = self.saved_grid_mapping(coord_system, tmp_path)
        assert attributes["scale_factor_at_projection_origin"] == 0.9
        assert "standard_parallel" not in attributes

    @pytest.mark.parametrize(
        ("scale_kwargs", "expected_scale"),
        [
            ({"true_scale_lat": 71.0}, {"true_scale_lat": 71.0}),
            (
                {"scale_factor_at_projection_origin": 0.9},
                {"scale_factor_at_projection_origin": 0.9},
            ),
            # Neither given: the saver writes a scale factor of 1.0 rather
            # than leaving the projection unscaled.
            ({}, {"scale_factor_at_projection_origin": 1.0}),
        ],
        ids=["true_scale_lat", "scale_factor", "neither"],
    )
    def test_polar_stereographic(self, scale_kwargs, expected_scale, tmp_path):
        coord_system = iris.coord_systems.PolarStereographic(
            central_lat=90.0,
            central_lon=-150.0,
            false_easting=13.0,
            false_northing=17.0,
            **scale_kwargs,
        )
        attributes = self.saved_grid_mapping(coord_system, tmp_path)
        expected = {
            "latitude_of_projection_origin": 90.0,
            "straight_vertical_longitude_from_pole": -150.0,
            "false_easting": 13.0,
            "false_northing": 17.0,
            **expected_scale,
        }
        assert {name: attributes.get(name) for name in expected} == expected


class TestStandardName:
    def test_standard_name_roundtrip(self, tmp_path):
        standard_name = "air_temperature detection_minimum"
        cube = iris.cube.Cube(1, standard_name=standard_name)
        fout = tmp_path / "standard_name.nc"
        iris.save(cube, fout)
        detection_limit_cube = iris.load_cube(fout)
        assert detection_limit_cube.standard_name == standard_name


class TestCalendar:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.calendar = Unit("days since 1970-01-01", calendar="360_day")
        self.cube = iris.cube.Cube(1, units=self.calendar)

    def test_calendar_roundtrip(self, tmp_path):
        fout = tmp_path / "calendar.nc"
        iris.save(self.cube, fout)
        detection_limit_cube = iris.load_cube(fout)
        assert detection_limit_cube.units == self.calendar
