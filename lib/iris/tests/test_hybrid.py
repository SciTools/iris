# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the hybrid vertical coordinate representations."""

import contextlib

import numpy as np
import pytest

import iris
from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.tests import _shared_utils
import iris.tests.stock
from iris.warnings import IrisIgnoringBoundsWarning


@_shared_utils.skip_plot
@_shared_utils.skip_data
class TestRealistic4d(_shared_utils.GraphicsTest):
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.cube = iris.tests.stock.realistic_4d()
        self.altitude = self.cube.coord("altitude")

    def test_metadata(self):
        assert self.altitude.units == "m"
        assert self.altitude.coord_system is None
        assert self.altitude.attributes == {"positive": "up"}

    def test_points(self):
        assert self.altitude.points.min() == pytest.approx(np.float32(191.84892))
        assert self.altitude.points.max() == pytest.approx(np.float32(40000))

    def test_transpose(self, request):
        _shared_utils.assert_CML(request, self.cube, ("stock", "realistic_4d.cml"))
        self.cube.transpose()
        _shared_utils.assert_CML(request, self.cube, ("derived", "transposed.cml"))

    def test_indexing(self, request):
        cube = self.cube[:, :, 0, 0]
        # Make sure the derived 'altitude' coordinate survived the indexing.
        _ = cube.coord("altitude")
        _shared_utils.assert_CML(request, cube, ("derived", "column.cml"))

    def test_removing_derived_coord(self, request):
        cube = self.cube
        cube.remove_coord("altitude")
        _shared_utils.assert_CML(
            request, cube, ("derived", "removed_derived_coord.cml")
        )

    def test_removing_sigma(self, request):
        # Check the cube remains OK when sigma is removed.
        cube = self.cube
        cube.remove_coord("sigma")
        _shared_utils.assert_CML(request, cube, ("derived", "removed_sigma.cml"))
        _shared_utils.assert_string(
            request, str(cube), ("derived", "removed_sigma.__str__.txt")
        )

        # Check the factory now only has surface_altitude and delta dependencies.
        factory = cube.aux_factory(name="altitude")
        t = [key for key, coord in factory.dependencies.items() if coord is not None]
        assert sorted(t) == sorted(["orography", "delta"])

    def test_removing_orography(self, request):
        # Check the cube remains OK when the orography is removed.
        cube = self.cube
        cube.remove_coord("surface_altitude")
        _shared_utils.assert_CML(request, cube, ("derived", "removed_orog.cml"))
        _shared_utils.assert_string(
            request, str(cube), ("derived", "removed_orog.__str__.txt")
        )

        # Check the factory now only has sigma and delta dependencies.
        factory = cube.aux_factory(name="altitude")
        t = [key for key, coord in factory.dependencies.items() if coord is not None]
        assert sorted(t) == sorted(["sigma", "delta"])

    def test_derived_coords(self):
        derived_coords = self.cube.derived_coords
        assert len(derived_coords) == 1
        altitude = derived_coords[0]
        assert altitude.standard_name == "altitude"
        assert altitude.attributes == {"positive": "up"}

    def test_aux_factory(self):
        factory = self.cube.aux_factory(name="altitude")
        assert factory.standard_name == "altitude"
        assert factory.attributes == {"positive": "up"}

    def test_aux_factory_var_name(self):
        factory = self.cube.aux_factory(name="altitude")
        factory.var_name = "alt"
        factory = self.cube.aux_factory(var_name="alt")
        assert factory.standard_name == "altitude"
        assert factory.attributes == {"positive": "up"}

    def test_no_orography(self, request):
        # Get rid of the normal hybrid-height factory.
        cube = self.cube
        factory = cube.aux_factory(name="altitude")
        cube.remove_aux_factory(factory)

        # Add a new one which only references level_height & sigma.
        delta = cube.coord("level_height")
        sigma = cube.coord("sigma")
        factory = HybridHeightFactory(delta, sigma)
        cube.add_aux_factory(factory)

        assert len(cube.aux_factories) == 1
        assert len(cube.derived_coords) == 1
        _shared_utils.assert_string(
            request, str(cube), ("derived", "no_orog.__str__.txt")
        )
        _shared_utils.assert_CML(request, cube, ("derived", "no_orog.cml"))

    def test_invalid_dependencies(self):
        # Must have either delta or orography
        with pytest.raises(ValueError, match="Unable to determine units"):
            _ = HybridHeightFactory()
        sigma = self.cube.coord("sigma")
        with pytest.raises(ValueError, match="Unable to determine units"):
            _ = HybridHeightFactory(sigma=sigma)

        # Orography must not have bounds
        with pytest.warns(IrisIgnoringBoundsWarning):
            with contextlib.suppress(ValueError):
                _ = HybridHeightFactory(orography=sigma)

    def test_bounded_orography(self):
        # Start with everything normal
        orog = self.cube.coord("surface_altitude")
        altitude = self.cube.coord("altitude")
        assert isinstance(altitude.bounds, np.ndarray)

        # Make sure altitude still works OK if orography was messed
        # with *after* altitude was created.
        orog.bounds = np.zeros(orog.shape + (4,))

        # Check that altitude derivation now issues a warning.
        msg = "Orography.* bounds.* being disregarded"
        with pytest.warns(IrisIgnoringBoundsWarning, match=msg):
            _ = self.cube.coord("altitude")


@_shared_utils.skip_data
class TestHybridPressure:
    @pytest.fixture(autouse=True)
    def _setup(self):
        # Convert the hybrid-height into hybrid-pressure...
        cube = iris.tests.stock.realistic_4d()

        # Get rid of the normal hybrid-height factory.
        factory = cube.aux_factory(name="altitude")
        cube.remove_aux_factory(factory)

        # Mangle the height coords into pressure coords.
        delta = cube.coord("level_height")
        delta.rename("level_pressure")
        delta.units = "Pa"
        sigma = cube.coord("sigma")
        ref = cube.coord("surface_altitude")
        ref.rename("surface_air_pressure")
        ref.units = "Pa"

        factory = HybridPressureFactory(delta, sigma, ref)
        cube.add_aux_factory(factory)
        self.cube = cube
        self.air_pressure = self.cube.coord("air_pressure")

    def test_metadata(self):
        assert self.air_pressure.units == "Pa"
        assert self.air_pressure.coord_system is None
        assert self.air_pressure.attributes == {}

    def test_points(self):
        points = self.air_pressure.points
        assert points.dtype == np.float32
        assert points.min() == pytest.approx(np.float32(191.84892))
        assert points.max() == pytest.approx(np.float32(40000))

        # Convert the reference surface to float64 and check the
        # derived coordinate becomes float64.
        temp = self.cube.coord("surface_air_pressure").points
        temp = temp.astype("f8")
        self.cube.coord("surface_air_pressure").points = temp
        points = self.cube.coord("air_pressure").points
        assert points.dtype == np.float64
        assert points.min() == pytest.approx(191.8489257)
        assert points.max() == pytest.approx(40000)

    def test_invalid_dependencies(self):
        # Must have either delta or surface_air_pressure
        with pytest.raises(ValueError, match="insufficient source coordinates"):
            _ = HybridPressureFactory()
        sigma = self.cube.coord("sigma")
        with pytest.raises(ValueError, match="insufficient source coordinates"):
            _ = HybridPressureFactory(sigma=sigma)

        # Surface pressure must not have bounds
        with pytest.warns(IrisIgnoringBoundsWarning):
            with contextlib.suppress(ValueError):
                _ = HybridPressureFactory(sigma=sigma, surface_air_pressure=sigma)

    def test_bounded_surface_pressure(self):
        # Start with everything normal
        surface_pressure = self.cube.coord("surface_air_pressure")
        pressure = self.cube.coord("air_pressure")
        assert isinstance(pressure.bounds, np.ndarray)

        # Make sure pressure still works OK if surface pressure was messed
        # with *after* pressure was created.
        surface_pressure.bounds = np.zeros(surface_pressure.shape + (4,))

        # Check that air_pressure derivation now issues a warning.
        msg = "Surface pressure.* bounds.* being disregarded"
        with pytest.warns(IrisIgnoringBoundsWarning, match=msg):
            self.cube.coord("air_pressure")
