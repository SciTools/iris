# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the
`iris.aux_factory.HybridLogPressureFactory` class.

"""

from unittest.mock import Mock

import cf_units
import numpy as np
import pytest

import iris
from iris.aux_factory import HybridLogPressureFactory


def create_default_sample_parts(self):
    self.eta = Mock(units=cf_units.Unit("1"), nbounds=0)
    self.sigma = Mock(units=cf_units.Unit("1"), nbounds=0)
    self.surface_air_pressure = Mock(units=cf_units.Unit("Pa"), nbounds=0)
    self.reference_air_pressure = Mock(units=cf_units.Unit("Pa"), nbounds=0)
    self.factory = HybridLogPressureFactory(
        eta=self.eta,
        sigma=self.sigma,
        surface_air_pressure=self.surface_air_pressure,
        reference_air_pressure=self.reference_air_pressure,
    )


class Test___init__:
    @pytest.fixture(autouse=True)
    def _setup(self):
        create_default_sample_parts(self)

    def test_insufficient_coords(self):
        msg = "Unable to construct hybrid log-pressure coordinate factory due to insufficient source coordinates."
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory()
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=None,
                sigma=self.sigma,
                surface_air_pressure=None,
                reference_air_pressure=None,
            )
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=None,
                sigma=None,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_incompatible_eta_units(self):
        self.eta.units = cf_units.Unit("m")
        msg = "Invalid units: eta must be dimensionless."
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=self.eta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_incompatible_sigma_units(self):
        self.sigma.units = cf_units.Unit("Pa")
        msg = "Invalid units: sigma must be dimensionless."
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=self.eta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_incompatible_surface_air_pressure_units(self):
        self.surface_air_pressure.units = cf_units.Unit("unknown")
        msg = "Incompatible units: reference_air_pressure and surface_air_pressure must have the same units."
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=self.eta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_different_pressure_units(self):
        self.reference_air_pressure.units = cf_units.Unit("hPa")
        self.surface_air_pressure.units = cf_units.Unit("Pa")
        msg = (
            "Incompatible units: reference_air_pressure and "
            "surface_air_pressure must have the same units."
        )
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=self.eta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_too_many_eta_bounds(self):
        self.eta.nbounds = 4
        msg = "Invalid eta coordinate: must have either 0 or 2 bounds."
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=self.eta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_too_many_sigma_bounds(self):
        self.sigma.nbounds = 4
        msg = "Invalid sigma coordinate: must have either 0 or 2 bounds."
        with pytest.raises(ValueError, match=msg):
            HybridLogPressureFactory(
                eta=self.eta,
                sigma=self.sigma,
                surface_air_pressure=self.surface_air_pressure,
                reference_air_pressure=self.reference_air_pressure,
            )

    def test_factory_metadata(self):
        factory = HybridLogPressureFactory(
            eta=self.eta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
            reference_air_pressure=self.reference_air_pressure,
        )
        assert factory.standard_name == "air_pressure"
        assert factory.long_name is None
        assert factory.var_name is None
        assert factory.units == self.reference_air_pressure.units
        assert factory.units == self.surface_air_pressure.units
        assert factory.coord_system is None
        assert factory.attributes == {}

    def test_promote_sigma_units_unknown_to_dimensionless(self):
        sigma = Mock(units=cf_units.Unit("unknown"), nbounds=0)
        factory = HybridLogPressureFactory(
            eta=self.eta,
            sigma=sigma,
            surface_air_pressure=self.surface_air_pressure,
            reference_air_pressure=self.reference_air_pressure,
        )
        assert factory.dependencies["sigma"].units == "1"


class Test_dependencies:
    @pytest.fixture(autouse=True)
    def _setup(self):
        create_default_sample_parts(self)

    def test_value(self):
        kwargs = dict(
            eta=self.eta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
            reference_air_pressure=self.reference_air_pressure,
        )
        factory = HybridLogPressureFactory(**kwargs)
        assert factory.dependencies == kwargs


class Test_make_coord:
    @staticmethod
    def coords_dims_func(coord):
        mapping = dict(
            level_pressure=(0,),
            sigma=(0,),
            surface_air_pressure=(1, 2),
            reference_air_pressure=(0,),
        )
        return mapping[coord.name()]

    @pytest.fixture(autouse=True)
    def _setup(self):
        # Create standard data objects for coord testing
        self.eta = iris.coords.DimCoord(
            [0.0, 1.0, 2.0], long_name="level_pressure", units="1"
        )
        self.sigma = iris.coords.DimCoord([1.0, 0.9, 0.8], long_name="sigma", units="1")
        self.surface_air_pressure = iris.coords.AuxCoord(
            np.arange(4).reshape(2, 2), "surface_air_pressure", units="Pa"
        )
        self.reference_air_pressure = iris.coords.AuxCoord(
            np.array(1), long_name="reference_air_pressure", units="Pa"
        )

    def test_points_only(self):
        # Determine expected coord by manually broadcasting coord points
        # knowing the dimension mapping.
        eta_pts = self.eta.points[..., np.newaxis, np.newaxis]
        sigma_pts = self.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = self.surface_air_pressure.points[np.newaxis, ...]
        ref_pts = self.reference_air_pressure.points
        expected_points = ref_pts * eta_pts * (surf_pts / ref_pts) ** sigma_pts
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridLogPressureFactory(
            eta=self.eta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
            reference_air_pressure=self.reference_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord

    def test_none_surface_air_pressure(self):
        # Note absence of broadcasting as multidimensional coord
        # is not present.
        expected_points = self.eta.points * 0
        expected_coord = iris.coords.AuxCoord(
            expected_points, standard_name="air_pressure", units="Pa"
        )
        factory = HybridLogPressureFactory(
            eta=self.eta,
            sigma=self.sigma,
            reference_air_pressure=self.reference_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)

        assert derived_coord == expected_coord

    def test_with_bounds(self):
        self.eta.guess_bounds(0)
        self.sigma.guess_bounds(0.5)
        # Determine expected coord by manually broadcasting coord points
        # and bounds based on the dimension mapping.
        eta_pts = self.eta.points[..., np.newaxis, np.newaxis]
        sigma_pts = self.sigma.points[..., np.newaxis, np.newaxis]
        surf_pts = self.surface_air_pressure.points[np.newaxis, ...]
        ref_pts = self.reference_air_pressure.points
        expected_points = ref_pts * eta_pts * (surf_pts / ref_pts) ** sigma_pts
        eta_vals = self.eta.bounds.reshape(3, 1, 1, 2)
        sigma_vals = self.sigma.bounds.reshape(3, 1, 1, 2)
        surf_vals = self.surface_air_pressure.points.reshape(1, 2, 2, 1)
        ref_vals = self.reference_air_pressure.points
        expected_bounds = ref_vals * eta_vals * (surf_vals / ref_vals) ** sigma_vals
        expected_coord = iris.coords.AuxCoord(
            expected_points,
            standard_name="air_pressure",
            units="Pa",
            bounds=expected_bounds,
        )
        factory = HybridLogPressureFactory(
            eta=self.eta,
            sigma=self.sigma,
            surface_air_pressure=self.surface_air_pressure,
            reference_air_pressure=self.reference_air_pressure,
        )
        derived_coord = factory.make_coord(self.coords_dims_func)
        assert derived_coord == expected_coord


class Test_update:
    @pytest.fixture(autouse=True)
    def _setup(self):
        create_default_sample_parts(self)

    def test_good_reference_air_pressure(self):
        new_ref_coord = Mock(units=cf_units.Unit("Pa"), nbounds=0)
        self.factory.update(self.reference_air_pressure, new_ref_coord)
        assert self.factory.reference_air_pressure is new_ref_coord

    def test_bad_reference_air_pressure(self):
        new_ref_coord = Mock(units=cf_units.Unit("1"), nbounds=0)
        msg = "Failed to update dependencies. Incompatible units: reference_air_pressure and surface_air_pressure must have the same units."
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.reference_air_pressure, new_ref_coord)

    def test_alternative_bad_eta(self):
        new_eta_coord = Mock(units=cf_units.Unit("Pa"), nbounds=4)
        msg = "Failed to update dependencies. Invalid eta coordinate: must have either 0 or 2 bounds."
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.eta, new_eta_coord)

    def test_good_surface_air_pressure(self):
        new_surface_p_coord = Mock(units=cf_units.Unit("Pa"), nbounds=0)
        self.factory.update(self.surface_air_pressure, new_surface_p_coord)
        assert self.factory.surface_air_pressure is new_surface_p_coord

    def test_bad_surface_air_pressure(self):
        new_surface_p_coord = Mock(units=cf_units.Unit("km"), nbounds=0)
        msg = "Failed to update dependencies. Incompatible units: reference_air_pressure and surface_air_pressure must have the same units."
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.surface_air_pressure, new_surface_p_coord)

    def test_non_dependency(self):
        old_coord = Mock()
        new_coord = Mock()
        orig_dependencies = self.factory.dependencies
        self.factory.update(old_coord, new_coord)
        assert self.factory.dependencies == orig_dependencies

    def test_none_eta(self):
        msg = "Failed to update dependencies. Unable to construct hybrid log-pressure coordinate factory due to insufficient source coordinates."
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.eta, None)

    def test_none_sigma(self):
        msg = "Failed to update dependencies. Unable to construct hybrid log-pressure coordinate factory due to insufficient source coordinates."
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.sigma, None)

    def test_insufficient_coords(self):
        self.factory.update(self.surface_air_pressure, None)
        msg = "Failed to update dependencies. Unable to construct hybrid log-pressure coordinate factory due to insufficient source coordinates."
        with pytest.raises(ValueError, match=msg):
            self.factory.update(self.reference_air_pressure, None)
