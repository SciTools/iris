# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.cf.CFGridMappingVariable`."""

import warnings

import pytest

from iris.fileformats.cf import CFGridMappingVariable
import iris.warnings

from .identify_mixins import _NetCDFVar, assert_warning_gated

CF_IDENTITY = "grid_mapping"


class TestIdentify:
    def test_no_coord_system_mappings_returns_empty(self):
        """When coord_system_mappings is absent, no results."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        crs_var = _NetCDFVar("crs_var")
        vars_all = {"crs_var": crs_var, "ref_source": ref_source}

        result = CFGridMappingVariable.identify(vars_all, coord_system_mappings=None)
        assert result == {}

    def test_no_mapping_entry_for_source_returns_empty(self):
        """Data var has grid_mapping attr but no entry in coord_system_mappings."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        crs_var = _NetCDFVar("crs_var")
        vars_all = {"crs_var": crs_var, "ref_source": ref_source}

        # Mapping dict exists but has no entry for "ref_source".
        result = CFGridMappingVariable.identify(
            vars_all, coord_system_mappings={"other_var": {"crs_var": [None]}}
        )
        assert result == {}

    def test_simple_mapping_none_coord_identified(self):
        """A mapping with coord=None (simple grid_mapping style) is accepted."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        crs_var = _NetCDFVar("crs_var")
        vars_all = {"crs_var": crs_var, "ref_source": ref_source}

        # {coord_name -> cs_name}; None coord means simple style.
        cs_mappings = {"ref_source": {None: "crs_var"}}

        result = CFGridMappingVariable.identify(
            vars_all, coord_system_mappings=cs_mappings
        )
        assert "crs_var" in result
        assert isinstance(result["crs_var"], CFGridMappingVariable)

    def test_valid_coord_ref_identified(self):
        """Mapping with a real coordinate reference that exists is accepted."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        crs_var = _NetCDFVar("crs_var")
        coord_var = _NetCDFVar("lat")
        vars_all = {
            "crs_var": crs_var,
            "lat": coord_var,
            "ref_source": ref_source,
        }

        cs_mappings = {"ref_source": {"lat": "crs_var"}}

        result = CFGridMappingVariable.identify(
            vars_all, coord_system_mappings=cs_mappings
        )
        assert "crs_var" in result

    def test_missing_mapping_variable_warns(self):
        """Missing grid mapping variable itself emits a warning."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        # crs_var is NOT in vars_all
        vars_all = {"ref_source": ref_source}

        cs_mappings = {"ref_source": {None: "crs_var"}}

        def operation(warn: bool):
            warnings.warn(
                "emit at least 1 warning",
                category=iris.warnings.IrisUserWarning,
            )
            CFGridMappingVariable.identify(
                vars_all, coord_system_mappings=cs_mappings, warn=warn
            )

        warn_regex = r"Missing CF-netCDF grid mapping variable 'crs_var'.*"
        assert_warning_gated(
            operation, iris.warnings.IrisCfMissingVarWarning, warn_regex
        )

    def test_missing_coord_ref_warns(self):
        """Missing coordinate associated with a grid mapping emits a warning.

        Note: this warning is not gated by the `warn` argument - it is always
        emitted when a referenced coordinate variable is absent.
        """
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        crs_var = _NetCDFVar("crs_var")
        # lat is NOT in vars_all
        vars_all = {"crs_var": crs_var, "ref_source": ref_source}

        cs_mappings = {"ref_source": {"lat": "crs_var"}}

        warn_regex = r"Missing CF-netCDF coordinate variable 'lat'.*"
        with pytest.warns(iris.warnings.IrisCfMissingVarWarning, match=warn_regex):
            CFGridMappingVariable.identify(vars_all, coord_system_mappings=cs_mappings)

    def test_ignore(self):
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, CF_IDENTITY, "crs_var")
        crs_var = _NetCDFVar("crs_var")
        vars_all = {"crs_var": crs_var, "ref_source": ref_source}

        cs_mappings = {"ref_source": {None: "crs_var"}}

        result = CFGridMappingVariable.identify(
            vars_all, ignore=["crs_var"], coord_system_mappings=cs_mappings
        )
        assert result == {}

    def test_target_unknown_raises(self):
        vars_all = {"ref_source": _NetCDFVar("ref_source")}

        message = "Cannot identify unknown target CF-netCDF variable 'unknown'"
        with pytest.raises(ValueError, match=message):
            CFGridMappingVariable.identify(vars_all, target="unknown")

    def test_target_wrong_type_raises(self):
        vars_all = {"ref_source": _NetCDFVar("ref_source")}

        message = "Expect a target CF-netCDF variable name"
        with pytest.raises(TypeError, match=message):
            CFGridMappingVariable.identify(vars_all, target=object())


class TestIdentifyGroupingByCRS:
    """Tests exercising the cs_coord_mappings grouping (lines 698-708 of cf.py).

    The grouping step inverts {coord -> cs_name} into {cs_name -> [coords]},
    so that each unique coordinate system is iterated once.
    """

    def test_multiple_coords_one_cs(self):
        """Two coordinates both mapping to the same CRS: one CRS identified."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, "grid_mapping", "crs_1")
        crs_1 = _NetCDFVar("crs_1")
        lat = _NetCDFVar("lat")
        lon = _NetCDFVar("lon")
        vars_all = {
            "crs_1": crs_1,
            "lat": lat,
            "lon": lon,
            "ref_source": ref_source,
        }

        # Both lat and lon reference the same coordinate system.
        cs_mappings = {"ref_source": {"lat": "crs_1", "lon": "crs_1"}}

        result = CFGridMappingVariable.identify(
            vars_all, coord_system_mappings=cs_mappings
        )
        assert list(result.keys()) == ["crs_1"]

    def test_multiple_cs_both_identified(self):
        """Two coordinates each referencing a different CRS: both CRSs identified."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, "grid_mapping", "crs_1 crs_2")
        crs_1 = _NetCDFVar("crs_1")
        crs_2 = _NetCDFVar("crs_2")
        lat = _NetCDFVar("lat")
        height = _NetCDFVar("height")
        vars_all = {
            "crs_1": crs_1,
            "crs_2": crs_2,
            "lat": lat,
            "height": height,
            "ref_source": ref_source,
        }

        cs_mappings = {"ref_source": {"lat": "crs_1", "height": "crs_2"}}

        result = CFGridMappingVariable.identify(
            vars_all, coord_system_mappings=cs_mappings
        )
        assert set(result.keys()) == {"crs_1", "crs_2"}

    def test_partial_coords_missing_cs_still_identified(self):
        """One coord present, one missing for the same CRS: CRS still identified
        (has_a_valid_coord is True from the present coord), but a warning is
        issued for the missing one.
        """
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, "grid_mapping", "crs_1")
        crs_1 = _NetCDFVar("crs_1")
        lat = _NetCDFVar("lat")
        # lon is intentionally absent from vars_all
        vars_all = {
            "crs_1": crs_1,
            "lat": lat,
            "ref_source": ref_source,
        }

        cs_mappings = {"ref_source": {"lat": "crs_1", "lon": "crs_1"}}

        warn_regex = r"Missing CF-netCDF coordinate variable 'lon'.*"
        with pytest.warns(iris.warnings.IrisCfMissingVarWarning, match=warn_regex):
            result = CFGridMappingVariable.identify(
                vars_all, coord_system_mappings=cs_mappings
            )
        # CRS still in result because lat was valid.
        assert "crs_1" in result

    def test_all_coords_missing_cs_excluded(self):
        """All coords missing for a CRS: has_a_valid_coord stays False, CRS excluded."""
        ref_source = _NetCDFVar("ref_source")
        setattr(ref_source, "grid_mapping", "crs_1")
        crs_1 = _NetCDFVar("crs_1")
        # Both lat and lon absent from vars_all
        vars_all = {"crs_1": crs_1, "ref_source": ref_source}

        cs_mappings = {"ref_source": {"lat": "crs_1", "lon": "crs_1"}}

        warn_regex = "Missing CF-netCDF coordinate variable"
        with pytest.warns(iris.warnings.IrisCfMissingVarWarning, match=warn_regex):
            result = CFGridMappingVariable.identify(
                vars_all, coord_system_mappings=cs_mappings
            )
        assert result == {}
