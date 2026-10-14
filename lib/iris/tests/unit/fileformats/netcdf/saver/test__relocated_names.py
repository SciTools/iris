# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Names that must survive the CF saver's move to iris.fileformats.cf.

Two separate promises. Four public names moved and are still reachable from
their old homes, warning when they are. Every method of the netCDF Saver kept
its name, so that code subclassing it keeps overriding what it overrode.
"""

import inspect
import warnings

import numpy as np
import pytest

import iris
from iris._deprecation import IrisDeprecation
from iris.cube import Cube
from iris.fileformats import netcdf
from iris.fileformats.cf import saver as cf_saver
from iris.fileformats.netcdf import saver as netcdf_saver
from iris.loading import LOAD_PROBLEMS

#: The four names spec section 4.8 relocates, with warn-on-use aliases left
#: behind at both their old homes.
RELOCATED = [
    "CFNameCoordMap",
    "CF_CONVENTIONS_VERSION",
    "MESH_ELEMENTS",
    "SPATIO_TEMPORAL_AXES",
]

#: Every method the netCDF Saver answered to before the generic half of it
#: became CFSaver. A name missing from the class is an override, somewhere
#: outside Iris, that has silently stopped overriding anything.
SAVER_METHODS = [
    "_add_ancillary_variables",
    "_add_aux_coords",
    "_add_aux_factories",
    "_add_cell_measures",
    "_add_dim_coords",
    "_add_grid_mapping_to_dataset",
    "_add_inner_related_vars",
    "_add_mesh",
    "_backend_write",
    "_cf_coord_standardised_units",
    "_create_cf_bounds",
    "_create_cf_cell_methods",
    "_create_cf_data_variable",
    "_create_cf_dimensions",
    "_create_cf_grid_mapping",
    "_create_generic_cf_array_var",
    "_create_mesh",
    "_create_string_variable",
    "_ensure_valid_dtype",
    "_flush_before_close",
    "_get_dim_names",
    "_get_element_variable_name",
    "_get_mesh_variable_name",
    "_increment_name",
    "_lazy_stream_data",
    "_patch_dataset",
    "_set_cf_var_attributes",
    "cf_valid_var_name",
    "check_attribute_compliance",
    "complete",
    "delayed_completion",
    "update_global_attributes",
    "write",
]


@pytest.mark.parametrize("name", RELOCATED)
class TestFromTheSaverModule:
    def test_resolves_to_the_same_object(self, name):
        with pytest.warns(IrisDeprecation):
            assert getattr(netcdf_saver, name) is getattr(cf_saver, name)

    def test_the_warning_names_the_new_home(self, name):
        with pytest.warns(IrisDeprecation, match=f"iris.fileformats.cf.saver.{name}"):
            getattr(netcdf_saver, name)

    def test_is_discoverable(self, name):
        # PEP 562 forwards the read but not dir(), so the module says so itself.
        assert name in dir(netcdf_saver)


@pytest.mark.parametrize("name", RELOCATED)
class TestFromThePackage:
    def test_resolves_to_the_same_object(self, name):
        with pytest.warns(IrisDeprecation):
            assert getattr(netcdf, name) is getattr(cf_saver, name)

    def test_is_discoverable(self, name):
        assert name in dir(netcdf)


class TestNamesThatDoNotMove:
    @pytest.mark.parametrize("name", ["Saver", "save", "save_mesh"])
    def test_the_saver_module_still_defines_it(self, name):
        # Spec section 4.8: netCDF API, and it stays that way. Defined here,
        # not forwarded, so reading it must not warn.
        assert name in vars(netcdf_saver)

    def test_an_unknown_name_raises_quietly(self, recwarn):
        with pytest.raises(AttributeError, match="no attribute"):
            netcdf_saver.not_a_real_name
        assert recwarn.list == []


class TestSaverMethods:
    @pytest.mark.parametrize("name", SAVER_METHODS)
    def test_the_netcdf_saver_still_answers_to_it(self, name):
        assert callable(getattr(netcdf_saver.Saver, name, None))

    def test_the_netcdf_saver_is_a_cf_saver(self):
        assert issubclass(netcdf_saver.Saver, cf_saver.CFSaver)

    def test_write_keeps_its_signature(self):
        # Documented API with defaults users rely on, and save() passes some
        # of them by position.
        parameters = inspect.signature(netcdf_saver.Saver.write).parameters
        assert list(parameters) == [
            "self",
            "cube",
            "local_keys",
            "unlimited_dimensions",
            "zlib",
            "complevel",
            "shuffle",
            "fletcher32",
            "contiguous",
            "chunksizes",
            "endian",
            "least_significant_digit",
            "packing",
            "fill_value",
        ]
        assert parameters["complevel"].default == 4
        assert parameters["shuffle"].default is True
        assert parameters["endian"].default == "native"


class TestNothingInsideIrisUsesTheOldNames:
    def test_a_save_emits_no_deprecation(self, tmp_path):
        # The aliases are for code outside Iris. A save that trips one means a
        # reader inside Iris was left pointing at the old home. Promoting to an
        # error is safe here: iris.save has no equivalent of LOAD_PROBLEMS, so
        # an exception reaches the test rather than being recorded.
        #
        # save_split_attrs is set because the legacy attribute mode is itself
        # deprecated, and its warning would fail this test for a reason that
        # has nothing to do with where the saver lives.
        with warnings.catch_warnings():
            warnings.simplefilter("error", IrisDeprecation)
            with iris.FUTURE.context(save_split_attrs=True):
                iris.save(
                    Cube(np.arange(3.0), var_name="payload"), tmp_path / "quiet.nc"
                )

    def test_a_load_emits_no_deprecation(self, tmp_path):
        # Recorded rather than promoted to an error, for the reason
        # SciTools/iris#7333 found: iris.load captures an exception raised
        # while building a cube into iris.loading.LOAD_PROBLEMS, so a filter of
        # "error" drops the offending coordinate and the test passes against an
        # implementation that is still warning. The LOAD_PROBLEMS assertion is
        # what keeps a silent load from being a degraded one.
        path = tmp_path / "quiet.nc"
        with iris.FUTURE.context(save_split_attrs=True):
            iris.save(Cube(np.arange(3.0), var_name="payload"), path)

        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            cubes = iris.load(path)

        deprecations = [
            str(warning.message)
            for warning in record
            if issubclass(warning.category, IrisDeprecation)
        ]
        assert deprecations == []
        assert LOAD_PROBLEMS.problems_by_file.get(str(path), []) == []
        assert len(cubes) == 1
