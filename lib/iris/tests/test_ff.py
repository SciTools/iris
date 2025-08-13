# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the Fieldsfile file loading plugin and FFHeader."""

import collections

import numpy as np
import pytest

import iris
import iris.fileformats._ff as ff
import iris.fileformats.pp as pp
from iris.tests import _shared_utils


class TestFF_HEADER:
    def test_initialisation(self):
        assert ff.FF_HEADER[0] == ("data_set_format_version", (0,))
        assert ff.FF_HEADER[17] == ("integer_constants", (99, 100))

    def test_size(self):
        assert len(ff.FF_HEADER) == 31


@_shared_utils.skip_data
class TestFFHeader:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.filename = _shared_utils.get_data_path(("FF", "n48_multi_field"))
        self.ff_header = ff.FFHeader(self.filename)
        self.valid_headers = (
            "integer_constants",
            "real_constants",
            "level_dependent_constants",
            "lookup_table",
            "data",
        )
        self.invalid_headers = (
            "row_dependent_constants",
            "column_dependent_constants",
            "fields_of_constants",
            "extra_constants",
            "temp_historyfile",
            "compressed_field_index1",
            "compressed_field_index2",
            "compressed_field_index3",
        )

    def test_constructor(self):
        # Test FieldsFile header attribute lookup.
        assert self.ff_header.data_set_format_version == 20
        assert self.ff_header.sub_model == 1
        assert self.ff_header.vert_coord_type == 5
        assert self.ff_header.horiz_grid_type == 0
        assert self.ff_header.dataset_type == 3
        assert self.ff_header.run_identifier == 0
        assert self.ff_header.experiment_number == -32768
        assert self.ff_header.calendar == 1
        assert self.ff_header.grid_staggering == 3
        assert self.ff_header.time_type == -32768
        assert self.ff_header.projection_number == -32768
        assert self.ff_header.model_version == 802
        assert self.ff_header.obs_file_type == -32768
        assert self.ff_header.last_fieldop_type == -32768
        assert self.ff_header.first_validity_time == (2011, 7, 10, 18, 0, 0, 191)
        assert self.ff_header.last_validity_time == (2011, 7, 10, 21, 0, 0, 191)
        assert self.ff_header.misc_validity_time == (2012, 4, 30, 18, 12, 13, -32768)
        assert self.ff_header.integer_constants.shape == (46,)
        assert self.ff_header.real_constants.shape == (38,)
        assert self.ff_header.level_dependent_constants.shape == (71, 8)
        assert self.ff_header.row_dependent_constants is None
        assert self.ff_header.column_dependent_constants is None
        assert self.ff_header.fields_of_constants is None
        assert self.ff_header.extra_constants is None
        assert self.ff_header.temp_historyfile is None
        assert self.ff_header.compressed_field_index1 is None
        assert self.ff_header.compressed_field_index2 is None
        assert self.ff_header.compressed_field_index3 is None
        assert self.ff_header.lookup_table == (909, 64, 5)
        assert self.ff_header.total_prognostic_fields == 3119
        assert self.ff_header.data == (2049, 2961, -32768)

    def test_str(self, request):
        _shared_utils.assert_string(
            request, str(self.ff_header), ("FF", "ffheader.txt")
        )

    def test_repr(self):
        target = "FFHeader('" + self.filename + "')"
        assert repr(self.ff_header) == target

    def test_shape(self):
        assert self.ff_header.shape("data") == (2961, -32768)


@_shared_utils.skip_data
class TestFF2PP2Cube:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.filename = _shared_utils.get_data_path(("FF", "n48_multi_field"))

    def test_unit_pass_0(self, request):
        # Test FieldsFile to PPFields cube load.
        cube_by_name = collections.defaultdict(int)
        cubes = iris.load(self.filename)
        while cubes:
            cube = cubes.pop(0)
            standard_name = cube.standard_name
            cube_by_name[standard_name] += 1
            filename = "{}_{}.cml".format(standard_name, cube_by_name[standard_name])
            _shared_utils.assert_CML(request, cube, ("FF", filename))

    def test_raw_to_table_count(self):
        filename = _shared_utils.get_data_path(("FF", "n48_multi_field_table_count"))
        cubes = iris.load_raw(filename)
        ff_header = ff.FFHeader(filename)
        table_count = ff_header.lookup_table[2]
        assert len(cubes) == table_count


@_shared_utils.skip_data
class TestFFieee32:
    def test_iris_loading(self):
        ff32_fname = _shared_utils.get_data_path(("FF", "n48_multi_field.ieee32"))
        ff64_fname = _shared_utils.get_data_path(("FF", "n48_multi_field"))

        ff32_cubes = iris.load(ff32_fname)
        ff64_cubes = iris.load(ff64_fname)

        for ff32, ff64 in zip(ff32_cubes, ff64_cubes):
            # load the data
            _, _ = ff32.data, ff64.data
            assert ff32 == ff64


@_shared_utils.skip_data
class TestFFVariableResolutionGrid:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.filename = _shared_utils.get_data_path(("FF", "n48_multi_field"))

        self.ff2pp = ff.FF2PP(self.filename)
        self.ff_header = self.ff2pp._ff_header

        data_shape = (73, 96)
        delta = np.sin(np.linspace(0, np.pi * 5, data_shape[1])) * 5
        lons = np.linspace(0, 180, data_shape[1]) + delta
        lons = np.vstack([lons[:-1], lons[:-1] + 0.5 * np.diff(lons)]).T
        lons = np.reshape(lons, lons.shape, order="F")

        delta = np.sin(np.linspace(0, np.pi * 5, data_shape[0])) * 5
        lats = np.linspace(-90, 90, data_shape[0]) + delta
        lats = np.vstack([lats[:-1], lats[:-1] + 0.5 * np.diff(lats)]).T
        lats = np.reshape(lats, lats.shape, order="F")

        self.ff_header.column_dependent_constants = lons
        self.ff_header.row_dependent_constants = lats

        self.U_grid_x = lons[:-1, 1]
        self.V_grid_y = lats[:-1, 1]
        self.P_grid_x = lons[:, 0]
        self.P_grid_y = lats[:, 0]

        self.orig_make_pp_field = pp.make_pp_field

        def new_make_pp_field(header):
            field = self.orig_make_pp_field(header)
            field.stash = self.ff2pp._custom_stash
            field.bdx = field.bdy = field.bmdi
            return field

        # Replace the pp module function with this new function;
        # this gets called in PP2FF.
        pp.make_pp_field = new_make_pp_field

    def teardown_method(self):
        pp.make_pp_field = self.orig_make_pp_field

    def _check_stash(self, stash, x_coord, y_coord):
        self.ff2pp._custom_stash = stash
        field = next(iter(self.ff2pp))
        _shared_utils.assert_array_equal(
            x_coord,
            field.x,
            ("x_coord was incorrect for stash {}".format(stash)),
        )
        _shared_utils.assert_array_equal(
            y_coord,
            field.y,
            ("y_coord was incorrect for stash {}".format(stash)),
        )

    def test_p(self):
        self._check_stash("m01s00i001", self.P_grid_x, self.P_grid_y)

    def test_u(self):
        self._check_stash("m01s00i002", self.U_grid_x, self.P_grid_y)

    def test_v(self):
        self._check_stash("m01s00i003", self.P_grid_x, self.V_grid_y)
