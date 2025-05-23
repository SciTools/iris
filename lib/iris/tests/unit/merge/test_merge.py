# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Test the cube merging mechanism."""

from collections.abc import Iterable
import itertools

import numpy as np
import numpy.ma as ma
import pytest

import iris
from iris._lazy_data import as_lazy_data
from iris.coords import AuxCoord, DimCoord
import iris.cube
from iris.cube import CubeAttrsDict
import iris.exceptions
from iris.tests._shared_utils import assert_array_equal, assert_masked_array_equal


class TestDataMergeCombos:
    def _make_data(
        self,
        data,
        dtype=np.dtype("int32"),
        fill_value=None,
        mask=None,
        lazy=False,
        N=3,
    ):
        if isinstance(data, Iterable):
            shape = (len(data), N, N)
            data = np.array(data).reshape(-1, 1, 1)
        else:
            shape = (N, N)
        if mask is not None:
            payload = ma.empty(shape, dtype=dtype, fill_value=fill_value)
            payload.data[:] = data
            if isinstance(mask, bool):
                payload.mask = mask
            else:
                payload[mask] = ma.masked
        else:
            payload = np.empty(shape, dtype=dtype)
            payload[:] = data
        if lazy:
            payload = as_lazy_data(payload)
        return payload

    def _make_cube(
        self,
        data,
        dtype=np.dtype("int32"),
        fill_value=None,
        mask=None,
        lazy=False,
        N=3,
    ):
        x = np.arange(N)
        y = np.arange(N)
        payload = self._make_data(
            data, dtype=dtype, fill_value=fill_value, mask=mask, lazy=lazy, N=N
        )
        cube = iris.cube.Cube(payload)
        lat = DimCoord(y, standard_name="latitude", units="degrees")
        cube.add_dim_coord(lat, 0)
        lon = DimCoord(x, standard_name="longitude", units="degrees")
        cube.add_dim_coord(lon, 1)
        height = DimCoord(data, standard_name="height", units="m")
        cube.add_aux_coord(height)
        return cube

    @staticmethod
    def _expected_fill_value(fill0="none", fill1="none"):
        result = None
        if fill0 != "none" or fill1 != "none":
            if fill0 == "none":
                result = fill1
            elif fill1 == "none":
                result = fill0
            elif fill0 == fill1:
                result = fill0
        return result

    def _check_fill_value(self, result, fill0="none", fill1="none"):
        expected_fill_value = self._expected_fill_value(fill0, fill1)
        if expected_fill_value is None:
            data = result.data
            if ma.isMaskedArray(data):
                np_fill_value = ma.masked_array(0, dtype=result.dtype).fill_value
                assert data.fill_value == np_fill_value
        else:
            data = result.data
            if ma.isMaskedArray(data):
                assert data.fill_value == expected_fill_value

    def setup_method(self):
        self.dtype = np.dtype("int32")
        fill_value = 1234
        self.lazy_combos = itertools.product([False, True], [False, True])
        fill_combos = itertools.product([None, fill_value], [fill_value, None])
        single_fill_combos = itertools.product([None, fill_value])
        self.combos = itertools.product(self.lazy_combos, fill_combos)
        self.mixed_combos = itertools.product(self.lazy_combos, single_fill_combos)

    def test__ndarray_ndarray(self):
        for lazy0, lazy1 in self.lazy_combos:
            cubes = iris.cube.CubeList()
            cubes.append(self._make_cube(0, dtype=self.dtype, lazy=lazy0))
            cubes.append(self._make_cube(1, dtype=self.dtype, lazy=lazy1))
            result = cubes.merge_cube()
            expected = self._make_data([0, 1], dtype=self.dtype)
            assert_array_equal(result.data, expected)
            assert result.dtype == self.dtype
            self._check_fill_value(result)

    def test__masked_masked(self):
        for (lazy0, lazy1), (fill0, fill1) in self.combos:
            cubes = iris.cube.CubeList()
            mask = ((0,), (0,))
            cubes.append(
                self._make_cube(
                    0,
                    mask=mask,
                    lazy=lazy0,
                    dtype=self.dtype,
                    fill_value=fill0,
                )
            )
            mask = ((1,), (1,))
            cubes.append(
                self._make_cube(
                    1,
                    mask=mask,
                    lazy=lazy1,
                    dtype=self.dtype,
                    fill_value=fill1,
                )
            )
            result = cubes.merge_cube()
            mask = ((0, 1), (0, 1), (0, 1))
            expected_fill_value = self._expected_fill_value(fill0, fill1)
            expected = self._make_data(
                [0, 1],
                mask=mask,
                dtype=self.dtype,
                fill_value=expected_fill_value,
            )
            assert_masked_array_equal(result.data, expected)
            assert result.dtype == self.dtype
            self._check_fill_value(result, fill0, fill1)

    def test__ndarray_masked(self):
        for (lazy0, lazy1), (fill,) in self.mixed_combos:
            cubes = iris.cube.CubeList()
            cubes.append(self._make_cube(0, lazy=lazy0, dtype=self.dtype))
            mask = [(0, 1), (0, 1)]
            cubes.append(
                self._make_cube(
                    1, mask=mask, lazy=lazy1, dtype=self.dtype, fill_value=fill
                )
            )
            result = cubes.merge_cube()
            mask = [(1, 1), (0, 1), (0, 1)]
            expected_fill_value = self._expected_fill_value(fill)
            expected = self._make_data(
                [0, 1],
                mask=mask,
                dtype=self.dtype,
                fill_value=expected_fill_value,
            )
            assert_masked_array_equal(result.data, expected)
            assert result.dtype == self.dtype
            self._check_fill_value(result, fill1=fill)

    def test__masked_ndarray(self):
        for (lazy0, lazy1), (fill,) in self.mixed_combos:
            cubes = iris.cube.CubeList()
            mask = [(0, 1), (0, 1)]
            cubes.append(
                self._make_cube(
                    0, mask=mask, lazy=lazy0, dtype=self.dtype, fill_value=fill
                )
            )
            cubes.append(self._make_cube(1, lazy=lazy1, dtype=self.dtype))
            result = cubes.merge_cube()
            mask = [(0, 0), (0, 1), (0, 1)]
            expected_fill_value = self._expected_fill_value(fill)
            expected = self._make_data(
                [0, 1],
                mask=mask,
                dtype=self.dtype,
                fill_value=expected_fill_value,
            )
            assert_masked_array_equal(result.data, expected)
            assert result.dtype == self.dtype
            self._check_fill_value(result, fill0=fill)

    def test_maksed_array_preserved(self):
        for (lazy0, lazy1), (fill,) in self.mixed_combos:
            cubes = iris.cube.CubeList()
            mask = False
            cubes.append(
                self._make_cube(
                    0, mask=mask, lazy=lazy0, dtype=self.dtype, fill_value=fill
                )
            )
            cubes.append(self._make_cube(1, lazy=lazy1, dtype=self.dtype))
            result = cubes.merge_cube()
            mask = False
            expected_fill_value = self._expected_fill_value(fill)
            expected = self._make_data(
                [0, 1],
                mask=mask,
                dtype=self.dtype,
                fill_value=expected_fill_value,
            )
            assert type(result.data) is ma.MaskedArray
            assert_masked_array_equal(result.data, expected)
            assert result.dtype == self.dtype
            self._check_fill_value(result, fill0=fill)

    def test_fill_value_invariant_to_order__same_non_none(self):
        fill_value = 1234
        cubes = [self._make_cube(i, mask=True, fill_value=fill_value) for i in range(3)]
        for combo in itertools.permutations(cubes):
            result = iris.cube.CubeList(combo).merge_cube()
            assert result.data.fill_value == fill_value

    def test_fill_value_invariant_to_order__all_none(self):
        cubes = [self._make_cube(i, mask=True, fill_value=None) for i in range(3)]
        for combo in itertools.permutations(cubes):
            result = iris.cube.CubeList(combo).merge_cube()
            np_fill_value = ma.masked_array(0, dtype=result.dtype).fill_value
            assert result.data.fill_value == np_fill_value

    def test_fill_value_invariant_to_order__different_non_none(self):
        cubes = [self._make_cube(0, mask=True, fill_value=1234)]
        cubes.append(self._make_cube(1, mask=True, fill_value=2341))
        cubes.append(self._make_cube(2, mask=True, fill_value=3412))
        cubes.append(self._make_cube(3, mask=True, fill_value=4123))
        for combo in itertools.permutations(cubes):
            result = iris.cube.CubeList(combo).merge_cube()
            np_fill_value = ma.masked_array(0, dtype=result.dtype).fill_value
            assert result.data.fill_value == np_fill_value

    def test_fill_value_invariant_to_order__mixed(self):
        cubes = [self._make_cube(0, mask=True, fill_value=None)]
        cubes.append(self._make_cube(1, mask=True, fill_value=1234))
        cubes.append(self._make_cube(2, mask=True, fill_value=4321))
        for combo in itertools.permutations(cubes):
            result = iris.cube.CubeList(combo).merge_cube()
            np_fill_value = ma.masked_array(0, dtype=result.dtype).fill_value
            assert result.data.fill_value == np_fill_value


class TestCubeMergeWithAncils:
    def _makecube(self, y, cm=False, av=False):
        cube = iris.cube.Cube([0, 0])
        cube.add_dim_coord(iris.coords.DimCoord([0, 1], long_name="x"), 0)
        cube.add_aux_coord(iris.coords.DimCoord(y, long_name="y"))
        if cm:
            cube.add_cell_measure(iris.coords.CellMeasure([1, 1], long_name="foo"), 0)
        if av:
            cube.add_ancillary_variable(
                iris.coords.AncillaryVariable([1, 1], long_name="bar"), 0
            )
        return cube

    def test_fail_missing_cell_measure(self):
        cube1 = self._makecube(0, cm=True)
        cube2 = self._makecube(1)
        cubes = iris.cube.CubeList([cube1, cube2]).merge()
        assert len(cubes) == 2

    def test_fail_missing_ancillary_variable(self):
        cube1 = self._makecube(0, av=True)
        cube2 = self._makecube(1)
        cubes = iris.cube.CubeList([cube1, cube2]).merge()
        assert len(cubes) == 2

    def test_fail_different_cell_measure(self):
        cube1 = self._makecube(0, cm=True)
        cube2 = self._makecube(1)
        cube2.add_cell_measure(iris.coords.CellMeasure([2, 2], long_name="foo"), 0)
        cubes = iris.cube.CubeList([cube1, cube2]).merge()
        assert len(cubes) == 2

    def test_fail_different_ancillary_variable(self):
        cube1 = self._makecube(0, av=True)
        cube2 = self._makecube(1)
        cube2.add_ancillary_variable(
            iris.coords.AncillaryVariable([2, 2], long_name="bar"), 0
        )
        cubes = iris.cube.CubeList([cube1, cube2]).merge()
        assert len(cubes) == 2

    def test_merge_with_cell_measure(self):
        cube1 = self._makecube(0, cm=True)
        cube2 = self._makecube(1, cm=True)
        cubes = iris.cube.CubeList([cube1, cube2]).merge()
        assert len(cubes) == 1
        assert cube1.cell_measures() == cubes[0].cell_measures()

    def test_merge_with_ancillary_variable(self):
        cube1 = self._makecube(0, av=True)
        cube2 = self._makecube(1, av=True)
        cubes = iris.cube.CubeList([cube1, cube2]).merge()
        assert len(cubes) == 1
        assert cube1.ancillary_variables() == cubes[0].ancillary_variables()

    def test_cell_measure_error_msg(self):
        msg = "cube.cell_measures differ"
        cube1 = self._makecube(0, cm=True)
        cube2 = self._makecube(1)
        with pytest.raises(iris.exceptions.MergeError, match=msg):
            _ = iris.cube.CubeList([cube1, cube2]).merge_cube()

    def test_ancillary_variable_error_msg(self):
        msg = "cube.ancillary_variables differ"
        cube1 = self._makecube(0, av=True)
        cube2 = self._makecube(1)
        with pytest.raises(iris.exceptions.MergeError, match=msg):
            _ = iris.cube.CubeList([cube1, cube2]).merge_cube()


class TestCubeMerge__split_attributes__error_messages:
    """Specific tests for the detection and wording of attribute-mismatch errors.

    In particular, the adoption of 'split' attributes with the new
    :class:`iris.cube.CubeAttrsDict` introduces some more subtle possible discrepancies
    in attributes, where this has also impacted the messaging, so this aims to probe
    those cases.
    """

    def _check_merge_error(self, attrs_1, attrs_2, expected_message):
        """Check the error from a merge failure caused by a mismatch of attributes.

        Build a pair of cubes with given attributes, merge them + check for a match
        to the expected error message.
        """
        cube_1 = iris.cube.Cube(
            [0],
            aux_coords_and_dims=[(AuxCoord([1], long_name="x"), None)],
            attributes=attrs_1,
        )
        cube_2 = iris.cube.Cube(
            [0],
            aux_coords_and_dims=[(AuxCoord([2], long_name="x"), None)],
            attributes=attrs_2,
        )
        with pytest.raises(iris.exceptions.MergeError, match=expected_message):
            iris.cube.CubeList([cube_1, cube_2]).merge_cube()

    def test_keys_differ__single(self):
        self._check_merge_error(
            attrs_1=dict(a=1, b=2),
            attrs_2=dict(a=1),
            # Note: matching key 'a' does *not* appear in the message
            expected_message="cube.attributes keys differ: 'b'",
        )

    def test_keys_differ__multiple(self):
        self._check_merge_error(
            attrs_1=dict(a=1, b=2),
            attrs_2=dict(a=1, c=2),
            expected_message="cube.attributes keys differ: 'b', 'c'",
        )

    def test_values_differ__single(self):
        self._check_merge_error(
            attrs_1=dict(a=1, b=2),  # Note: matching key 'a' does not appear
            attrs_2=dict(a=1, b=3),
            expected_message="cube.attributes values differ for keys: 'b'",
        )

    def test_values_differ__multiple(self):
        self._check_merge_error(
            attrs_1=dict(a=1, b=2),
            attrs_2=dict(a=12, b=22),
            expected_message="cube.attributes values differ for keys: 'a', 'b'",
        )

    def test_splitattrs_keys_local_global_mismatch(self):
        # Since Cube.attributes is now a "split-attributes" dictionary, it is now
        # possible to have "cube1.attributes != cube1.attributes", but also
        # "set(cube1.attributes.keys()) == set(cube2.attributes.keys())".
        # I.E. it is now necessary to specifically compare ".globals" and ".locals" to
        # see *what* differs between two attributes dictionaries.
        self._check_merge_error(
            attrs_1=CubeAttrsDict(globals=dict(a=1), locals=dict(b=2)),
            attrs_2=CubeAttrsDict(locals=dict(a=2)),
            expected_message="cube.attributes keys differ: 'a', 'b'",
        )

    def test_splitattrs_keys_local_match_masks_global_mismatch(self):
        self._check_merge_error(
            attrs_1=CubeAttrsDict(globals=dict(a=1), locals=dict(a=3)),
            attrs_2=CubeAttrsDict(globals=dict(a=2), locals=dict(a=3)),
            expected_message="cube.attributes values differ for keys: 'a'",
        )


@pytest.mark.parametrize(
    "dtype", [np.int16, np.int32, np.int64, np.float32, np.float64]
)
class TestCubeMerge_masked_scalar:
    """Test for merging of scalar coordinates containing masked data."""

    def _build_cube(self, scalar_data):
        return iris.cube.Cube(
            np.arange(5),
            standard_name="air_pressure",
            aux_coords_and_dims=[
                (AuxCoord(points=scalar_data, standard_name="realization"), None)
            ],
        )

    def test_merge_scalar_coords_all_masked(self, dtype):
        """Test merging of scalar aux coords all with masked data."""
        n = 5
        cubes = iris.cube.CubeList(
            [self._build_cube(np.ma.masked_all(1, dtype=dtype)) for i in range(n)]
        )
        merged = cubes.merge_cube()
        c = merged.coord("realization")
        assert np.ma.isMaskedArray(c.points)
        assert np.all(c.points.mask)
        assert c.points.dtype.type is dtype

    def test_merge_scalar_coords_some_masked(self, dtype):
        """Test merging of scalar aux coords with mix of masked and unmasked data."""
        n = 5
        cubes = iris.cube.CubeList(
            [
                self._build_cube(np.ma.masked_array(i, dtype=dtype, mask=i % 2))
                for i in range(n)
            ]
        )
        merged = cubes.merge_cube()
        c = merged.coord("realization")
        assert np.ma.isMaskedArray(c.points)
        assert all([c.points.mask[i] == i % 2 for i in range(n)])
        assert c.points.dtype.type is dtype
