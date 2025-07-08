# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for merging cubes."""

import datetime

import numpy as np
import pytest

from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube, CubeList
from iris.tests._shared_utils import (
    assert_array_equal,
    assert_CML,
    get_data_path,
    skip_data,
)
import iris.tests.stock


class MergeMixin:
    """Mix-in class for attributes & utilities common to these test cases."""

    def test_normal_cubes(self, request):
        cubes = iris.load(self._data_path)
        assert len(cubes) == self._num_cubes
        assert_CML(request, cubes, ["merge", self._prefix + ".cml"])

    def test_remerge(self):
        # After the merge process the coordinates within each cube can be in a
        # different order. Until that changes we can't compare the cubes
        # directly or with the CML ... so we just make sure the count stays
        # the same.
        cubes = iris.load(self._data_path)
        cubes2 = cubes.merge()
        assert len(cubes) == len(cubes2)

    def test_duplication(self):
        cubes = iris.load(self._data_path)
        pytest.raises(iris.exceptions.DuplicateDataError, (cubes + cubes).merge)
        cubes2 = (cubes + cubes).merge(unique=False)
        assert len(cubes2) == 2 * len(cubes)


@skip_data
class TestSingleCube(MergeMixin):
    def setup_method(self):
        self._data_path = get_data_path(("PP", "globClim1", "theta.pp"))
        self._num_cubes = 1
        self._prefix = "theta"


@skip_data
class TestMultiCube(MergeMixin):
    def setup_method(self):
        self._data_path = get_data_path(("PP", "globClim1", "dec_subset.pp"))
        self._num_cubes = 4
        self._prefix = "dec"

    def test_coord_attributes(self):
        def custom_coord_callback(cube, field, filename):
            cube.coord("time").attributes["monty"] = "python"
            cube.coord("time").attributes["brain"] = "hurts"

        # Load slices, decorating a coord with custom attributes
        cubes = iris.load_raw(self._data_path, callback=custom_coord_callback)
        # Merge
        merged = iris.cube.CubeList(cubes).merge()
        # Check the custom attributes are in the merged cube
        for cube in merged:
            assert cube.coord("time").attributes["monty"] == "python"
            assert cube.coord("time").attributes["brain"] == "hurts"


@skip_data
class TestColpex:
    def setup_method(self):
        self._data_path = get_data_path(("PP", "COLPEX", "small_colpex_theta_p_alt.pp"))

    def test_colpex(self, request):
        cubes = iris.load(self._data_path)
        assert len(cubes) == 3
        assert_CML(request, cubes, ("COLPEX", "small_colpex_theta_p_alt.cml"))


@skip_data
class TestDataMerge:
    def test_extended_proxy_data(self, request):
        # Get the empty theta cubes for T+1.5 and T+2
        data_path = get_data_path(("PP", "COLPEX", "theta_and_orog_subset.pp"))
        phenom_constraint = iris.Constraint("air_potential_temperature")
        datetime_1 = datetime.datetime(2009, 9, 9, 17, 20)
        datetime_2 = datetime.datetime(2009, 9, 9, 17, 50)
        time_constraint1 = iris.Constraint(time=datetime_1)
        time_constraint2 = iris.Constraint(time=datetime_2)
        time_constraint_1_and_2 = iris.Constraint(
            time=lambda c: c in (datetime_1, datetime_2)
        )
        cube1 = iris.load_cube(data_path, phenom_constraint & time_constraint1)
        cube2 = iris.load_cube(data_path, phenom_constraint & time_constraint2)

        # Merge the two halves
        cubes = iris.cube.CubeList([cube1, cube2]).merge(True)
        assert_CML(request, cubes, ("merge", "theta_two_times.cml"))

        # Make sure we get the same result directly from load
        cubes = iris.load_cube(data_path, phenom_constraint & time_constraint_1_and_2)
        assert_CML(request, cubes, ("merge", "theta_two_times.cml"))

    def test_real_data(self, request):
        data_path = get_data_path(("PP", "globClim1", "theta.pp"))
        cubes = iris.load_raw(data_path)
        # Force the source 2-D cubes to load their data before the merge
        for cube in cubes:
            _ = cube.data
        cubes = cubes.merge()
        assert_CML(request, cubes, ["merge", "theta.cml"])


class TestDimensionSplitting:
    def _make_cube(self, a, b, c, data):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3, 4], dtype=np.int32),
                long_name="x",
                units="1",
            ),
            1,
        )
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3], dtype=np.int32),
                long_name="y",
                units="1",
            ),
            0,
        )
        cube.add_aux_coord(
            DimCoord(np.array([a], dtype=np.int32), long_name="a", units="1")
        )
        cube.add_aux_coord(
            DimCoord(np.array([b], dtype=np.int32), long_name="b", units="1")
        )
        cube.add_aux_coord(
            DimCoord(np.array([c], dtype=np.int32), long_name="c", units="1")
        )
        return cube

    def test_single_split(self, request):
        # Test what happens when a cube forces a simple, two-way split.
        cubes = []
        cubes.append(self._make_cube(0, 0, 0, 0))
        cubes.append(self._make_cube(0, 1, 1, 1))
        cubes.append(self._make_cube(1, 0, 2, 2))
        cubes.append(self._make_cube(1, 1, 3, 3))
        cubes.append(self._make_cube(2, 0, 4, 4))
        cubes.append(self._make_cube(2, 1, 5, 5))
        cube = iris.cube.CubeList(cubes).merge()
        assert_CML(request, cube, ("merge", "single_split.cml"))

    def test_multi_split(self, request):
        # Test what happens when a cube forces a three-way split.
        cubes = []
        cubes.append(self._make_cube(0, 0, 0, 0))
        cubes.append(self._make_cube(0, 0, 1, 1))
        cubes.append(self._make_cube(0, 1, 0, 2))
        cubes.append(self._make_cube(0, 1, 1, 3))
        cubes.append(self._make_cube(1, 0, 0, 4))
        cubes.append(self._make_cube(1, 0, 1, 5))
        cubes.append(self._make_cube(1, 1, 0, 6))
        cubes.append(self._make_cube(1, 1, 1, 7))
        cubes.append(self._make_cube(2, 0, 0, 8))
        cubes.append(self._make_cube(2, 0, 1, 9))
        cubes.append(self._make_cube(2, 1, 0, 10))
        cubes.append(self._make_cube(2, 1, 1, 11))
        cube = iris.cube.CubeList(cubes).merge()
        assert_CML(request, cube, ("merge", "multi_split.cml"))


class TestCombination:
    def _make_cube(self, a, b, c, d, data=0):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3, 4], dtype=np.int32),
                long_name="x",
                units="1",
            ),
            1,
        )
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3], dtype=np.int32),
                long_name="y",
                units="1",
            ),
            0,
        )

        for name, value in zip(["a", "b", "c", "d"], [a, b, c, d]):
            dtype = np.str_ if isinstance(value, str) else np.float32
            cube.add_aux_coord(
                AuxCoord(np.array([value], dtype=dtype), long_name=name, units="1")
            )

        return cube

    def test_separable_combination(self, request):
        cubes = iris.cube.CubeList()
        cubes.append(
            self._make_cube("2005", "ECMWF", "HOPE-E, Sys 1, Met 1, ENSEMBLES", 0)
        )
        cubes.append(
            self._make_cube("2005", "ECMWF", "HOPE-E, Sys 1, Met 1, ENSEMBLES", 1)
        )
        cubes.append(
            self._make_cube("2005", "ECMWF", "HOPE-E, Sys 1, Met 1, ENSEMBLES", 2)
        )
        cubes.append(
            self._make_cube(
                "2026", "UK Met Office", "HadGEM2, Sys 1, Met 1, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2026", "UK Met Office", "HadGEM2, Sys 1, Met 1, ENSEMBLES", 1
            )
        )
        cubes.append(
            self._make_cube(
                "2026", "UK Met Office", "HadGEM2, Sys 1, Met 1, ENSEMBLES", 2
            )
        )
        cubes.append(
            self._make_cube("2002", "CERFACS", "GELATO, Sys 0, Met 1, ENSEMBLES", 0)
        )
        cubes.append(
            self._make_cube("2002", "CERFACS", "GELATO, Sys 0, Met 1, ENSEMBLES", 1)
        )
        cubes.append(
            self._make_cube("2002", "CERFACS", "GELATO, Sys 0, Met 1, ENSEMBLES", 2)
        )
        cubes.append(
            self._make_cube("2002", "IFM-GEOMAR", "ECHAM5, Sys 1, Met 10, ENSEMBLES", 0)
        )
        cubes.append(
            self._make_cube("2002", "IFM-GEOMAR", "ECHAM5, Sys 1, Met 10, ENSEMBLES", 1)
        )
        cubes.append(
            self._make_cube("2002", "IFM-GEOMAR", "ECHAM5, Sys 1, Met 10, ENSEMBLES", 2)
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 10, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 11, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 12, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 13, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 14, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 15, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 16, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 17, ENSEMBLES", 0
            )
        )
        cubes.append(
            self._make_cube(
                "2502", "UK Met Office", "HadCM3, Sys 51, Met 18, ENSEMBLES", 0
            )
        )
        cube = cubes.merge()
        assert_CML(
            request, cube, ("merge", "separable_combination.cml"), checksum=False
        )


class TestDimSelection:
    def _make_cube(self, a, b, data=0, a_dim=False, b_dim=False):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3, 4], dtype=np.int32),
                long_name="x",
                units="1",
            ),
            1,
        )
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3], dtype=np.int32),
                long_name="y",
                units="1",
            ),
            0,
        )

        for name, value, dim in zip(["a", "b"], [a, b], [a_dim, b_dim]):
            dtype = np.str_ if isinstance(value, str) else np.float32
            ctype = DimCoord if dim else AuxCoord
            coord = ctype(np.array([value], dtype=dtype), long_name=name, units="1")
            cube.add_aux_coord(coord)

        return cube

    def test_string_a_with_aux(self, request):
        templates = (("a", 0), ("b", 1), ("c", 2), ("d", 3))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "string_a_with_aux.cml"), checksum=False)
        assert isinstance(cube.coord("a"), AuxCoord)
        assert isinstance(cube.coord("b"), DimCoord)
        assert cube.coord("b") in cube.dim_coords

    def test_string_b_with_aux(self, request):
        templates = ((0, "a"), (1, "b"), (2, "c"), (3, "d"))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "string_b_with_aux.cml"), checksum=False)
        assert isinstance(cube.coord("a"), DimCoord)
        assert cube.coord("a") in cube.dim_coords
        assert isinstance(cube.coord("b"), AuxCoord)

    def test_string_a_with_dim(self, request):
        templates = (("a", 0), ("b", 1), ("c", 2), ("d", 3))
        cubes = [self._make_cube(a, b, b_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "string_a_with_dim.cml"), checksum=False)
        assert isinstance(cube.coord("a"), AuxCoord)
        assert isinstance(cube.coord("b"), DimCoord)
        assert cube.coord("b") in cube.dim_coords

    def test_string_b_with_dim(self, request):
        templates = ((0, "a"), (1, "b"), (2, "c"), (3, "d"))
        cubes = [self._make_cube(a, b, a_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "string_b_with_dim.cml"), checksum=False)
        assert isinstance(cube.coord("a"), DimCoord)
        assert cube.coord("a") in cube.dim_coords
        assert isinstance(cube.coord("b"), AuxCoord)

    def test_string_a_b(self, request):
        templates = (("a", "0"), ("b", "1"), ("c", "2"), ("d", "3"))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "string_a_b.cml"), checksum=False)
        assert isinstance(cube.coord("a"), AuxCoord)
        assert isinstance(cube.coord("b"), AuxCoord)

    def test_a_aux_b_aux(self, request):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "a_aux_b_aux.cml"), checksum=False)
        assert isinstance(cube.coord("a"), DimCoord)
        assert cube.coord("a") in cube.dim_coords
        assert isinstance(cube.coord("b"), DimCoord)
        assert cube.coord("b") in cube.aux_coords

    def test_a_aux_b_dim(self, request):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b, b_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "a_aux_b_dim.cml"), checksum=False)
        assert isinstance(cube.coord("a"), DimCoord)
        assert cube.coord("a") in cube.aux_coords
        assert isinstance(cube.coord("b"), DimCoord)
        assert cube.coord("b") in cube.dim_coords

    def test_a_dim_b_aux(self, request):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b, a_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "a_dim_b_aux.cml"), checksum=False)
        assert isinstance(cube.coord("a"), DimCoord)
        assert cube.coord("a") in cube.dim_coords
        assert isinstance(cube.coord("b"), DimCoord)
        assert cube.coord("b") in cube.aux_coords

    def test_a_dim_b_dim(self, request):
        templates = ((0, 10), (1, 11), (2, 12), (3, 13))
        cubes = [self._make_cube(a, b, a_dim=True, b_dim=True) for a, b in templates]
        cube = iris.cube.CubeList(cubes).merge()[0]
        assert_CML(request, cube, ("merge", "a_dim_b_dim.cml"), checksum=False)
        assert isinstance(cube.coord("a"), DimCoord)
        assert cube.coord("a") in cube.dim_coords
        assert isinstance(cube.coord("b"), DimCoord)
        assert cube.coord("b") in cube.aux_coords


class TestTimeTripleMerging:
    def _make_cube(self, a, b, c, data=0):
        cube_data = np.empty((4, 5), dtype=np.float32)
        cube_data[:] = data
        cube = iris.cube.Cube(cube_data)
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3, 4], dtype=np.int32),
                long_name="x",
                units="1",
            ),
            1,
        )
        cube.add_dim_coord(
            DimCoord(
                np.array([0, 1, 2, 3], dtype=np.int32),
                long_name="y",
                units="1",
            ),
            0,
        )
        cube.add_aux_coord(
            DimCoord(
                np.array([a], dtype=np.int32),
                standard_name="forecast_period",
                units="1",
            )
        )
        cube.add_aux_coord(
            DimCoord(
                np.array([b], dtype=np.int32),
                standard_name="forecast_reference_time",
                units="1",
            )
        )
        cube.add_aux_coord(
            DimCoord(np.array([c], dtype=np.int32), standard_name="time", units="1")
        )
        return cube

    def _test_triples(self, triples, filename, request):
        cubes = [self._make_cube(fp, rt, t) for fp, rt, t in triples]
        cube = iris.cube.CubeList(cubes).merge()
        assert_CML(
            request, cube, ("merge", "time_triple_" + filename + ".cml"), checksum=False
        )

    def test_single_forecast(self, request):
        # A single forecast series (i.e. from a single reference time)
        # => fp, t: 4; rt: scalar
        triples = (
            (0, 10, 10),
            (1, 10, 11),
            (2, 10, 12),
            (3, 10, 13),
        )
        self._test_triples(triples, "single_forecast", request)

    def test_successive_forecasts(self, request):
        # Three forecast series from successively later reference times
        # => rt, t: 3; fp, t: 4
        triples = (
            (0, 10, 10),
            (1, 10, 11),
            (2, 10, 12),
            (3, 10, 13),
            (0, 11, 11),
            (1, 11, 12),
            (2, 11, 13),
            (3, 11, 14),
            (0, 12, 12),
            (1, 12, 13),
            (2, 12, 14),
            (3, 12, 15),
        )
        self._test_triples(triples, "successive_forecasts", request)

    def test_time_vs_ref_time(self, request):
        # => fp, t: 4; fp, rt: 3
        triples = (
            (2, 10, 12),
            (3, 10, 13),
            (4, 10, 14),
            (5, 10, 15),
            (1, 11, 12),
            (2, 11, 13),
            (3, 11, 14),
            (4, 11, 15),
            (0, 12, 12),
            (1, 12, 13),
            (2, 12, 14),
            (3, 12, 15),
        )
        self._test_triples(triples, "time_vs_ref_time", request)

    def test_time_vs_forecast(self, request):
        # => rt, t: 4, fp, rt: 3
        triples = (
            (0, 10, 10),
            (0, 11, 11),
            (0, 12, 12),
            (0, 13, 13),
            (1, 9, 10),
            (1, 10, 11),
            (1, 11, 12),
            (1, 12, 13),
            (2, 8, 10),
            (2, 9, 11),
            (2, 10, 12),
            (2, 11, 13),
        )
        self._test_triples(triples, "time_vs_forecast", request)

    def test_time_non_dim_coord(self, request):
        # => rt: 1 fp, t (bounded): 2
        triples = (
            (5, 0, 2.5),
            (10, 0, 5),
        )
        cubes = [self._make_cube(fp, rt, t) for fp, rt, t in triples]
        for end_time, cube in zip([5, 10], cubes):
            cube.coord("time").bounds = [0, end_time]
        (cube,) = iris.cube.CubeList(cubes).merge()
        assert_CML(
            request,
            cube,
            ("merge", "time_triple_time_non_dim_coord.cml"),
            checksum=False,
        )
        # make sure that forecast_period is the dimensioned coordinate (as time becomes an AuxCoord)
        assert cube.coord(dimensions=0, dim_coords=True).name() == "forecast_period"

    def test_independent(self, request):
        # => fp: 2; rt: 2; t: 2
        triples = (
            (0, 10, 10),
            (0, 11, 10),
            (0, 10, 11),
            (0, 11, 11),
            (1, 10, 10),
            (1, 11, 10),
            (1, 10, 11),
            (1, 11, 11),
        )
        self._test_triples(triples, "independent", request)

    def test_series(self, request):
        # => fp, rt, t: 5 (with only t being definitive).
        triples = (
            (0, 10, 10),
            (0, 11, 11),
            (0, 12, 12),
            (1, 12, 13),
            (2, 12, 14),
        )
        self._test_triples(triples, "series", request)

    def test_non_expanding_dimension(self, request):
        triples = (
            (0, 10, 0),
            (0, 20, 1),
            (0, 20, 0),
        )
        # => fp: scalar; rt, t: 3 (with no time being definitive)
        self._test_triples(triples, "non_expanding", request)

    def test_duplicate_data(self, request):
        # test what happens when we have repeated time coordinates (i.e. duplicate data)
        cube1 = self._make_cube(0, 10, 0)
        cube2 = self._make_cube(1, 20, 1)
        cube3 = self._make_cube(1, 20, 1)

        # check that we get a duplicate data error when unique is True
        with pytest.raises(iris.exceptions.DuplicateDataError):
            iris.cube.CubeList([cube1, cube2, cube3]).merge()

        cubes = iris.cube.CubeList([cube1, cube2, cube3]).merge(unique=False)
        assert_CML(
            request, cubes, ("merge", "time_triple_duplicate_data.cml"), checksum=False
        )

    def test_simple1(self, request):
        cube1 = self._make_cube(0, 10, 0)
        cube2 = self._make_cube(1, 20, 1)
        cube3 = self._make_cube(2, 20, 0)
        cube = iris.cube.CubeList([cube1, cube2, cube3]).merge()
        assert_CML(request, cube, ("merge", "time_triple_merging1.cml"), checksum=False)

    def test_simple2(self, request):
        cubes = iris.cube.CubeList(
            [
                self._make_cube(0, 0, 0),
                self._make_cube(1, 0, 1),
                self._make_cube(2, 0, 2),
                self._make_cube(0, 1, 3),
                self._make_cube(1, 1, 4),
                self._make_cube(2, 1, 5),
            ]
        )
        cube = cubes.merge()[0]
        assert_CML(request, cube, ("merge", "time_triple_merging2.cml"), checksum=False)

        cube = iris.cube.CubeList(cubes[:-1]).merge()[0]
        assert_CML(request, cube, ("merge", "time_triple_merging3.cml"), checksum=False)

    def test_simple3(self, request):
        cubes = iris.cube.CubeList(
            [
                self._make_cube(0, 0, 0),
                self._make_cube(0, 1, 1),
                self._make_cube(0, 2, 2),
                self._make_cube(1, 0, 3),
                self._make_cube(1, 1, 4),
                self._make_cube(1, 2, 5),
            ]
        )
        cube = cubes.merge()[0]
        assert_CML(request, cube, ("merge", "time_triple_merging4.cml"), checksum=False)

        cube = iris.cube.CubeList(cubes[:-1]).merge()[0]
        assert_CML(request, cube, ("merge", "time_triple_merging5.cml"), checksum=False)


class TestCubeMergeTheoretical:
    def test_simple_bounds_merge(self, request):
        cube1 = iris.tests.stock.simple_2d()
        cube2 = iris.tests.stock.simple_2d()

        cube1.add_aux_coord(DimCoord(np.int32(10), long_name="pressure", units="Pa"))
        cube2.add_aux_coord(DimCoord(np.int32(11), long_name="pressure", units="Pa"))

        r = iris.cube.CubeList([cube1, cube2]).merge()
        assert_CML(request, r, ("cube_merge", "test_simple_bound_merge.cml"))

    def test_simple_multidim_merge(self, request):
        cube1 = iris.tests.stock.simple_2d_w_multidim_coords()
        cube2 = iris.tests.stock.simple_2d_w_multidim_coords()

        cube1.add_aux_coord(DimCoord(np.int32(10), long_name="pressure", units="Pa"))
        cube2.add_aux_coord(DimCoord(np.int32(11), long_name="pressure", units="Pa"))

        r = iris.cube.CubeList([cube1, cube2]).merge()[0]
        assert_CML(request, r, ("cube_merge", "multidim_coord_merge.cml"))

        # try transposing the cubes first
        cube1.transpose([1, 0])
        cube2.transpose([1, 0])
        r = iris.cube.CubeList([cube1, cube2]).merge()[0]
        assert_CML(request, r, ("cube_merge", "multidim_coord_merge_transpose.cml"))

    def test_simple_points_merge(self, request):
        cube1 = iris.tests.stock.simple_2d(with_bounds=False)
        cube2 = iris.tests.stock.simple_2d(with_bounds=False)

        cube1.add_aux_coord(DimCoord(np.int32(10), long_name="pressure", units="Pa"))
        cube2.add_aux_coord(DimCoord(np.int32(11), long_name="pressure", units="Pa"))

        r = iris.cube.CubeList([cube1, cube2]).merge()
        assert_CML(request, r, ("cube_merge", "test_simple_merge.cml"))

        # check that the unique merging raises a Duplicate data error
        pytest.raises(
            iris.exceptions.DuplicateDataError,
            iris.cube.CubeList([cube1, cube1]).merge,
            unique=True,
        )

        # check that non unique merging returns both cubes
        r = iris.cube.CubeList([cube1, cube1]).merge(unique=False)
        assert_CML(request, r[0], ("cube_merge", "test_orig_point_cube.cml"))
        assert_CML(request, r[1], ("cube_merge", "test_orig_point_cube.cml"))

        # test attribute merging
        cube1.attributes["my_attr1"] = "foo"
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        assert_CML(request, r, ("cube_merge", "test_simple_attributes1.cml"))

        cube2.attributes["my_attr1"] = "bar"
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 2 cubes
        assert_CML(request, r, ("cube_merge", "test_simple_attributes2.cml"))

        cube2.attributes["my_attr1"] = "foo"
        r = iris.cube.CubeList([cube1, cube2]).merge()
        # result should be 1 cube
        assert_CML(request, r, ("cube_merge", "test_simple_attributes3.cml"))


class TestContiguous:
    def test_form_contiguous_dimcoord(self):
        # Test that cube sliced up and remerged in the opposite order maintains
        # contiguity.
        cube1 = Cube([1, 2, 3], "air_temperature", units="K")
        coord1 = DimCoord([3, 2, 1], long_name="spam")
        coord1.guess_bounds()
        cube1.add_dim_coord(coord1, 0)
        cubes = CubeList(cube1.slices_over("spam"))
        cube2 = cubes.merge_cube()
        coord2 = cube2.coord("spam")

        assert coord2.is_contiguous()
        assert_array_equal(coord2.points, [1, 2, 3])
        assert_array_equal(coord2.bounds, coord1.bounds[::-1, ::-1])
