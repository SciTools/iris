# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Integration tests for delayed saving."""

import warnings

from cf_units import Unit
import dask.array as da
import dask.config
import distributed
import numpy as np
import pytest

import iris
from iris.fileformats.netcdf._thread_safe_nc import default_fillvals
import iris.tests
from iris.tests.stock import realistic_4d


def are_dask_collections(obj):
    result = dask.is_dask_collection(obj)
    if not result and hasattr(obj, "__len__"):
        result = all([dask.is_dask_collection(item) for item in obj])
    return result


class Test__lazy_stream_data:
    # Ensure all saves are done with split-atttribute saving,
    # -- because some of these tests are sensitive to unexpected warnings.
    @pytest.fixture(autouse=True)
    def all_saves_with_split_attrs(self):
        with iris.FUTURE.context(save_split_attrs=True):
            yield

    @pytest.fixture(autouse=True)
    def output_path(self, tmp_path):
        # A temporary output netcdf-file path, **unique to each test call**.
        self.temp_output_filepath = tmp_path / "tmp.nc"
        yield self.temp_output_filepath

    @pytest.fixture(autouse=True, scope="module")
    def all_vars_lazy(self):
        # For the operation of these tests, we want to force all netcdf variables
        # to load as lazy data, i.e. **don't** use real data for 'small' ones.
        old_value = iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES
        iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES = 0
        yield
        iris.fileformats.netcdf.loader._LAZYVAR_MIN_BYTES = old_value

    @staticmethod
    @pytest.fixture(params=[False, True], ids=["SaveImmediate", "SaveDelayed"])
    def save_is_delayed(request):
        return request.param

    @staticmethod
    def make_testcube(
        include_lazy_content=True,
        ensure_fillvalue_collision=False,
        data_is_maskedbytes=False,
        include_extra_coordlikes=False,
    ):
        cube = realistic_4d()

        def fix_array(array):
            """Make a new, custom array to replace the provided cube/coord data.
            Optionally provide default-fill-value collisions, and/or replace with lazy
            content.
            """
            if array is not None:
                if data_is_maskedbytes:
                    dmin, dmax = 0, 255
                else:
                    dmin, dmax = array.min(), array.max()
                array = np.random.default_rng().uniform(dmin, dmax, size=array.shape)

                if data_is_maskedbytes:
                    array = array.astype("u1")
                    array = np.ma.masked_array(array)
                    # To trigger, it must also have at least one *masked point*.
                    array[tuple([0] * array.ndim)] = np.ma.masked

                if ensure_fillvalue_collision:
                    # Set point at midpoint index = default-fill-value
                    fill_value = default_fillvals[array.dtype.str[1:]]
                    inds = tuple(dim // 2 for dim in array.shape)
                    array[inds] = fill_value

                if include_lazy_content:
                    # Make the array lazy.
                    # Ensure we always have multiple chunks (relatively small ones).
                    chunks = list(array.shape)
                    chunks[0] = 1
                    array = da.from_array(array, chunks=chunks)

            return array

        # Replace the cube data, and one aux-coord, according to the control settings.
        cube.data = fix_array(cube.data)
        auxcoord = cube.coord("surface_altitude")
        auxcoord.points = fix_array(auxcoord.points)

        if include_extra_coordlikes:
            # Also concoct + attach an ancillary variable and a cell-measure, so we can
            #  check that they behave the same as coordinates.
            ancil_dims = [0, 2]
            cm_dims = [0, 3]
            ancil_shape = [cube.shape[idim] for idim in ancil_dims]
            cm_shape = [cube.shape[idim] for idim in cm_dims]
            from iris.coords import AncillaryVariable, CellMeasure

            ancil = AncillaryVariable(
                fix_array(np.zeros(ancil_shape)), long_name="sample_ancil"
            )
            cube.add_ancillary_variable(ancil, ancil_dims)
            cm = CellMeasure(fix_array(np.zeros(cm_shape)), long_name="sample_cm")
            cube.add_cell_measure(cm, cm_dims)
        return cube

    def test_realfile_loadsave_equivalence(self, save_is_delayed, output_path):
        input_filepath = iris.tests.get_data_path(
            ["NetCDF", "global", "xyz_t", "GEMS_CO2_Apr2006.nc"]
        )
        original_cubes = iris.load(input_filepath)

        # Preempt some standard changes that an iris save will impose.
        for cube in original_cubes:
            if cube.units == Unit("-"):
                # replace 'unknown unit' with 'no unit'.
                cube.units = Unit("?")
            # Fix conventions attribute to what iris.save outputs.
            cube.attributes["Conventions"] = "CF-1.7"

        original_cubes = sorted(original_cubes, key=lambda cube: cube.name())
        result = iris.save(original_cubes, output_path, compute=not save_is_delayed)
        if save_is_delayed:
            # In this case, must also "complete" the save.
            dask.compute(result)
        reloaded_cubes = iris.load(output_path)
        reloaded_cubes = sorted(reloaded_cubes, key=lambda cube: cube.name())
        assert reloaded_cubes == original_cubes
        # NOTE: it might be nicer to use assertCDL, but unfortunately importing
        # unittest.TestCase seems to lose us the ability to use fixtures.

    @classmethod
    @pytest.fixture(
        params=[
            "ThreadedScheduler",
            "DistributedScheduler",
            "SingleThreadScheduler",
        ]
    )
    def scheduler_type(cls, request):
        sched_typename = request.param
        if sched_typename == "ThreadedScheduler":
            config_name = "threads"
        elif sched_typename == "SingleThreadScheduler":
            config_name = "single-threaded"
        else:
            assert sched_typename == "DistributedScheduler"
            config_name = "distributed"

        if config_name == "distributed":
            _distributed_client = distributed.Client()

        with dask.config.set(scheduler=config_name):
            yield sched_typename

        if config_name == "distributed":
            _distributed_client.close()

    def test_scheduler_types(self, output_path, scheduler_type, save_is_delayed):
        # Check operation works and behaves the same with different schedulers,
        # especially including distributed.

        # Just check that the dask scheduler is setup as 'expected'.
        if scheduler_type == "ThreadedScheduler":
            expected_dask_scheduler = "threads"
        elif scheduler_type == "SingleThreadScheduler":
            expected_dask_scheduler = "single-threaded"
        else:
            assert scheduler_type == "DistributedScheduler"
            expected_dask_scheduler = "distributed"

        assert dask.config.get("scheduler") == expected_dask_scheduler

        # Use a testcase that produces delayed warnings (and check those too).
        cube = self.make_testcube(
            include_lazy_content=True, ensure_fillvalue_collision=True
        )
        result = iris.save(cube, output_path, compute=not save_is_delayed)

        if not save_is_delayed:
            assert result is None
        else:
            assert are_dask_collections(result)

    def test_time_of_writing(self, save_is_delayed, output_path, scheduler_type):
        # Check when lazy data is *actually* written :
        #  - in 'immediate' mode, on initial file write
        #  - in 'delayed' mode, only when the delayed-write is computed.
        original_cube = self.make_testcube(include_extra_coordlikes=True)
        assert original_cube.has_lazy_data()
        assert original_cube.coord("surface_altitude").has_lazy_points()
        assert original_cube.cell_measure("sample_cm").has_lazy_data()
        assert original_cube.ancillary_variable("sample_ancil").has_lazy_data()

        result = iris.save(
            original_cube,
            output_path,
            compute=not save_is_delayed,
        )
        assert save_is_delayed == (result is not None)

        # Read back : NOTE avoid loading the separate surface-altitude cube.
        readback_cube = iris.load_cube(output_path, "air_potential_temperature")
        # Check the components to be tested *are* lazy. See: self.all_vars_lazy().
        assert readback_cube.has_lazy_data()
        assert readback_cube.coord("surface_altitude").has_lazy_points()
        assert readback_cube.cell_measure("sample_cm").has_lazy_data()
        assert readback_cube.ancillary_variable("sample_ancil").has_lazy_data()

        # If 'delayed', the lazy content should all be masked, otherwise none of it.
        def getmask(cube_or_coord):
            cube_or_coord = cube_or_coord.copy()  # avoid realising the original
            if hasattr(cube_or_coord, "points"):
                data = cube_or_coord.points
            else:
                data = cube_or_coord.data
            return np.ma.getmaskarray(data)

        test_components = [
            readback_cube,
            readback_cube.coord("surface_altitude"),
            readback_cube.ancillary_variable("sample_ancil"),
            readback_cube.cell_measure("sample_cm"),
        ]

        def fetch_masks():
            data_mask, coord_mask, ancil_mask, cm_mask = [
                getmask(data) for data in test_components
            ]
            return data_mask, coord_mask, ancil_mask, cm_mask

        data_mask, coord_mask, ancil_mask, cm_mask = fetch_masks()
        if save_is_delayed:
            assert np.all(data_mask)
            assert np.all(coord_mask)
            assert np.all(ancil_mask)
            assert np.all(cm_mask)
        else:
            assert np.all(~data_mask)
            assert np.all(~coord_mask)
            assert np.all(~ancil_mask)
            assert np.all(~cm_mask)

        if save_is_delayed:
            # Complete the write.
            dask.compute(result)

            # Re-fetch the lazy arrays.  The data should now **not be masked**.
            data_mask, coord_mask, ancil_mask, cm_mask = fetch_masks()
            # All written now ?
            assert np.all(~data_mask)
            assert np.all(~coord_mask)
            assert np.all(~ancil_mask)
            assert np.all(~cm_mask)

    def test_no_delayed_writes(self, output_path):
        # Just check that a delayed save returns a usable 'delayed' object, even when
        # there is no lazy content = no delayed writes to perform.
        cube = self.make_testcube(include_lazy_content=False)
        warnings.simplefilter("error")
        result = iris.save(cube, output_path, compute=False)
        assert are_dask_collections(result)
