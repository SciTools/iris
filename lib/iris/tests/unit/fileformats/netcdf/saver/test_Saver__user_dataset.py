# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Tests for saving into a dataset the caller opened, or is only pretending to.

``iris.fileformats.netcdf.save(cube, dataset, compute=False)`` accepts an
open dataset in place of a path. Two kinds of caller do this:

* one holding a real, open netCDF4 dataset, who wants Iris to add to it;
* one holding an object that merely *looks* like a netCDF4 dataset, and
  collects what Iris writes rather than storing it - the "Xarray bridge",
  https://github.com/SciTools/iris/issues/4994, which is how
  https://github.com/pp-mo/ncdata translates between Iris and Xarray.

The second is the reason ``Saver`` may not assume its dataset is netCDF4,
and it is entirely untested elsewhere.

"""

import dask
import numpy as np
import pytest

import iris
from iris.coords import DimCoord
from iris.cube import Cube
import iris.fileformats.netcdf as inetcdf
from iris.fileformats.netcdf import _thread_safe_nc as threadsafe_nc


@pytest.fixture(autouse=True)
def split_attrs():
    # Avoid the legacy-attribute deprecation warning; irrelevant here.
    with iris.FUTURE.context(save_split_attrs=True):
        yield


@pytest.fixture
def cube():
    cube = Cube(
        np.arange(6.0).reshape(2, 3),
        standard_name="air_temperature",
        units="K",
        var_name="air",
    )
    cube.add_dim_coord(
        DimCoord(np.arange(2.0), standard_name="latitude", units="degrees"), 0
    )
    cube.add_dim_coord(
        DimCoord(np.arange(3.0), standard_name="longitude", units="degrees"), 1
    )
    return cube


class TestRealDataset:
    """A dataset the caller opened, and closes themselves."""

    def test_save_into_an_open_dataset(self, cube, tmp_path):
        path = tmp_path / "user.nc"
        dataset = threadsafe_nc.DatasetWrapper(path, mode="w", format="NETCDF4")
        delayed = inetcdf.save(cube, dataset, compute=False)
        # The caller owns the dataset, so the caller closes it - and the
        # delayed writes only resolve once they have.
        dataset.close()
        dask.compute(delayed)

        result = iris.load_cube(path)
        assert result.standard_name == "air_temperature"
        assert result.units == "K"
        assert np.array_equal(result.data, cube.data)
        assert [coord.name() for coord in result.coords()] == [
            "latitude",
            "longitude",
        ]

    def test_saver_does_not_close_what_it_was_given(self, cube, tmp_path):
        path = tmp_path / "user.nc"
        dataset = threadsafe_nc.DatasetWrapper(path, mode="w", format="NETCDF4")
        inetcdf.save(cube, dataset, compute=False)
        assert dataset.isopen()
        dataset.close()

    def test_compute_true_is_refused(self, cube, tmp_path):
        path = tmp_path / "user.nc"
        dataset = threadsafe_nc.DatasetWrapper(path, mode="w", format="NETCDF4")
        with pytest.raises(ValueError, match="Cannot save to a user-provided dataset"):
            inetcdf.save(cube, dataset, compute=True)
        dataset.close()


class _EmulatedDimension:
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size

    def isunlimited(self):
        return self._size is None


class _EmulatedVariable:
    """The least a netCDF4.Variable emulator can be and still be written to.

    ``_data_array`` is the whole point: an emulator receives its data by
    having this attribute set, never by ``__setitem__``. ``__setitem__``
    raises here so that a regression shows up as a failure rather than as
    data quietly going nowhere.

    """

    def __init__(self, name, datatype, dimensions, shape):
        self.name = name
        self.datatype = np.dtype(datatype)
        self.dtype = self.datatype
        self.dimensions = tuple(dimensions)
        self.shape = shape
        self.size = int(np.prod(shape)) if shape else 1
        self._data_array = None
        self._attrs = {}

    def setncattr(self, name, value):
        self._attrs[name] = value

    def getncattr(self, name):
        return self._attrs[name]

    def ncattrs(self):
        return list(self._attrs)

    def chunking(self):
        return "contiguous"

    def __setitem__(self, keys, values):
        raise AssertionError("An emulated variable must receive _data_array.")


class _EmulatedDataset:
    """A netCDF4.Dataset emulator, in the shape ncdata presents.

    Deliberately *not* a ``_thread_safe_nc`` wrapper and deliberately without
    ``THREAD_SAFE_FLAG``, so that the wrapping branch of
    ``NetCDFDataset.from_existing`` is the one under test.

    """

    def __init__(self):
        self.variables = {}
        self.dimensions = {}
        self.file_format = "NETCDF4"
        self._attrs = {}
        self._open = True

    def createDimension(self, name, size):
        self.dimensions[name] = _EmulatedDimension(size)
        return self.dimensions[name]

    def createVariable(self, name, datatype, dimensions=(), **kwargs):
        shape = tuple(len(self.dimensions[name_]) for name_ in dimensions)
        variable = _EmulatedVariable(name, datatype, dimensions, shape)
        self.variables[name] = variable
        return variable

    def setncattr(self, name, value):
        self._attrs[name] = value

    def getncattr(self, name):
        return self._attrs[name]

    def ncattrs(self):
        return list(self._attrs)

    def sync(self):
        pass

    def close(self):
        self._open = False

    def isopen(self):
        return self._open

    def filepath(self):
        return "<emulated>"

    def set_auto_chartostring(self, onoff):
        pass


class TestEmulatedDataset:
    """An object that only looks like a dataset - the Xarray bridge."""

    @pytest.fixture
    def written(self, cube):
        dataset = _EmulatedDataset()
        inetcdf.save(cube, dataset, compute=False)
        return dataset

    def test_variables_are_created(self, written):
        assert sorted(written.variables) == ["air", "latitude", "longitude"]

    def test_dimensions_are_created(self, written):
        assert {name: len(dim) for name, dim in written.dimensions.items()} == {
            "latitude": 2,
            "longitude": 3,
        }

    def test_attributes_reach_the_emulator_as_bytes(self, written):
        # _bytes_if_ascii: an ASCII string attribute is offered as bytes, so
        # that netCDF4 gives it type NC_CHAR. An emulator sees the same.
        assert written.variables["air"]._attrs == {
            "standard_name": b"air_temperature",
            "units": b"K",
        }
        assert written._attrs == {"Conventions": b"CF-1.7"}

    def test_data_arrives_as_a_data_array(self, written, cube):
        # Not via __setitem__, which the emulated variable refuses.
        assert np.array_equal(written.variables["air"]._data_array, cube.data)
        assert np.array_equal(written.variables["latitude"]._data_array, np.arange(2.0))

    def test_dimensions_are_recorded_on_the_variable(self, written):
        assert written.variables["air"].dimensions == ("latitude", "longitude")
