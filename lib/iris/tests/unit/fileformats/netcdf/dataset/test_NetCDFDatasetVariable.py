# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for :class:`iris.fileformats.netcdf._dataset.NetCDFDatasetVariable`."""

from collections.abc import MutableMapping
import warnings

import numpy as np
import pytest

from iris._deprecation import IrisDeprecation
from iris.fileformats.cf.dataset import CFDatasetVariable
from iris.fileformats.netcdf import _bytecoding_datasets, _dataset, _thread_safe_nc

from .conftest import SAMPLE_AIR, SAMPLE_LABELS


@pytest.fixture
def raw(sample_path):
    dataset = _thread_safe_nc.DatasetWrapper(sample_path, mode="r")
    yield dataset
    dataset.close()


@pytest.fixture
def encoded(sample_path):
    dataset = _bytecoding_datasets.EncodedDataset(sample_path, mode="r")
    yield dataset
    dataset.close()


def wrap(dataset, name, location, write_lock_factory=None):
    return _dataset.NetCDFDatasetVariable(
        dataset.variables[name], location, write_lock_factory=write_lock_factory
    )


@pytest.fixture
def air(raw, sample_location):
    return wrap(raw, "air_temperature", sample_location)


@pytest.fixture
def height(raw, sample_location):
    return wrap(raw, "height", sample_location)


class TestStorageProperties:
    def test_is_a_cf_dataset_variable(self, air):
        assert isinstance(air, CFDatasetVariable)

    def test_name(self, air):
        assert air.name == "air_temperature"

    def test_location(self, air, sample_location):
        assert air.location == sample_location

    def test_dimensions_are_a_tuple(self, air):
        # netCDF4 answers with a tuple already, but the contract says tuple
        # and CFVariable.spans does set arithmetic on it, so pin it.
        assert air.dimensions == ("time", "lat")
        assert isinstance(air.dimensions, tuple)

    def test_shape_dtype_size_ndim(self, air):
        assert air.shape == (3, 4)
        assert air.dtype == np.dtype("f4")
        assert air.size == 12
        assert air.ndim == 2

    def test_len(self, air):
        assert len(air) == 3

    def test_scalar_has_no_len(self, height):
        assert height.shape == ()
        with pytest.raises(TypeError, match="unsized"):
            len(height)

    def test_fill_value(self, air):
        assert air.fill_value == -999.0

    def test_no_fill_value_is_none(self, height):
        assert height.fill_value is None

    def test_chunking_is_a_tuple(self, air):
        # netCDF4 answers with a list; the contract says a shape.
        assert air.chunking == (1, 4)

    def test_contiguous_chunking_is_none(self, height):
        # netCDF4 answers "contiguous" here, and loader.py already treats
        # that and None identically, so both collapse to None.
        assert height.chunking is None


class TestData:
    def test_getitem(self, air):
        np.testing.assert_array_equal(air[:], SAMPLE_AIR)

    def test_getitem_indexed(self, air):
        np.testing.assert_array_equal(air[0], SAMPLE_AIR[0])


class TestAttributes:
    def test_is_a_plain_mutable_mapping(self, air):
        # Tracking belongs to CFVariable, not here: a CFDatasetVariable can
        # have two CFVariables promoted over it - see finding F1.
        assert isinstance(air.attributes, MutableMapping)
        assert not hasattr(air.attributes, "read")

    def test_contents(self, air):
        assert dict(air.attributes) == {
            "_FillValue": -999.0,
            "units": "K",
            "standard_name": "air_temperature",
            "coordinates": "height",
        }

    def test_is_built_once(self, air):
        assert air.attributes is air.attributes

    def test_values_are_read_once(self, mocker, sample_location):
        # Pins the contract _NetCDFAttributes's own docstring states: values
        # are read once, at construction, because the interface promises a
        # mapping whose keys/items cost nothing. test_is_built_once above
        # only pins that the mapping object is cached, not that its values
        # are materialised rather than fetched again on every read - call
        # counts are the only way to see that.
        variable = mocker.Mock()
        variable.ncattrs.return_value = ["units"]
        variable.getncattr.return_value = "K"
        wrapped = _dataset.NetCDFDatasetVariable(variable, sample_location)
        _ = wrapped.attributes["units"]
        _ = wrapped.attributes["units"]
        assert variable.ncattrs.call_count == 1
        assert variable.getncattr.call_count == 1

    def test_missing_key_raises_key_error(self, air):
        with pytest.raises(KeyError, match="nonesuch"):
            air.attributes["nonesuch"]

    def test_unreadable_attribute_becomes_empty_string(self, mocker, sample_location):
        # ncattrs() can list a name that getncattr then refuses. _getncattr in
        # cf/_reader.py tolerated exactly this with a "" default, and that
        # tolerance has to survive the move.
        #
        # A real VariableWrapper cannot be mocked this way: its __setattr__
        # forwards every set to the contained netCDF4 object, so patching
        # one of its methods would write a same-named attribute into the
        # file instead. A bare stand-in with the two methods
        # _NetCDFAttributes actually calls is enough.
        variable = mocker.Mock()
        variable.ncattrs.return_value = ["units", "broken"]
        variable.getncattr.side_effect = lambda name: (
            "m" if name == "units" else _raise(AttributeError(name))
        )
        wrapped = _dataset.NetCDFDatasetVariable(variable, sample_location)
        assert wrapped.attributes["broken"] == ""


def _raise(exception):
    raise exception


class _EmulatedVariable:
    """A stand-in for the Xarray bridge's own variable object.

    Never a real, file-backed
    :class:`~iris.fileformats.netcdf._thread_safe_nc.VariableWrapper`: that
    class's ``__setattr__`` forwards every set to the actual netCDF4 object,
    so it cannot carry an ad-hoc ``_data_array`` of its own.
    """

    def ncattrs(self):
        return []


class TestNetCDFOnlyMembers:
    def test_variable_exposes_the_backing_wrapper(self, raw, sample_location):
        # raw.variables constructs a fresh VariableWrapper on every access,
        # so the check has to compare against the one instance actually
        # passed in, not a second, independently fetched wrapper.
        variable = raw.variables["air_temperature"]
        wrapped = _dataset.NetCDFDatasetVariable(variable, sample_location)
        assert wrapped.variable is variable

    def test_not_variable_length(self, air):
        assert air.is_variable_length is False

    def test_not_emulated(self, air):
        assert air.is_emulated is False

    def test_emulated_data_array_raises_when_not_emulated(self, air):
        with pytest.raises(AttributeError, match="_data_array"):
            air.emulated_data_array

    def test_setting_emulated_data_array_raises_when_not_emulated(self, air):
        # The setter refuses for the same reason the getter does, and the
        # consequence of not refusing is worse: VariableWrapper.__setattr__
        # forwards to the contained object, so the set would write a file
        # attribute named "_data_array".
        with pytest.raises(AttributeError, match="_data_array"):
            air.emulated_data_array = np.ones(3)
        assert "_data_array" not in air.variable.ncattrs()

    def test_emulated_round_trip(self, sample_location):
        # The Xarray bridge hook, issue #4994: an emulating variable carries
        # its own array instead of file storage.
        variable = _EmulatedVariable()
        wrapped = _dataset.NetCDFDatasetVariable(variable, sample_location)
        variable._data_array = np.zeros(3)
        assert wrapped.is_emulated is True
        np.testing.assert_array_equal(wrapped.emulated_data_array, np.zeros(3))
        wrapped.emulated_data_array = np.ones(3)
        np.testing.assert_array_equal(variable._data_array, np.ones(3))


class TestDeprecatedNetcdfMember:
    def test_a_reach_through_warns(self, air):
        with pytest.warns(IrisDeprecation, match="ncattrs"):
            names = air.deprecated_netcdf_member("ncattrs")()
        assert sorted(names) == [
            "_FillValue",
            "coordinates",
            "standard_name",
            "units",
        ]

    def test_a_missing_name_raises_and_does_not_warn(self, air):
        # hasattr() probes land here, and a probe that comes back False is
        # not a use of anything - warning about it would be noise. The
        # underlying netCDF4 variable raises its own AttributeError, which
        # does not name the attribute - only the type matters here.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(AttributeError):
                air.deprecated_netcdf_member("no_such_netcdf_member")

    def test_the_message_names_the_replacement(self, air):
        with pytest.warns(IrisDeprecation, match=r"cf_data\.variable"):
            air.deprecated_netcdf_member("ncattrs")


class TestCharacterData:
    def test_raw_wrapper_sees_the_char_dimension(self, raw, sample_location):
        label = wrap(raw, "label", sample_location)
        assert label.dimensions == ("time", "nchars")
        assert label.shape == (3, 8)
        assert label.dtype == np.dtype("S1")

    def test_encoded_wrapper_sees_strings(self, encoded, sample_location):
        # EncodedVariable drops the trailing char dimension and reports a
        # string dtype. The CFDatasetVariable must pass that through, not
        # reach around it to the contained netCDF4 variable.
        label = wrap(encoded, "label", sample_location)
        assert label.dimensions == ("time",)
        assert label.shape == (3,)
        assert label.dtype == np.dtype("U8")
        assert [text.strip() for text in label[:]] == SAMPLE_LABELS
