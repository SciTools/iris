# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""One test body, run against every :class:`CFDataset` implementation.

Subclass :class:`CFDatasetContract` in a module named for the implementation,
supply the three fixtures it declares, and every test here runs against it.
The class deliberately has no ``Test`` prefix, so pytest does not collect it
where it is written, only where it is subclassed.

Spec section 6: the interface is only worth having if both sides of it agree,
and agreement is cheapest to check by asking the same questions twice.
"""

import numpy as np
import pytest

from iris.fileformats.cf.dataset import CFDataset, CFDatasetVariable

#: The canonical dataset every implementation is checked against.
CONTRACT_DIMENSIONS = {"y": 3, "x": 2}
CONTRACT_DATA = np.arange(6, dtype="f8").reshape(3, 2)
CONTRACT_VARIABLE_ATTRS = {"units": "K", "long_name": "surface temperature"}
CONTRACT_GLOBALS = {"Conventions": "CF-1.7", "title": "contract"}


def populate(dataset):
    """Fill an empty writable dataset with the canonical contents.

    Uses only the CFDataset API, so this doubles as the write half of the
    contract: an implementation that cannot build this cannot be used to save.
    """
    for name, size in CONTRACT_DIMENSIONS.items():
        dataset.create_dimension(name, size)
    for name, value in CONTRACT_GLOBALS.items():
        dataset.attributes[name] = value

    variable = dataset.create_variable(
        "air_temperature", np.dtype("f8"), tuple(CONTRACT_DIMENSIONS), fill_value=-1.0
    )
    for name, value in CONTRACT_VARIABLE_ATTRS.items():
        variable.attributes[name] = value
    variable[:] = CONTRACT_DATA

    # A dimensionless variable: what a grid mapping is, and the one shape a
    # CFDatasetVariable must handle without a leading dimension.
    scalar = dataset.create_variable("grid", np.dtype("i4"))
    scalar.attributes["grid_mapping_name"] = "latitude_longitude"
    scalar[()] = 0
    return dataset


class CFDatasetContract:
    """What every CFDataset implementation must do, whatever it stores into."""

    @pytest.fixture
    def readable(self):
        """Return an open read-mode dataset holding what populate() writes."""
        raise NotImplementedError("supply a 'readable' fixture")

    @pytest.fixture
    def writable(self):
        """Return an open, empty, write-mode dataset."""
        raise NotImplementedError("supply a 'writable' fixture")

    @pytest.fixture
    def reopen(self):
        """Return a callable giving a fresh read-mode dataset over 'writable'."""
        raise NotImplementedError("supply a 'reopen' fixture")

    # -- The dataset itself ------------------------------------------------

    def test_is_a_cf_dataset(self, readable):
        assert isinstance(readable, CFDataset)

    def test_location_is_a_non_empty_string(self, readable):
        assert isinstance(readable.location, str)
        assert readable.location

    def test_mode_is_a_read_mode(self, readable):
        assert readable.mode in ("r", "r+", "a")

    def test_starts_open(self, readable):
        assert readable.closed is False

    def test_close_then_closed(self, writable):
        writable.close()
        assert writable.closed is True

    def test_close_is_idempotent(self, writable):
        writable.close()
        writable.close()
        assert writable.closed is True

    def test_context_manager_returns_self_and_closes(self, writable):
        with writable as entered:
            assert entered is writable
        assert writable.closed is True

    def test_dimensions(self, readable):
        assert dict(readable.dimensions) == CONTRACT_DIMENSIONS

    def test_global_attributes(self, readable):
        assert dict(readable.attributes) == CONTRACT_GLOBALS

    def test_variable_names(self, readable):
        assert sorted(readable.variables) == ["air_temperature", "grid"]

    # -- A variable --------------------------------------------------------

    @pytest.fixture
    def variable(self, readable):
        return readable.variables["air_temperature"]

    @pytest.fixture
    def scalar(self, readable):
        return readable.variables["grid"]

    def test_variable_is_a_cf_dataset_variable(self, variable):
        assert isinstance(variable, CFDatasetVariable)

    def test_variable_name(self, variable):
        assert variable.name == "air_temperature"

    def test_variable_location_matches_its_dataset(self, variable, readable):
        assert variable.location == readable.location

    def test_variable_dimensions(self, variable):
        assert variable.dimensions == tuple(CONTRACT_DIMENSIONS)

    def test_variable_shape(self, variable):
        assert variable.shape == tuple(CONTRACT_DIMENSIONS.values())

    def test_variable_dtype(self, variable):
        assert variable.dtype == np.dtype("f8")

    def test_variable_size(self, variable):
        assert variable.size == CONTRACT_DATA.size

    def test_variable_ndim(self, variable):
        assert variable.ndim == CONTRACT_DATA.ndim

    def test_variable_len(self, variable):
        assert len(variable) == CONTRACT_DATA.shape[0]

    def test_variable_data(self, variable):
        np.testing.assert_array_equal(variable[:], CONTRACT_DATA)

    def test_variable_data_indexed(self, variable):
        np.testing.assert_array_equal(variable[1], CONTRACT_DATA[1])

    def test_variable_attributes(self, variable):
        for name, value in CONTRACT_VARIABLE_ATTRS.items():
            assert variable.attributes[name] == value

    def test_variable_attributes_omit_unset_names(self, variable):
        assert "nonesuch" not in variable.attributes
        with pytest.raises(KeyError):
            variable.attributes["nonesuch"]

    def test_variable_fill_value(self, variable):
        assert variable.fill_value == -1.0

    def test_variable_chunking_is_none_or_a_shape(self, variable):
        chunking = variable.chunking
        assert chunking is None or (
            isinstance(chunking, tuple)
            and len(chunking) == len(variable.shape)
            and all(isinstance(size, int) for size in chunking)
        )

    def test_scalar_variable(self, scalar):
        assert scalar.dimensions == ()
        assert scalar.shape == ()
        with pytest.raises(TypeError, match="unsized"):
            len(scalar)

    # -- Writing -----------------------------------------------------------

    def test_populate_then_read_back(self, writable, reopen):
        populate(writable)
        writable.close()
        with reopen() as reader:
            np.testing.assert_array_equal(
                reader.variables["air_temperature"][:], CONTRACT_DATA
            )
            assert dict(reader.dimensions) == CONTRACT_DIMENSIONS
            assert dict(reader.attributes) == CONTRACT_GLOBALS

    def test_created_variable_is_visible_on_the_dataset(self, writable):
        writable.create_dimension("x", 2)
        created = writable.create_variable("thing", np.dtype("f4"), ("x",))
        assert writable.variables["thing"].name == created.name

    def test_create_variable_without_dimensions(self, writable):
        # Finding F4: saver.py:2080 supplies only a name and a dtype.
        variable = writable.create_variable("grid", np.dtype("i4"))
        assert variable.dimensions == ()

    def test_setitem_then_getitem(self, writable):
        writable.create_dimension("x", 3)
        variable = writable.create_variable("thing", np.dtype("f4"), ("x",))
        variable[:] = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(variable[:], [1.0, 2.0, 3.0])

    def test_attribute_write_through(self, writable, reopen):
        populate(writable)
        writable.variables["air_temperature"].attributes["comment"] = "added"
        writable.close()
        with reopen() as reader:
            assert reader.variables["air_temperature"].attributes["comment"] == "added"

    def test_write_handle_works_after_close(self, writable, reopen):
        populate(writable)
        handle = writable.variables["air_temperature"].write_handle()
        writable.close()

        payload = CONTRACT_DATA * 10
        handle[:] = payload
        with reopen() as reader:
            np.testing.assert_array_equal(
                reader.variables["air_temperature"][:], payload
            )

    def test_sync_and_finalise_are_callable(self, writable):
        populate(writable)
        assert writable.sync() is None
        assert writable.finalise() is None
        assert writable.closed is False
