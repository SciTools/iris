# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :class:`iris.common.metadata.CubeMetadata`.

"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from copy import deepcopy
import unittest.mock as mock
from unittest.mock import sentinel

import pytest

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata, CubeMetadata


def _make_metadata(
    standard_name=None,
    long_name=None,
    var_name=None,
    attributes=None,
    force_mapping=True,
):
    if force_mapping:
        if attributes is None:
            attributes = {}
        else:
            attributes = dict(STASH=attributes)

    return CubeMetadata(
        standard_name=standard_name,
        long_name=long_name,
        var_name=var_name,
        units=None,
        attributes=attributes,
        cell_methods=None,
    )


class Test(tests.IrisTest):
    def setUp(self):
        self.standard_name = mock.sentinel.standard_name
        self.long_name = mock.sentinel.long_name
        self.var_name = mock.sentinel.var_name
        self.units = mock.sentinel.units
        self.attributes = mock.sentinel.attributes
        self.cell_methods = mock.sentinel.cell_methods
        self.cls = CubeMetadata

    def test_repr(self):
        metadata = self.cls(
            standard_name=self.standard_name,
            long_name=self.long_name,
            var_name=self.var_name,
            units=self.units,
            attributes=self.attributes,
            cell_methods=self.cell_methods,
        )
        fmt = (
            "CubeMetadata(standard_name={!r}, long_name={!r}, var_name={!r}, "
            "units={!r}, attributes={!r}, cell_methods={!r})"
        )
        expected = fmt.format(
            self.standard_name,
            self.long_name,
            self.var_name,
            self.units,
            self.attributes,
            self.cell_methods,
        )
        self.assertEqual(expected, repr(metadata))

    def test__fields(self):
        expected = (
            "standard_name",
            "long_name",
            "var_name",
            "units",
            "attributes",
            "cell_methods",
        )
        self.assertEqual(self.cls._fields, expected)

    def test_bases(self):
        self.assertTrue(issubclass(self.cls, BaseMetadata))


@pytest.fixture(params=CubeMetadata._fields)
def fieldname(request):
    return request.param


@pytest.fixture(params=["strict", "lenient"])
def op_leniency(request):
    return request.param


class Test___eq__:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lvalues = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            # Must be a mapping.
            attributes=dict(),
            cell_methods=sentinel.cell_methods,
        )
        # Setup another values tuple with all-distinct content objects.
        self.rvalues = deepcopy(self.lvalues)
        self.dummy = sentinel.dummy
        self.cls = CubeMetadata

    def test_wraps_docstring(self):
        assert self.cls.__eq__.__doc__ == BaseMetadata.__eq__.__doc__

    def test_lenient_service(self):
        qualname___eq__ = _qualname(self.cls.__eq__)
        assert qualname___eq__ in _LENIENT
        assert _LENIENT[qualname___eq__]
        assert _LENIENT[self.cls.__eq__]

    def test_call(self):
        other = sentinel.other
        return_value = sentinel.return_value
        metadata = self.cls(*(None,) * len(self.cls._fields))
        with mock.patch.object(
            BaseMetadata, "__eq__", return_value=return_value
        ) as mocker:
            result = metadata.__eq__(other)

        assert return_value == result
        assert mocker.call_args_list == [mock.call(other)]

    def test_op_same(self, op_leniency):
        # Check op all-same content, but all-new data.
        # NOTE: test for both strict/lenient, should both work the same.
        is_lenient = op_leniency == "lenient"
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check equality both l==r and r==l.
            assert lmetadata.__eq__(rmetadata)
            assert rmetadata.__eq__(lmetadata)

    def test_op_different__none(self, fieldname, op_leniency):
        # Compare when a given field is set with 'None', both strict + lenient.
        if fieldname == "attributes":
            # Must be a dict, cannot be None.
            pytest.skip()
        else:
            is_lenient = op_leniency == "lenient"
            lmetadata = self.cls(**self.lvalues)
            self.rvalues.update({fieldname: None})
            rmetadata = self.cls(**self.rvalues)
            if fieldname in ("cell_methods", "standard_name", "units"):
                # These ones are compared strictly
                expect_success = False
            elif fieldname in ("var_name", "long_name"):
                # For other 'normal' fields : lenient succeeds, strict does not.
                expect_success = is_lenient
            else:
                # Ensure we are handling all the different field cases
                raise ValueError(
                    f"{self.__name__} unhandled fieldname : {fieldname}"
                )

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=is_lenient
            ):
                # Check equality both l==r and r==l.
                assert lmetadata.__eq__(rmetadata) == expect_success
                assert rmetadata.__eq__(lmetadata) == expect_success

    def test_op_different__value(self, fieldname, op_leniency):
        # Compare when a given field value is changed, both strict + lenient.
        if fieldname == "attributes":
            # Dicts have more possibilities: handled separately.
            pytest.skip()
        else:
            is_lenient = op_leniency == "lenient"
            lmetadata = self.cls(**self.lvalues)
            self.rvalues.update({fieldname: self.dummy})
            rmetadata = self.cls(**self.rvalues)
            if fieldname in (
                "cell_methods",
                "standard_name",
                "units",
                "long_name",
            ):
                # These ones are compared strictly
                expect_success = False
            elif fieldname == "var_name":
                # For other 'normal' fields : lenient succeeds, strict does not.
                expect_success = is_lenient
            else:
                # Ensure we are handling all the different field cases
                raise ValueError(
                    f"{self.__name__} unhandled fieldname : {fieldname}"
                )

            with mock.patch(
                "iris.common.metadata._LENIENT", return_value=is_lenient
            ):
                # Check equality both l==r and r==l.
                assert lmetadata.__eq__(rmetadata) == expect_success
                assert rmetadata.__eq__(lmetadata) == expect_success

    def test_op_different__attribute_extra(self, op_leniency):
        # Check when one set of attributes has an extra entry.
        is_lenient = op_leniency == "lenient"
        lmetadata = self.cls(**self.lvalues)
        self.rvalues["attributes"]["_extra_"] = 1
        rmetadata = self.cls(**self.rvalues)
        # This counts as equal *only* in the lenient case.
        expect_success = is_lenient
        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check equality both l==r and r==l.
            assert lmetadata.__eq__(rmetadata) == expect_success
            assert rmetadata.__eq__(lmetadata) == expect_success

    def test_op_different__attribute_value(self, op_leniency):
        # Check when one set of attributes has an extra entry.
        is_lenient = op_leniency == "lenient"
        self.lvalues["attributes"]["_extra_"] = mock.sentinel.value1
        self.rvalues["attributes"]["_extra_"] = mock.sentinel.value2
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)
        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # This should ALWAYS fail.
            assert not lmetadata.__eq__(rmetadata)
            assert not rmetadata.__eq__(lmetadata)


class Test___lt__(tests.IrisTest):
    def setUp(self):
        self.cls = CubeMetadata
        self.one = self.cls(1, 1, 1, 1, 1, 1)
        self.two = self.cls(1, 1, 1, 2, 1, 1)
        self.none = self.cls(1, 1, 1, None, 1, 1)
        self.attributes_cm = self.cls(1, 1, 1, 1, 10, 10)

    def test__ascending_lt(self):
        result = self.one < self.two
        self.assertTrue(result)

    def test__descending_lt(self):
        result = self.two < self.one
        self.assertFalse(result)

    def test__none_rhs_operand(self):
        result = self.one < self.none
        self.assertFalse(result)

    def test__none_lhs_operand(self):
        result = self.none < self.one
        self.assertTrue(result)

    def test__ignore_attributes_cell_methods(self):
        result = self.one < self.attributes_cm
        self.assertFalse(result)
        result = self.attributes_cm < self.one
        self.assertFalse(result)


class Test_combine:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.lvalues = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            cell_methods=sentinel.cell_methods,
        )
        # Get a second copy with all-new objects.
        self.rvalues = deepcopy(self.lvalues)
        self.dummy = sentinel.dummy
        self.cls = CubeMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        assert self.cls.combine.__doc__ == BaseMetadata.combine.__doc__

    def test_lenient_service(self):
        qualname_combine = _qualname(self.cls.combine)
        assert qualname_combine in _LENIENT
        assert _LENIENT[qualname_combine]
        assert _LENIENT[self.cls.combine]

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = self.none.combine(other)

        assert return_value == result
        assert mocker.call_args_list == [mock.call(other, lenient=None)]

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "combine", return_value=return_value
        ) as mocker:
            result = self.none.combine(other, lenient=lenient)

        assert return_value == result
        assert mocker.call_args_list == [mock.call(other, lenient=lenient)]

    def test_op_same(self, op_leniency):
        # Result is same as either input, both strict + lenient.
        is_lenient = op_leniency == "lenient"
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)
        expected = self.lvalues

        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_different__none(self, fieldname, op_leniency):
        # One field has a given field set to 'None', both strict + lenient.
        if fieldname == "attributes":
            # Can't be None : Tested separately
            pytest.skip()

        is_lenient = op_leniency == "lenient"

        lmetadata = self.cls(**self.lvalues)
        # Cancel one setting in the rhs argument.
        self.rvalues[fieldname] = None
        rmetadata = self.cls(**self.rvalues)

        if fieldname in ("cell_methods", "units"):
            # NB cell-methods and units *always* strict behaviour.
            # strict form : take only those which both have set
            strict_result = True
        elif fieldname in ("standard_name", "long_name", "var_name"):
            strict_result = not is_lenient
        else:
            # Ensure we are handling all the different field cases
            raise ValueError(
                f"{self.__name__} unhandled fieldname : {fieldname}"
            )

        if strict_result:
            # include only those which both have
            expected = self.rvalues
        else:
            # also include those which only 1 has
            expected = self.lvalues

        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_different__value(self, fieldname, op_leniency):
        # One field has different value for lhs/rhs, both strict + lenient.
        if fieldname == "attributes":
            # Can't be None : Tested separately
            pytest.skip()

        is_lenient = op_leniency == "lenient"

        self.lvalues[fieldname] = mock.sentinel.value1
        self.rvalues[fieldname] = mock.sentinel.value2
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        # In all cases, this field should be None in the result
        expected = self.lvalues.copy()
        expected[fieldname] = None

        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_different__attribute_extra(self, op_leniency):
        # One field has an extra attribute, both strict + lenient.
        is_lenient = op_leniency == "lenient"

        self.lvalues["attributes"] = {"_a_common_": mock.sentinel.dummy}
        self.rvalues["attributes"] = self.lvalues["attributes"].copy()
        self.rvalues["attributes"]["_extra_"] = mock.sentinel.testvalue
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        if is_lenient:
            # the extra attribute should appear in the result ..
            expected = self.rvalues
        else:
            # .. it should not
            expected = self.lvalues

        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_different__attribute_value(self, op_leniency):
        # lhs and rhs have different values for an attribute, both strict + lenient.
        is_lenient = op_leniency == "lenient"

        self.lvalues["attributes"] = {
            "_a_common_": self.dummy,
            "_b_common_": mock.sentinel.value1,
        }
        self.lvalues["attributes"] = {
            "_a_common_": self.dummy,
            "_b_common_": mock.sentinel.value2,
        }
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        # Result contains entirely EMPTY attributes (either strict or lenient).
        # TODO: is this maybe a mistake of the existing implementation ?
        expected = self.lvalues.copy()
        expected["attributes"] = None

        with mock.patch(
            "iris.common.metadata._LENIENT", return_value=is_lenient
        ):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected


class Test_difference(tests.IrisTest):
    def setUp(self):
        self.values = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=sentinel.attributes,
            cell_methods=sentinel.cell_methods,
        )
        self.dummy = sentinel.dummy
        self.cls = CubeMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        self.assertEqual(
            BaseMetadata.difference.__doc__, self.cls.difference.__doc__
        )

    def test_lenient_service(self):
        qualname_difference = _qualname(self.cls.difference)
        self.assertIn(qualname_difference, _LENIENT)
        self.assertTrue(_LENIENT[qualname_difference])
        self.assertTrue(_LENIENT[self.cls.difference])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)

    def test_op_lenient_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertIsNone(lmetadata.difference(rmetadata))
            self.assertIsNone(rmetadata.difference(lmetadata))

    def test_op_lenient_same_none(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["var_name"] = None
        rmetadata = self.cls(**right)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertIsNone(lmetadata.difference(rmetadata))
            self.assertIsNone(rmetadata.difference(lmetadata))

    def test_op_lenient_same_cell_methods_none(self):
        lmetadata = self.cls(**self.values)
        right = self.values.copy()
        right["cell_methods"] = None
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["cell_methods"] = (sentinel.cell_methods, None)
        rexpected = deepcopy(self.none)._asdict()
        rexpected["cell_methods"] = (None, sentinel.cell_methods)

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_different(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["units"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["units"] = (left["units"], right["units"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["units"] = lexpected["units"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_lenient_different_cell_methods(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["cell_methods"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["cell_methods"] = (
            left["cell_methods"],
            right["cell_methods"],
        )
        rexpected = deepcopy(self.none)._asdict()
        rexpected["cell_methods"] = lexpected["cell_methods"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=True):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_same(self):
        lmetadata = self.cls(**self.values)
        rmetadata = self.cls(**self.values)

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertIsNone(lmetadata.difference(rmetadata))
            self.assertIsNone(rmetadata.difference(lmetadata))

    def test_op_strict_different(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_cell_methods(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["cell_methods"] = self.dummy
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["cell_methods"] = (
            left["cell_methods"],
            right["cell_methods"],
        )
        rexpected = deepcopy(self.none)._asdict()
        rexpected["cell_methods"] = lexpected["cell_methods"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_none(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["long_name"] = None
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["long_name"] = (left["long_name"], right["long_name"])
        rexpected = deepcopy(self.none)._asdict()
        rexpected["long_name"] = lexpected["long_name"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )

    def test_op_strict_different_measure_none(self):
        left = self.values.copy()
        lmetadata = self.cls(**left)
        right = self.values.copy()
        right["cell_methods"] = None
        rmetadata = self.cls(**right)
        lexpected = deepcopy(self.none)._asdict()
        lexpected["cell_methods"] = (
            left["cell_methods"],
            right["cell_methods"],
        )
        rexpected = deepcopy(self.none)._asdict()
        rexpected["cell_methods"] = lexpected["cell_methods"][::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=False):
            self.assertEqual(
                lexpected, lmetadata.difference(rmetadata)._asdict()
            )
            self.assertEqual(
                rexpected, rmetadata.difference(lmetadata)._asdict()
            )


class Test_equal(tests.IrisTest):
    def setUp(self):
        self.cls = CubeMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        self.assertEqual(BaseMetadata.equal.__doc__, self.cls.equal.__doc__)

    def test_lenient_service(self):
        qualname_equal = _qualname(self.cls.equal)
        self.assertIn(qualname_equal, _LENIENT)
        self.assertTrue(_LENIENT[qualname_equal])
        self.assertTrue(_LENIENT[self.cls.equal])

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = self.none.equal(other)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=None), kwargs)

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "equal", return_value=return_value
        ) as mocker:
            result = self.none.equal(other, lenient=lenient)

        self.assertEqual(return_value, result)
        self.assertEqual(1, mocker.call_count)
        (arg,), kwargs = mocker.call_args
        self.assertEqual(other, arg)
        self.assertEqual(dict(lenient=lenient), kwargs)


class Test_name(tests.IrisTest):
    def setUp(self):
        self.default = CubeMetadata.DEFAULT_NAME

    def test_standard_name(self):
        token = "standard_name"
        metadata = _make_metadata(standard_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_standard_name__invalid_token(self):
        token = "nope nope"
        metadata = _make_metadata(standard_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_long_name(self):
        token = "long_name"
        metadata = _make_metadata(long_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_long_name__invalid_token(self):
        token = "nope nope"
        metadata = _make_metadata(long_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_var_name(self):
        token = "var_name"
        metadata = _make_metadata(var_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_var_name__invalid_token(self):
        token = "nope nope"
        metadata = _make_metadata(var_name=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_attributes(self):
        token = "stash"
        metadata = _make_metadata(attributes=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, token)

    def test_attributes__invalid_token(self):
        token = "nope nope"
        metadata = _make_metadata(attributes=token)
        result = metadata.name()
        self.assertEqual(result, token)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_attributes__non_mapping(self):
        metadata = _make_metadata(force_mapping=False)
        self.assertIsNone(metadata.attributes)
        emsg = "Invalid 'CubeMetadata.attributes' member, must be a mapping."
        with self.assertRaisesRegex(AttributeError, emsg):
            _ = metadata.name()

    def test_default(self):
        metadata = _make_metadata()
        result = metadata.name()
        self.assertEqual(result, self.default)
        result = metadata.name(token=True)
        self.assertEqual(result, self.default)

    def test_default__invalid_token(self):
        token = "nope nope"
        metadata = _make_metadata()
        result = metadata.name(default=token)
        self.assertEqual(result, token)
        emsg = "Cannot retrieve a valid name token"
        with self.assertRaisesRegex(ValueError, emsg):
            _ = metadata.name(default=token, token=True)


class Test__names(tests.IrisTest):
    def test_standard_name(self):
        token = "standard_name"
        metadata = _make_metadata(standard_name=token)
        expected = (token, None, None, None)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_long_name(self):
        token = "long_name"
        metadata = _make_metadata(long_name=token)
        expected = (None, token, None, None)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_var_name(self):
        token = "var_name"
        metadata = _make_metadata(var_name=token)
        expected = (None, None, token, None)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_attributes(self):
        token = "stash"
        metadata = _make_metadata(attributes=token)
        expected = (None, None, None, token)
        result = metadata._names
        self.assertEqual(expected, result)

    def test_attributes__non_mapping(self):
        metadata = _make_metadata(force_mapping=False)
        self.assertIsNone(metadata.attributes)
        emsg = "Invalid 'CubeMetadata.attributes' member, must be a mapping."
        with self.assertRaisesRegex(AttributeError, emsg):
            _ = metadata._names

    def test_None(self):
        metadata = _make_metadata()
        expected = (None, None, None, None)
        result = metadata._names
        self.assertEqual(expected, result)


if __name__ == "__main__":
    tests.main()
