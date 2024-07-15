# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.common.metadata.CubeMetadata`."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
from typing import Any, ClassVar

import iris.tests as tests  # isort:skip

from copy import deepcopy
import unittest.mock as mock
from unittest.mock import sentinel

import pytest

from iris.common.lenient import _LENIENT, _qualname
from iris.common.metadata import BaseMetadata, CubeMetadata
from iris.cube import CubeAttrsDict


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


@pytest.fixture(params=CubeMetadata._fields)  # type: ignore[attr-defined]
def fieldname(request):
    """Parametrize testing over all CubeMetadata field names."""
    return request.param


@pytest.fixture(params=["strict", "lenient"])
def op_leniency(request):
    """Parametrize testing over strict or lenient operation."""
    return request.param


@pytest.fixture(params=["primaryAA", "primaryAX", "primaryAB"])
def primary_values(request):
    """Parametrize over the possible non-trivial pairs of operation values.

    The parameters all provide two attribute values which are the left- and right-hand
    arguments to the tested operation.  The attribute values are single characters from
    the end of the parameter name -- except that "X" denotes a "missing" attribute.

    The possible cases are:

    * one side has a value and the other is missing
    * left and right have the same non-missing value
    * left and right have different non-missing values
    """
    return request.param


@pytest.fixture(params=[False, True], ids=["primaryLocal", "primaryGlobal"])
def primary_is_global_not_local(request):
    """Parametrize split-attribute testing over "global" or "local" attribute types."""
    return request.param


@pytest.fixture(params=[False, True], ids=["leftrightL2R", "leftrightR2L"])
def order_reversed(request):
    """Parametrize split-attribute testing over "left OP right" or "right OP left"."""
    return request.param


# Define the expected results for split-attribute testing.
# This dictionary records the expected results for the various possible arrangements of
# values of a single attribute in the "left" and "right" inputs of a CubeMetadata
# operation.
# The possible operations are "equal", "combine" or "difference", and may all be
# performed "strict" or "lenient".
# N.B. the *same* results should also apply when left+right are swapped, with a suitable
# adjustment to the result value.  Likewise, results should be the same for either
# global- or local-style attributes.
_ALL_RESULTS: dict[str, dict[str, dict[str, Any]]] = {
    "equal": {
        "primaryAA": {"lenient": True, "strict": True},
        "primaryAX": {"lenient": True, "strict": False},
        "primaryAB": {"lenient": False, "strict": False},
    },
    "combine": {
        "primaryAA": {"lenient": "A", "strict": "A"},
        "primaryAX": {"lenient": "A", "strict": None},
        "primaryAB": {"lenient": None, "strict": None},
    },
    "difference": {
        "primaryAA": {"lenient": None, "strict": None},
        "primaryAX": {"lenient": None, "strict": ("A", None)},
        "primaryAB": {"lenient": ("A", "B"), "strict": ("A", "B")},
    },
}
# A fixed attribute name used for all the split-attribute testing.
_TEST_ATTRNAME = "_test_attr_"


def extract_attribute_value(split_dict, extract_global):
    """Extract a test-attribute value from a split-attribute dictionary.

    Parameters
    ----------
    split_dict : CubeAttrsDict
        a split dictionary from an operation result
    extract_global : bool
        whether to extract values of the global, or local, `_TEST_ATTRNAME` attribute

    Returns
    -------
        str | None
    """
    if extract_global:
        result = split_dict.globals.get(_TEST_ATTRNAME, None)
    else:
        result = split_dict.locals.get(_TEST_ATTRNAME, None)
    return result


def extract_result_value(input, extract_global):
    """Extract the values(s) of the main test attribute from an operation result.

    Parameters
    ----------
    input : bool | CubeMetadata
        an operation result : the structure varies for the three different operations.
    extract_global : bool
        whether to return values of a global, or local, `_TEST_ATTRNAME` attribute.

    Returns
    -------
    None | bool | str | tuple[None | str]
        result value(s)
    """
    if not isinstance(input, CubeMetadata):
        # Result is either boolean (for "equals") or a None (for "difference").
        result = input
    else:
        # Result is a CubeMetadata.  Get the value(s) of the required attribute.
        result = input.attributes

        if isinstance(result, CubeAttrsDict):
            result = extract_attribute_value(result, extract_global)
        else:
            # For "difference", input.attributes is a *pair* of dictionaries.
            assert isinstance(result, tuple)
            result = tuple(
                [extract_attribute_value(dic, extract_global) for dic in result]
            )
            if result == (None, None):
                # This value occurs when the desired attribute is *missing* from a
                # difference result, but other (secondary) attributes were *different*.
                # We want only differences of the *target* attribute, so convert these
                # to a plain 'no difference', for expected-result testing purposes.
                result = None

    return result


def make_attrsdict(value):
    """Return a dictionary containing a test attribute with the given value.

    If the value is "X", the attribute is absent (result is empty dict).
    """
    if value == "X":
        # Translate an "X" input as "missing".
        result = {}
    else:
        result = {_TEST_ATTRNAME: value}
    return result


def check_splitattrs_testcase(
    operation_name: str,
    check_is_lenient: bool,
    primary_inputs: str = "AA",  # two character values
    secondary_inputs: str = "XX",  # two character values
    check_global_not_local: bool = True,
    check_reversed: bool = False,
):
    """Test a metadata operation with split-attributes against known expected results.

    Parameters
    ----------
    operation_name : str
        One of "equal", "combine" or "difference.
    check_is_lenient : bool
        Whether the tested operation is performed 'lenient' or 'strict'.
    primary_inputs : str
        A pair of characters defining left + right attribute values for the operands of
        the operation.
    secondary_inputs : str
        A further pair of values for an attribute of the same name but "other" type
        ( i.e. global/local when the main test is local/global ).
    check_global_not_local : bool
        If `True` then the primary operands, and the tested result values, are *global*
        attributes, and the secondary ones are local.
        Otherwise, the other way around.
    check_reversed : bool
        If True, the left and right operands are exchanged, and the expected value
        modified according.

    Notes
    -----
    The expected result of an operation is mostly defined by :  the operation applied;
    the main "primary" inputs; and the lenient/strict mode.

    In the case of the "equals" operation, however, the expected result is simply
    set to `False` if the secondary inputs do not match.

    Calling with different values for the keywords aims to show that the main operation
    has the expected value, from _ALL_RESULTS, the ***same in essentially all cases***
    ( though modified in specific ways for some factors ).

    This regularity also demonstrates the required independence over the other
    test-factors, i.e. global/local attribute type, and right-left order.
    """
    # Just for comfort, check that inputs are all one of a few single characters.
    assert all((item in list("ABCDX")) for item in (primary_inputs + secondary_inputs))
    # Interpret "primary" and "secondary" inputs as "global" and "local" attributes.
    if check_global_not_local:
        global_values, local_values = primary_inputs, secondary_inputs
    else:
        local_values, global_values = primary_inputs, secondary_inputs

    # Form 2 inputs to the operation :  Make left+right split-attribute input
    # dictionaries, with both the primary and secondary attribute value settings.
    input_dicts = [
        CubeAttrsDict(
            globals=make_attrsdict(global_value),
            locals=make_attrsdict(local_value),
        )
        for global_value, local_value in zip(global_values, local_values)
    ]
    # Make left+right CubeMetadata with those attributes, other fields all blank.
    input_l, input_r = [
        CubeMetadata(
            **{
                field: attrs if field == "attributes" else None
                for field in CubeMetadata._fields  # type: ignore[attr-defined]
            }
        )
        for attrs in input_dicts
    ]

    if check_reversed:
        # Swap the inputs to perform a 'reversed' calculation.
        input_l, input_r = input_r, input_l

    # Run the actual operation
    result = getattr(input_l, operation_name)(input_r, lenient=check_is_lenient)

    if operation_name == "difference" and check_reversed:
        # Adjust the result of a "reversed" operation to the 'normal' way round.
        # ( N.B. only "difference" results are affected by reversal. )
        if isinstance(result, CubeMetadata):
            result = result._replace(attributes=result.attributes[::-1])  # type: ignore[attr-defined]

    # Extract, from the operation result, the value to be tested against "expected".
    result = extract_result_value(result, check_global_not_local)

    # Get the *expected* result for this operation.
    which = "lenient" if check_is_lenient else "strict"
    primary_key = "primary" + primary_inputs
    expected = _ALL_RESULTS[operation_name][primary_key][which]
    if operation_name == "equal" and expected:
        # Account for the equality cases made `False` by mismatched secondary values.
        left, right = list(
            secondary_inputs
        )  # see https://github.com/python/mypy/issues/13823
        secondaries_same = left == right or (check_is_lenient and "X" in (left, right))
        if not secondaries_same:
            expected = False

    # Check that actual extracted operation result matches the "expected" one.
    assert result == expected


class MixinSplitattrsMatrixTests:
    """Define split-attributes tests to perform on all the metadata operations.

    This is inherited by the testclass for each operation :
    i.e. Test___eq__, Test_combine and Test_difference
    """

    # Define the operation name : set in each inheritor
    operation_name: ClassVar[str]

    def test_splitattrs_cases(
        self,
        op_leniency,
        primary_values,
        primary_is_global_not_local,
        order_reversed,
    ):
        """Check the basic operation against the expected result from _ALL_RESULTS.

        Parametrisation checks this for all combinations of various factors :

        * possible arrangements of the primary values
        * strict and lenient
        * global- and local-type attributes
        * left-to-right or right-to-left operation order.
        """
        primary_inputs = primary_values[-2:]
        check_is_lenient = {"strict": False, "lenient": True}[op_leniency]
        check_splitattrs_testcase(
            operation_name=self.operation_name,
            check_is_lenient=check_is_lenient,
            primary_inputs=primary_inputs,
            secondary_inputs="XX",
            check_global_not_local=primary_is_global_not_local,
            check_reversed=order_reversed,
        )

    @pytest.mark.parametrize(
        "secondary_values",
        [
            "secondaryXX",
            "secondaryCX",
            "secondaryXC",
            "secondaryCC",
            "secondaryCD",
        ],
        # NOTE: test CX as well as XC, since primary choices has "AX" but not "XA".
    )
    def test_splitattrs_global_local_independence(
        self,
        op_leniency,
        primary_values,
        secondary_values,
    ):
        """Check that results are (mostly) independent of the "other" type attributes.

        The operation on attributes of the 'primary' type (global/local) should be
        basically unaffected by those of the 'secondary' type (--> local/global).

        This is not really true for equality, so we adjust those results to compensate.
        See :func:`check_splitattrs_testcase` for explanations.

        Notes
        -----
        We provide this *separate* test for global/local attribute independence,
        parametrized over selected relevant arrangements of the 'secondary' values.
        We *don't* test with reversed order or "local" primary inputs, because matrix
        testing over *all* relevant factors produces too many possible combinations.
        """
        primary_inputs = primary_values[-2:]
        secondary_inputs = secondary_values[-2:]
        check_is_lenient = {"strict": False, "lenient": True}[op_leniency]
        check_splitattrs_testcase(
            operation_name=self.operation_name,
            check_is_lenient=check_is_lenient,
            primary_inputs=primary_inputs,
            secondary_inputs=secondary_inputs,
            check_global_not_local=True,
            check_reversed=False,
        )


class Test___eq__(MixinSplitattrsMatrixTests):
    operation_name = "equal"

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

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            # Check equality both l==r and r==l.
            assert lmetadata.__eq__(rmetadata)
            assert rmetadata.__eq__(lmetadata)

    def test_op_different__none(self, fieldname, op_leniency):
        # One side has field=value, and the other field=None, both strict + lenient.
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
                raise ValueError(f"{self.__name__} unhandled fieldname : {fieldname}")

            with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
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
                raise ValueError(f"{self.__name__} unhandled fieldname : {fieldname}")

            with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
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
        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            # Check equality both l==r and r==l.
            assert lmetadata.__eq__(rmetadata) == expect_success
            assert rmetadata.__eq__(lmetadata) == expect_success

    def test_op_different__attribute_value(self, op_leniency):
        # lhs and rhs have different values for an attribute, both strict + lenient.
        is_lenient = op_leniency == "lenient"
        self.lvalues["attributes"]["_extra_"] = mock.sentinel.value1
        self.rvalues["attributes"]["_extra_"] = mock.sentinel.value2
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)
        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
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


class Test_combine(MixinSplitattrsMatrixTests):
    operation_name = "combine"

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

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_different__none(self, fieldname, op_leniency):
        # One side has field=value, and the other field=None, both strict + lenient.
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
            raise ValueError(f"{self.__name__} unhandled fieldname : {fieldname}")

        if strict_result:
            # include only those which both have
            expected = self.rvalues
        else:
            # also include those which only 1 has
            expected = self.lvalues

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected

    def test_op_different__value(self, fieldname, op_leniency):
        # One field has different value for lhs/rhs, both strict + lenient.
        if fieldname == "attributes":
            # Attribute behaviours are tested separately
            pytest.skip()

        is_lenient = op_leniency == "lenient"

        self.lvalues[fieldname] = mock.sentinel.value1
        self.rvalues[fieldname] = mock.sentinel.value2
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        # In all cases, this field should be None in the result : leniency has no effect
        expected = self.lvalues.copy()
        expected[fieldname] = None

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
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

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
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

        # Result has entirely EMPTY attributes (whether strict or lenient).
        # TODO: is this maybe a mistake of the existing implementation ?
        expected = self.lvalues.copy()
        expected["attributes"] = None

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            # Check both l+r and r+l
            assert lmetadata.combine(rmetadata)._asdict() == expected
            assert rmetadata.combine(lmetadata)._asdict() == expected


class Test_difference(MixinSplitattrsMatrixTests):
    operation_name = "difference"

    @pytest.fixture(autouse=True)
    def setup(self):
        self.lvalues = dict(
            standard_name=sentinel.standard_name,
            long_name=sentinel.long_name,
            var_name=sentinel.var_name,
            units=sentinel.units,
            attributes=dict(),  # MUST be a dict
            cell_methods=sentinel.cell_methods,
        )
        # Make a copy with all-different objects in it.
        self.rvalues = deepcopy(self.lvalues)
        self.dummy = sentinel.dummy
        self.cls = CubeMetadata
        self.none = self.cls(*(None,) * len(self.cls._fields))

    def test_wraps_docstring(self):
        assert self.cls.difference.__doc__ == BaseMetadata.difference.__doc__

    def test_lenient_service(self):
        qualname_difference = _qualname(self.cls.difference)
        assert qualname_difference in _LENIENT
        assert _LENIENT[qualname_difference]
        assert _LENIENT[self.cls.difference]

    def test_lenient_default(self):
        other = sentinel.other
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other)

        assert return_value == result
        assert mocker.call_args_list == [mock.call(other, lenient=None)]

    def test_lenient(self):
        other = sentinel.other
        lenient = sentinel.lenient
        return_value = sentinel.return_value
        with mock.patch.object(
            BaseMetadata, "difference", return_value=return_value
        ) as mocker:
            result = self.none.difference(other, lenient=lenient)

        assert return_value == result
        assert mocker.call_args_list == [mock.call(other, lenient=lenient)]

    def test_op_same(self, op_leniency):
        is_lenient = op_leniency == "lenient"
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            assert lmetadata.difference(rmetadata) is None
            assert rmetadata.difference(lmetadata) is None

    def test_op_different__none(self, fieldname, op_leniency):
        # One side has field=value, and the other field=None, both strict + lenient.
        if fieldname in ("attributes",):
            # These cannot properly be set to 'None'.  Tested elsewhere.
            pytest.skip()

        is_lenient = op_leniency == "lenient"

        lmetadata = self.cls(**self.lvalues)
        self.rvalues[fieldname] = None
        rmetadata = self.cls(**self.rvalues)

        if fieldname in ("units", "cell_methods"):
            # These ones are always "strict"
            strict_result = True
        elif fieldname in ("standard_name", "long_name", "var_name"):
            strict_result = not is_lenient
        else:
            # Ensure we are handling all the different field cases
            raise ValueError(f"{self.__name__} unhandled fieldname : {fieldname}")

        if strict_result:
            diffentry = tuple([getattr(mm, fieldname) for mm in (lmetadata, rmetadata)])
            # NOTE: in these cases, the difference metadata will fail an == operation,
            # because of the 'None' entries.
            # But we can use metadata._asdict() and test that.
            lexpected = self.none._asdict()
            lexpected[fieldname] = diffentry
            rexpected = lexpected.copy()
            rexpected[fieldname] = diffentry[::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            if strict_result:
                assert lmetadata.difference(rmetadata)._asdict() == lexpected
                assert rmetadata.difference(lmetadata)._asdict() == rexpected
            else:
                # Expect NO differences
                assert lmetadata.difference(rmetadata) is None
                assert rmetadata.difference(lmetadata) is None

    def test_op_different__value(self, fieldname, op_leniency):
        # One field has different value for lhs/rhs, both strict + lenient.
        if fieldname == "attributes":
            # Attribute behaviours are tested separately
            pytest.skip()

        self.lvalues[fieldname] = mock.sentinel.value1
        self.rvalues[fieldname] = mock.sentinel.value2
        lmetadata = self.cls(**self.lvalues)
        rmetadata = self.cls(**self.rvalues)

        # In all cases, this field should show a difference : leniency has no effect
        ldiff_values = (mock.sentinel.value1, mock.sentinel.value2)
        ldiff_metadata = self.none._asdict()
        ldiff_metadata[fieldname] = ldiff_values
        rdiff_metadata = self.none._asdict()
        rdiff_metadata[fieldname] = ldiff_values[::-1]

        # Check both l+r and r+l
        assert lmetadata.difference(rmetadata)._asdict() == ldiff_metadata
        assert rmetadata.difference(lmetadata)._asdict() == rdiff_metadata

    def test_op_different__attribute_extra(self, op_leniency):
        # One field has an extra attribute, both strict + lenient.
        is_lenient = op_leniency == "lenient"
        self.lvalues["attributes"] = {"_a_common_": self.dummy}
        lmetadata = self.cls(**self.lvalues)
        rvalues = deepcopy(self.lvalues)
        rvalues["attributes"]["_b_extra_"] = mock.sentinel.extra
        rmetadata = self.cls(**rvalues)

        if not is_lenient:
            # In this case, attributes returns a "difference dictionary"
            diffentry = tuple([{}, {"_b_extra_": mock.sentinel.extra}])
            lexpected = self.none._asdict()
            lexpected["attributes"] = diffentry
            rexpected = lexpected.copy()
            rexpected["attributes"] = diffentry[::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            if is_lenient:
                # It recognises no difference
                assert lmetadata.difference(rmetadata) is None
                assert rmetadata.difference(lmetadata) is None
            else:
                # As calculated above
                assert lmetadata.difference(rmetadata)._asdict() == lexpected
                assert rmetadata.difference(lmetadata)._asdict() == rexpected

    def test_op_different__attribute_value(self, op_leniency):
        # lhs and rhs have different values for an attribute, both strict + lenient.
        is_lenient = op_leniency == "lenient"
        self.lvalues["attributes"] = {
            "_a_common_": self.dummy,
            "_b_extra_": mock.sentinel.value1,
        }
        lmetadata = self.cls(**self.lvalues)
        self.rvalues["attributes"] = {
            "_a_common_": self.dummy,
            "_b_extra_": mock.sentinel.value2,
        }
        rmetadata = self.cls(**self.rvalues)

        # In this case, attributes returns a "difference dictionary"
        diffentry = tuple(
            [
                {"_b_extra_": mock.sentinel.value1},
                {"_b_extra_": mock.sentinel.value2},
            ]
        )
        lexpected = self.none._asdict()
        lexpected["attributes"] = diffentry
        rexpected = lexpected.copy()
        rexpected["attributes"] = diffentry[::-1]

        with mock.patch("iris.common.metadata._LENIENT", return_value=is_lenient):
            # As calculated above -- same for both strict + lenient
            assert lmetadata.difference(rmetadata)._asdict() == lexpected
            assert rmetadata.difference(lmetadata)._asdict() == rexpected


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
