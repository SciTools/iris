# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Test function :obj:`iris.util.lift_empty_masks`"""

from dask import array as da
import numpy as np
import pytest

from iris._lazy_data import as_concrete_data
from iris.cube import Cube
from iris.util import has_mask, lift_empty_masks


def make_masked(
    with_mask=True,
    some_masked=False,
    scalar_mask=False,
    dask=False,
    cube=False,
    fill_value=None,
):
    if some_masked:
        mask = [True, False]
    elif scalar_mask:
        mask = np.ma.nomask
    else:
        mask = [False, False]

    if with_mask:
        if dask:
            func = da.ma.masked_array
        else:
            func = np.ma.masked_array
        array = func([1, 2], mask=mask, fill_value=fill_value)
    else:
        if dask:
            func = da.array
        else:
            func = np.array
        array = func([1, 2])

    if cube:
        result = Cube(array)
    else:
        result = array

    return result


class TestInputs:
    @staticmethod
    @lift_empty_masks
    def input_checker(*args, **kwargs):
        def get_has_mask(input_):
            if isinstance(input_, Cube):
                input_ = input_.core_data()
            return has_mask(input_)

        interior_args_have_masks = [get_has_mask(a) for a in args]
        interior_kwargs_have_masks = [get_has_mask(v) for v in kwargs.values()]

        return interior_args_have_masks, interior_kwargs_have_masks

    @pytest.mark.parametrize(
        "with_mask, some_masked, scalar_mask, expected_masked",
        (
            [True, False, True, False],
            [True, False, False, False],
            [True, True, False, True],
            [False, False, False, False],
        ),
        ids=[
            "scalar_false_mask",
            "full_false_mask",
            "full_true_mask",
            "no_mask",
        ],
    )
    @pytest.mark.parametrize("dask", [False, True], ids=["numpy", "dask"])
    @pytest.mark.parametrize(
        "cube", [False, True], ids=["array_in_Cube", "raw_array"]
    )
    @pytest.mark.parametrize(
        "fill_value", [None, 999], ids=["default_fill_value", "alt_fill_value"]
    )
    def test_single_arg(
        self,
        with_mask,
        some_masked,
        scalar_mask,
        dask,
        cube,
        fill_value,
        expected_masked,
    ):
        input_arg = make_masked(
            with_mask, some_masked, scalar_mask, dask, cube
        )
        args_masked = self.input_checker(input_arg)[0]
        assert args_masked == [expected_masked]

    def test_multi_arg(self):
        args = [make_masked(some_masked=b) for b in (False, True)]
        args_masked = self.input_checker(*args)[0]
        assert args_masked == [False, True]

    def test_kwarg(self):
        kwarg = dict(a=make_masked(some_masked=False))
        _, kwargs_masked = self.input_checker(**kwarg)
        assert kwargs_masked == [False]

    def test_multi_kwarg(self):
        kwargs = {
            str(i): make_masked(some_masked=b)
            for i, b in enumerate([False, True])
        }
        _, kwargs_masked = self.input_checker(**kwargs)
        assert kwargs_masked == [False, True]

    def test_args_kwargs(self):
        args = [make_masked(some_masked=b) for b in (False, True)]
        kwargs = {
            str(i): make_masked(some_masked=b)
            for i, b in enumerate([False, True])
        }
        args_masked, kwargs_masked = self.input_checker(*args, **kwargs)
        assert args_masked == [False, True]
        assert kwargs_masked == [False, True]

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_with_without_masks(self, dask):
        arg1 = make_masked(with_mask=True, some_masked=False)
        arg2 = make_masked(with_mask=False)
        arg3 = make_masked(with_mask=True, some_masked=True)
        args_masked = self.input_checker(arg1, arg2, arg3)[0]
        assert args_masked == [False, False, True]

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_mixed_masks(self, dask):
        arg1 = make_masked(some_masked=False, scalar_mask=False)
        arg2 = make_masked(some_masked=False, scalar_mask=True)
        with pytest.warns(match="Inconsistent false mask types"):
            args_masked = self.input_checker(arg1, arg2)[0]
        assert args_masked == [True, True]

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_mixed_fill_values(self, dask):
        arg1 = make_masked(some_masked=False)
        arg2 = make_masked(some_masked=False, fill_value=999)
        with pytest.warns(match="Inconsistent fill_values"):
            args_masked = self.input_checker(arg1, arg2)[0]
        assert args_masked == [True, True]


class TestReturned:
    @staticmethod
    def compare_masks(observed_array, expected_array):
        observed_array = as_concrete_data(observed_array)
        expected_array = as_concrete_data(expected_array)

        observed_mask = getattr(observed_array, "mask", None)
        expected_mask = getattr(expected_array, "mask", None)

        observed_fill_value = getattr(observed_array, "fill_value", None)
        expected_fill_value = getattr(expected_array, "fill_value", None)

        assert observed_fill_value == expected_fill_value
        if expected_mask is None:
            assert observed_mask is None
        else:
            assert np.array_equal(observed_mask, expected_mask)

    @staticmethod
    @lift_empty_masks
    def returner(*args, **kwargs):
        result = [
            *[a + 1 for a in args],
            *[v + 1 for v in kwargs.values()],
            # Double up to demonstrate that symmetry is not required.
            *[a + 1 for a in args],
            *[v + 1 for v in kwargs.values()],
        ]
        return tuple(result)

    @staticmethod
    @lift_empty_masks
    def returner_single(arg):
        return arg + 1

    @staticmethod
    @lift_empty_masks
    def returns_scalar(array):
        return np.max(array)

    @pytest.mark.parametrize(
        "with_mask, some_masked, scalar_mask",
        (
            [True, False, True],
            [True, False, False],
            [True, True, False],
            [False, False, False],
        ),
        ids=[
            "scalar_false_mask",
            "full_false_mask",
            "full_true_mask",
            "no_mask",
        ],
    )
    @pytest.mark.parametrize("dask", [False, True], ids=["numpy", "dask"])
    @pytest.mark.parametrize(
        "cube", [False, True], ids=["array_in_Cube", "raw_array"]
    )
    def test_single_arg(self, with_mask, some_masked, scalar_mask, dask, cube):
        input_arg = make_masked(
            with_mask, some_masked, scalar_mask, dask, cube
        )
        output = self.returner_single(input_arg)

        expected = input_arg
        observed = output
        if cube:
            expected = expected.core_data()
            observed = output.core_data()

        self.compare_masks(observed, expected)

    def test_multi_arg(self):
        args = [make_masked(some_masked=b) for b in (False, True)]
        outputs = self.returner(*args)
        for arg, output in zip(args, outputs):
            self.compare_masks(output, arg)

    def test_kwarg(self):
        kwarg = dict(a=make_masked(some_masked=False))
        output = self.returner(**kwarg)[0]
        self.compare_masks(output, kwarg["a"])

    def test_multi_kwarg(self):
        kwargs = {
            str(i): make_masked(some_masked=b)
            for i, b in enumerate([False, True])
        }
        outputs = self.returner(**kwargs)
        for kwarg, output in zip(list(kwargs.values()), outputs):
            self.compare_masks(output, kwarg)

    def test_args_kwargs(self):
        args = [make_masked(some_masked=b) for b in (False, True)]
        kwargs = {
            str(i): make_masked(some_masked=b)
            for i, b in enumerate([False, True])
        }
        outputs = self.returner(*args, **kwargs)
        expected = [*args, *list(kwargs.values())]
        for output, expected in zip(outputs, expected):
            self.compare_masks(output, expected)

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_without_masks(self, dask):
        """No lifting or re-application takes place."""
        args = [make_masked(with_mask=False)] * 2
        outputs = self.returner(*args)
        for output in outputs:
            assert not has_mask(output)

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_with_without_masks(self, dask):
        """All unmasked outputs get given the same false mask."""
        arg1 = make_masked(with_mask=True, some_masked=False)
        arg2 = make_masked(with_mask=False)
        arg3 = make_masked(with_mask=True, some_masked=True)
        outputs = self.returner(arg1, arg2, arg3)
        self.compare_masks(outputs[0], arg1)
        self.compare_masks(outputs[1], arg1)
        self.compare_masks(outputs[2], arg3)

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_mixed_masks(self, dask):
        arg1 = make_masked(some_masked=False, scalar_mask=False)
        arg2 = make_masked(some_masked=False, scalar_mask=True)
        with pytest.warns(match="Inconsistent false mask types"):
            outputs = self.returner(arg1, arg2)
        self.compare_masks(outputs[0], arg1)
        self.compare_masks(outputs[1], arg2)

    @pytest.mark.parametrize(
        "dask", [False, True], ids=["numpy_array", "dask_array"]
    )
    def test_mixed_fill_values(self, dask):
        arg1 = make_masked(some_masked=False)
        arg2 = make_masked(some_masked=False, fill_value=999)
        with pytest.warns(match="Inconsistent fill_values"):
            outputs = self.returner(arg1, arg2)
        self.compare_masks(outputs[0], arg1)
        self.compare_masks(outputs[1], arg2)

    def test_asymmetrical(self):
        arg = make_masked(some_masked=False)
        outputs = self.returner(arg)
        for output in outputs:
            self.compare_masks(output, arg)

    def test_scalar_not_re_masked(self):
        arg = make_masked(some_masked=False)
        output = self.returns_scalar(arg)
        assert not has_mask(output)


class TestCubesRestored:
    @staticmethod
    @lift_empty_masks
    def cube_eater(*cubes):
        for cube in cubes:
            assert not has_mask(cube.core_data())

    @staticmethod
    @lift_empty_masks
    def cube_errorer(*cubes):
        for cube in cubes:
            assert not has_mask(cube.core_data())
        raise Exception

    def test_single_cube(self):
        cube = make_masked(some_masked=False, cube=True)
        assert has_mask(cube.core_data())
        self.cube_eater(cube)
        assert has_mask(cube.core_data())

    def test_multi_cube(self):
        cubes = [
            make_masked(some_masked=False, cube=True),
            make_masked(some_masked=False, dask=True, cube=True),
        ]
        for cube in cubes:
            assert has_mask(cube.core_data())
        self.cube_eater(*cubes)
        for cube in cubes:
            assert has_mask(cube.core_data())

    def test_multi_cube_error(self):
        """Restored even when an error takes place."""
        cubes = [
            make_masked(some_masked=False, cube=True),
            make_masked(some_masked=False, dask=True, cube=True),
        ]
        for cube in cubes:
            assert has_mask(cube.core_data())
        with pytest.raises(Exception):
            self.cube_errorer(*cubes)
        for cube in cubes:
            assert has_mask(cube.core_data())
