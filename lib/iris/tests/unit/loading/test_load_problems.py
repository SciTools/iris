# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :class:`iris.loading.LoadProblems` class."""

from traceback import TracebackException

import pytest

from iris.coords import DimCoord
from iris.cube import Cube
from iris.loading import LoadProblems
from iris.warnings import IrisLoadWarning


@pytest.fixture
def error():
    return ValueError("Example ValueError")


@pytest.fixture
def stack_trace(error):
    try:
        raise error
    except ValueError as raised_error:
        return TracebackException.from_exception(raised_error)


@pytest.fixture(
    params=[dict, Cube, DimCoord, None],
    ids=["loaded_dict", "loaded_Cube", "loaded_DimCoord", "loaded_None"],
)
def loaded_object(request):
    lookup = {
        dict: {"long_name": "foo"},
        Cube: Cube([1.0], long_name="foo"),
        DimCoord: DimCoord([1.0], long_name="foo"),
        None: None,
    }
    return lookup[request.param]


@pytest.fixture
def first_filename():
    return "test.nc"


@pytest.fixture
def destination():
    return LoadProblems.Problem.Destination(Cube, "foo")


@pytest.fixture
def problem_instance(first_filename, loaded_object, stack_trace, destination):
    return LoadProblems.Problem(
        filename=first_filename,
        loaded=loaded_object,
        stack_trace=stack_trace,
        destination=destination,
    )


@pytest.fixture
def load_problems_instance(problem_instance):
    problem2 = LoadProblems.Problem(
        filename="test2.nc",
        loaded=problem_instance.loaded,
        stack_trace=problem_instance.stack_trace,
        destination=problem_instance.destination,
    )
    problem3 = LoadProblems.Problem(
        filename=problem_instance.filename,
        loaded=None,
        stack_trace=problem_instance.stack_trace,
        destination=problem_instance.destination,
    )
    result = LoadProblems()
    result._problems = [problem_instance, problem2, problem3]
    return result


def test_problem_str(problem_instance):
    if isinstance(problem_instance.loaded, (Cube, DimCoord)):
        expected_loaded = problem_instance.loaded.summary(shorten=True)
    else:
        expected_loaded = str(problem_instance.loaded)

    expected = (
        f'{problem_instance.filename}: "{problem_instance.stack_trace}", '
        f"{expected_loaded}"
    )
    assert str(problem_instance) == expected


def test_load_problems_str(load_problems_instance):
    expected_lines = [
        f"{repr(load_problems_instance)}:",
        *[f"  {problem}" for problem in load_problems_instance.problems],
    ]
    expected = "\n".join(expected_lines)
    assert str(load_problems_instance) == expected


def test_problems_property(load_problems_instance):
    assert load_problems_instance.problems == load_problems_instance._problems


def test_problems_by_file_property(load_problems_instance):
    filenames = [p.filename for p in load_problems_instance._problems]
    expected = dict.fromkeys(filenames)
    for filename in filenames:
        expected[filename] = [
            p for p in load_problems_instance._problems if p.filename == filename
        ]
    assert load_problems_instance.problems_by_file == expected


@pytest.mark.parametrize("handled", [True, False], ids=["handled", "not_handled"])
def test_record(
    load_problems_instance, loaded_object, error, stack_trace, destination, handled
):
    def check_equality(problem: LoadProblems.Problem, expected: LoadProblems.Problem):
        assert problem.filename == expected.filename
        assert problem.loaded == expected.loaded
        assert str(problem.stack_trace) == str(expected.stack_trace)
        assert problem.destination is expected.destination
        assert problem.handled == expected.handled

    file_names = ["test3.nc", "test4.nc"]

    expected_additions = [
        LoadProblems.Problem(
            filename=filename,
            loaded=loaded_object,
            stack_trace=stack_trace,
            destination=destination,
            handled=handled,
        )
        for filename in file_names
    ]
    expected_problems = load_problems_instance._problems + expected_additions

    for ix, filename in enumerate(file_names):
        result = load_problems_instance.record(
            filename=filename,
            loaded=loaded_object,
            exception=error,
            destination=destination,
            handled=handled,
        )
        check_equality(result, expected_additions[ix])

    for ix, problem in enumerate(load_problems_instance._problems):
        check_equality(problem, expected_problems[ix])


def test_warning(load_problems_instance, loaded_object, error, destination):
    with pytest.warns(
        expected_warning=IrisLoadWarning,
        match="Not all file objects were parsed correctly.",
    ):
        load_problems_instance.record(
            filename="test3.nc",
            loaded=loaded_object,
            exception=error,
            destination=destination,
        )


def test_reset(load_problems_instance):
    assert load_problems_instance._problems != []
    load_problems_instance.reset()
    assert load_problems_instance._problems == []


def test_reset_with_filename(load_problems_instance, first_filename):
    original_problems = [p for p in load_problems_instance._problems]
    load_problems_instance.reset(first_filename)

    for problem in original_problems:
        if problem.filename == first_filename:
            assert problem not in load_problems_instance._problems
        else:
            assert problem in load_problems_instance._problems


def test_global_instance():
    from iris.loading import LOAD_PROBLEMS

    assert isinstance(LOAD_PROBLEMS, LoadProblems)
