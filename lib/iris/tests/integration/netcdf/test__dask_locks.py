# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Unit tests for the :mod:`iris.fileformats.netcdf._dask_locks` package.

Note: these integration tests replace any unit testing of this module, due to its total
dependence on Dask, and even on Dask's implementation details rather than supported
and documented API and behaviour.
So (a) it is essential to check the module's behaviour against actual Dask operation,
and (b) mock-ist testing of the implementation code in isolation would not add anything
of much value.
"""
import dask
import dask.config
import distributed
import pytest

from iris.fileformats.netcdf._dask_locks import (
    DaskSchedulerTypeError,
    dask_scheduler_is_distributed,
    get_dask_array_scheduler_type,
    get_worker_lock,
)


@pytest.fixture(
    params=[
        "UnspecifiedScheduler",
        "ThreadedScheduler",
        "SingleThreadScheduler",
        "ProcessScheduler",
        "DistributedScheduler",
    ]
)
def dask_scheduler(request):
    # Control Dask to enable a specific scheduler type.
    sched_typename = request.param
    if sched_typename == "UnspecifiedScheduler":
        config_name = None
    elif sched_typename == "SingleThreadScheduler":
        config_name = "single-threaded"
    elif sched_typename == "ThreadedScheduler":
        config_name = "threads"
    elif sched_typename == "ProcessScheduler":
        config_name = "processes"
    else:
        assert sched_typename == "DistributedScheduler"
        config_name = "distributed"

    if config_name == "distributed":
        _distributed_client = distributed.Client()

    if config_name is None:
        context = None
    else:
        context = dask.config.set(scheduler=config_name)
        context.__enter__()

    yield sched_typename

    if context:
        context.__exit__(None, None, None)

    if config_name == "distributed":
        _distributed_client.close()


def test_dask_scheduler_is_distributed(dask_scheduler):
    result = dask_scheduler_is_distributed()
    # Should return 'True' only with a distributed scheduler.
    expected = dask_scheduler == "DistributedScheduler"
    assert result == expected


def test_get_dask_array_scheduler_type(dask_scheduler):
    result = get_dask_array_scheduler_type()
    expected = {
        "UnspecifiedScheduler": "threads",
        "ThreadedScheduler": "threads",
        "ProcessScheduler": "processes",
        "SingleThreadScheduler": "single-threaded",
        "DistributedScheduler": "distributed",
    }[dask_scheduler]
    assert result == expected


def test_get_worker_lock(dask_scheduler):
    test_identity = "<dummy-filename>"
    error = None
    try:
        result = get_worker_lock(test_identity)
    except DaskSchedulerTypeError as err:
        error = err
        result = None

    if dask_scheduler == "ProcessScheduler":
        assert result is None
        assert isinstance(error, DaskSchedulerTypeError)
        msg = 'scheduler type is "processes", which is not supported'
        assert msg in error.args[0]
    else:
        assert error is None
        assert result is not None
        if dask_scheduler == "DistributedScheduler":
            assert isinstance(result, distributed.Lock)
            assert result.name == test_identity
        else:
            # low-level object doesn't have a readily available class for isinstance
            assert all(
                hasattr(result, att)
                for att in ("acquire", "release", "locked")
            )
