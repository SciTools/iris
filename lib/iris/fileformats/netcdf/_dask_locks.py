# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Module containing code to create locks enabling dask workers to co-operate.

This matter is complicated by needing different solutions for different dask
scheduler types, i.e. local 'threads' scheduler, local 'processes' or
distributed.

In any case, an "iris.fileformats.netcdf.saver.Saver" object contains a
netCDF4.Dataset targeting an output file, and creates a Saver.file_write_lock
object to serialise write-accesses to the file from dask tasks :  All dask-task
file writes go via a "iris.fileformats.netcdf.saver.NetCDFWriteProxy" object,
which also contains a link to the Saver.file_write_lock, and uses it to prevent
workers from fouling each other.

For each chunk written, the NetCDFWriteProxy acquires the common per-file lock;
opens a Dataset on the file; performs a write to the relevant variable; closes
the Dataset and then releases the lock.  This process is obviously very similar
to what the NetCDFDataProxy does for reading lazy chunks.

For a threaded scheduler, the Saver.lock is a simple threading.Lock().  The
workers (threads) execute tasks which contain a NetCDFWriteProxy, as above.
All of those contain the common lock, and this is simply **the same object**
for all workers, since they share an address space.

For a distributed scheduler, the Saver.lock is a `distributed.Lock()` which is
identified with the output filepath.  This is distributed to the workers by
serialising the task function arguments, which will include the
NetCDFWriteProxy.  A worker behaves like a process, though it may execute on a
remote machine.  When a distributed.Lock is deserialised to reconstruct the
worker task, this creates an object that communicates with the scheduler.
These objects behave as a single common lock, as they all have the same string
'identity', so the scheduler implements inter-process communication so that
they can mutually exclude each other.

It is also *conceivable* that multiple processes could write to the same file in
parallel, if the operating system supports it.  However, this also requires
that the libnetcdf C library is built with parallel access option, which is
not common.  With the "ordinary" libnetcdf build, a process which attempts to
open for writing a file which is _already_ open for writing simply raises an
access error.  In any case, Iris netcdf saver will not support this mode of
operation, at present.

We don't currently support a local "processes" type scheduler.  If we did, the
behaviour should be very similar to a distributed scheduler.  It would need to
use some other serialisable shared-lock solution in place of
'distributed.Lock', which requires a distributed scheduler to function.

"""

import threading

import dask.array
import dask.base
import dask.multiprocessing
import dask.threaded


# A dedicated error class, allowing filtering and testing of errors raised here.
class DaskSchedulerTypeError(ValueError):  # noqa: D101
    pass


def dask_scheduler_is_distributed():
    """Return whether a distributed.Client is active."""
    # NOTE: this replicates logic in `dask.base.get_scheduler` : if a distributed client
    # has been created + is still active, then the default scheduler will always be
    # "distributed".
    is_distributed = False
    # NOTE: must still work when 'distributed' is not available.
    try:
        import distributed

        client = distributed.get_client()
        is_distributed = client is not None
    except (ImportError, ValueError):
        pass
    return is_distributed


def get_dask_array_scheduler_type():
    """Work out what type of scheduler an array.compute*() will use.

    Returns one of 'distributed', 'threads' or 'processes'.
    The return value is a valid argument for dask.config.set(scheduler=<type>).
    This cannot distinguish between distributed local and remote clusters --
    both of those simply return 'distributed'.

    Notes
    -----
    This takes account of how dask is *currently* configured.  It will
    be wrong if the config changes before the compute actually occurs.

    """
    if dask_scheduler_is_distributed():
        result = "distributed"
    else:
        # Call 'get_scheduler', which respects the config settings, but pass an array
        # so we default to the default scheduler for that type of object.
        trial_dask_array = dask.array.zeros(1)
        get_function = dask.base.get_scheduler(collections=[trial_dask_array])
        # Detect the ones which we recognise.
        if get_function == dask.threaded.get:
            result = "threads"
        elif get_function == dask.local.get_sync:
            result = "single-threaded"
        elif get_function == dask.multiprocessing.get:
            result = "processes"
        else:
            msg = f"Dask default scheduler for arrays is unrecognised : {get_function}"
            raise DaskSchedulerTypeError(msg)

    return result


def get_worker_lock(identity: str):
    """Return a mutex Lock which can be shared by multiple Dask workers.

    The type of Lock generated depends on the dask scheduler type, which must
    therefore be set up before this is called.

    Parameters
    ----------
    identity : str

    """
    scheduler_type = get_dask_array_scheduler_type()
    if scheduler_type == "distributed":
        from dask.distributed import Lock as DistributedLock

        lock: DistributedLock | threading.Lock = DistributedLock(identity)
    elif scheduler_type in ("threads", "single-threaded"):
        # N.B. the "identity" string is never used in this case, as the same actual
        # lock object is used by all workers.
        lock = threading.Lock()
    else:
        msg = (
            "The configured dask array scheduler type is "
            f'"{scheduler_type}", '
            "which is not supported by the Iris netcdf saver."
        )
        raise DaskSchedulerTypeError(msg)

    # NOTE: not supporting 'processes' scheduler, for now.
    return lock
