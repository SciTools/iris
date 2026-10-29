# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Locks that let dask workers co-operate when writing to one target.

A saver writing lazy data hands each chunk to a dask worker, and the workers
must not write over each other. What that takes depends on the dask scheduler,
so this module works out which scheduler is in play and returns a lock to
match. The scheduler must therefore be configured before a lock is asked for.

The "processes" scheduler is refused by :func:`get_worker_lock`, and writing
to one target from genuinely separate processes is not supported.

.. maintainer-notes::

   A dataset implementation asks for one lock per target, identified by its
   path or URL, and gives it to every write handle it builds. The netCDF write
   handle reopens the file, acquires the lock, writes its chunk, closes the
   file and releases the lock, which is the mirror image of what the read path
   does for a lazy chunk. A store whose chunks are separate objects needs no
   such coordination, and asks for no lock.

   For a threaded scheduler the lock is a plain :class:`threading.Lock`. The
   workers are threads sharing one address space, so they all hold the same
   object, and the identity string is never used.

   For a distributed scheduler the lock is a ``distributed.Lock`` named for
   the target. It reaches the workers by being serialised along with the task,
   and every copy carries the same name, so the scheduler can make them
   exclude one another. A worker behaves like a process, and may run on
   another machine.

   Supporting the "processes" scheduler would need a serialisable shared lock
   that does not require a distributed scheduler to function. Writing from
   genuinely separate processes is a further step again: for netCDF it needs
   libnetcdf built with the parallel access option, which is uncommon, since
   with an ordinary build opening a file for writing that is already open for
   writing simply raises an access error.

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
