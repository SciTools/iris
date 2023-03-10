# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Module containing code to create locks enabling dask workers to co-operate.

This matter is complicated by needing different solutions for different dask scheduler
types, i.e. local 'threads' scheduler, local 'processes' or distributed.

"""
import threading


def get_worker_lock(identity: str):
    """
    Return a mutex Lock which can be shared amongst Dask workers.

    The type of Lock generated depends on the dask scheduler type, which must therefore
    be set up before this is called.

    """
    return threading.Lock()
