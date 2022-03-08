# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Common code for benchmarks."""
from functools import wraps
from os import environ
import resource

ARTIFICIAL_DIM_SIZE = int(10e3)  # For all artificial cubes, coords etc.


def disable_repeat_between_setup(benchmark_object):
    """
    Decorator for benchmarks where object persistence would be inappropriate.

    E.g:
        * Benchmarking data realisation
        * Benchmarking Cube coord addition

    Can be applied to benchmark classes/methods/functions.

    https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks

    """
    # Prevent repeat runs between setup() runs - object(s) will persist after 1st.
    benchmark_object.number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    benchmark_object.repeat = (5, 30, 20.0)
    # ASV uses warmup to estimate benchmark time before planning the real run.
    #  Prevent this, since object(s) will persist after first warmup run,
    #  which would give ASV misleading info (warmups ignore ``number``).
    benchmark_object.warmup_time = 0.0

    return benchmark_object


class TrackAddedMemoryAllocation:
    """
    Context manager which measures by how much process resident memory grew,
    during execution of its enclosed code block.

    Obviously limited as to what it actually measures : Relies on the current
    process not having significant unused (de-allocated) memory when the
    tested codeblock runs, and only reliable when the code allocates a
    significant amount of new memory.

    Example:
        with TrackAddedMemoryAllocation() as mb:
            initial_call()
            other_call()
        result = mb.addedmem_mb()

    """

    @staticmethod
    def process_resident_memory_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def __enter__(self):
        self.mb_before = self.process_resident_memory_mb()
        return self

    def __exit__(self, *_):
        self.mb_after = self.process_resident_memory_mb()

    def addedmem_mb(self):
        """Return measured memory growth, in Mb."""
        return self.mb_after - self.mb_before

    @staticmethod
    def decorator(changed_params: list = None):
        """
        Decorates this benchmark to track growth in resident memory during execution.

        Intended for use on ASV ``track_`` benchmarks. Applies the
        :class:`TrackAddedMemoryAllocation` context manager to the benchmark
        code, sets the benchmark ``unit`` attribute to ``Mb``. Optionally
        replaces the benchmark ``params`` attribute with ``changed_params`` -
        useful to avoid testing very small memory volumes, where the results
        are vulnerable to noise.

        Parameters
        ----------
        changed_params : list
            Replace the benchmark's ``params`` attribute with this list.

        """
        if changed_params:
            # Must make a copy for re-use safety!
            _changed_params = list(changed_params)
        else:
            _changed_params = None

        def _inner_decorator(decorated_func):
            @wraps(decorated_func)
            def _inner_func(*args, **kwargs):
                assert decorated_func.__name__[:6] == "track_"
                # Run the decorated benchmark within the added memory context manager.
                with TrackAddedMemoryAllocation() as mb:
                    decorated_func(*args, **kwargs)
                return mb.addedmem_mb()

            if _changed_params:
                # Replace the params if replacement provided.
                _inner_func.params = _changed_params
            _inner_func.unit = "Mb"
            return _inner_func

        return _inner_decorator


def on_demand_benchmark(benchmark_object):
    """
    Decorator. Disables these benchmark(s) unless ON_DEMAND_BENCHARKS env var is set.

    For benchmarks that, for whatever reason, should not be run by default.
    E.g:
        * Require a local file
        * Used for scalability analysis instead of commit monitoring.

    Can be applied to benchmark classes/methods/functions.

    """
    if "ON_DEMAND_BENCHMARKS" in environ:
        return benchmark_object
