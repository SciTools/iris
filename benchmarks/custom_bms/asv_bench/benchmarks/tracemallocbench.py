# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Benchmark for growth in process resident memory, repeating for accuracy.

Uses a modified version of the repeat logic in
:class:`asv_runner.benchmarks.time.TimeBenchmark`.
"""

import re
import tracemalloc
from typing import Callable

from asv_runner.benchmarks.time import TimeBenchmark


class DuckTimerForMemory:
    """Measures how much process resident memory grows, during execution.

    Provides a `timeit` method duck-typed to mimic :meth:`timeit.Timer.timeit`,
    for use in overriding :class:`asv_runner.benchmarks.time.TimeBenchmark`.
    """

    def __init__(self, func: Callable):
        """Store the func to be called during the benchmark."""
        self.func = func

    def timeit(self, number: int) -> int:
        """Measure process resident memory growth while executing the given ``func``.

        Parameters
        ----------
        number : int
            The number of times to call the function. Combined memory growth
            for ALL the calls is measured.

        Returns
        -------
        int
            How much process resident memory grew, in bytes.
        """
        tracemalloc.start()
        for _ in range(number):
            self.func()
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak_mem_bytes


class TracemallocBenchmark(TimeBenchmark):
    """Benchmark for growth in process resident memory, repeating for accuracy.

    Inherits behaviour from :class:`asv_runner.benchmarks.time.TimeBenchmark`,
    see https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks.

    Attributes
    ----------
    Mostly identical to :class:`asv_runner.benchmarks.time.TimeBenchmark`. See
    https://asv.readthedocs.io/en/stable/writing_benchmarks.html#benchmark-attributes.
    Make sure to use the inherited ``repeat`` attribute if greater accuracy
    is needed. Below are the attributes where inherited behaviour is
    overridden.

    number : int
        The number of times the benchmarked operation will be called per
        ``repeat``. The combined memory growth for all calls is measured. A
        floor of ``1`` is enforced to prevent this benchmark
        inheriting unwanted default behaviour from
        :class:`asv_runner.benchmarks.time.TimeBenchmark`.
    warmup_time : float = 0
        Always set to ``0``, as this feature is not needed for memory
        measurement. ``0`` also prevents this benchmark inheriting unwanted
        default behaviour from
        :class:`asv_runner.benchmarks.time.TimeBenchmark`.
    type : str = "tracemalloc"
        The type of this benchmark. All benchmark operations prefixed with
        ``tracemalloc_`` will use this benchmark class.
    unit : str = "bytes"
        The units of the measured metric (i.e. the growth in memory).

    Notes
    -----
    :class:`asv_runner.benchmarks.time.TimeBenchmark`, behaviour is overridden
    byt replacing :class:`timeit.Timer` instance with a custom implementation -
    :class:`DuckTimerForMemory`.

    """

    # TODO: confirm that this can both detect regressions and dial-out noise.

    name_regex = re.compile("^(Tracemalloc[A-Z_].+)|(tracemalloc_.+)$")

    def __init__(self, name, func, attr_sources):
        super().__init__(name, func, attr_sources)
        self.type = "tracemalloc"
        self.unit = "bytes"

        # No easy way to detect if these settings are being used, or if we're
        #  just getting the default (only way would involve private methods).
        #  Don't want to pollute with warnings, so we rely solely on the
        #  docstring to keep developers informed.
        self.number = max(1, self.number)
        self.warmup_time = 0

    def _get_timer(self, *param):
        """Override parent method to return a :class:`DuckTimerForMemory` instance."""
        if param:

            def func():
                self.func(*param)

        else:
            func = self.func
        return DuckTimerForMemory(func=func)


# https://asv.readthedocs.io/projects/asv-runner/en/latest/development/benchmark_plugins.html
export_as_benchmark = [TracemallocBenchmark]
