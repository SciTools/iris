# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

import re
import tracemalloc

from asv_runner.benchmarks.time import TimeBenchmark


# TODO: docstrings and naming.


class FakeTimer:
    def __init__(self, func: callable):
        self.func = func

    def timeit(self, number: int):
        tracemalloc.start()
        self.func()
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak_mem_bytes


class TracemallocBenchmark(TimeBenchmark):
    # Inheriting from TimeBenchmark, and 'impersonating' the timeit timer,
    #  provides the same repetition functionality as time benchmarks. A
    #  developer wanting more accuracy for their tracemalloc benchmark could
    #  increase the `repeat` number via an attribute, as documented in ASV.
    #  https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks
    # TODO: confirm that this can both detect regressions and dial-out noise.

    name_regex = re.compile("^(Tracemalloc[A-Z_].+)|(tracemalloc_.+)$")

    def __init__(self, name, func, attr_sources):
        super().__init__(name, func, attr_sources)
        self.type = "tracemalloc"
        self.unit = "bytes"
        # TODO: warnings or assertions that detect if number or warmup_time is
        #  ever set to anything else - the developer needs to know this will
        #  not have an effect.
        self.number = 1
        self.warmup_time = 0

    def _get_timer(self, *param):
        if param:

            def func():
                self.func(*param)

        else:
            func = self.func
        return FakeTimer(func=func)


# https://asv.readthedocs.io/projects/asv-runner/en/latest/development/benchmark_plugins.html
export_as_benchmark = [TracemallocBenchmark]
