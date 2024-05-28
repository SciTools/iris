import re
import tracemalloc

from asv_runner.benchmarks.time import TimeBenchmark


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
    name_regex = re.compile("^(Tracemalloc[A-Z_].+)|(tracemalloc_.+)$")

    def __init__(self, name, func, attr_sources):
        super().__init__(name, func, attr_sources)
        self.type = "tracemalloc"
        self.unit = "bytes"
        self.number = 1
        self.warmup_time = 0

    def _get_timer(self, *param):
        if param:

            def func():
                self.func(*param)

        else:
            func = self.func
        return FakeTimer(func=func)


export_as_benchmark = [TracemallocBenchmark]
