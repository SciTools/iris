import re

from asv_runner.benchmarks.time import TimeBenchmark


class NewBenchmark(TimeBenchmark):
    name_regex = re.compile("^(New[A-Z_].+)|(new_.+)$")


export_as_benchmark = [NewBenchmark]
