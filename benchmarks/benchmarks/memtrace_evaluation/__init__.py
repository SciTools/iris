# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Benchmarks to evaluate tracemalloc/rss methods of memory measurement."""

import numpy as np

from .. import TrackAddedMemoryAllocation
from .memory_exercising_task import SampleParallelTask


class MemcheckCommon:
    # Basic controls over the test calculation
    default_params = {
        "measure": "tracemalloc",  # alternate: "rss"
        "runtype": "threads",  # alternate: "processes"
        "ysize": 10000,
        "nx": 2000,
        "nblocks": 6,
        "nworkers": 3,
    }

    def _setup(self, **kwargs):
        params = self.default_params.copy()
        params.update(kwargs)
        measure = params["measure"]
        runtype = params["runtype"]
        ysize = params["ysize"]
        nx = params["nx"]
        nblocks = params["nblocks"]
        nworkers = params["nworkers"]

        ny_task = ysize // nblocks
        use_processes = {"threads": False, "processes": True}[runtype]
        self.task = SampleParallelTask(
            n_blocks=nblocks,
            outerdim=ny_task,
            innerdim=nx,
            n_workers=nworkers,
            use_process_workers=use_processes,
        )
        self.use_tracemalloc = {"tracemalloc": True, "rss": False}[measure]

    def run_time_calc(self):
        # This usage is a bit crap, as we don't really care about the runtype.
        self.task.perform()

    def run_addedmem_calc(self):
        with TrackAddedMemoryAllocation(
            use_tracemalloc=self.use_tracemalloc,
            result_min_mb=0.0,
        ) as tracer:
            self.task.perform()
        return tracer.addedmem_mb()


def memory_units_mib(func):
    func.unit = "Mib"
    return func


class MemcheckRunstyles(MemcheckCommon):
    # only some are parametrised, or it's just too complicated!
    param_names = [
        "measure",
        "runtype",
        "ysize",
    ]
    params = [
        # measure
        ["tracemalloc", "rss"],
        # runtype
        ["threads", "processes"],
        # ysize
        [10000, 40000],
    ]

    def setup(self, measure, runtype, ysize):
        self._setup(measure=measure, runtype=runtype, ysize=ysize)

    def time_calc(self, measure, runtype, ysize):
        self.run_time_calc()

    @memory_units_mib
    def track_addmem_calc(self, measure, runtype, ysize):
        return self.run_addedmem_calc()


class MemcheckBlocksAndWorkers(MemcheckCommon):
    # only some are parametrised, or it's just too complicated!
    param_names = [
        "nblocks",
        "nworkers",
    ]
    params = [
        # nblocks
        [1, 4, 9],
        # nworkers
        [1, 2, 3, 4],
    ]

    def setup(self, nblocks, nworkers):
        self.default_params["ysize"] = 20000
        self._setup(
            nblocks=nblocks,
            nworkers=nworkers,
        )

    def time_calc(self, nblocks, nworkers):
        self.run_time_calc()

    @memory_units_mib
    def track_addmem_calc(self, nblocks, nworkers):
        return self.run_addedmem_calc()


class MemcheckBlocksAndWorkers_processes(MemcheckBlocksAndWorkers):
    def setup(self, nblocks, nworkers):
        self.default_params["runtype"] = "processes"
        super().setup(nblocks, nworkers)


class MemcheckBlocksAndWorkers_Rss(MemcheckBlocksAndWorkers):
    def setup(self, nblocks, nworkers):
        self.default_params["measure"] = "rss"
        super().setup(
            nblocks=nblocks,
            nworkers=nworkers,
        )


class MemcheckTaskRepeats(MemcheckCommon):
    param_names = ["nreps"]
    params = [1, 2, 3, 4]

    def setup(self, nreps):
        self._extra_allocated_mem = []
        self._setup()

    def _test_task(self):
        odd_array = np.zeros([1000, 1000], dtype=np.float32)
        odd_array[1, 1] = 1
        self._extra_allocated_mem.extend(odd_array)
        self.task.perform()

    @memory_units_mib
    def track_mem(self, nreps):
        with TrackAddedMemoryAllocation() as tracer:
            for _ in range(nreps):
                self._test_task()
        return tracer.addedmem_mb()
