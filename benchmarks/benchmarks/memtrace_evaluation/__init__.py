from .. import TrackAddedMemoryAllocation
from .memory_exercising_task import SampleParallelTask

class MemcheckOnesize:
    # Basic controls over the test calculation:
    # param_names = ["measure", "runtype", "nyfull", "nx", "nblocks", "nworkers"]
    # The actual params are used differently in different subclasses
    # params = [
    #     # measure
    #     ["tracemalloc", "rss"],
    #     ["threads", "processes"],
    #     [10000, 40000],
    #     [1, 4, 10, 40],
    #     [1, 2, 4, 20],
    # ]

    param_names = ["measure", "runtype", "ysize"]
    params = [
        # measure
        ["tracemalloc", "rss"],
        # runtype
        ["threads", "processes"],
        # ysize
        [10000, 40000],
    ]

    def setup(self, measure, runtype, ysize):
        nx = 250
        nblocks = 6
        nworkers = 4
        nyfull = ysize // nblocks
        use_processes = {
            "threads": False,
            "processes": True
        }[runtype]
        self.task = SampleParallelTask(
            n_blocks=nblocks,
            outerdim=nyfull // nblocks,
            innerdim=nx,
            n_workers=nworkers,
            use_process_workers=use_processes
        )
        self.use_tracemalloc = {
            "tracemalloc": True,
            "rss": False
        }[measure]

    def time_calc(self, measure, runtype, ysize):
        # This usage is a bit crap, as we don't really care about the runtype.
        self.task.perform()

    def track_addedmem_calc(self, measure, runtype, ysize):
        with TrackAddedMemoryAllocation(
                use_tracemalloc=self.use_tracemalloc,
                result_min_mb=0.0,
        ) as tracer:
            self.task.perform()
        return tracer.addedmem_mb()
