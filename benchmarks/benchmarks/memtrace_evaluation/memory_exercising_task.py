# Provide standard parallel calculations for testing the memory tracing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

"""
the basic operation is to for each worker to construct a (NY, NX) numpy 
random array, of which it calculates and returns the mean(axis=0)
  --> (NX,) result
The results are then collected --> (N_BLOCKS, NX),
and a mean over all calculated --> (NX,)
The final (single-value) result is the *minimum* of that.
"""

# _SHOW_DEBUG = True
_SHOW_DEBUG = False


def debug(msg):
    if _SHOW_DEBUG:
        print(msg)


def subtask_operation(arg):
    i_task, ny, nx = arg
    debug(f"\nRunning #{i_task}({ny}, {nx}) ..")
    data = np.random.uniform(0.0, 1.0, size=(ny, nx))
    sub_result = data.mean(axis=0)
    debug(f"\n.. completed #{i_task}")
    return sub_result


class SampleParallelTask:
    def __init__(
        self,
        n_blocks=5,
        outerdim=1000,
        innerdim=250,
        n_workers=4,
        use_process_workers=False,
    ):
        self.n_blocks = n_blocks
        self.outerdim = outerdim
        self.innerdim = innerdim
        self.n_workers = n_workers
        if use_process_workers:
            self.pool_type = ProcessPoolExecutor
        else:
            self.pool_type = ThreadPoolExecutor
        self._setup_calc()

    def _setup_calc(self):
        self._pool = self.pool_type(self.n_workers)

    def perform(self):
        partial_results = self._pool.map(
            subtask_operation,
            [
                (i_task + 1, self.outerdim, self.innerdim)
                for i_task in range(self.n_blocks)
            ],
        )
        combined = np.stack(list(partial_results))
        result = np.mean(combined, axis=0)
        result = result.min()
        return result


if __name__ == "__main__":
    nb = 12
    nw = 3
    ny, nx = 1000, 200
    dims = (ny, nx)
    use_processes = False
    typ = "process" if use_processes else "thread"
    msg = f"Starting: blocks={nb} workers={nw} size={dims} type={typ}"
    print(msg)
    calc = SampleParallelTask(
        n_blocks=nb,
        outerdim=ny,
        innerdim=nx,
        n_workers=nw,
        use_process_workers=use_processes,
    )
    debug("Created.")
    debug("Run..")
    result = calc.perform()
    debug("\n.. Run DONE.")
    debug(f"result = {result}")
