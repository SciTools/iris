# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Benchmarks for the SPerf scheme of the UK Met Office's NG-VAT project.

SPerf = assessing performance against a series of increasingly large LFRic
datasets.
"""
from iris import load_cube

# TODO: remove uses of PARSE_UGRID_ON_LOAD once UGRID parsing is core behaviour.
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

from .. import TrackAddedMemoryAllocation, on_demand_benchmark
from ..generate_data.ugrid import make_cubesphere_testfile


class FileMixin:
    """For use in any benchmark classes that work on a file."""

    params = [
        [12, 384, 640, 960, 1280, 1668],
        [1, 36, 72],
        [1, 3, 36, 72],
    ]
    param_names = ["cubesphere_C<N>", "N levels", "N time steps"]
    # cubesphere_C<N>: notation refers to faces per panel.
    #  e.g. C1 is 6 faces, 8 nodes

    def setup(self, c_size, n_levels, n_times):
        self.file_path = make_cubesphere_testfile(
            c_size=c_size, n_levels=n_levels, n_times=n_times
        )

    def load_cube(self):
        with PARSE_UGRID_ON_LOAD.context():
            return load_cube(str(self.file_path))


@on_demand_benchmark
class CubeComparison(FileMixin):
    """
    Benchmark time and memory costs of comparing :class:`~iris.cube.Cube`\\ s
     with attached :class:`~iris.experimental.ugrid.mesh.Mesh`\\ es.

    Uses :class:`FileMixin` as the realistic case will be comparing
    :class:`~iris.cube.Cube`\\ s that have been loaded from file.

    """

    # Cut down paremt parameters.
    params = [FileMixin.params[0]]

    def setup(self, c_size, n_levels=1, n_times=1):
        super().setup(c_size, n_levels, n_times)
        self.cube = self.load_cube()
        self.other_cube = self.load_cube()

    def peakmem_eq(self, n_cube):
        _ = self.cube == self.other_cube

    def time_eq(self, n_cube):
        _ = self.cube == self.other_cube
