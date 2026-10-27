# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""File saving benchmarks."""

from iris import save
from iris.mesh import save_mesh

from .generate_data.ugrid import make_cube_like_2d_cubesphere

#: Target name and save keywords per format. Zarr needs mode="w" because its
#:  saver defaults to mode="w-" and raises FileExistsError on a target that
#:  exists - and ASV calls a timing function many times between setup() calls.
SAVE_TARGETS = {
    "NetCDF": ("tmp.nc", {}),
    "Zarr": ("tmp.zarr", {"mode": "w"}),
}


class Save:
    params = [[50, 600], [False, True], list(SAVE_TARGETS)]
    param_names = ["cubesphere-N", "is_unstructured", "file_format"]

    def setup(self, n_cubesphere, is_unstructured, file_format):
        self.cube = make_cube_like_2d_cubesphere(
            n_cube=n_cubesphere, with_mesh=is_unstructured
        )
        self.save_target, self.save_kwargs = SAVE_TARGETS[file_format]

    def _save_data(self, cube, do_copy=True):
        if do_copy:
            # Copy the cube, to avoid distorting the results by changing it
            # Because we known that older Iris code realises lazy coords
            cube = cube.copy()
        save(cube, self.save_target, **self.save_kwargs)

    def _save_mesh(self, cube):
        # In this case, we are happy that the mesh is *not* modified
        save_mesh(cube.mesh, "mesh.nc")

    def time_save_cube(self, n_cubesphere, is_unstructured, file_format):
        self._save_data(self.cube)

    def time_save_mesh(self, n_cubesphere, is_unstructured, file_format):
        # iris.mesh.save_mesh is defined in iris.fileformats.netcdf.saver and
        #  writes netCDF only, so there is nothing to time for Zarr. A cube
        #  that *has* a mesh still saves it through time_save_cube, with
        #  is_unstructured=True.
        if is_unstructured and file_format == "NetCDF":
            self._save_mesh(self.cube)

    def tracemalloc_save(self, n_cubesphere, is_unstructured, file_format):
        # Don't need to copy the cube here since track_ benchmarks don't
        #  do repeats between self.setup() calls.
        self._save_data(self.cube, do_copy=False)
