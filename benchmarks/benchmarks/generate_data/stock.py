# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Wrappers for using :mod:`iris.tests.stock` methods for benchmarking.

See :mod:`benchmarks.generate_data` for an explanation of this structure.
"""

from pathlib import Path
import pickle

from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD, load_mesh

from . import BENCHMARK_DATA, REUSE_DATA, run_function_elsewhere


def create_file__xios_2d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1
):
    """
    Wrapper for :meth:`iris.tests.stock.netcdf.create_file__xios_2d_face_half_levels`.

    Have taken control of temp_file_dir

    todo: is create_file__xios_2d_face_half_levels still appropriate now we can
     properly save Mesh Cubes?
    """

    def _external(*args, **kwargs):
        from iris.tests.stock.netcdf import (
            create_file__xios_2d_face_half_levels,
        )

        print(create_file__xios_2d_face_half_levels(*args, **kwargs), end="")

    args_list = [dataset_name, n_faces, n_times]
    args_hash = hash(str(args_list))
    save_path = (
        BENCHMARK_DATA / f"create_file__xios_2d_face_half_levels_{args_hash}"
    ).with_suffix(".nc")
    if not REUSE_DATA or not save_path.is_file():
        # create_file__xios_2d_face_half_levels takes control of save location
        #  so need to move to a more specific name that allows re-use.
        actual_path = run_function_elsewhere(
            _external, str(BENCHMARK_DATA), *args_list
        )
        Path(actual_path.decode()).replace(save_path)
    return save_path


def sample_mesh(n_nodes=None, n_faces=None, n_edges=None, lazy_values=False):
    """Wrapper for :meth:iris.tests.stock.mesh.sample_mesh`."""

    def _external(*args, **kwargs):
        from iris.experimental.ugrid import save_mesh
        from iris.tests.stock.mesh import sample_mesh

        save_path_ = kwargs.pop("save_path")
        new_mesh = sample_mesh(*args, **kwargs)
        save_mesh(new_mesh, save_path_)

    arg_list = [n_nodes, n_faces, n_edges, lazy_values]
    args_hash = hash(str(arg_list))
    save_path = (BENCHMARK_DATA / f"sample_mesh_{args_hash}").with_suffix(
        ".nc"
    )
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            _external, *arg_list, save_path=str(save_path)
        )
    with PARSE_UGRID_ON_LOAD.context():
        return load_mesh(str(save_path))


def sample_meshcoord(sample_mesh_kwargs=None, location="face", axis="x"):
    """
    Wrapper for :meth:`iris.tests.stock.mesh.sample_meshcoord`.

    Inputs deviate from the original as cannot pass a
    :class:`iris.experimental.ugrid.Mesh to the separate Python instance - must
    instead generate the Mesh as well.
    """

    def _external(**kwargs):
        from pathlib import Path
        import pickle

        from iris.tests.stock.mesh import sample_mesh, sample_meshcoord

        sample_mesh_kwargs_ = kwargs.pop("sample_mesh_kwargs", None)
        if sample_mesh_kwargs_:
            kwargs["mesh"] = sample_mesh(**sample_mesh_kwargs_)

        pickle_path_ = Path(kwargs.pop("pickle_path"))
        new_meshcoord = sample_meshcoord(**kwargs)
        with pickle_path_.open("wb") as write_file:
            pickle.dump(new_meshcoord, write_file)

    args_hash = hash(str([sample_mesh_kwargs, location, axis]))
    pickle_path = BENCHMARK_DATA / f"sample_meshcoord_f{args_hash}"
    # No file re-use - risky with pickle objects.
    _ = run_function_elsewhere(
        _external,
        sample_mesh_kwargs=sample_mesh_kwargs,
        location=location,
        axis=axis,
        pickle_path=str(pickle_path),
    )
    with pickle_path.open("rb") as read_file:
        return pickle.load(read_file)
