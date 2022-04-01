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

from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD, load_mesh

from . import BENCHMARK_DATA, REUSE_DATA, load_realised, run_function_elsewhere


def _create_file__xios_common(func_name, **kwargs):
    def _external(func_name_, temp_file_dir, **kwargs_):
        from iris.tests.stock import netcdf

        func = getattr(netcdf, func_name_)
        print(func(temp_file_dir, **kwargs_), end="")

    args_hash = hash(str(kwargs))
    save_path = (BENCHMARK_DATA / f"{func_name}_{args_hash}").with_suffix(
        ".nc"
    )
    if not REUSE_DATA or not save_path.is_file():
        # The xios functions take control of save location so need to move to
        #  a more specific name that allows re-use.
        actual_path = run_function_elsewhere(
            _external,
            func_name_=func_name,
            temp_file_dir=str(BENCHMARK_DATA),
            **kwargs,
        )
        Path(actual_path.decode()).replace(save_path)
    return save_path


def create_file__xios_2d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1
):
    """
    Wrapper for :meth:`iris.tests.stock.netcdf.create_file__xios_2d_face_half_levels`.

    Have taken control of temp_file_dir

    todo: is create_file__xios_2d_face_half_levels still appropriate now we can
     properly save Mesh Cubes?
    """

    return _create_file__xios_common(
        func_name="create_file__xios_2d_face_half_levels",
        dataset_name=dataset_name,
        n_faces=n_faces,
        n_times=n_times,
    )


def create_file__xios_3d_face_half_levels(
    temp_file_dir, dataset_name, n_faces=866, n_times=1, n_levels=38
):
    """
    Wrapper for :meth:`iris.tests.stock.netcdf.create_file__xios_3d_face_half_levels`.

    Have taken control of temp_file_dir

    todo: is create_file__xios_3d_face_half_levels still appropriate now we can
     properly save Mesh Cubes?
    """

    return _create_file__xios_common(
        func_name="create_file__xios_3d_face_half_levels",
        dataset_name=dataset_name,
        n_faces=n_faces,
        n_times=n_times,
        n_levels=n_levels,
    )


def sample_mesh(n_nodes=None, n_faces=None, n_edges=None, lazy_values=False):
    """Wrapper for :meth:iris.tests.stock.mesh.sample_mesh`."""

    def _external(*args, **kwargs):
        from iris.experimental.ugrid import save_mesh
        from iris.tests.stock.mesh import sample_mesh

        save_path_ = kwargs.pop("save_path")
        # Always saving, so laziness is irrelevant. Use lazy to save time.
        kwargs["lazy_values"] = True
        new_mesh = sample_mesh(*args, **kwargs)
        save_mesh(new_mesh, save_path_)

    arg_list = [n_nodes, n_faces, n_edges]
    args_hash = hash(str(arg_list))
    save_path = (BENCHMARK_DATA / f"sample_mesh_{args_hash}").with_suffix(
        ".nc"
    )
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            _external, *arg_list, save_path=str(save_path)
        )
    with PARSE_UGRID_ON_LOAD.context():
        if not lazy_values:
            # Realise everything.
            with load_realised():
                mesh = load_mesh(str(save_path))
        else:
            mesh = load_mesh(str(save_path))
    return mesh


def sample_meshcoord(sample_mesh_kwargs=None, location="face", axis="x"):
    """
    Wrapper for :meth:`iris.tests.stock.mesh.sample_meshcoord`.

    Parameters deviate from the original as cannot pass a
    :class:`iris.experimental.ugrid.Mesh to the separate Python instance - must
    instead generate the Mesh as well.

    MeshCoords cannot be saved to file, so the _external method saves the
    MeshCoord's Mesh, then the original Python instance loads in that Mesh and
    regenerates the MeshCoord from there.
    """

    def _external(sample_mesh_kwargs_, save_path_):
        from iris.experimental.ugrid import save_mesh
        from iris.tests.stock.mesh import sample_mesh, sample_meshcoord

        if sample_mesh_kwargs_:
            input_mesh = sample_mesh(**sample_mesh_kwargs_)
        else:
            input_mesh = None
        # Don't parse the location or axis arguments - only saving the Mesh at
        #  this stage.
        new_meshcoord = sample_meshcoord(mesh=input_mesh)
        save_mesh(new_meshcoord.mesh, save_path_)

    args_hash = hash(str(sample_mesh_kwargs))
    save_path = (
        BENCHMARK_DATA / f"sample_mesh_coord_{args_hash}"
    ).with_suffix(".nc")
    if not REUSE_DATA or not save_path.is_file():
        _ = run_function_elsewhere(
            _external,
            sample_mesh_kwargs_=sample_mesh_kwargs,
            save_path_=str(save_path),
        )
    with PARSE_UGRID_ON_LOAD.context():
        with load_realised():
            source_mesh = load_mesh(str(save_path))
    # Regenerate MeshCoord from its Mesh, which we saved.
    return source_mesh.to_MeshCoord(location=location, axis=axis)
