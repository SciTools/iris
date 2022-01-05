# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Common code for benchmarks."""
import resource

from generate_data import BENCHMARK_DATA, run_function_elsewhere

from iris import load_cube as iris_loadcube
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD

ARTIFICIAL_DIM_SIZE = int(10e3)  # For all artificial cubes, coords etc.


def disable_repeat_between_setup(benchmark_object):
    """
    Decorator for benchmarks where object persistence would be inappropriate.

    E.g:
        * Benchmarking data realisation
        * Benchmarking Cube coord addition

    Can be applied to benchmark classes/methods/functions.

    https://asv.readthedocs.io/en/stable/benchmarks.html#timing-benchmarks

    """
    # Prevent repeat runs between setup() runs - object(s) will persist after 1st.
    benchmark_object.number = 1
    # Compensate for reduced certainty by increasing number of repeats.
    #  (setup() is run between each repeat).
    #  Minimum 5 repeats, run up to 30 repeats / 20 secs whichever comes first.
    benchmark_object.repeat = (5, 30, 20.0)
    # ASV uses warmup to estimate benchmark time before planning the real run.
    #  Prevent this, since object(s) will persist after first warmup run,
    #  which would give ASV misleading info (warmups ignore ``number``).
    benchmark_object.warmup_time = 0.0

    return benchmark_object


def generate_cube_like_2d_cubesphere(
    n_cube: int, with_mesh: bool, output_path: str
):
    """
    Construct and save to file an LFRIc cubesphere-like cube for a given
    cubesphere size, *or* a simpler structured (UM-like) cube of equivalent
    size.

    NOTE: this function is *NEVER* called from within this actual package.
    Instead, it is to be called via benchmarks.remote_data_generation,
    so that it can use up-to-date facilities, independent of the ASV controlled
    environment which contains the "Iris commit under test".
    This means:
      * it must be completely self-contained : i.e. it includes all its
        own imports, and saves results to an output file.

    """
    from iris import save
    from iris.tests.stock.mesh import sample_mesh, sample_mesh_cube

    n_face_nodes = n_cube * n_cube
    n_faces = 6 * n_face_nodes

    # Set n_nodes=n_faces and n_edges=2*n_faces
    # : Not exact, but similar to a 'real' cubesphere.
    n_nodes = n_faces
    n_edges = 2 * n_faces
    if with_mesh:
        mesh = sample_mesh(
            n_nodes=n_nodes, n_faces=n_faces, n_edges=n_edges, lazy_values=True
        )
        cube = sample_mesh_cube(mesh=mesh, n_z=1)
    else:
        cube = sample_mesh_cube(nomesh_faces=n_faces, n_z=1)

    # Strip off the 'extra' aux-coord mapping the mesh, which sample-cube adds
    # but which we don't want.
    cube.remove_coord("mesh_face_aux")

    # Save the result to a named file.
    save(cube, output_path)


def make_cube_like_2d_cubesphere(n_cube: int, with_mesh: bool):
    """
    Generate an LFRIc cubesphere-like cube for a given cubesphere size,
    *or* a simpler structured (UM-like) cube of equivalent size.

    All the cube data, coords and mesh content are LAZY, and produced without
    allocating large real arrays (to allow peak-memory testing).

    NOTE: the actual cube generation is done in a stable Iris environment via
    benchmarks.remote_data_generation, so it is all channeled via cached netcdf
    files in our common testdata directory.

    """
    identifying_filename = (
        f"cube_like_2d_cubesphere_C{n_cube}_Mesh={with_mesh}.nc"
    )
    filepath = BENCHMARK_DATA / identifying_filename
    if not filepath.exists():
        # Create the required testfile, by running the generation code remotely
        # in a 'fixed' python environment.
        run_function_elsewhere(
            generate_cube_like_2d_cubesphere,
            n_cube,
            with_mesh=with_mesh,
            output_path=filepath,
        )

    # File now *should* definitely exist: content is simply the desired cube.
    with PARSE_UGRID_ON_LOAD.context():
        cube = iris_loadcube(filepath)
    return cube


class TrackAddedMemoryAllocation:
    """
    Context manager which measures by how much process resident memory grew,
    during execution of its enclosed code block.

    Obviously limited as to what it actually measures : Relies on the current
    process not having significant unused (de-allocated) memory when the
    tested codeblock runs, and only reliable when the code allocates a
    significant amount of new memory.

    Example:
        with TrackAddedMemoryAllocation() as mb:
            initial_call()
            other_call()
        result = mb.addedmem_mb()

    """

    @staticmethod
    def process_resident_memory_mb():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    def __enter__(self):
        self.mb_before = self.process_resident_memory_mb()
        return self

    def __exit__(self, *_):
        self.mb_after = self.process_resident_memory_mb()

    def addedmem_mb(self):
        """Return measured memory growth, in Mb."""
        return self.mb_after - self.mb_before
