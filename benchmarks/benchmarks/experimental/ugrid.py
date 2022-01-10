# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
Benchmark tests for the experimental.ugrid module.

"""

from copy import deepcopy

import numpy as np

from iris.experimental import ugrid

from .. import ARTIFICIAL_DIM_SIZE, disable_repeat_between_setup
from ..generate_data.stock import sample_mesh, sample_meshcoord


class UGridCommon:
    """
    A base class running a generalised suite of benchmarks for any ugrid object.
    Object to be specified in a subclass.

    ASV will run the benchmarks within this class for any subclasses.

    ASV will not benchmark this class as setup() triggers a NotImplementedError.
    (ASV has not yet released ABC/abstractmethod support - asv#838).

    """

    params = [
        6,  # minimal cube-sphere
        int(1e6),  # realistic cube-sphere size
    ]
    param_names = ["number of faces"]

    def setup(self, *params):
        self.object = self.create()

    def create(self):
        raise NotImplementedError

    def time_create(self, *params):
        """Create an instance of the benchmarked object. create() method is
        specified in the subclass."""
        self.create()

    def time_return(self, *params):
        """Return an instance of the benchmarked object."""
        _ = self.object


class Connectivity(UGridCommon):
    def setup(self, n_faces):
        self.array = np.zeros([n_faces, 3], dtype=np.int)
        super().setup(n_faces)

    def create(self):
        return ugrid.Connectivity(
            indices=self.array, cf_role="face_node_connectivity"
        )

    def time_indices(self, n_faces):
        _ = self.object.indices

    def time_src_lengths(self, n_faces):
        _ = self.object.src_lengths()

    def time_validate_indices(self, n_faces):
        self.object.validate_indices()


@disable_repeat_between_setup
class ConnectivityLazy(Connectivity):
    """Lazy equivalent of :class:`Connectivity`."""

    def setup(self, n_faces):
        super().setup(n_faces)
        self.array = self.object.lazy_indices()
        self.object = self.create()


class Mesh(UGridCommon):
    def setup(self, n_faces, lazy=False):
        n_nodes = n_faces + 2
        n_edges = n_faces * 2
        self.mesh_kwargs = dict(
            n_nodes=n_nodes, n_faces=n_faces, n_edges=n_edges, lazy_values=lazy
        )

        super().setup(n_faces)

        self.face_node = self.object.face_node_connectivity
        self.node_x = self.object.node_coords.node_x
        # Kwargs for reuse in search and remove methods.
        self.connectivities_kwarg = dict(cf_role="edge_node_connectivity")
        self.coords_kwarg = dict(include_faces=True)

        # TODO: an opportunity for speeding up runtime if needed, since
        #  eq_object is not needed for all benchmarks. Just don't generate it
        #  within a benchmark - the execution time is large enough that it
        #  could be a significant portion of the benchmark - makes regressions
        #  smaller and could even pick up regressions in copying instead!
        self.eq_object = deepcopy(self.object)

    def create(self):
        return sample_mesh(**self.mesh_kwargs)

    def time_add_connectivities(self, n_faces):
        self.object.add_connectivities(self.face_node)

    def time_add_coords(self, n_faces):
        self.object.add_coords(node_x=self.node_x)

    def time_connectivities(self, n_faces):
        _ = self.object.connectivities(**self.connectivities_kwarg)

    def time_coords(self, n_faces):
        _ = self.object.coords(**self.coords_kwarg)

    def time_eq(self, n_faces):
        _ = self.object == self.eq_object

    def time_remove_connectivities(self, n_faces):
        self.object.remove_connectivities(**self.connectivities_kwarg)

    def time_remove_coords(self, n_faces):
        self.object.remove_coords(**self.coords_kwarg)


@disable_repeat_between_setup
class MeshLazy(Mesh):
    """Lazy equivalent of :class:`Mesh`."""

    def setup(self, n_faces, lazy=True):
        super().setup(n_faces, lazy=lazy)


class MeshCoord(UGridCommon):
    # Add extra parameter value to match AuxCoord benchmarking.
    params = UGridCommon.params + [ARTIFICIAL_DIM_SIZE]

    def setup(self, n_faces, lazy=False):
        self.mesh_kwargs = dict(
            n_nodes=n_faces + 2,
            n_edges=n_faces * 2,
            n_faces=n_faces,
            lazy_values=lazy,
        )

        super().setup(n_faces)

    def create(self):
        return sample_meshcoord(sample_mesh_kwargs=self.mesh_kwargs)

    def time_points(self, n_faces):
        _ = self.object.points

    def time_bounds(self, n_faces):
        _ = self.object.bounds


@disable_repeat_between_setup
class MeshCoordLazy(MeshCoord):
    """Lazy equivalent of :class:`MeshCoord`."""

    def setup(self, n_faces, lazy=True):
        super().setup(n_faces, lazy=lazy)
