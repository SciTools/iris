# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the :class:`iris.experimental.ugrid.Connectivity` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import dask.array as da
import numpy as np

from iris.cube import Cube
from iris.coords import AuxCoord, DimCoord


def array_index_with_missing(array, index_array):
    # Index the first array with the second,
    # EXCEPT putting NaNs where any index is < 0.
    # Works on *either* real or lazy arrays.
    result = da.where(index_array < 0, np.nan, array[index_array])
    return result


def create_meshcoordlike(
    cube, face_x_coordname, node_x_coordname, face_node_inds_coordname
):
    """
    Proof of Concept to create an AuxCoord which mimics intended MeshCoord
    behaviour, for fetching points + bounds.

    It contains lazy points and bounds definitions, which access coord data
    from a cube when computed.
    A real MeshCoord will access the appropriate data, dynamically, from a
    Mesh object.  We simulate this by fetching data from given coords of a cube,
    accessing them dynamically by name.

    """
    # NOTE: the 'cube' arg is used as the analogue of the 'mesh' object
    # -- all data is accessed from it, dynamically.
    # We create an actual AuxCoord object, whose 'points' and 'bounds'
    # defer to accesses on the "mesh" (i.e. cube).
    # Using Dask, in the most straightforward way
    # (i.e. no efficiency considerations, not even chunking)

    class ArrayMimic:
        """
        An array-like object which gets its data from a function.

        Note: the basic access interface is the __getitem__ call.
        The function also receives the getitem 'keys', in case that is useful.

        """

        def __init__(self, access_func, shape, dtype, *axs_args, **axs_kwargs):
            """
            Args:
            * dtype (np.dtype), shape (list of int):
                properties of the "whole" array mimicked.
                Must match those of 'self.[:]'.

            * access_func (()(keys, *axs_args, **axs_kwargs) --> ndarray):
                Function called with indexing keys, returning a concrete array
                result.
                Must have the assumed signature, and return an array of the
                correct shape, like that of 'np.zeros(self.shape)[keys]'.

            """
            # Define sufficient instance properties for this to be recognised
            # as an array-like by Dask.
            self.shape = shape
            self.dtype = dtype
            self.ndim = len(
                shape
            )  # Though this one may look redundant, Dask seems to require it.
            self._axs_func = access_func
            self._axs_args = axs_args
            self._axs_kwargs = axs_kwargs

        def __getitem__(self, keys):
            # Array access : return a section of the notional 'whole' array.
            return self._axs_func(keys, *self._axs_args, **self._axs_kwargs)

    # Define a array accessor function for each input to our calculations.
    # N.B. these function definitions are specific to each call, referencing
    # the supplied cube and coord names arguments (like a partial call).
    def fetch_face_xs(keys):
        # Get lazy points of the face-x coord in this cube.
        lazy_points = cube.coord(face_x_coordname).core_points()
        return lazy_points[keys]

    def fetch_node_xs(keys):
        # Get lazy points of the face-x coord from the cube.
        # NOTE: to fit this into a "normal" cube, the cube needs a 'nodes'
        # dimension, to which this will be attached.
        lazy_points = cube.coord(node_x_coordname).core_points()
        return lazy_points[keys]

    def fetch_face_node_inds(keys):
        # Get lazy indices from the face-nodes-connectivity coord in the cube.
        # NOTE: to fit the indices into a "normal" cube, we store them as the
        # *bounds* of a face-nodes-connectivity coord, which is mapped to the
        # 'faces' dimension.
        lazy_indices = cube.coord(face_node_inds_coordname).core_bounds()
        return lazy_indices[keys]

    # Work out what the actual shapes and dtypes of our data arrays will be.
    co = cube.coord(face_x_coordname)
    face_x_shape, face_x_dtype = co.shape, co.dtype
    co = cube.coord(node_x_coordname)
    node_x_shape, node_x_dtype = co.shape, co.dtype
    co = cube.coord(face_node_inds_coordname)
    facenodeinds_shape, facenodeinds_dtype = co.core_bounds().shape, co.dtype

    # Do some quick sanity checks.
    assert len(face_x_shape) == 1
    assert len(node_x_shape) == 1
    assert len(facenodeinds_shape) == 2
    assert facenodeinds_shape == (face_x_shape[0], 4)
    assert np.dtype(facenodeinds_dtype).kind == "i"

    # Wrap the basic access operations as deferred-access arrays, using the
    # ArrayMimic (func --> arraylike) and Dask (arraylike --> da.Array).
    face_x_arraylike = ArrayMimic(
        fetch_face_xs, shape=face_x_shape, dtype=face_x_dtype
    )
    lazy_face_x = da.from_array(face_x_arraylike, chunks="auto")
    node_x_arraylike = ArrayMimic(
        fetch_node_xs, shape=node_x_shape, dtype=node_x_dtype
    )
    lazy_node_x = da.from_array(node_x_arraylike, chunks="auto")
    facenodeinds_arraylike = ArrayMimic(
        fetch_face_node_inds,
        dtype=facenodeinds_dtype,
        shape=facenodeinds_shape,
    )
    lazy_facenodeinds = da.from_array(facenodeinds_arraylike, chunks="auto")

    # Construct the appropriate lazy bounds calculation for the MeshCoord.
    # This type of multidimensional indexing is not supported...
    #    "lazy_face_bounds_x = lazy_node_x[lazy_facenodeinds]"
    # ...so we build that calculation by iterating over the final (bounds)
    # index, and stacking those to reconstruct the output.
    # Within that operation, we also apply our 'array_index_with_missing'
    # function, which ensures that any -1 indices yield NaN points.
    lazy_face_bounds_x = da.stack(
        [
            array_index_with_missing(
                lazy_node_x, lazy_facenodeinds[:, i_bound]
            )
            for i_bound in range(4)
        ],
        axis=-1,
    )
    assert lazy_face_bounds_x.shape == facenodeinds_shape

    # Create an AuxCoord wrapping the appropriate calculations.
    result = AuxCoord(
        points=lazy_face_x,
        bounds=lazy_face_bounds_x,
        long_name="mesh_coord_x",
        units=cube.coord(face_x_coordname).units,
    )
    return result


class Test_MeshCoord__dataview(tests.IrisTest):
    def setUp(self):
        face_nodes_array = np.array(
            [
                [0, 2, 1, 3],
                [1, 3, 10, 13],
                [2, 7, 9, 19],
                [
                    3,
                    4,
                    7,
                    -1,
                ],  # This one has a "missing" point (it's a triangle)
                [8, 1, 7, 2],
            ]
        )
        n_faces = face_nodes_array.shape[0]
        n_nodes = int(face_nodes_array.max() + 1)
        face_xs = 500.0 + np.arange(n_faces)
        node_xs = 100.0 + np.arange(n_nodes)
        # Record all these for re-use in tests
        self.n_faces = n_faces
        self.n_nodes = n_nodes
        self.face_xs = face_xs
        self.node_xs = node_xs
        self.face_nodes_array = face_nodes_array

        # Buld a cube with this info stored in various coordinates.
        # Note: the intended cube looks something like this :
        """
        face_nodes_map / (1)           (faces: ff, nodes: nn)
            Dimension coords:
                face_number                    x           -
                node_number                    -           x
            Aux coords
                face_x                         x           -
                node_x                         -           x
                face_node_connectivity         x           -
                  # index stored in *bounds*, which is the right shape
                mesh_coord_x                   x           -
                  # this is the new coord generated
        """
        cube = Cube(np.zeros((n_faces, n_nodes)))
        cube.add_dim_coord(
            DimCoord(np.arange(n_faces), long_name="face_number", units=1), 0
        )
        cube.add_dim_coord(
            DimCoord(np.arange(n_nodes), long_name="node_number", units=1), 1
        )
        # Note: others are all AuxCoords, so we can check for lazy behaviour.
        cube.add_aux_coord(AuxCoord(face_xs, long_name="face_x", units=1), 0)
        cube.add_aux_coord(AuxCoord(node_xs, long_name="node_x", units=1), 1)
        cube.add_aux_coord(
            AuxCoord(
                points=np.zeros(n_faces, dtype=face_nodes_array.dtype),
                bounds=face_nodes_array,
                long_name="face_node_connectivity",
                units=1,
            ),
            0,
        )  # NOTE: here we use the *bounds*, which has the wanted shape

        # Construct the new aux coord.
        mesh_coord = create_meshcoordlike(
            cube=cube,
            face_x_coordname="face_x",
            node_x_coordname="node_x",
            face_node_inds_coordname="face_node_connectivity",
        )
        # Put it also into the cube (though we don't really need it there).
        cube.add_aux_coord(mesh_coord, 0)
        self.mesh_coord = mesh_coord
        self.cube = cube

    def assertArraysNanAllClose(self, arr1, arr2, fill=-999.0):
        # Test 2 arrays for ~equal values, including matching any NaNs.
        wherenans = np.isnan(arr1)
        self.assertArrayAllClose(np.isnan(arr2), wherenans)
        arr1 = np.where(wherenans, fill, arr1)
        arr2 = np.where(wherenans, fill, arr2)
        self.assertArrayAllClose(arr1, arr2)

    def test_points_values(self):
        # Basic points content check.
        mesh_coord = self.mesh_coord
        self.assertTrue(mesh_coord.has_lazy_points())
        # The points are just the face_x-s
        self.assertArrayAllClose(mesh_coord.points, self.face_xs)

    def test_bounds_values(self):
        # Basic bounds content check.
        mesh_coord = self.mesh_coord
        self.assertTrue(mesh_coord.has_lazy_bounds())
        # The bounds are selected node_x-s :  all == node_number + 100.0
        result = mesh_coord.bounds
        expected = np.where(
            self.face_nodes_array < 0, np.nan, 100.0 + self.face_nodes_array
        )
        self.assertArraysNanAllClose(result, expected)

    def test_points_deferred_access(self):
        # Check that MeshCoord.points always fetches from the current "face_x"
        # coord in the cube.
        cube = self.cube
        mesh_coord = self.mesh_coord
        fetch_without_realise = mesh_coord.lazy_points().compute()
        all_points_vals = self.face_xs
        self.assertArrayAllClose(fetch_without_realise, all_points_vals)

        # Replace 'face_x' coord with one having different values, same name
        face_x_coord = self.cube.coord("face_x")
        face_x_coord_2 = face_x_coord.copy()
        all_points_vals_2 = np.array(
            all_points_vals + 1.0, dtype=int
        )  # Change both values and dtype.
        face_x_coord_2.points = all_points_vals_2
        dims = cube.coord_dims(face_x_coord)
        cube.remove_coord(face_x_coord)
        cube.add_aux_coord(face_x_coord_2, dims)

        # Check that new values + different dtype are now produced by the
        # MeshCoord bounds access.
        fetch_without_realise = mesh_coord.lazy_points().compute()
        self.assertArrayAllClose(fetch_without_realise, all_points_vals_2)
        self.assertEqual(fetch_without_realise.dtype, all_points_vals_2.dtype)
        self.assertNotEqual(fetch_without_realise.dtype, all_points_vals.dtype)

    def test_bounds_deferred_access__node_x(self):
        # Show that MeshCoord.points always fetches from the current "node_x"
        # coord in the cube.
        cube = self.cube
        mesh_coord = self.mesh_coord
        fetch_without_realise = mesh_coord.lazy_bounds().compute()
        all_bounds_vals = array_index_with_missing(
            self.node_xs, self.face_nodes_array
        )
        self.assertArraysNanAllClose(fetch_without_realise, all_bounds_vals)

        # Replace 'node_x' coord with one having different values, same name
        face_x_coord = self.cube.coord("node_x")
        face_x_coord_2 = face_x_coord.copy()
        all_face_points = face_x_coord.points
        all_face_points_2 = np.array(
            all_face_points + 1.0
        )  # Change the values.
        self.assertFalse(np.allclose(all_face_points_2, all_face_points))
        face_x_coord_2.points = all_face_points_2
        dims = cube.coord_dims(face_x_coord)
        cube.remove_coord(face_x_coord)
        cube.add_aux_coord(face_x_coord_2, dims)

        # Check that new, different values are now delivered by the MeshCoord.
        expected_new_values = all_bounds_vals + 1.0
        fetch_without_realise = mesh_coord.lazy_bounds().compute()
        self.assertArraysNanAllClose(
            fetch_without_realise, expected_new_values
        )

    def test_bounds_deferred_access__facenodes(self):
        # Show that MeshCoord.points always fetches from the current
        # "face_node_connectivity" coord in the cube.
        cube = self.cube
        mesh_coord = self.mesh_coord
        fetch_without_realise = mesh_coord.lazy_bounds().compute()
        all_bounds_vals = array_index_with_missing(
            self.node_xs, self.face_nodes_array
        )
        self.assertArraysNanAllClose(fetch_without_realise, all_bounds_vals)

        # Replace the index coord with one having different values, same name
        face_nodes_coord = self.cube.coord("face_node_connectivity")
        face_nodes_coord_2 = face_nodes_coord.copy()
        conns = face_nodes_coord.bounds
        conns_2 = np.array(conns % 10)  # Change some values
        self.assertFalse(np.allclose(conns, conns_2))
        face_nodes_coord_2.bounds = conns_2
        dims = cube.coord_dims(face_nodes_coord)
        cube.remove_coord(face_nodes_coord)
        cube.add_aux_coord(face_nodes_coord_2, dims)

        # Check that new + different values are now delivered by the MeshCoord.
        expected_new_values = self.node_xs[conns_2]
        fetch_without_realise = mesh_coord.lazy_bounds().compute()
        self.assertArrayAllClose(fetch_without_realise, expected_new_values)

    def test_meshcoord_leaves_originals_lazy(self):
        cube = self.cube

        # Ensure all the source coords are lazy.
        source_coords = ("face_x", "node_x", "face_node_connectivity")
        for name in source_coords:
            co = cube.coord(name)
            co.points = co.lazy_points()
            co.bounds = co.lazy_bounds()

        # Check all the source coords are lazy.
        for name in source_coords:
            co = cube.coord(name)
            co.points = co.lazy_points()
            self.assertTrue(co.has_lazy_points())
            if co.has_bounds():
                self.assertTrue(co.has_lazy_bounds())

        # Calculate both points + bounds of the meshcoord
        mesh_coord = self.mesh_coord
        self.assertTrue(mesh_coord.has_lazy_points())
        self.assertTrue(mesh_coord.has_lazy_bounds())
        mesh_coord.points
        mesh_coord.bounds
        self.assertFalse(mesh_coord.has_lazy_points())
        self.assertFalse(mesh_coord.has_lazy_bounds())

        # Check all the source coords are still lazy.
        for name in source_coords:
            co = cube.coord(name)
            co.points = co.lazy_points()
            self.assertTrue(co.has_lazy_points())
            if co.has_bounds():
                self.assertTrue(co.has_lazy_bounds())

    @staticmethod
    def access_wrapped_array(array, chunks="auto"):
        # Wrap an array with a getitem wrapper that records all accesses.
        class ArrayWrapper:
            def __init__(self, array):
                self._array = array
                # Define enough properties to satisfy da.from_array
                self.shape = array.shape
                self.dtype = array.dtype
                self.ndim = array.ndim
                self.accesses = []

            def __getitem__(self, keys):
                self.accesses.append(keys)
                return self._array.__getitem__(keys)

        wrapper = ArrayWrapper(array)
        lazy_array = da.from_array(
            wrapper, chunks=chunks, meta=type(array)
        )  # NB this avoids an initial 0-length access.
        return lazy_array, wrapper

    def test_partial_access_points(self):
        # TODO: this does not achieve what we hoped for, i.e.
        # " Fetching part of MeshCoord.points uses only part of face_x. "
        cube = self.cube
        co_face_x = cube.coord("face_x")
        lazy_wrapped, wrapper = self.access_wrapped_array(co_face_x.points)
        co_face_x.points = lazy_wrapped

        mesh_coord = self.mesh_coord
        self.assertEqual(wrapper.accesses, [])
        mesh_coord.points[1:2]
        # NOTE: this does *not* work as desired, at present..
        #   self.assertEqual(wrapper.accesses, [(slice(1,2),)])
        # ..instead it fetches the whole thing :
        self.assertEqual(wrapper.accesses, [(slice(0, self.n_faces),)])

    def test_partial_access_bounds__nodex(self):
        # TODO:
        # Fetching part of MeshCoord.bounds uses only part of node_x.
        pass

    def test_partial_access_bounds__facenodes(self):
        # TODO:
        # Fetching part of MeshCoord.bounds fetches only part of
        # face_node_connectivity.
        pass


if __name__ == "__main__":
    tests.run()
