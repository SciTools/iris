# Valid Submeshes From Indexing a Cube on a Mesh

**Created:** 2026-11-17

Claims marked **[verified]** were checked by running code against this
checkout in the `py313` conda environment. Everything else is read from source.


## 1. Intent

Indexing a cube along its mesh dimension, with
`iris.experimental.mesh_coord_indexing.SETTING` set to `NEW_MESH`, should give
a cube whose `MeshXY` is a valid submesh of the original. Every connectivity on
the original mesh should survive. Every surviving connectivity should point
only at elements of the new mesh.

`NEW_MESH` builds the new mesh with `_MeshIndexSet.as_mesh()`. That calls
`indexed(frozen=True)` on the original mesh's coordinate manager and
connectivity manager.

Only `face_node_connectivity` and `edge_node_connectivity` come through that
path correctly. `_MeshConnectivityManagerBase.indexed` remaps the values of
every connectivity through a node lookup, whatever its `connected` element.
`face_face_connectivity`, `face_edge_connectivity` and `edge_face_connectivity`
therefore come out with wrong values. A face subset also selects no edges, so
the edge coordinates and `edge_node_connectivity` are dropped.


## 2. Goals

- `face_face_connectivity`, `face_edge_connectivity` and
  `edge_face_connectivity` are remapped correctly for any subset.
- A subset keeps the topology dimension of the original mesh.
- Every subset of the same cube gives a mesh with the same set of
  connectivities.
- A connectivity entry that refers to an element outside the submesh is
  masked.
- The `MESH_INDEX_SET` setting gets the same fix. It goes through the same
  `indexed` methods with `frozen=False`.


## 3. Non-goals

- `boundary_node_connectivity`. It is deferred to later work. Its `location`
  is `"boundary"`, which `indexed` does not handle, so it raises a `KeyError`
  today.
- The `volume_*` connectivities. `MeshXY` does not hold them.
- Lifting the existing requirement that node subsets use strictly increasing
  indices.
- Making the rest of Iris work with zero-length mesh dimensions. Section 9
  lists what is tested.


## 4. Anti-goals

- A 2D mesh must not become 1D because a subset contains no complete face.
- A subset must never add elements at the cube's own location. The number of
  elements at that location must equal the length of the cube's mesh
  dimension.
- The `MESH_INDEX_SET` path must not realise lazy data. Its managers raise an
  error when they find realised connectivities or coordinates.


## 5. Approaches

1. **One lookup per element type. Picked.** Build an "original index to new
   index" lookup for each of node, edge and face. Remap each connectivity's
   values through the lookup for its `connected` element type. Mask values
   that point outside the subset. UGRID allows missing values in these
   connectivities.
2. **Drop any connectivity whose `connected` elements are not all selected.**
   Rejected: it discards the connectivities this work is for.
3. **Rebuild the mesh from coordinates with `MeshXY.from_coords`.** Rejected:
   it only produces `*_node` connectivities.

Within approach 1, a subset may contain no complete higher-dimensional
element. An edge subset that does not fully enclose any face is a common
example.

- **Allow zero-length connectivities. Picked.** The face dimension of the
  submesh has length zero and the mesh stays 2D.
- **Drop to 1D only when nothing qualifies.** Rejected: the topology dimension
  of the mesh would depend on which elements were picked.


## 6. Which Elements a Subset Keeps

Call the cube's mesh location the *selected location*. The cube's indexing
keys give the elements at that location. They may be in any order and may
repeat.

Two rules decide the other element types:

- A lower-dimensional element is kept when a selected element refers to it.
- A higher-dimensional element is kept only when every one of its bounding
  elements at the selected location is selected.

| Selected location | Nodes kept | Edges kept | Faces kept |
|---|---|---|---|
| node | the given nodes | edges whose nodes are all selected | faces whose nodes are all selected |
| edge | nodes of the selected edges | the given edges | faces whose edges are all selected |
| face | nodes of the selected faces | edges of the selected faces | the given faces |

The node column is today's behaviour. The other derived cells are new.

Derived elements keep their original relative order and appear once each. This
matches how nodes are derived today, through a boolean membership array over
the original nodes.

### 6.1 When an edge belongs to a face

`face_node_connectivity` and `edge_node_connectivity` define which edges belong
to which faces. `face_edge_connectivity` and `edge_face_connectivity` are
shortcuts that should be derivable from them.

An edge with nodes `(p, q)` belongs to a face with nodes `(a, b, c, d)` if and
only if `{p, q}` equals one of the face's sides: `{a, b}`, `{b, c}`, `{c, d}`
or `{d, a}`. The same holds for faces with any number of nodes. Masked slots
in a ragged `face_node_connectivity` are skipped, and the last unmasked node
joins back to the first.

Nodes alone never make an edge belong to a face. Two selected faces that
share only a corner each with a given edge do not bring in that edge.

Sections 6.2 and 6.3 always use this definition, even when
`face_edge_connectivity` or `edge_face_connectivity` is present.

### 6.2 Finding the edges of selected faces

Keep every edge that belongs to a selected face, by section 6.1.

### 6.3 Finding the faces whose edges are all selected

Keep a face only if every one of its sides is a selected edge, by section 6.1.
A face with a side that matches no edge in the mesh is never kept.

### 6.4 Meshes without some element types

A 2D mesh may have no edges. A submesh of it has no edges either. The set of
connectivities on the submesh is the set on the original, minus
`boundary_node_connectivity`.

A mesh has edges only if it has `edge_node_connectivity`. UGRID requires
`edge_node_connectivity` whenever `face_edge_connectivity` or
`edge_face_connectivity` is present. Iris does not enforce that. If a mesh
breaks the rule, its submesh drops `face_edge_connectivity`,
`edge_face_connectivity` and the edge coordinates. This is the one exception
to keeping the original set of connectivities.


## 7. Remapping Connectivities

For each element type, build a lookup over the original elements. It gives the
element's index in the submesh, or a masked value if the element is not in the
submesh.

For each connectivity:

- Its rows are those of the elements kept at its `location`.
- Its values are remapped through the lookup for its `connected` element type.
- Values that point outside the submesh become masked.
- `start_index` and `location_axis` are preserved, as `indexed` does today.

If the selected indices repeat, the lookup points at one of the copies. Which
copy is not specified.

`boundary_node_connectivity` is left out of the submesh.


## 8. Zero-Length Connectivities

`Connectivity` cannot be created with a zero-length location axis. Its
`_validate_indices` method calls `indices.min()`, which raises
`ValueError: zero-size array to reduction operation minimum which has no
identity` on an empty array **[verified]**. Skip the minimum-index check when
the array is empty.

A zero-length `AuxCoord` can already be created **[verified]**.

An `edge_face_connectivity` whose entries are all masked can already be
created **[verified]**.

`as_mesh()` decides the topology dimension from which connectivities exist.
`has_edges = hasattr(conn_man, "edge_node")` is true for every 2D manager,
even when `edge_node` is `None`. With this change, `as_mesh()` should take the
topology dimension from the original mesh. It should decide whether to pass
edge coordinates by whether `edge_node_connectivity` is present.

A node subset of a 2D mesh will now produce a 2D mesh. Today `as_mesh()`
raises `NotImplementedError` for node subsets. The existing test that pins
that error changes with it.


## 9. Testing

Unit tests, against `indexed` and `_MeshIndexSet`:

- `face_face_connectivity`, `face_edge_connectivity` and
  `edge_face_connectivity` remap correctly for node, edge and face subsets.
- Entries pointing outside the submesh are masked.
- `start_index=1` and transposed connectivities.
- Faces that share only a corner with some edge do not bring in those edges.
- A face with a side that matches no edge is not kept by an edge subset.
- Ragged faces, where `face_node_connectivity` has masked slots.
- An edge subset that encloses no face gives a 2D mesh with zero faces.
- With `frozen=False`, the results stay lazy.
- `Connectivity` accepts a zero-length location axis.

Integration tests, through `NEW_MESH` on a cube:

- The new mesh has the same connectivities as the original, minus
  `boundary_node_connectivity`.
- A cube on a mesh with zero faces saves to NetCDF. The netCDF4 library treats
  a dimension of size 0 as unlimited, so this may fail. If it does, record it
  as follow-up work rather than fixing it here.

`Connectivity.validate_indices()` requires at least 3 unmasked entries per face
for every face-located connectivity, including `face_face_connectivity`. A face
whose neighbours are all outside the submesh fails that check. Boundary faces
of an ordinary mesh can fail it already. That check is out of scope.

A changelog fragment is required (see `changelog/AGENTS.md`).


## 10. Left to the Builder

- How the lookups and derived selections are built, eagerly or lazily, as long
  as the `MESH_INDEX_SET` path stays lazy.
- Whether `indexed` keeps its signature and accepts boolean masks for the edge
  and face arguments, or takes a new argument shape.
- How `_indexed_manager` chooses the manager class, given that the topology
  dimension is kept.
- Test fixtures.


## 11. Calls Made Without the Requester

- A node subset keeps the edges and faces whose nodes are all selected. This
  changes `NEW_MESH` on node data from an error to a working result.
- Edge-face membership always comes from `face_node_connectivity` and
  `edge_node_connectivity`, even when `face_edge_connectivity` is present.
  This keeps one code path. If the two disagree on the original mesh, entries
  of the remapped `face_edge_connectivity` that point at edges the submesh
  did not keep are masked.
- A face with a side that matches no edge is never kept by an edge subset.
- `boundary_node_connectivity` is left out of the submesh instead of raising.
- A mesh with `face_edge_connectivity` or `edge_face_connectivity` but no
  `edge_node_connectivity` is treated as having no edges (section 6.4).
- Repeated indices: the lookup points at an unspecified copy.
- A NetCDF save failure for zero-length dimensions is recorded, not fixed.
