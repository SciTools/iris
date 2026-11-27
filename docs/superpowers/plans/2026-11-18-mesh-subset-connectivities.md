# Valid submeshes from indexing a cube on a mesh: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to carry out this plan task by task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Indexing a cube on a mesh gives a submesh in which every
connectivity points at the right elements of the submesh.

**Architecture:** `_MeshIndexSet` works out which nodes, edges and faces the
submesh keeps. Edge and face selections become concrete integer index arrays,
derived from `face_node` sides and `edge_node` pairs. Then
`_MeshConnectivityManagerBase.indexed` builds one lookup per element type. It
remaps each connectivity through the lookup of its `connected` type, so
anything outside the submesh is masked. `as_mesh()` keeps the original
topology dimension.

**Tech Stack:** Python 3.12+, NumPy masked arrays, Dask, pytest.

**Spec:**
[`docs/superpowers/specs/2026-11-17-mesh-subset-connectivities-design.md`](../specs/2026-11-17-mesh-subset-connectivities-design.md)

**Global Constraints:**
- Follow `AGENTS.md`, `lib/iris/AGENTS.md` and `lib/iris/tests/AGENTS.md`.
  Prose follows `.github/WRITING.md`.
- Never name this plan, its tasks or the spec in code, tests or comments.
- Connectivity and coord objects on a `_MeshIndexSet` stay lazy when the
  original mesh's are lazy.
- Edge and face selections are concrete NumPy integer arrays.
  `_MeshConnectivityManagerBase.add` calls `len()` on every connectivity, so
  row counts must be known.
- An edge belongs to a face if and only if its node pair matches one side of
  that face in `face_node`. A side joins consecutive unmasked slots, and the
  last one wraps to the first. Masked slots are assumed to trail. `face_edge`
  and `edge_face` are never used to decide membership.
- `boundary_node_connectivity` is left out of every submesh.
- Test command, from the repo root (written as `PYTEST` below):
  `conda run -n py313 env PYTHONPATH=lib OVERRIDE_TEST_DATA_REPOSITORY=/home/users/stephen.worsley/old-home/Repos/iris-test-data/test_data python -m pytest -q -p no:cacheprovider`
- Pre-commit: `conda run -n py313 pre-commit run --files <paths>`.
- Commit after each task. Don't touch the untracked files already in the
  working tree.

**Review Focus:**
- Repeated indices (`cube[:, [0, 0]]`): no error, two faces, and
  connectivities that point at the repeated element resolve to one of its
  copies.
- A lazy original mesh through `MESH_INDEX_SET`: every connectivity on the
  index set still has lazy indices, and the values match the real case.
- `start_index=1` with masked slots: the remap adds the start index to
  unmasked values only.
- `location_axis=1` (transposed) connectivities come back transposed, with the
  same values.
- Ragged faces (`varied_mesh`): a side running from the last unmasked slot back
  to the first is matched correctly.

---

### Task 1: allow zero-length connectivities

**Files:**
- Modify: `lib/iris/mesh/components.py` (`Connectivity._validate_indices`)
- Test: `lib/iris/tests/unit/mesh/components/test_Connectivity.py`
  (`TestValidations`)

**Interfaces:** produces a `Connectivity` that accepts indices of shape
`(0, n)` (`location_axis=0`) or `(n, 0)` (`location_axis=1`), real or lazy.

- [ ] **Step 1: write the failing tests.** Add these to `TestValidations`:
  - `test_zero_length_real` checks that `Connectivity(np.zeros((0, 4), int), cf_role="face_node_connectivity")` builds, and that `shape == (0, 4)`.
  - `test_zero_length_lazy` does the same with `da.zeros((0, 4), dtype=int)`, and checks that `has_lazy_indices()` is true.
  - `test_zero_length_transposed` uses shape `(2, 0)` with `location_axis=1`.
- [ ] **Step 2: run them.** `PYTEST lib/iris/tests/unit/mesh/components/test_Connectivity.py -k zero_length`.
  Expect a ValueError: "zero-size array to reduction operation minimum which
  has no identity".
- [ ] **Step 3: implement.** In `_validate_indices`, skip the minimum-value
  check when `indices.size == 0`. Leave the other checks as they are.
- [ ] **Step 4: run the whole test file.** Expect it all to pass.
- [ ] **Step 5: commit** with "Allow zero-length mesh connectivities".

---

### Task 2: derive the edge and face selections

**Files:**
- Modify: `lib/iris/mesh/components.py` (new module-level helpers;
  `_MeshIndexSet._calculate_edge_indices` and `_calculate_face_indices`)
- Modify: `lib/iris/tests/stock/mesh.py` (new `sample_strip_mesh`)
- Test: `lib/iris/tests/unit/mesh/components/test_MeshIndexSet.py`
  (`Test_index_calculations`, `Test_unusual_connectivities`)

**Interfaces:**
- `sample_strip_mesh(lazy_values: bool = False, start_index: int = 0, location_axis: int = 0) -> MeshXY`.
  It returns the 3×1 strip below, with all five connectivities and node, edge
  and face coords. `start_index` is added to every unmasked index.
  `location_axis=1` stores each connectivity transposed.
- `_MeshIndexSet._calculate_edge_indices() -> np.ndarray | None` returns 1-D
  integer indices into the original edges, or `None` when the mesh has no
  `edge_node_connectivity`.
- `_MeshIndexSet._calculate_face_indices() -> np.ndarray | None` does the same
  for faces, keyed on `face_node_connectivity`.
- For the selected location, both return the given indices unchanged. For any
  other location, they return the derived elements in their original order,
  once each.

The strip has nodes 0–3 along y=0 and 4–7 along y=1, at x=0–3. Faces A, B and
C sit between them.

```
4---5---6---7
| A | B | C |
0---1---2---3
```

The literal arrays for `sample_strip_mesh` are below. In `face_face` and
`edge_face`, `-` is masked.

| Connectivity | Rows |
|---|---|
| face_node | `[0,1,5,4] [1,2,6,5] [2,3,7,6]` |
| edge_node | `[0,1] [1,2] [2,3] [4,5] [5,6] [6,7] [0,4] [1,5] [2,6] [3,7]` |
| face_edge | `[0,7,3,6] [1,8,4,7] [2,9,5,8]` |
| face_face | `[-,1,-,-] [-,2,-,0] [-,-,-,1]` |
| edge_face | `[0,-] [1,-] [2,-] [0,-] [1,-] [2,-] [0,-] [0,1] [1,2] [2,-]` |

Face coords are the face centres. Edge coords are the edge midpoints.

The new helpers in `components.py` take zero-based, location-major, concrete
values. Build that from any connectivity with
`np.ma.asanyarray(as_concrete_data(c.indices_by_location(c.core_indices()))) - c.start_index`.
- `_edge_keys(edge_nodes, n_nodes)` gives one int64 key per edge:
  `min(p, q) * n_nodes + max(p, q)`.
- `_face_side_keys(face_nodes, n_nodes)` gives a masked
  `(n_faces, n_slots)` int64 array of side keys built the same way. Slot `k`
  pairs with slot `k + 1` when that slot is unmasked, and with slot 0
  otherwise. A key is masked wherever its slot is masked.

Each derivation uses the rule below.

| Selected | Edges kept | Faces kept |
|---|---|---|
| node | `node_mask[edge_nodes].all(axis=1)` | every unmasked slot's node is in `node_mask` |
| edge | the given indices | every unmasked side key is in `edge_keys[indices]` (`np.isin`) |
| face | `np.isin(edge_keys, side_keys[indices].compressed())` | the given indices |

`node_mask` is the concrete result of `_calculate_node_bool_index()`. Turn
masks into indices with `np.flatnonzero`.

- [ ] **Step 1: add `sample_strip_mesh`** to `lib/iris/tests/stock/mesh.py`,
  with a NumPy-style docstring that includes the diagram.
- [ ] **Step 2: write the failing tests.**
  - Add `Test_index_calculations.test_strip_derived_selections`. It is
    parametrised over the cases below, plus `start_index` (0 or 1) and
    `location_axis` (0 or 1). It uses the autouse `lazy_values`.

    | Location, indices | Edge indices | Face indices |
    |---|---|---|
    | face `[0, 2]` | `[0, 2, 3, 5, 6, 7, 8, 9]` | `[0, 2]` |
    | face `[1, 0]` | `[0, 1, 3, 4, 6, 7, 8]` | `[1, 0]` |
    | edge `[0, 3, 6, 7, 9]` | `[0, 3, 6, 7, 9]` | `[0]` |
    | edge `[1, 4]` | `[1, 4]` | `[]` |
    | node `[0, 1, 4, 5]` | `[0, 3, 6, 7]` | `[0]` |

  - Add `Test_unusual_connectivities.test_varied_derived_selections`. It is
    parametrised over `varied_mesh` and `varied_mesh_1_indexed`. Face `[3]`
    gives edges `[0, 1, 2, 3]`. Edge `[0, 1, 2, 3]` gives faces `[]`, because
    the pentagon's closing side (5, 0) has no edge.
  - Add `test_no_edges_gives_none`: `sample_mesh(n_edges=0)` with a face
    subset gives `_calculate_edge_indices() is None`.
  - Add `test_1d_mesh_no_faces`: `mesh_1d` with a node subset gives
    `_calculate_face_indices() is None`.
  - Rewrite `test_calculate_edge_indices`. It uses indices `[1]` over
    `meshes_locs_all`. Edge gives `[1]`, face (on `mesh_2d`) gives `[0]`, and
    node gives `[]`.
  - Rewrite `test_calculate_face_indices`. It uses `mesh_2d` and indices `[1]`.
    Face gives `[1]`, and edge and node both give `[]`.
- [ ] **Step 3: run them.**
  `PYTEST lib/iris/tests/unit/mesh/components/test_MeshIndexSet.py -k "derived or calculate or no_edges or no_faces"`.
  Expect failures where `None` is returned in place of an array.
- [ ] **Step 4: implement** the two helpers and the two `_calculate_*`
  methods.
- [ ] **Step 5: run the whole file.** Expect the new tests to pass. Expect
  `test_connectivity_manager_subset` to still pass, because `indexed()` doesn't
  use the derived selections yet.
- [ ] **Step 6: pre-commit** on the changed files, then **commit** with
  "Derive submesh edge and face selections from face sides".

---

### Task 3: remap each connectivity through its connected type

**Files:**
- Modify: `lib/iris/mesh/components.py` (`_MeshConnectivityManagerBase.indexed`)
- Test: `lib/iris/tests/unit/mesh/components/test_MeshIndexSet.py`

**Interfaces:**
- Consumes `indexed(node_bool_index, edge_indices, face_indices, frozen)`. The
  node argument is a boolean mask. Edges and faces are integer arrays or
  `None`.
- Produces a manager whose connectivities each have values in the submesh's
  numbering of their `connected` type. Values outside the submesh are masked.
  Each connectivity keeps its `start_index` and `location_axis`.

Implementation notes:
- Build one masked lookup per type, from original index to submesh index.
  - Keep the existing cumsum path for the node boolean mask.
  - For integer selections, use `lookup = np.ma.masked_all(n, np.intp)`, then
    `lookup[sel] = np.arange(len(sel))`.
  - Take `n` for edges and faces from a member connectivity at that location.
- Drop a connectivity, so it gives `None` in the new manager, when any of
  these holds:
  - its `location` is not node, edge or face;
  - its location's selection is `None`;
  - its `connected` selection is `None`.
- Remap with the existing `_index_conn_array`, through `lookups[c.connected]`.
- Leave `_indexed_manager` as it is.

Tests: add `Test_strip_connectivities`, using `sample_strip_mesh`.
- Read all connectivities from a single `index_set._connectivity_manager`
  access.
- Normalise each one with
  `np.ma.asanyarray(as_concrete_data(c.indices_by_location(c.core_indices()))) - c.start_index`.
  Reading `.indices` would realise a view, and the next access to the manager
  would then raise.
- Compare with `_shared_utils.assert_masked_array_equal`.

Expected results are below, with `-` for masked. "none (k)" means zero rows of
width k.

| Case | face_node | edge_node | face_edge | face_face | edge_face |
|---|---|---|---|---|---|
| face `[0, 2]` | `[0,1,5,4] [2,3,7,6]` | `[0,1] [2,3] [4,5] [6,7] [0,4] [1,5] [2,6] [3,7]` | `[0,5,2,4] [1,7,3,6]` | all masked, (2, 4) | `[0,-] [1,-] [0,-] [1,-] [0,-] [0,-] [-,1] [1,-]` |
| face `[1, 0]` | `[1,2,5,4] [0,1,4,3]` | `[0,1] [1,2] [3,4] [4,5] [0,3] [1,4] [2,5]` | `[1,6,3,5] [0,5,2,4]` | `[-,-,-,1] [-,0,-,-]` | `[1,-] [0,-] [1,-] [0,-] [1,-] [1,0] [0,-]` |
| edge `[0, 3, 6, 7, 9]` | `[0,1,4,3]` | `[0,1] [3,4] [0,3] [1,4] [2,5]` | `[0,3,1,2]` | all masked, (1, 4) | `[0,-] [0,-] [0,-] [0,-] [-,-]` |
| edge `[1, 4]` | none (4) | `[0,1] [2,3]` | none (4) | none (4) | all masked, (2, 2) |
| node `[0, 1, 4, 5]` | `[0,1,3,2]` | `[0,1] [2,3] [0,2] [1,3]` | `[0,3,1,2]` | all masked, (1, 4) | `[0,-] [0,-] [0,-] [0,-]` |

- [ ] **Step 1: write the failing tests.**
  - Add `test_strip_cases`, parametrised over the table rows plus
    `start_index` (0 or 1) and `location_axis` (0 or 1). It also checks that
    each result's `start_index` and `location_axis` match the original's.
  - Add `test_strip_repeated_faces`. Face `[0, 0]` gives face_node
    `[0,1,3,2] [0,1,3,2]`, and face_face is entirely masked with shape (2, 4).
  - Add `test_strip_stays_lazy`. With `lazy_values=True`, every connectivity
    on the index set has `is_lazy_data(c.core_indices())`.
  - Add `test_boundary_node_dropped`. Add a `boundary_node_connectivity` to a
    strip mesh. A face subset gives a manager with no boundary connectivity,
    and raises no KeyError.
  - Add `test_no_edge_node_drops_edge_connectivities`. Remove
    `edge_node_connectivity` from a strip mesh. A face subset then has no
    `edge_node`, `face_edge` or `edge_face`, but keeps `face_face`. If
    `MeshXY` refuses that mesh, report it and skip the test.
- [ ] **Step 2: update `test_connectivity_manager_subset`.**
  - For a node subset `[0, 2]`, `edge_node.shape[0] == 0`. On `mesh_2d`, also
    `face_node.shape[0] == 0`.
  - For an edge subset on `mesh_2d`, `face_node.shape[0] == 0`, in place of
    `not hasattr(..., "face_node")`. On `mesh_1d`, keep the `hasattr` check.
  - For a face subset, `edge_node.shape[0] == 1`, in place of
    `edge_node is None`. Edge 2 is a side of face 2.
- [ ] **Step 3: run them.** `PYTEST lib/iris/tests/unit/mesh/components/test_MeshIndexSet.py`.
  Expect the new and updated tests to fail.
- [ ] **Step 4: implement** the lookups and the drop rule in `indexed()`.
- [ ] **Step 5: run** `test_MeshIndexSet.py` and
  `lib/iris/tests/unit/mesh/`. Expect everything to pass except
  `Test_as_mesh.test_creation` for node, which Task 4 handles.
- [ ] **Step 6: pre-commit**, then **commit** with "Remap mesh subset
  connectivities through their connected element type".

---

### Task 4: keep the topology dimension in `as_mesh()`

**Files:**
- Modify: `lib/iris/mesh/components.py` (`_MeshIndexSet.as_mesh`)
- Test: `lib/iris/tests/unit/mesh/components/test_MeshIndexSet.py`
  (`Test_as_mesh`)
- Test: `lib/iris/tests/integration/mesh/test_indexing_meshes.py`

**Interfaces:** `as_mesh()` returns a `MeshXY` with
`topology_dimension == self.mesh.topology_dimension` for every location. It
holds the same set of connectivity `cf_role`s as the original, less
`boundary_node_connectivity`, and less any connectivity whose element type the
subset doesn't have.

Implementation notes:
- Take `topology_dimension` from `self.mesh`.
- Set `has_edges = getattr(conn_man, "edge_node", None) is not None`.
- Pass edge or face coords only when the coordinate manager has them.
- Remove the `NotImplementedError` for node subsets.

- [ ] **Step 1: update `Test_as_mesh.test_creation`.** Node now returns a
  `MeshXY`, and its node coords have the index set's shape. Delete the
  `pytest.raises` branch.
- [ ] **Step 2: write the failing integration tests.** Add these to
  `test_indexing_meshes.py` under `NEW_MESH`, each on
  `sample_mesh_cube(mesh=sample_strip_mesh(), location=...)`:
  - `test_strip_submesh_keeps_connectivities`, parametrised over face
    `[0, 2]`, edge `[0, 3, 6, 7, 9]` and node `[0, 1, 4, 5]`. The result's mesh
    has `topology_dimension == 2` and the same five `cf_role`s.
  - `test_strip_submesh_zero_faces`. Edge `[1, 4]` gives a 2D mesh with
    `face_node_connectivity.shape == (0, 4)` and face coords of length 0.
  - `test_strip_submesh_zero_faces_save`. Save that cube with
    `iris.save(cube, tmp_path / "zero_faces.nc")` and reload it with
    `iris.load_cube`. The reloaded mesh is 2D with zero faces. If saving or
    loading fails, mark the test `xfail(strict=True)`, with the exception as
    the reason, and report it as follow-up work.
  - Add `cube_mesh_node` to `test_subset_indexing_new_mesh`, and delete the
    "invalid thing to do" comment.
- [ ] **Step 3: run them.**
  `PYTEST lib/iris/tests/unit/mesh/components/test_MeshIndexSet.py lib/iris/tests/integration/mesh/test_indexing_meshes.py`.
  Expect node and zero-face failures.
- [ ] **Step 4: implement** the `as_mesh()` changes.
- [ ] **Step 5: run** the same command, then `PYTEST -n auto lib/iris/tests/unit/mesh lib/iris/tests/integration/mesh lib/iris/tests/unit/experimental`.
  Expect everything to pass, or to xfail as recorded.
- [ ] **Step 6: pre-commit**, then **commit** with "Keep the topology
  dimension when building a submesh".

---

### Task 5: final checks

**Files:** none changed, unless a check fails.

- [ ] **Step 1:** `conda run -n py313 pre-commit run --files <every file changed on the branch>`.
- [ ] **Step 2:** `PYTEST -n auto lib/iris/tests/unit lib/iris/tests/integration/mesh`.
  Expect no new failures against `main`.
- [ ] **Step 3:** run the doctests once, with
  `cd docs/src && conda run -n py313 env PYTHONPATH=../../lib make doctest`.
- [ ] **Step 4:** report what is still outstanding:
  - The changelog fragment `changelog/<PR>.bugfix.rst`. It needs the pull
    request number and the author's GitHub handle.
  - Any xfail added in Task 4.
