# Making Iris merge and concatenate robust, efficient and configurable

```{readingtime}
```

> **Living document.** This spec is updated as the design evolves; it is not a
> point-in-time record of what was agreed on any particular day. Cite sections
> of it as `merge spec §N.N`. A bare `§N.N` inside this document refers to this
> document.

- **Date:** 2026-09-10
- **Status:** design agreed; no pull request opened yet. §6 is the live progress
  record, and is updated as each pull request lands.
- **Issues:** {issue}`2761`, {issue}`5375`, {issue}`6790`, {issue}`7063`,
  {issue}`7241`
- **Applies to:** `lib/iris/_merge.py`, `lib/iris/_concatenate.py`,
  `lib/iris/_combine.py`
- **Baseline:** `greenfield` at `253510b37`, Iris `3.17.0.dev4`. Line references
  in this document are accurate as of that commit.

(merge-spec-1)=

## 1. Summary

`iris.cube.CubeList.merge` and `CubeList.concatenate` are the core of Iris's
input workflow and the source of a long tail of user complaints. They are two
independent engines that share no code, carry an open correctness bug from 2017,
realise lazy arrays repeatedly, and expose almost no configuration.

This document specifies a programme of twelve small, independently reviewable
pull requests. Each is either a behaviour-preserving refactor or a standalone
user-visible improvement. Together they build a shared substrate beneath both
engines, without requiring anyone to agree to a rewrite up front.

The six goals, in the requester's words, are that merging datasets into a hyper
cube should be **more robust, efficient, configurable at the API level,
extensible, lazy and lenient**.

(merge-spec-2)=

## 2. Why the current design is at its limit

(merge-spec-2-1)=

### 2.1 Two engines that share nothing

`lib/iris/_merge.py` (1863 lines) and `lib/iris/_concatenate.py` (1702 lines)
each define their own private `_CubeSignature`, `_CoordSignature`, `_ProtoCube`,
`_CoordAndDims` and `_CoordMetaData`. They share no comparison routine, no
signature type and no error type.

`lib/iris/_combine.py` (added 2025) sequences them but unifies nothing — its
`_combine_cubes` loop calls `.merge()` and `.concatenate()` in an order given by
the `merge_concat_sequence` option.

A third comparison implementation, `iris.common.resolve.Resolve` (2607 lines),
was built to be the single source of truth for cube comparison and combination
and is used only by cube maths. Iris therefore carries three independent answers
to "are these two cubes compatible, and how do I combine them?".

(merge-spec-2-2)=

### 2.2 Merge is not lazy

Merge compares live `Coord` objects and cell measures / ancillary variables with
`==`:

- `_merge.py:192` — `_CoordPayload._coords_msgs`, reached from
  `match_signature` at lines 262 and 267 with tuples of actual `Coord` objects.
- `_merge.py:450-453` — `_CubeSignature.match`, on `cell_measures_and_dims` and
  `ancillary_variables_and_dims`.

`_DimensionalMetadata.__eq__` (`coords.py:659-690`) ends in
`iris.util.array_equal(..., withnans=True)`, so every one of these comparisons
realises lazy arrays. They run once per candidate-cube × ProtoCube pair, with no
caching.

Concatenate solved exactly this problem in {pull}`5926` using `xxhash` array
hashing and a single batched `dask.compute` (`_compute_hashes`,
`_concatenate.py:479`). Merge never received it. Reported impact on ECMWF
hybrid-pressure loading is roughly 5×, and more than 10× worse again under
`_LAZY_DERIVED_LOADING`, with over 90% of time in
`cubes.combine → merge → coord equality → dask compute` ({issue}`7063`,
{issue}`7241`).

(merge-spec-2-3)=

### 2.3 Merge is barely configurable

`concatenate()` has four escape hatches — `check_aux_coords`,
`check_cell_measures`, `check_ancils`, `check_derived_coords`. `merge()` has one
keyword, `unique`. This asymmetry is directly user-visible in {issue}`6790`,
where cubes refuse to merge but concatenate happily after `new_axis`.

(merge-spec-2-4)=

### 2.4 Merge can be silently wrong

{issue}`2761`, open since 2017. `build_indexes` (`_merge.py:585-604`) records,
for each scalar value, the *set* of values it co-occurs with. Set membership
discards the structure needed to decide separability, so unrelated candidate
dimensions are judged separable. See §5.6 for the confirmed diagnosis and §5.7
for the fix.

(merge-spec-2-5)=

### 2.5 Leniency has been requested for a decade

{issue}`1987` (2016) → {issue}`4446` (merge, 2021) → {issue}`5392`
(concatenate, 2023). Never delivered, in part because retrofitting it onto two
unrelated engines means doing the work twice in two idioms.

(merge-spec-3)=

## 3. Constraints

1. **No large pull requests.** Iris core developers are sceptical of agentic
   contributions, and the AUX-Coord minutes of 2026-01-07 record "Easy to raise
   a PR, but hard to get reviewed". Review capacity is the scarce resource being
   optimised for. Every item below must stand alone.
2. **Backwards compatible with v3.x.** No breaking changes. The one deliberate
   behaviour change (§5.7) is a bug fix and is treated as such.
3. **Honest labelling.** Agentic work is labelled as agentic; precedent exists in
   {pull}`7161`.

(merge-spec-4)=

## 4. Decisions taken

(merge-spec-4-1)=

### 4.1 Where configurability is exposed

Each capability is implemented once in the shared substrate, exposed as explicit
keywords on `merge()` / `concatenate()`, and reaches load time through
`merge_kwargs` / `concatenate_kwargs` forwarding dicts on `CombineOptions`.

This mirrors the existing `equalise_cubes_kwargs` rather than inventing a new
mechanism. `CombineOptions.OPTION_KEYS` is already a list carrying the comment
"so we can update it in an inheriting class" (`_combine.py:182`), and dict-valued
options are already special-cased at `_combine.py:232`.

Rejected: `CombineOptions`-only, which leaves {issue}`6790` unfixed for direct
callers and buries configuration in ambient thread-local state; and
method-keywords-only, which leaves load-time users unable to reach the new
behaviour.

(merge-spec-4-2)=

### 4.2 How far leniency goes

Leniency covers **metadata** (via the existing `metadata.equal(lenient=...)`
semantics) and **structure** (tolerating missing or extra aux coords, cell
measures and ancillary variables).

Coercion of dtypes, units and calendars stays out of the engine, in
`iris.util.equalise_cubes`, which is the existing extension point for it.

(merge-spec-4-3)=

### 4.3 Substrate built bottom-up, not designed top-down

The substrate is grown one PR at a time starting from the hashing layer, rather
than specified in advance. The alternative — designing a unified v4.0 engine
first — was considered and parked at
<https://github.com/bjlittle/iris/issues/333>.

Adopting `Resolve` wholesale was also rejected as a *starting* point: it is
pairwise and merge is N-ary, so whether it scales is unresolved research risk,
and nothing user-visible would ship until it was answered.

(merge-spec-5)=

## 5. Design

(merge-spec-5-1)=

### 5.1 The substrate module

A single private module, `lib/iris/_combine_common.py`, sibling to `_merge.py`,
`_concatenate.py`, `_combine.py` and `_lazy_data.py`. Promoted to a package only
if it exceeds roughly 800 lines.

Rejected placements:

- **`iris/common/`** — that subpackage is documented and public-facing; putting
  engine internals there implies an API commitment.
- **Nesting under a `_combine/` package** — this inverts the dependency.
  `_merge.py` would import `iris._combine.hashing`, executing
  `iris/_combine/__init__.py`, which reaches `iris.cube`. The substrate sits
  *below* merge and concatenate; `_combine.py` sits *above* them.

The substrate must import nothing from Iris. The hashing code's only Iris
dependency is `iris.coords` for type hints in `array_id`, which goes behind
`TYPE_CHECKING`. This is what makes the merge-side import risk-free.

(merge-spec-5-2)=

### 5.2 What is shared, and what is not

The two cube-level signatures are **not** unified. They encode different
questions: merge's asks "same kind of cube, identical shape?", because merge adds
a dimension and all sources have the same shape; concatenate's asks "joinable
along an axis?" and carries extents, dim order and dim mapping to answer it.

The genuinely shared kernel sits one level down:

1. **Element signature** — a hashed, comparable representation of a single
   dimensional-metadata object (coord, cell measure, ancillary variable):
   its `metadata`, its dims, `has_bounds()`, and hashes of points/bounds/data.
2. **Array-collection walk** — the traversal of a cube yielding arrays to hash.
   Concatenate's `add_coords` (`_concatenate.py:601-611`) and merge's equivalent
   are the same function written twice.
3. **Comparison result** — a value carrying *why* two elements differ, which
   feeds diagnostics.

(merge-spec-5-3)=

### 5.3 Merge needs a driver function

Concatenate's loop lives in `iris._concatenate.concatenate()`, which is why it
can hash everything up front. Merge's loop lives in `CubeList.merge()` at
`cube.py:443-467`; there is no `iris._merge.merge()`.

A new `iris._merge.merge(cubes, unique=True)` takes that loop, and
`CubeList.merge()` delegates — mirroring `CubeList.concatenate()`. This is also
the landing site for every keyword added later, which is why it comes early.

`merge_cube()` does not share this loop: it builds a single ProtoCube from
`self[0]` with `error_on_mismatch=True` and no name grouping
(`cube.py:364-373`). It is left alone initially and revisited when its error path
becomes relevant.

(merge-spec-5-4)=

### 5.4 Hash-based comparison in merge

Arrays are collected up front in `_merge.merge()`, mirroring concatenate, and
`hashes` is threaded through `ProtoCube.register()` to the two consumption sites
in §2.2. `self.scalar.defns` (`_merge.py:256`) is left alone — it is already
metadata-only and cheap.

**Exactly one `compute_hashes()` call per invocation.** This is not stylistic:
`ArrayHash.__eq__` raises `ValueError` on same-shape/different-chunks, and only a
single batched call guarantees chunk unification across every input cube.

**This is semantics-preserving.** `_DimensionalMetadata.__eq__`
(`coords.py:659-690`) is metadata equality, then `has_bounds()` parity, then
`array_equal(..., withnans=True)` on values and bounds. And
`lib/iris/tests/unit/concatenate/test_hashing.py:88` already asserts that hash
equality is equivalent to `array_equal(a, b, withnans=True)` across a shared
corpus. The metadata and `has_bounds()` checks stay exactly where they are; only
the two `array_equal` calls become hash lookups.

**The `id()` hazard.** `array_id` is `f"{id(coord)}{bound}"`, and merge calls
`_extract_coord_payload` once per *(cube, ProtoCube)* pair, so ids must be stable
across calls. They are: `Cube.dim_coords` and `Cube.aux_coords`
(`cube.py:3158,3182`) return a fresh tuple of the *same* objects.
`Cube.derived_coords` (`cube.py:3195`) does not — it regenerates from factories
on every access. Merge never array-compares derived coords, going through
`factory_defns` (metadata only, `_merge.py:271`), so it is unaffected. The driver
must nonetheless hold a reference to every input cube for the call's duration,
and this needs a code comment.

**Risks to state in the PR:** metadata comparison must remain ordered first,
because hashes unify numerical dtypes (`float32([1])` and `bool([1])` hash equal
— consistent with `array_equal`, but only if ordering is unchanged); and the
chunk-mismatch `ValueError` becomes reachable from merge for the first time.

(merge-spec-5-5)=

### 5.5 Measurement

`benchmarks/benchmarks/merge_concat.py::Merge` merges two cubes and explicitly
strips cell measures and ancillary variables. It cannot demonstrate this
improvement. The benchmark is extended in a separate PR *before* the change, so
the baseline is committed to `main` and the speedup is reproducible.

(merge-spec-5-6)=

### 5.6 Separability: the diagnosis

Confirmed by direct probe against the reporter's case in {issue}`2761`:

```text
A = [1, 2, 1, 2, 1, 2]
B = [3, 4, 4, 4, 4, 3]
C = [5, 5, 6, 6, 7, 7]
```

Current Iris derives:

```text
a: separable=['b', 'c']  inseparable=[]
b: separable=['a']       inseparable=['c']
c: separable=['a']       inseparable=['b']
```

and `CubeList.merge()` raises
`ValueError: You must specify the meta or dtype of the array` — a raw Dask error
with no connection to the actual problem. That is the user experience today.

The current predicate is `_separable_pair` (`_merge.py:607-631`): X and Y are
separable iff every value of X co-occurs with the same *set* of Y values.

(merge-spec-5-7)=

### 5.7 Separability: the fix

**Replacing the set with a multiset does not work.** The multiset
co-occurrences are also equal in this case (`a=1 → {3:1, 4:2}`,
`a=2 → {3:1, 4:2}`). The correct criterion is about the pair grid, not per-value
co-occurrence:

> X and Y are separable if and only if the observed (X, Y) pairs form the
> complete Cartesian product `values(X) × values(Y)` with **uniform
> multiplicity**.

Uniform, not one: in a full three-way product each (X, Y) pair recurs once per
value of the third dimension, and requiring multiplicity 1 would wrongly reject
it.

Verified against five topologies:

| topology | expected | result |
|---|---|---|
| 2×2 product | separable | ✓ |
| {issue}`2761` case | only `a`–`c` separable | ✓ |
| 2×2×2 full product | all pairs separable (multiplicity 2, uniform) | ✓ |
| `b = f(a)` | inseparable | ✓ |
| incomplete 2×2 (3 cells) | inseparable | ✓ |

Verified end-to-end by simulating the predicate through a real
`CubeList.merge()`: six cubes become one, `(a: 2; c: 3; y: 2; x: 2)` with `b` as
a 2-D auxiliary coordinate spanning `a` and `c` — precisely what the reporter
said the answer should be. The rest of the algebra (`derive_groups`,
`_derive_separable_group`, `_derive_consistent_groups`, `_is_dependent`) already
handles it correctly once the predicate is right.

**Blast radius is one call site**: `_merge.py:1215-1220` in `ProtoCube.merge()`,
where `positions` is already in scope — it is passed to `derive_space` on the
next line. Only `derive_relation_matrix` changes shape. `build_indexes` stays,
because `_define_space` still consumes it (`_merge.py:1446`).

**This algebra has essentially no unit tests.** Grepping `lib/iris/tests` finds
exactly one reference: `integration/merge/test_merge.py:329`
`test_separable_combination`. That is why a 2017 bug survived to 2026.
Characterisation tests therefore land *before* the fix, so the fix's diff shows
exactly which expectations change.

**Known edge case to pin with a test:** duplicated source cubes make a pair grid
non-uniform, so a case that was "separable, then duplicate-detected" becomes
"inseparable, then duplicate-detected". Probably the same user-visible outcome,
but it must be verified rather than assumed.

(merge-spec-5-8)=

### 5.8 Diagnostics

`merge()` explains nothing today — it returns N cubes with no indication why they
did not become one. The `msgs` lists built in `_CubeSignature.match` and
`_CoordPayload.match_signature` are constructed and discarded whenever
`error_on_mismatch=False`.

The work is therefore mostly plumbing: capture and surface what is already
computed, using the comparison-result type from §5.2. `merge()`'s return type is
unchanged; a separate diagnostic entry point is added alongside
`iris.util.describe_diff` (`util.py:205`), which already does pairwise cube
diffs. What is missing is the group-level account: these three formed one cube,
this fourth stayed out because X.

(merge-spec-5-9)=

### 5.9 Leniency

The primitive already exists: every metadata class has
`equal(other, lenient=None)` (`metadata.py:660`), decorated `@lenient_service`.
Merge hand-compares metadata field by field in `_CubeSignature._defn_msgs`
(`_merge.py:350-406`); concatenate has its own equivalent. Both become
`equal(lenient=...)`.

`lenient=True` is passed **explicitly**. The `LENIENT` thread-local
(`iris/common/lenient.py:667`) is not used, so no ambient state enters the
engine's hot path — consistent with §4.1.

On {issue}`5394` and {issue}`5395` the claim must be precise: this adopts the
lenient metadata *semantics* that `Resolve` is built on, not `Resolve` itself.
They are partially addressed, not closed.

Structural leniency — reconciling elements present in some cubes but not others,
by intersection or union — is separated from the `check_*` flags, which only say
"do not compare". It lands last among the behaviour changes because it is the one
item with genuine semantic debate in it.

(merge-spec-6)=

## 6. Plan roadmap

Twelve pull requests deliver the design. Each gets its own implementation plan in
`docs/src/developers_guide/plans/`, shipped as its own pull request ahead of the
implementation so that the implementation PR carries only the change under
review, and a PR is merged before any PR that *depends on it* is written. The dependencies form a partial order, not a chain:
PRs 1, 2, 3 and 6 have no predecessors and may proceed in parallel. The ordering
follows §5's layering — the substrate first (PR 1), then the seam that lets merge
use it (PRs 2–3), then the payoff (PR 4), then the correctness and capability
work that the substrate makes cheap.

This table is the programme's live progress record and is updated as each PR
lands. Status vocabulary: **not started** · **in progress** · **✅ complete** ·
**blocked** · **abandoned**, each carrying its {pull} citation once one exists.

| # | PR | Scope (spec §) | Type | Closes | Depends on | Status |
|---|---|---|---|---|---|---|
| 1 | Hashing machinery → `_combine_common.py` | §5.1 — move `_concatenate.py:305-541` (~237 lines) and `tests/unit/concatenate/test_hashing.py`; names lose their underscore prefix | internal | — | — | not started |
| 2 | `iris._merge.merge()` driver function | §5.3 — lift the loop out of `CubeList.merge()` (`cube.py:443-467`); `CubeList.merge` delegates, mirroring `CubeList.concatenate` | internal | — | — | not started |
| 3 | Extend merge benchmark | §5.5 — `benchmarks/benchmarks/merge_concat.py::Merge` currently strips cell measures and ancillaries, so it cannot show the PR 4 effect | internal | — | — | not started |
| 4 | **Hash-based comparison in merge** | §5.4 — collect arrays up front, thread `hashes` through `ProtoCube.register()` to the two `==` sites of §2.2 | performance | {issue}`7063`, {issue}`7241` | 1, 2, 3 | not started |
| 5 | Shared element signature + collection walk | §5.2 items 1–3 — one hashed element signature, one array-collection walk, one comparison-result type, for both engines | internal | — | 1, 4 | not started |
| 6 | Separability characterisation tests + doctest cleanup | §5.6, §7 — the five-topology table as tests against *current* behaviour; fix the Python-2 doctests at `_merge.py:691` | internal | — | — | not started |
| 7 | **Separability fix** | §5.7 — replace `_separable_pair`'s set criterion with complete-grid-and-uniform-multiplicity, at the single call site `_merge.py:1215-1220` | bugfix | {issue}`2761`, likely {issue}`5768` | 6 | not started |
| 8 | Structured diagnostics | §5.8 — capture the `msgs` already built and discarded; a group-level account beside `iris.util.describe_diff` | feature | {issue}`5375` | 2, 5 | not started |
| 9 | Metadata leniency | §5.9 — `metadata.equal(lenient=...)` replaces hand-rolled field comparison in both engines | feature | {issue}`4446`, {issue}`5392`; part of {issue}`5394` / {issue}`5395` | 2, 5 | not started |
| 10 | `check_*` parity for merge | §2.3 — the four `check_*` keywords on merge, with the `cube.py:427-433` documentation change in the same PR | feature | {issue}`6790` | 2 | not started |
| 11 | Structural leniency | §5.9 — reconcile elements present in some cubes but not others, by intersection or union | feature | — | 9, 10 | not started |
| 12 | `CombineOptions` forwarding dicts | §4.1 — `merge_kwargs` / `concatenate_kwargs` in all four `SETTINGS` dicts, defaulting to `None` | feature | — | 10, 11 | not started |

**PRs 1–4 are the proving tranche.** They end with a measured speedup against a
committed baseline and no behaviour change whatsoever. If core developers do not
engage with those, the remainder should not be attempted.

Cross-cutting rules (apply to every row rather than one of them):

- **Each PR cites this spec** and says which row it is, so a reviewer can see the
  shape of the programme without reading twelve descriptions, and says plainly
  that it is agentic work — {pull}`7161` is the precedent (§3).
- **Every PR targets the `greenfield` feature branch, not `main`.** The
  programme is long-running and deliberately incremental, so its parts land
  together rather than trickling into `main` one at a time; `greenfield` is then
  merged back once the tranche it carries is coherent. Each PR is labelled
  `Feature: Merge/Concatenate`, `Type: Feature Branch` and `Agentic`, so the
  agentic ones are filterable — both for reviewers who want them and for those
  who would rather not.
- **Changelog fragment ships with its PR**, typed as the Type column says, per
  `changelog/AGENTS.md`.
- **Refactor PRs change no test expectations** (rows 1, 2, 5). Test files move and
  imports update; assertions do not. Behaviour PRs land their tests with them.
- **Performance claims are quoted, not asserted.** Any row claiming an effect
  quotes before/after numbers from `benchmarks/benchmarks/merge_concat.py`
  against a baseline already on `main` (§5.5).

Notes on individual items:

- **PR 1** moves seven names (`_concatenate.py:305-541`, ~237 lines) and the test
  file `tests/unit/concatenate/test_hashing.py`. Names lose their underscore
  prefix on the way out, following `_lazy_data.py`; this makes it a move *and* a
  rename, which is a reasonable thing for a reviewer to object to and a cheap
  thing to concede.
- **PR 1** must state plainly that it is the first step of a sequence and link
  the performance issues. Its justification is PR 4.
- **PR 7** will change CML fixtures under `tests/results/cube_merge/`. Cases
  previously called separable become inseparable; most raised or produced wrong
  cubes, but some may have been accidentally right. Every changed fixture needs
  an individual justification. This is the review-cost centre of the programme,
  which is why it follows the cheap wins.
- **PR 10** requires a documentation change in the same PR: the note at
  `cube.py:427-433` states that aux coords, cell measures and ancillaries "must
  be identical in every input cube". `check_*=False` is exactly the relaxation of
  that sentence.
- **PR 12** must add each new option to all four `SETTINGS` dicts
  (`_combine.py:190-219`), defaulting to `None` in all four so load-time
  behaviour is unchanged.

Outside the roadmap:

- A v4.0 unified engine is parked at <https://github.com/bjlittle/iris/issues/333>
  (§4.3, §9) and is not a row here.
- The coercion work under {issue}`6383` is adjacent and separately owned: §4.2
  routes dtype, unit and calendar coercion to `iris.util.equalise_cubes` rather
  than into either engine.
- Core-developer engagement is the gating condition, not a deliverable. PR 1's
  reception decides whether PRs 5–12 are attempted (§3).

### Assumptions and open decisions

Enumerated so they are visible decisions rather than silent drift. Items 1–5 are
decisions this design makes; the remainder are questions assigned to the PR that
must answer them, ordered by owning PR. Items carry a tracking issue where one
already exists; the rest name their owning row, and get an issue when that PR is
opened.

1. **Resolved** (2026-09-10) — **Incremental substrate, not a designed engine.**
   The alternative — specify a unified v4.0 engine, then build it — was
   considered and rejected as a *starting* point, because nothing user-visible
   ships until the whole thing is agreed. It is preserved rather than discarded,
   at <https://github.com/bjlittle/iris/issues/333>, and §4.3 carries the
   reasoning. Adopting `Resolve` wholesale was rejected on the same ground: it is
   pairwise, merge is N-ary, and whether it scales is unresolved research risk.
2. **Resolved** (2026-09-10) — **Configurability is keywords plus forwarding
   dicts.** Each capability is implemented once in the substrate, exposed as an
   explicit keyword on `merge()` / `concatenate()`, and reaches load time through
   `merge_kwargs` / `concatenate_kwargs` on `CombineOptions` (§4.1). The two
   rejected alternatives are recorded there: `CombineOptions`-only leaves
   {issue}`6790` unfixed for direct callers, and method-keywords-only leaves
   load-time users unable to reach the behaviour.
3. **Resolved** (2026-09-10) — **Leniency covers metadata and structure;
   coercion stays out.** §4.2. Dtype, unit and calendar coercion belongs in
   `iris.util.equalise_cubes`, which is the existing extension point for it.
4. **Resolved** (2026-09-10) — **The substrate is a flat private module.**
   `lib/iris/_combine_common.py`, sibling to `_merge.py` and `_concatenate.py`,
   promoted to a package only above roughly 800 lines. `iris/common/` was
   rejected as public-facing, and a `_combine/` package as inverting the
   dependency — the substrate sits *below* the two engines, `_combine.py` sits
   above them (§5.1).
5. **Resolved** (2026-09-10) — **The two cube-level signatures are not
   unified.** They encode different questions, and forcing one type on both would
   be a rewrite wearing a refactor's clothes (§5.2). {issue}`3234` therefore
   stays open; this programme narrows it rather than closing it.
6. **Open** (PR 1) — **Whether the moved names drop their underscore prefix.**
   Following `_lazy_data.py` makes PR 1 a move *and* a rename, which is a
   reasonable thing for a reviewer to object to and a cheap thing to concede. The
   location is what matters, not the name (§8).
7. **Open** (PR 1) — **Whether the programme proceeds past the proving tranche.**
   §3 names review capacity as the scarce resource. PR 1 is a pure move with no
   behaviour change and its justification is PR 4, so its reception is the signal
   for everything after it. Nothing in PRs 5–12 is load-bearing for PRs 1–4.
8. **Open** (PR 2) — **Whether `merge_cube()` joins the driver.** §5.3 leaves it
   alone: it builds a single ProtoCube from `self[0]` with
   `error_on_mismatch=True` and no name grouping (`cube.py:364-373`). Its error
   path is exactly where PR 8's diagnostics are most visible, so the question
   re-opens there rather than being settled once.
9. **Open** (PR 4) — **What merge does if `ArrayHash.__eq__` raises.** §5.4
   argues that exactly one batched `compute_hashes()` call per invocation makes
   the same-shape/different-chunks `ValueError` unreachable. PR 4 must prove that
   with a test rather than assert it, and decide the fallback if the argument is
   wrong — this exception becomes reachable from merge for the first time.
10. **Open** (PR 5, §9) — **Whether `hexdigest` reconciliation stays out of
    scope.** `_concatenate.py` uses both `iris.common.metadata.hexdigest` (line
    807, diagnostic path only) and the array-hashing layer. §9 keeps them apart so
    that a mechanical move does not become a semantic argument. If PR 5's shared
    element signature makes the duplication glaring, revisit it there.
11. **Open** (PR 6) — **How many CML fixtures PR 7 changes.** Unknown until the
    characterisation tests exist. Cases previously called separable become
    inseparable; most raised or produced wrong cubes, but some may have been
    accidentally right, and every changed fixture needs an individual
    justification. §5.7 calls this the programme's review-cost centre; PR 6
    measures it and PR 7 justifies it.
12. **Open** (PR 6, PR 7) — **Duplicated source cubes under the new criterion.**
    Duplication makes a pair grid non-uniform, so a case that was "separable,
    then duplicate-detected" becomes "inseparable, then duplicate-detected".
    Believed to be the same user-visible outcome, but §5.7 requires it be pinned
    by a test rather than assumed.
13. **Open** (PR 7, {issue}`5768`) — **Whether the separability fix closes
    {issue}`5768`.** The roadmap says "likely", which is not good enough for a PR
    description. The claim needs a reproduction against the fixed predicate before
    it is made.
14. **Open** (PR 10) — **Whether the new keywords also go on `merge_cube()`.**
    Downstream of item 8. If `merge_cube()` never joins the driver, it acquires
    the keywords separately or not at all, and {issue}`6790` is only partly
    answered.
15. **Open** (PR 11) — **Intersection or union for structural leniency.**
    Reconciling elements present in some cubes but not others has two defensible
    answers, and the choice is user-visible. §5.9 separates this from the
    `check_*` flags, which only say "do not compare". It lands last precisely
    because it is the one item with genuine semantic debate in it.

(merge-spec-7)=

## 7. Testing and validation

Per `lib/iris/tests/AGENTS.md`: pytest style, plain assertions,
`pytest.raises(..., match=...)`, `pytest.warns`, explicit laziness assertions, no
network.

- **Refactor PRs (1, 2, 5)** must change no test expectations. Test files move
  and imports update; assertions do not.
- **PR 4** needs explicit tests for masked arrays, NaN, and dtype edge cases at
  the hash boundary, plus a test that the chunk-mismatch `ValueError` does not
  fire in normal use. The existing equivalence test
  (`tests/unit/concatenate/test_hashing.py:88`) is the anchor.
- **PR 6** introduces the topology table of §5.7 as characterisation tests
  against *current* behaviour, and fixes the dead Python-2 doctests in
  `build_indexes` and `derive_relation_matrix` (`_merge.py:691` uses
  `matrix.iteritems()`, which cannot ever have run).
- **PR 7** flips those expectations, with each change justified.
- **Benchmarks**: `benchmarks/benchmarks/merge_concat.py`, extended in PR 3, with
  before/after numbers quoted in PR 4.
- Every PR carries a `changelog/<PR-number>.<type>.rst` fragment per
  `changelog/AGENTS.md`.

(merge-spec-8)=

## 8. Risks

| Risk | Mitigation |
|---|---|
| Core devs reject the shared module in PR 1 | PR 1 is a pure move with no behaviour change; if the module name or the rename is contested, concede both — the location matters, not the name |
| PR 7 changes more CML fixtures than expected | Characterisation tests land first (PR 6) so the delta is visible before the fix; each fixture justified individually |
| Hash equality diverges from `Coord.__eq__` in an untested corner | Metadata and `has_bounds()` checks unchanged; only `array_equal` calls replaced; existing equivalence test is the anchor |
| Programme stalls after the proving tranche | PRs 1–4 are independently valuable; nothing later depends on the programme continuing |
| Structural leniency semantics prove contentious | It is last, and nothing else depends on it |

(merge-spec-9)=

## 9. Out of scope

- Unifying `_CubeSignature` across the two engines (§5.2).
- Adopting `Resolve` as the N-ary comparison engine (§4.3).
- Reconciling `iris.common.metadata.hexdigest` with the array-hashing layer.
  `_concatenate.py` uses both — `hexdigest` at line 807, but only in the
  diagnostic path that names which coords differ, not in the equality path.
  Worth doing eventually; bundling it would turn a mechanical move into a
  semantic argument.
- Coercion of dtypes, units and calendars (belongs in
  `iris.util.equalise_cubes`).
- A v4.0 unified engine — parked at
  <https://github.com/bjlittle/iris/issues/333>.

(merge-spec-10)=

## 10. Open questions

Questions raised by the research behind this spec that the roadmap does not
answer. Unlike §6's assumptions — each of which is owned by a PR that must
resolve it — these are open against the programme as a whole, and some will
outlive it. Every one carries a status and, where it exists, a citation; those
without one say so.

- **Open** ({issue}`3234`) — **Should the two engines ever become one?** §5.2
  says no *now*: the two cube-level signatures encode different questions, and
  the shared substrate captures the real duplication without forcing a common
  type. Whether that remains the right answer once PRs 1, 5 and 9 have landed is
  worth re-asking against the codebase they leave behind, not against today's.
- **Parked** (<https://github.com/bjlittle/iris/issues/333>) — **Does `Resolve`
  scale from pairwise to N-ary comparison?** `Resolve` is the modern, tested,
  public machinery for reconciling two cubes, and merge is inherently N-ary.
  Nobody has established what pairwise-folding an N-ary problem costs, in
  performance or in semantics. This is the central unknown behind {issue}`5394`
  and {issue}`5395`, and it is research risk, not implementation risk — which is
  why §4.3 keeps it off the roadmap rather than pretending to have costed it.
- **Open** ({issue}`5375`) — **Should merge diagnostics be public API?** PR 8
  makes the reasons a group failed to merge *available*; it does not settle
  whether they are a documented, stable, user-facing surface or an error-message
  improvement. The two have very different maintenance costs, and the answer
  determines whether the diagnostics need a deprecation policy.
- **Open** ({issue}`6383`) — **Does `equalise_cubes` absorb enough coercion to
  keep engine leniency narrow?** §4.2 draws the line at metadata and structure on
  the strength of `equalise_cubes` covering dtype, unit and calendar. If its
  sub-issues stall, pressure returns to put coercion in the engines, and the line
  has to be re-argued rather than silently redrawn.
- **Open** ({issue}`7063`) — **What is the measured speedup from hash-based
  comparison in merge?** {issue}`7063` and {issue}`7241` report the problem;
  {pull}`5926` reports roughly 5× for the concatenate case. Merge's profile is
  different — many small scalar-coord comparisons rather than a few large array
  ones — so the concatenate figure is not transferable. PR 3 exists to make this
  answerable before PR 4 claims anything (§5.5).
- **Open** (no tracking issue yet; raised with PR 4) — **Can cube registration
  stay lazy end to end?** Hashing still computes, so PR 4 replaces
  many realisations with one batched one rather than removing realisation.
  Whether merge could instead decide most registrations on metadata alone and
  touch array values only for genuine ties is a larger question that PR 4 does
  not foreclose but also does not answer.
- **Open** (no tracking issue yet; raised with PR 7) — **Does the separability
  fix need a deprecation cycle?** §5.7 treats it as a bugfix: the current
  criterion produces silently wrong cubes, and there is no defensible way to
  keep producing them. But it changes results for existing data without any
  opt-in, which is the kind of change core developers reasonably want staged.
  Warn-then-change is available; whether it is warranted is a judgement for the
  PR 7 review, and the answer is worth recording here when it comes.

(merge-spec-11)=

## 11. References

- {issue}`2761` — Merge Problems (separability)
- {issue}`3234` — Unify merge and concatenate
- {issue}`4446`, {issue}`5392` — `LENIENT` merge / concatenate
- {issue}`5375` — Various cube merge/concatenate issues
- {issue}`5394`, {issue}`5395` — trial `resolve` in merge / concatenate
- {issue}`5768` — dimensions mashed together on merge
- {pull}`5926` — array hashing in concatenate
- {issue}`6383` — additional `equalise_cubes` functionality
- {issue}`6790` — merge/concatenate asymmetry
- {issue}`7063`, {issue}`7241` — merge performance
- {discussion}`6881` — AUX-Coord minutes, 2026-01-07
