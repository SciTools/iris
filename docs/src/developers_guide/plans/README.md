# Implementation plans

Implementation plans for the design specs in [`../specs/`](../specs).

Unlike a spec, a plan is **not** a living document.  It records the sequence of
steps agreed at the time it was written, and is frozen once the work it
describes has merged.  Plans are tracked in the repository for provenance, but
are excluded from the Sphinx build via `exclude_patterns` in `docs/src/conf.py`.

A plan ships as its **own** pull request, ahead of the implementation it
describes, so that the implementation pull request carries only the change under
review.  A reviewer opening that pull request should see a diff they can hold in
their head; a plan bundled alongside is not evidence about the change, and makes
a small diff look like a large one.  If implementation shows the plan to be
wrong, correct it on the plan's own branch — a plan is frozen when its work
merges, not while the work is under way.

Naming: `YYYY-MM-DD-<topic>.md`.
