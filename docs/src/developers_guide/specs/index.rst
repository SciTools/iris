.. include:: ../../common_links.inc

.. _developers_guide_specs:

Design Specs
============

A **design spec** records what a substantial piece of Iris work is building and
why, and is agreed before any of the implementation lands.  Specs are written
in `MyST`_ Markdown rather than reStructuredText; they are the only part of the
documentation that is.

Specs are **living documents**.  They are revised as the design evolves, so a
spec always describes the current intent rather than a snapshot of a past
discussion.  Each carries a **plan roadmap** recording the pull requests that
deliver it and the status of each, together with the assumptions, open decisions
and open questions the work has surfaced.  A spec is therefore the place to look
for how far a programme of work has got, not only what it intends.

Each spec declares a **citation prefix** so that its sections can be referred to
unambiguously from issues, pull requests and other specs.  Within a spec, a bare
``§N.N`` always means that same document.

Give every cited section its own ``§``, and separate a run of them with a comma
or a solidus — ``§5.6, §5.7`` or ``§5.6 / §5.7``, never ``§5.6-5.7``.  A dashed
range reads as a single citation with a stray suffix, so only its first section
is unambiguously identified; spelling both out costs one character and leaves
nothing for a reader, or a future cross-reference tool, to guess at.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Citation
     - Spec
   * - ``merge spec §…``
     - :doc:`2026-09-10-merge-concatenate-design`

.. note::

   Implementation plans live alongside these specs, in
   ``docs/src/developers_guide/plans/``.  Unlike a spec, a plan is a
   point-in-time record that is frozen once the work it describes has merged,
   so plans are tracked in the repository but are **not** published here.

.. _MyST: https://myst-parser.readthedocs.io/

.. toctree::
   :maxdepth: 1
   :hidden:

   2026-09-10-merge-concatenate-design
