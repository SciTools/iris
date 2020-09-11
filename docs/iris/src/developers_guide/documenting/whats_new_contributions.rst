.. _whats_new_contributions:

=================================
Contributing a "What's New" entry
=================================

Iris uses a file named ``latest.rst`` to keep a draft of upcoming changes
that will form the next release.  Contributions to the :ref:`iris_whatsnew`
document are written by the developer most familiar with the change made.
The contribution should be included as part of the Iris Pull Request that
introduces the change.

The ``latest.rst`` and the past release notes are kept in
``docs/iris/src/whatsnew/``.


Writing a contribution
======================

As introduced above, a contribution is the description of a change to Iris
which improved Iris in some way. As such, a single Iris Pull Request may
contain multiple changes that are worth highlighting as contributions to the
what's new document.

Each contribution will ideally be written as a single concise bullet point
in a reStructuredText format. Where possible do not exceed **column 80** and
ensure that any subsequent lines of the same bullet point are aligned with the
first. The content should target an Iris user as the audience. The required
content, in order, is as follows:

* Names of those who contributed the change. These should be their GitHub
  display name, or if that is not available use their GitHub user name. Link
  the name to their GitHub profile. E.g.
  ```Bill Little <https://github.com/bjlittle>`_ and
  `tkknight <https://github.com/tkknight>`_ changed...``

* The new/changed behaviour

* Context to the change. Possible examples include: what this fixes, why
  something was added, issue references (e.g. ``:issue:`9999```), more specific
  detail on the change itself.

* Pull request references, bracketed, following the final period. E.g.
  ``(:pull:`1111`, :pull:`9999`)``

* A trailing blank line (standard reStructuredText bullet format)

For example::

  * `Bill Little <https://github.com/bjlittle>`_ and
    `tkknight <https://github.com/tkknight>`_ changed changed argument ``x``
    to be optional in :class:`~iris.module.class` and
    :meth:`iris.module.method`. This allows greater flexibility as requested in
    :issue:`9999`. (:pull:`1111`, :pull:`9999`)


The above example also demonstrates some of the possible syntax for including
links to code. For more inspiration on possible content and references please
examine past what's :ref:`iris_whatsnew` entries.

.. note:: The reStructuredText syntax will be checked as part of building
          the documentation.  Any warnings should be corrected.
          `travis-ci`_ will automatically build the documentation when
          creating a pull request, however you can also manually
          :ref:`build <contributing.documentation.building>` the documentation.

.. _travis-ci: https://travis-ci.org/github/SciTools/iris


Contribution categories
=======================

The structure of the what's new release note should be easy to read by
users.  To achieve this several categories may be used.

*Features*
  Features that are new or changed to add functionality.

*Bug Fixes*
  A bug fix.

*Incompatible Changes*
  A change that causes an incompatibility with prior versions of Iris.

*Deprecations*
  Deprecations of functionality.

*Dependencies*
  Additions, removals and version changes in Iris' package dependencies.

*Documentation*
  Changes to documentation.

*Internal*
  Changes to any internal or development related topics, such as testing,
  environment dependencies etc
