Changelog Fragments
===================

The root ``changelog/`` directory contains `towncrier <https://towncrier.readthedocs.io/>`_
news fragment files. Each file represents a single changelog entry for the
next release.

Fragment File Naming
--------------------

Files must be named as::

    <PR-number>.<type>.rst

Where ``<PR-number>`` is the pull request number and ``<type>`` is one of
the following fragment types:

``announcement``
    📢 General news and announcements to the Iris community.

``feature``
    ✨ Features that are new or changed to add functionality.

``bugfix``
    🐛 A bug fix.

``breaking``
    💣 A change that causes an incompatibility with prior versions of Iris.

``performance``
    🚀 A performance enhancement.

``deprecation``
    🔥 Deprecation of functionality.

``dependency``
    🔗 Additions, removals and version changes in Iris' package dependencies.

``doc``
    📚 Changes to documentation.

``internal``
    💼 Changes to any internal or development related topics, such as
    testing, environment dependencies etc.

Fragment File Content
---------------------

Each file should contain a short reStructuredText description of the change.
For example, a file named ``7146.feature.rst`` might contain::

    :user:`bjlittle` extended the :meth:`~iris.coords.Coord.cell` and
    :meth:`~iris.coords.Coord.cells` methods to allow users to specify that
    they want :class:`~datetime.datetime` compatible objects returned within
    each generated :class:`~iris.coords.Coord.Cell` from a temporal
    coordinate. (:issue:`7112`)

Notes
-----
* You do not need to include the PR number in the fragment as it is already in
  the fragment filename.
* Multiple fragments may reference the same PR number if the PR makes
  changes across different categories.
* For multiple PRs that reference the same change, simply create a separate
  changlog fragment file for each PR with **identical** contents.
* Use ``:issue:`NNNN``` for issue references, ``:pull:`NNNN``` for PR
  references, and ``:user:`github-name``` for user references.
* The rendered changelog can be previewed with::
    
    > towncrier build --draft

* The changelog can be published with::
    
    > cd docs/src/whatsnew
    > mkdir <major.minor>
    > git mv highlights.rst <major.minor>
    > towncrier build --version <major.minor>

* For further details see the
  `Command Line Reference <https://towncrier.readthedocs.io/en/stable/cli.html>`__
  for ``towncrier``.
