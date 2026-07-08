.. include:: ../../common_links.inc

.. _whats_new_contributions:

=================================
Contributing a "What's New" Entry
=================================

.. readingtime::

Please include a "What's New" changelog fragment for **any** change that you
make to Iris. **Even if it is not relevant to users** - the
`Fragment Types`_ include ``internal`` for this - the page is read by
contributors as well as users, and it reveals the work needed to keep a
project going.

Iris uses `towncrier <https://towncrier.readthedocs.io/>`_ to manage changelog
entries. Each pull request adds a small file (a "fragment") to the root
``changelog/`` directory. At release time, ``towncrier`` collects the fragments
and renders the full What's New page.

See this docs section for all What's New pages: :ref:`iris_whatsnew`.

How it Works
============

Instead of editing a shared file, each contributor creates a small
reStructuredText file in the ``changelog/`` directory at the root of the
repository. This avoids the merge conflicts that were common with the
previous approach.

Creating a Fragment
===================

1. **Name your file** using the pattern::

       <PR-number>.<type>.rst

   For example, if your pull request number is ``7200`` and you are adding a
   feature, create::

       changelog/7200.feature.rst

   .. hint::

       If you have not yet created the pull request, you can guess what the
       next PR number may be using::

         > curl -s "https://api.github.com/repos/SciTools/iris/issues?sort=created&direction=desc&per_page=1" | jq -r '.[0].number + 1'


2. **Write a short description** of your change in the file. The content is
   reStructuredText. For example::

       :user:`tkknight` added a new option to :func:`iris.plot.pcolormesh`
       for controlling the colorbar orientation. (:issue:`9999`)

   Notes:

   * Use ``:user:`github-name``` to credit contributors.
   * Use ``:issue:`NNNN``` to reference issues.
   * The pull request reference is added automatically by ``towncrier`` based on
     the fragment filename - you do **not** need to include ``:pull:`` in your
     content unless you are referencing another pull request.
   * Where possible, do not exceed **column 80**.

3. **Multiple fragments per PR** are allowed if a single pull request makes
   changes across different categories. For example, a PR might have both
   ``7200.feature.rst`` and ``7200.doc.rst``.

4. **Multiple PRs per fragment** are automatically collated by ``towncrier``.
   Create separate fragment files per PR with **identical** contents.

Fragment Types
==============

The following fragment types are available, matching the rendered section
headings in the What's New page:

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
    💼 Changes to any internal or development related topics, such as testing,
    environment dependencies etc.

Highlights
==========

The release highlights associated with a ``towncrier`` changelog are defined
in the ``docs/src/whatsnew/highlights.rst`` file.

Manually update the ``hightlights.rst`` with any notable release information
that you want to share with the community.

Previewing the Changelog
========================

You can manually preview how the changelog will render by running::

    > towncrier build --draft

This will print the rendered reStructuredText to **stdout** without modifying any
files or removing fragment files.

.. note::

    The reStructuredText syntax will be checked as part of building the
    documentation. Any warnings should be corrected. The
    `Iris GitHub Actions`_ will automatically build the documentation when
    creating a pull request, however you can also manually
    :ref:`build <contributing.documentation.building>` the documentation.

.. tip::

    ``towncrier`` and the
    `sphinx-changlog <https://sphinx-changelog.readthedocs.io/en/latest/#>`__
    directive will automatically render the latest development changelog
    whenever the documentation is built.


Building the Changelog
======================

To build the release changelog

#. Change directory to ``docs/src/whatsnew/``.

#. Ensure that the ``hightlights.rst`` is populated.

#. Create the changelog release directory e.g., ``mkdir <major.minor>``.

#. Relocate the ``hightlights.rst`` i.e., ``git mv highlights.rst <major.minor>``.

#. Build the changelog i.e., ``towncrier build --version <major.minor>``. Note
   that this will create a rendered ``<major.minor>/<major.minor>.rst`` changelog
   and automatically stage this file with ``git``. The changelog news fragment
   files will also be automatically removed.

#. Remove the latest development changelog i.e., ``git rm latest.rst``.

#. Update the "What's New" ``index.rst`` replacing all references to ``latest.rst``
   with ``<major.minor>/<major.minor>.rst``.

.. tip::

    Using the ``--keep`` command line argument when building the changelog allows
    you to review the rendered release changelog and keep all the changelog news
    fragment files, allowing you to backtrack and make changes, if necessary.

Configuration
=============

``towncrier`` is configured within the ``[tool.towncrier]`` table of the root
``pyproject.toml``.

The ``changelog/template.rst`` file contains the ``jinja2`` template used by
``towncrier`` to render the changelog news fragments, sections, title and
include the associated ``hightlights.rst``.
