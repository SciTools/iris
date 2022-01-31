.. _whats_new_contributions:

=================================
Contributing a "What's New" Entry
=================================

Iris uses a file named ``dev.rst`` to keep a draft of upcoming development changes
that will form the next stable release.  Contributions to the :ref:`iris_whatsnew`
document are written by the developer most familiar with the change made.
The contribution should be included as part of the Iris Pull Request that
introduces the change.

The ``dev.rst`` and the past release notes are kept in the
``docs/src/whatsnew/`` directory. If you are writing the first contribution after
an Iris release: **create the new** ``dev.rst`` by copying the content from
``dev.rst.template`` in the same directory.

.. note::

  Ensure that the symbolic link ``latest.rst`` references the ``dev.rst`` file
  within the ``docs/src/whatsnew`` directory.

Since the `Contribution categories`_ include Internal changes, **all** Iris
Pull Requests should be accompanied by a "What's New" contribution.


Git Conflicts
=============

If changes to ``dev.rst`` are being suggested in several simultaneous
Iris Pull Requests, Git will likely encounter merge conflicts. If this
situation is thought likely (large PR, high repo activity etc.):

* PR author: Do not include a "What's New" entry. Mention in the PR text that a
  "What's New" entry is pending

* PR reviewer: Review the PR as normal. Once the PR is acceptable, ask that
  a **new pull request** be created specifically for the "What's New" entry,
  which references the main pull request and titled (e.g. for PR#9999):

    What's New for #9999

* PR author: create the "What's New" pull request

* PR reviewer: once the "What's New" PR is created, **merge the main PR**.
  (this will fix any `cirrus-ci`_ linkcheck errors where the links in the
  "What's New" PR reference new features introduced in the main PR)

* PR reviewer: review the "What's New" PR, merge once acceptable

These measures should mean the suggested ``dev.rst`` changes are outstanding
for the minimum time, minimising conflicts and minimising the need to rebase or
merge from trunk.


Writing a Contribution
======================

As introduced above, a contribution is the description of a change to Iris
which improved Iris in some way. As such, a single Iris Pull Request may
contain multiple changes that are worth highlighting as contributions to the
what's new document.

The appropriate contribution for a pull request might in fact be an addition or
change to an existing "What's New" entry.

Each contribution will ideally be written as a single concise entry using a
reStructuredText auto-enumerated list ``#.`` directive. Where possible do not
exceed **column 80** and ensure that any subsequent lines of the same entry are
aligned with the first. The content should target an Iris user as the audience.
The required content, in order, is as follows:

* Names of those who contributed the change. These should be their GitHub
  user name. Link the name to their GitHub profile. E.g.
  ```@tkknight <https://github.com/tkknight>`_ changed...``

* The new/changed behaviour

* Context to the change. Possible examples include: what this fixes, why
  something was added, issue references (e.g. ``:issue:`9999```), more specific
  detail on the change itself.

* Pull request references, bracketed, following the final period. E.g.
  ``(:pull:`1111`, :pull:`9999`)``

* A trailing blank line (standard reStructuredText list format)

For example::

  #. `@tkknight <https://github.com/tkknight>`_ changed changed argument ``x``
     to be optional in :class:`~iris.module.class` and
     :meth:`iris.module.method`. This allows greater flexibility as requested in
     :issue:`9999`. (:pull:`1111`, :pull:`9999`)


The above example also demonstrates some of the possible syntax for including
links to code. For more inspiration on possible content and references, please
examine past what's :ref:`iris_whatsnew` entries.

.. note:: The reStructuredText syntax will be checked as part of building
          the documentation.  Any warnings should be corrected.
          `cirrus-ci`_ will automatically build the documentation when
          creating a pull request, however you can also manually
          :ref:`build <contributing.documentation.building>` the documentation.

.. _cirrus-ci: https://cirrus-ci.com/github/SciTools/iris


Contribution Categories
=======================

The structure of the what's new release note should be easy to read by
users.  To achieve this several categories may be used.

**📢 Announcements**
  General news and announcements to the Iris community.

**✨ Features**
  Features that are new or changed to add functionality.

**🐛 Bug Fixes**
  A bug fix.

**💣 Incompatible Changes**
  A change that causes an incompatibility with prior versions of Iris.

**🔥 Deprecations**
  Deprecations of functionality.

**🔗 Dependencies**
  Additions, removals and version changes in Iris' package dependencies.

**📚 Documentation**
  Changes to documentation.

**💼 Internal**
  Changes to any internal or development related topics, such as testing,
  environment dependencies etc.
