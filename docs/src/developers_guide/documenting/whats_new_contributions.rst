.. include:: ../../common_links.inc

.. _whats_new_contributions:

=================================
Contributing a "What's New" Entry
=================================

Please include a "What's New" contribution in
``docs/src/whatsnew/latest.rst`` for **any** change that you make to Iris.
**Even if it is not relevant to users** - the `Contribution categories`_
include ``Internal`` for this - the page is read by contributors as well as
users, and it reveals the work needed to keep a project going.

See this docs section for all What's New pages: :ref:`iris_whatsnew`.

What Should it Look Like?
=========================

It should describe your change in a few sentences, with particular focus on
what the change means for users who might read this. For formatting guidance:
hundreds of examples can be found in existing ``docs/src/whatsnew/`` files,
or read the `Detail`_ section below for precise instructions.

.. hint::

    Our standard format includes the number of the pull request making the
    change. If you have not yet created the pull request, you can work out
    what the next PR number (i.e. your number) will be using this command::

      $ curl -s "https://api.github.com/repos/SciTools/iris/issues?sort=created&direction=desc&per_page=1" | jq -r '.[0].number + 1'

Git Conflicts
=============

Because every pull request includes a What's New entry, there are often
conflicts for the ``latest.rst`` file. Thankfully What's New files are simple!
GitHub's '`Resolve conflicts`_' button on the pull request provides an easy
interface for fixing these. Or feel free to use a different approach if you
prefer.

**If you are unsure, say so in a comment on your pull request and the Iris
development team will be happy to help.**

Detail
======

Iris uses a file named ``latest.rst`` to keep a draft of upcoming development
changes that will form the next stable release.  Contributions to the
:ref:`iris_whatsnew` document are written by the developer most familiar
with the change made.  The contribution should be included as part of
the Iris Pull Request that introduces the change.

The ``latest.rst`` and the past release notes are kept in the
``docs/src/whatsnew/`` directory.

Writing a Contribution
----------------------

A contribution is the short description of a change introduced to Iris
which improved it in some way. As such, a single Iris Pull Request may
contain multiple changes that are worth highlighting as contributions to the
what's new document.

The appropriate contribution for a pull request might in fact be an addition or
change to an existing "What's New" entry.

Each contribution will ideally be written as a single concise entry using a
reStructuredText auto-enumerated list ``#.`` directive. Where possible do not
exceed **column 80** and ensure that any subsequent lines of the same entry are
aligned with the first. The content should target an Iris user as the audience.
The required content, in order, is as follows:

* Use your discretion to decide on the names of all those that you want to
  acknowledge as part of your contribution. Also consider the efforts of the
  reviewer. Please use GitHub user names that link to their GitHub profile
  e.g.,

  ```@tkknight`_ Lorem ipsum dolor sit amet ...``

  Also add a full reference in the following section at the end of the ``latest.rst``::

    .. comment
       Whatsnew author names (@github name) in alphabetical order. Note that,
       core dev names are automatically included by the common_links.inc:

    .. _@tkknight: https://github.com/tkknight

  .. hint::

    Alternatively adopt the ``:user:`` `extlinks`_ convenience instead.

    For example to reference the ``github`` user ``tkknight`` simply use
    :literal:`:user:\`tkknight\``.

    This will be rendered as :user:`tkknight`.

    In addition, there is now no need to add a full reference to the user within
    the documentation.

* A succinct summary of the new/changed behaviour.

* Context to the change. Possible examples include: what this fixes, why
  something was added, issue references (e.g. ``:issue:`9999```), more specific
  detail on the change itself.

* Pull request references, bracketed, following the final period e.g.,
  ``(:pull:`1111`, :pull:`9999`)``

* A trailing blank line (standard reStructuredText list format).

For example::

  #. `@tkknight <https://github.com/tkknight>`_ and
     `@trexfeathers <https://github.com/trexfeathers>`_ (reviewer) changed
     argument ``x`` to be optional in :class:`~iris.module.class` and
     :meth:`iris.module.method`. This allows greater flexibility as requested in
     :issue:`9999`. (:pull:`1111`, :pull:`9999`)


The above example also demonstrates some of the possible syntax for including
links to code. For more inspiration on possible content and references, please
examine past what's :ref:`iris_whatsnew` entries.

.. note:: The reStructuredText syntax will be checked as part of building
          the documentation.  Any warnings should be corrected. The
          `Iris GitHub Actions`_ will automatically build the documentation when
          creating a pull request, however you can also manually
          :ref:`build <contributing.documentation.building>` the documentation.

Contribution Categories
-----------------------

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

Making a File
-------------

This is usually handled as part of the :ref:`iris_development_releases` process.
But if you are making the first contribution to a new minor or major release,
and you find that no ``docs/src/whatsnew/latest.rst`` file exists:
**create the new** ``latest.rst`` by copying the content from
``latest.rst.template`` in the same directory.


.. _Resolve conflicts: https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-on-github
