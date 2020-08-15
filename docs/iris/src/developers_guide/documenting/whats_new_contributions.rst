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
in a reStructuredText format with a trailing blank line.  For example::

  * Fixed :issue:`9999`.  Lorem ipsum dolor sit amet, consectetur adipiscing 
    elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
    <blank line>

Note that this example also cites the related issue, optionally you may also
include the pull request using the notation ``:pull:`9999```.  Where possible
do not exceed **column 80** and ensure that any subsequent lines
of the same bullet point is aligned with the first.  

The content of the bullet point should highlight the change that has been made
to Iris, targeting an Iris user as the audience.

For inspiration that may include adding links to code please examine past
what's :ref:`iris_whatsnew` entries.  

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

*Internal*
  Changes to any internal or development related topics, such as testing,
  environment dependencies etc

*Deprecations*
  Deprecations of functionality.

*Documentation*
  Changes to documentation.
