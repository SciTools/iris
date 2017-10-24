.. _whats_new_contributions:

=================================
Contributing a "What's New" entry
=================================

Iris has an aggregator for building a draft what's new document for each
release. The draft what's new document is built from contributions by code authors.
This means contributions to the what's new document are written by the
developer most familiar with the change made.

A contribution provides an entry in the what's new document, which describes a
change that improved Iris in some way. This change may be a new feature in Iris
or the fix for a bug introduced in a previous release. The contribution should
be included as part of the Iris Pull Request that introduces the change.

When a new release is prepared, the what's new contributions are combined into
a draft what's new document for the release.


Writing a Contribution
======================

As introduced above, a contribution is the description of a change to Iris
which improved Iris in some way. As such, a single Iris Pull Request may
contain multiple changes that are worth highlighting as contributions to the
what's new document.

Each contribution will ideally be written as a single concise bullet point.
The content of the bullet point should highlight the change that has been made
to Iris, targeting an Iris user as the audience.

A contribution is a feature summary by the code author, which avoids the
release developer having to personally review the change in detail :
It is not in itself the final documentation content,
so it does not have to be perfect or complete in every respect.


Adding Contribution Files
=========================

Each release must have a directory called ``contributions_<release number>``,
which should be created following the release of the current version of Iris. Each
release directory must be placed in ``docs/iris/src/whatsnew/``.
Contributions to the what's new must be written in markdown and placed into this
directory in text files. The filename for each item should be structured as follows:

``<category>_<date>_<summary>.txt``

Category
--------
The category must be one of the following:

*newfeature*
  Features that are new or changed to add functionality.
*bugfix*
  A bugfix.
*incompatiblechange*
  A change that causes an incompatibility with prior versions of Iris.
*deprecate*
  Deprecations of functionality.
*docchange*
  Changes to documentation.

Date
----

The date must be a hyphen-separated date in the format of:

 * a four digit year,
 * a three character month name, and
 * a two digit day.

For example:

 * 2012-Jan-30
 * 2014-May-03
 * 2015-Feb-19

Summary
-------

The summary can be any remaining filename characters, and simply provides a
short identifying description of the change.

For example:

 * whats-new-aggregator
 * using_mo_pack
 * correction-to-bilinear-regrid
 * GRIB2_pdt11


Complete Examples
-----------------

Some sample what's new contribution filenames:

 * bugfix_2015-Aug-18_partial_pp_constraints.txt
 * deprecate_2015-Nov-01_unit-module.txt
 * incompatiblechange_2015-Oct-12_GRIB_optional_Python3_unavailable.txt
 * newfeature_2015-Jul-03_pearsonr_rewrite.txt

.. note::
    A test in the standard test suite ensures that all the contents of the
    latest contributions directory conform to this naming scheme.


Compiling a Draft
=================

Compiling a draft from the supplied contributions should be done when preparing
a release. Running ``docs/iris/src/whatsnew/aggregate_directory.py`` with the
release number as the argument will create a draft what's new with the name
``<release>.rst`` file for the specified release, by aggregating the individual
contributions from the relevant folder.
Omitting the release number will build the latest version for which a
contributions folder is present.
This command fails if a file with the relevant name already exists.

The resulting draft document is only a starting point, which the release
developer will then edit to produce the final 'What's new in Iris x.x'
documentation.
