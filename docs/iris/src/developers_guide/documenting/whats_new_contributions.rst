=================================
Contributing a "What's New" entry
=================================

Iris has an aggregator for building a draft what's new document for each
release. The draft what's new document is built from contributions by code authors.
This means contributions to the what's new document are written by the
developer most familiar with the change made.

A contribution is an entry in the what's new document that describes a change to
Iris that improved Iris in some way. This change may be a new feature in Iris or
the fix for a bug introduced in a previous release. The contribution should be
included as part of the Iris Pull Request that introduces the change to Iris.

Creating a Contribution File
============================

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

The date must be a hyphen-separated ISO8601 date in the format of:

 * a four digit year,
 * a two digit month, and
 * a two digit day.

For example:

 * 2012-01-30
 * 2014-05-03
 * 2015-03-19

Summary
-------

The summary should be short and separated by hyphens.

For example:

 * whats-new-aggregator
 * netcdf-streaming
 * correction-to-bilinear-regrid
 * removed-stash-translation

Writing a Contribution
======================

As introduced above, a contribution is the description of a change to Iris that
improved Iris in some way. As such, a single Iris Pull Request may contain multiple
changes that are worth highlighting as contributions to the what's new document.

Each contribution will ideally be written as a single concise bullet point.
The content of the bullet point should highlight the change that has been made
to Iris, targeting an Iris user as the audience.

Compiling a Draft
=================

Compiling a draft from the supplied contributions should be done when preparing
a release. Running ``docs/iris/src/whatsnew/aggregate_directory.py`` with the
release number as the argument will check for the existence of a 
``<release>.rst`` file for the specified release. If none exists it will create
one by aggregating the individual contributions from the relevant folder.
This document can then be edited by the developer performing the release.


