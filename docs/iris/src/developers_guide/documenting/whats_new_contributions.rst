=================================
Contributing a "What's New" entry
=================================

Iris has an aggregator for building a draft "What's New" document for each
release, from sections contributed by code authors. This means entries for the
release documentation are written by the developer most familiar with the
change made. 

Creating a Contribution File
============================

Each release must have a directory called ``contributions_<release number>``,
which should be created following the release of the current version. Each
release directory must be placed in ``docs/iris/src/whatsnew/``. Items for
inclusion into the release documentation should be placed into this directory
in RST format.
The filename for each item should be structured as follows:

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

Each contribution should be written in the form of one concise bullet point.
This bullet point should describe the change that has been made, to an audience
of an Iris user.

Compiling a Draft
=================

Compiling a draft from the supplied contributions should be done when preparing
a release. Running ``docs/iris/src/whatsnew/aggregate_directory.py`` with the
release number as the argument will check for the existence of a 
``<release>.rst`` file for the specified release. If none exists it will create
one by aggregating the individual contributions from the relevant folder.
This document can then be edited by the developer performing the release.


