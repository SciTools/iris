.. _iris-philosophy:

****************
Iris' Philosophy
****************

.. _code-maintenance:

Code Maintenance
================

From a user point of view "code maintenance" means ensuring that your existing
working code stays working, in the face of changes to Iris.


Stability and Change
---------------------

In practice, as Iris develops, most users will want to periodically upgrade
their installed version to access new features or at least bug fixes.

This is obvious if you are still developing other code that uses Iris, or using
code from other sources.
However, even if you have only legacy code that remains untouched, some code
maintenance effort is probably still necessary:

* On the one hand, *in principle*, working code will go on working, as long
  as you don't change anything else.

* However, such "version stasis" can easily become a growing burden; if you
  are simply waiting until an update becomes unavoidable, often that will
  eventually occur when you need to update some other software component
  for some completely unconnected reason.


Principles of Change Management
-------------------------------

When you upgrade software to a new version, you often find that you need to
rewrite your legacy code, simply to keep it working.

In Iris, however, we aim to reduce code maintenance problems to an absolute
minimum by following defined change management rules.
These ensure that, *within a major release number* :

* you can be confident that your code will still work with subsequent minor
  releases

* you will be aware of future incompatibility problems in advance

* you can defer making code compatibility changes for some time, until it
  suits you

The above applies to *minor version upgrades* : e.g. code that works with version
"1.4.2" should still work with a subsequent minor release such as "1.5.0" or
"1.7.2".

A *major* release however, e.g. "v2.0.0" or "v3.0.0", can include more
significant changes, including so-called "breaking" changes:  This means that
existing code may need to be modified to make it work with the new version.

Since breaking change can only occur at major releases, these are the *only*
times we can alter or remove existing behaviours (even deprecated
ones).  This is what a major release is for: it enables the removal and
replacement of old features.

Of course, even at a major release, we do still aim to keep breaking changes to
a minimum.

.. _load-problems-explanation:

Loading Invalid File Content
============================

As discussed in :ref:`load-problems`, Iris will not attempt to load file content
that is malformed or non-conformant with relevant standards, instead redirecting
the content to :data:`iris.loading.LOAD_PROBLEMS`.
In many cases, a sensible workaround for loading 'problem content' would be
obvious, especially given the flexibility of the Iris data model. But instead,
this stricter approach from Iris on file quality has several benefits:

(See also: :issue:`5165`).

Raised Awareness
----------------

The Iris developers aspire to a world with maximum file compatibility - where
files can be correctly parsed by different parties and even different
software, without the need for caveats, notes, or workarounds. This is why Iris
conforms to file standards wherever they are available:
:term:`CF conventions`, :ref:`UGRID<ugrid>`, :term:`GRIB Format`,
etcetera.

Iris makes users aware of any non-compliance with a file standard by not
loading it directly into the data model (instead redirecting to
:data:`iris.loading.LOAD_PROBLEMS`). Awareness gives data consumers and
providers the opportunity to collaborate on improving file quality, increasing
the ease with which the file can be loaded by ANY appropriate software.

(Any workarounds or 'agnosticism' would only increase the ease of *Iris*
loading the file, hiding the fact that other software and other collaborators
might not understand it).

Maintainability
---------------

Well written standards allow the loading code to be written with assumptions
about what file content to expect. This code is much simpler than either fully
'agnostic' code which can load anything, or code which embeds various
workarounds for known problems. Simpler code takes less resource/expertise to
maintain, increasing the long-term sustainability of Iris.

Robustness
----------

Redirecting problem content to :data:`iris.loading.LOAD_PROBLEMS` occurs in
places where Iris would otherwise raise an exception. This means that
Iris can continue to load all the valid parts of the file, and the user has
a way to fix problems **within Iris**, rather than learning a NetCDF tool or
similar.

This will not only handle file problems, but also any current or future bugs in
the Iris codebase, until they are fixed in the next release.

User Discretion
---------------

File malformations/non-conformances are by-definition not covered by any
standard for that file type - there is no consensus on the correct way to
represent this information. By avoiding encoding workarounds into Iris'
codebase, we avoid imposing one party's opinion onto other Iris users, who may
believe the problem should be handled differently.

Diversity
---------

Several less 'opinionated' libraries are already available for those users that
want to load all content from their file, regardless of quality or meaning.
These libraries give the user the freedom to customise the handling of their
files as they see fit, but also put the onus on the user to understand the file
content and write code to handle it. Iris would be adding little new to the
ecosystem if it had an identical philosophy.

Examples include: :term:`netCDF4<NetCDF Format>`, :term:`Xarray`, `ecCodes`_.

Instead, when working with the Iris data model, users can be confident in
the validity, and precise meaning (from the :term:`CF conventions`) of this
information.

.. _filtering-warnings-explanation:

Verbose Warnings
================

Different people use Iris for very different purposes, from quick file
visualisation to extract-transform-load to statistical analysis. These
contrasting priorities mean disagreement on which Iris problems can be ignored
and which are critically important.

For problems that prevent Iris functioning: **Concrete Exceptions** are raised, which
stop code from running any further - no debate here. For less catastrophic
problems: **Warnings** are raised,
which notify you (in ``stderr``) but allow code to continue running. The Warnings are
there because Iris may **OR may not** function in the way you expect,
depending on what you need - e.g. a problem might prevent data being saved to
NetCDF, but statistical analysis will still work fine.

This means that Iris' default behaviour is to raise Warnings
for anything that might be a problem for **any** user, since it cannot predict
specific user needs. It is designed to work with the user to ``ignore`` Warnings
which are not considered helpful in their specific use case.

**How to ignore unwanted warnings:** :ref:`filtering-warnings`

Examples of Iris Warnings
-------------------------

- If you attempt to plot un-bounded point data as a ``pcolormesh``: Iris will
  guess appropriate bounds around each point so that quadrilaterals can be
  plotted. This permanently modifies the relevant coordinates, so the you are
  warned in case downstream operations assume un-bounded coordinates.
- If you load a NetCDF file where a CF variable references another variable -
  e.g. ``my_var:coordinates = "depth_var" ;`` - but the referenced variable
  (``depth_var``) is not in the file: Iris will still construct
  its data model, but without this reference relationship. You are warned since
  the file includes an error and the loaded result might therefore not be as
  expected.

.. _ecCodes: https://github.com/ecmwf/eccodes

Planning Procedures
===================

Planning procedures encompass some of the methods Iris' core developers make
decisions about how and what changes are made to Iris, and how we go about
prioritising these.

User Experience
---------------

Often, improving and updating the existing user experience can fall behind fixing create new features,
or quashing pesky bugs. To combat this, we plan to have regular development discussions to ensure
UX doesn't fall behind. These pages offer some insight into what sort of things we might be discussing,
and hopefully offers some extra transparency behind our development process.

See our more detailed page here: :ref:`ux_guide`.
