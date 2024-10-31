.. include:: ../common_links.inc

.. _which_regridder_to_use:

======================
Which Regridder to Use
======================

This section compares all the regridding schemes which exist in `Iris`_, and
externally in `iris-esmf-regrid`_ with a view to helping you to choose the right
regridder for your workflow. The choice of regridder
is usually limited by the kind of data you are going from and to, but there are
also factors of performance and numerical accuracy to consider. This section
provides a reference for how each of the regridders differ with respect to
these factors, beginning with a set of short tables going into their differences
in brief and ending in a more in depth look at how these differences might
play out in different contexts.

For an introduction on using regridders, see the :ref:`user guide<interpolation_and_regridding>`.

Regridder Comparison
====================

We will highlight here some of the properties of each regridder in a table of
the following form:

+-----------------+-----------------------------------------------------------+
| **API**         | Link to API documentation.                                |
+-----------------+-----------------------------------------------------------+
| **Method**      | The type of algorithm used to calculate the result.       |
|                 | See section on `comparing methods`_.                      |
+-----------------+-----------------------------------------------------------+
| **Source Grid** | The type of **coordinates** required on the ``src`` cube. |
+-----------------+-----------------------------------------------------------+
| **Target Grid** | The type of **coordinates** required on the ``tgt`` cube. |
+-----------------+-----------------------------------------------------------+
| **Coordinate    | The type of **coordinate system** required on the         |
| System**        | ``src``/``tgt`` cube coordinates.                         |
+-----------------+-----------------------------------------------------------+
| **Lazy          | If the result is calculated lazily. See                   |
| Regridding**    | :doc:`real and lazy data </userguide/real_and_lazy_data>`.|
+-----------------+-----------------------------------------------------------+
| **Weights       | See `regridder performance`_.                             |
| Caching**       |                                                           |
+-----------------+-----------------------------------------------------------+
| **Notes**       | Additional details.                                       |
+-----------------+-----------------------------------------------------------+

AreaWeighted
------------

+-----------------+--------------------------------------------------------+
| **API**         | :class:`~iris.analysis.AreaWeighted`                   |
+-----------------+--------------------------------------------------------+
| **Method**      | Conservative                                           |
+-----------------+--------------------------------------------------------+
| **Source Grid** | Pair of 1D lat/lon coordinates, must have bounds.      |
+-----------------+--------------------------------------------------------+
| **Target Grid** | Pair of 1D lat/lon coordinates, must have bounds.      |
+-----------------+--------------------------------------------------------+
| **Coordinate    | Must be equal on ``src`` and ``tgt``, may be ``None``. |
| System**        |                                                        |
+-----------------+--------------------------------------------------------+
| **Lazy          | ``True``                                               |
| Regridding**    |                                                        |
+-----------------+--------------------------------------------------------+
| **Weights       | ``True``                                               |
| Caching**       |                                                        |
+-----------------+--------------------------------------------------------+
| **Notes**       | Supports masked data with ``mdtol`` argument.          |
|                 | See `area conservation`_.                              |
+-----------------+--------------------------------------------------------+

Linear
------

+-----------------+----------------------------------------------------------------+
| **API**         | :class:`~iris.analysis.Linear`                                 |
+-----------------+----------------------------------------------------------------+
| **Method**      | Linear                                                         |
+-----------------+----------------------------------------------------------------+
| **Source Grid** | Pair of 1D lat/lon coordinates.                                |
+-----------------+----------------------------------------------------------------+
| **Target Grid** | Pair of 1D lat/lon coordinates.                                |
+-----------------+----------------------------------------------------------------+
| **Coordinate    | May be present on both ``src`` and ``tgt`` or both be ``None``.|
| System**        | May be different.                                              |
+-----------------+----------------------------------------------------------------+
| **Lazy          | ``True``                                                       |
| Regridding**    |                                                                |
+-----------------+----------------------------------------------------------------+
| **Weights       | ``False``                                                      |
| Caching**       |                                                                |
+-----------------+----------------------------------------------------------------+
| **Notes**       | Supports extrapolation outside source data bounds.             |
+-----------------+----------------------------------------------------------------+

Nearest
-------

+-----------------+----------------------------------------------------------------+
| **API**         | :class:`~iris.analysis.Nearest`                                |
+-----------------+----------------------------------------------------------------+
| **Method**      | Nearest (destination to source)                                |
+-----------------+----------------------------------------------------------------+
| **Source Grid** | Pair of 1D lat/lon coordinates.                                |
+-----------------+----------------------------------------------------------------+
| **Target Grid** | Pair of 1D lat/lon coordinates.                                |
+-----------------+----------------------------------------------------------------+
| **Coordinate    | May be present on both ``src`` and ``tgt`` or both be ``None``.|
| System**        | May be different.                                              |
+-----------------+----------------------------------------------------------------+
| **Lazy          | ``True``                                                       |
| Regridding**    |                                                                |
+-----------------+----------------------------------------------------------------+
| **Weights       | ``False``                                                      |
| Caching**       |                                                                |
+-----------------+----------------------------------------------------------------+

UnstructuredNearest
-------------------

+-----------------+----------------------------------------------------+
| **API**         | :class:`~iris.analysis.UnstructuredNearest`        |
+-----------------+----------------------------------------------------+
| **Method**      | Nearest (destination to source)                    |
+-----------------+----------------------------------------------------+
| **Source Grid** | Pair of lat/lon coordinates with any dimensionality|
|                 | (e.g., 1D or 2D). Must be associated to the same   |
|                 | axes on the source cube.                           |
+-----------------+----------------------------------------------------+
| **Target Grid** | Pair of 1D lat/lon coordinates.                    |
+-----------------+----------------------------------------------------+
| **Coordinate    | Must be equal on ``src`` and ``tgt``, may be       |
| System**        | ``None``.                                          |
+-----------------+----------------------------------------------------+
| **Lazy          | ``False``                                          |
| Regridding**    |                                                    |
+-----------------+----------------------------------------------------+
| **Weights       | ``False``                                          |
| Caching**       |                                                    |
+-----------------+----------------------------------------------------+

PointInCell
-----------

+-----------------+----------------------------------------------------+
| **API**         | :class:`~iris.analysis.PointInCell`                |
+-----------------+----------------------------------------------------+
| **Method**      | Point in cell                                      |
+-----------------+----------------------------------------------------+
| **Source Grid** | Pair of lat/lon coordinates with any dimensionality|
|                 | (e.g., 1D or 2D). Must be associated to the same   |
|                 | axes on the source cube.                           |
+-----------------+----------------------------------------------------+
| **Target Grid** | Pair of 1D lat/lon coordinates, must have bounds.  |
+-----------------+----------------------------------------------------+
| **Coordinate    | Must be equal on ``srs`` and ``tgt``, may be       |
| System**        | ``None``.                                          |
+-----------------+----------------------------------------------------+
| **Lazy          | ``False``                                          |
| Regridding**    |                                                    |
+-----------------+----------------------------------------------------+
| **Weights       | ``True``                                           |
| Caching**       |                                                    |
+-----------------+----------------------------------------------------+

External Regridders
===================

ESMFAreaWeighted
----------------

+-----------------+-------------------------------------------------------------------------+
| **API**         | :class:`~iris-esmf-regrid:esmf_regrid.schemes.ESMFAreaWeighted`         |
+-----------------+-------------------------------------------------------------------------+
| **Method**      | Conservative                                                            |
+-----------------+-------------------------------------------------------------------------+
| **Source Grid** | May be either:                                                          |
|                 |                                                                         |
|                 | - A pair of 1D x/y coordinates on different axes. Must have bounds.     |
|                 | - A pair of 2D x/y coordinates on the same axes. Must have bounds.      |
|                 | - An unstructured mesh located on cell faces.                           |
+-----------------+-------------------------------------------------------------------------+
| **Target Grid** | Any of the above. May be a different type to ``src`` grid.              |
+-----------------+-------------------------------------------------------------------------+
| **Coordinate    | ``src`` and ``tgt`` grid may have any coordinate system or ``None``.    |
| System**        |                                                                         |
+-----------------+-------------------------------------------------------------------------+
| **Lazy          | ``True``                                                                |
| Regridding**    |                                                                         |
+-----------------+-------------------------------------------------------------------------+
| **Weights       | ``True``                                                                |
| Caching**       |                                                                         |
+-----------------+-------------------------------------------------------------------------+
| **Notes**       | Supports masked data with ``mdtol`` argument (see `area conservation`_).|
|                 | Differs numerically to :class:`~iris.analysis.AreaWeighted` due to      |
|                 | representing edges as great circle arcs rather than lines of            |
|                 | latitude/longitude. This causes less difference at higher resolutions.  |
|                 | This can be mitigated somewhat by using the                             |
|                 | ``src_resolution`` / ``tgt_resolution`` arguments.                      |
+-----------------+-------------------------------------------------------------------------+

ESMFBilinear
------------

+-----------------+---------------------------------------------------------------------+
| **API**         | :class:`~iris-esmf-regrid:esmf_regrid.schemes.ESMFBilinear`         |
+-----------------+---------------------------------------------------------------------+
| **Method**      | Linear                                                              |
+-----------------+---------------------------------------------------------------------+
| **Source Grid** | May be either:                                                      |
|                 |                                                                     |
|                 | - A pair of 1D x/y coordinates on different axes.                   |
|                 | - A pair of 2D x/y coordinates on the same axes.                    |
|                 | - An unstructured mesh located on cell faces.                       |
+-----------------+---------------------------------------------------------------------+
| **Target Grid** | Any of the above. May be a different type to ``src`` grid.          |
+-----------------+---------------------------------------------------------------------+
| **Coordinate    | ``src`` and ``tgt`` grid may have any coordinate system or ``None``.|
| System**        |                                                                     |
+-----------------+---------------------------------------------------------------------+
| **Lazy          | ``True``                                                            |
| Regridding**    |                                                                     |
+-----------------+---------------------------------------------------------------------+
| **Weights       | ``True``                                                            |
| Caching**       |                                                                     |
+-----------------+---------------------------------------------------------------------+

ESMFNearest
-----------

+-----------------+---------------------------------------------------------------------+
| **API**         | :class:`~iris-esmf-regrid:esmf_regrid.schemes.ESMFNearest`          |
+-----------------+---------------------------------------------------------------------+
| **Method**      | Nearest (destination to source)                                     |
+-----------------+---------------------------------------------------------------------+
| **Source Grid** | May be either:                                                      |
|                 |                                                                     |
|                 | - A pair of 1D x/y coordinates on different axes.                   |
|                 | - A pair of 2D x/y coordinates on the same axes.                    |
|                 | - An unstructured mesh located on cell faces                        |
+-----------------+---------------------------------------------------------------------+
| **Target Grid** | Any of the above. May be a different type to ``src`` grid.          |
+-----------------+---------------------------------------------------------------------+
| **Coordinate    | ``src`` and ``tgt`` grid may have any coordinate system or ``None``.|
| System**        |                                                                     |
+-----------------+---------------------------------------------------------------------+
| **Lazy          | ``True``                                                            |
| Regridding**    |                                                                     |
+-----------------+---------------------------------------------------------------------+
| **Weights       | ``True``                                                            |
| Caching**       |                                                                     |
+-----------------+---------------------------------------------------------------------+

.. _comparing methods:

Comparing Methods
=================

The various regridding algorithms are implementations of the following
methods. While there may be slight differences in the way each regridder
implements a given method, each regridder broadly follows the principles
of that method. We give here a very brief overview of what situations
each method are best suited to followed by a more detailed discussion.

Conservative
------------

Good for representing the *entirety* of the underlying data.
Designed for data represented by cell faces. A fuller description of
what it means to be *conservative* can be found in the section on
`area conservation`_.

Linear
------

Good for approximating data represented at *precise points* in space and in
cases where it is desirable for the resulting data to be smooth. For more
detail, see the section on `regridder smoothness`_.

Nearest
-------

Tends to be the fastest regridding method. Ensures each resulting data value
represents a data value in the source. Good in cases where averaging is
inappropriate, e.g., for discontinuous data.

Point in cell
-------------

Similarly to the conservative method, represents the entirety of the underlying
data. Works well with data whose source is an unstructured series of points.

.. _numerical accuracy:

Numerical Accuracy
==================

An important thing to understand when regridding is that no regridding method
is perfect. That is to say, you will tend to lose information when you regrid
so that if you were to regrid from a source grid to a target and then back onto
the original source, you will usually end up with slightly different data.
Furthermore, statistical properties such as min, max and standard deviation are
not guaranteed to be preserved. While regridding is inherently imperfect, there
are some properties which can be better preserved by choosing the appropriate
regridding method. These include:

.. _area conservation:

Global Area Weighted Average
----------------------------
Area weighted regridding schemes such as :class:`~iris.analysis.AreaWeighted` and
:class:`~iris-esmf-regrid:esmf_regrid.schemes.ESMFAreaWeighted`
use *conservative* regridding schemes. The property which these regridders
*conserve* is the global area weighted average of the data (or equivalently,
the area weighted sum). More precisely, this means that::

   When regridding from a source cube to a target cube defined
   over the same area (e.g., the entire globe), assuming there
   are no masked data points, the area weighted average
   (weighted by the area covered by each data point) of the
   source cube ought to be equal (within minor tolerances)
   to the area weighted average of the result.

This property will be particularly important to consider if you are intending to
calculate global properties such as average temperature or total rainfall over a
given area. It may be less important if you are only interested in local behaviour,
e.g., temperature at particular locations.

When there are masked points in your data, the same global conservative properties
no longer strictly hold. This is because the area which the unmasked points in the
source cover is no longer the same as the area covered by unmasked points in the
target. With the keyword argument ``mdtol=0`` this means that there will be an area
around the source mask which will be masked in the result and therefore unaccounted
for in the area weighted average calculation. Conversely, with the keyword argument
``mdtol=1`` there will be an unmasked area in the result that is masked in the source.
This may be particularly important if you are intending to calculate properties
which depend area e.g., calculating the total global rainfall based on data in units
of ``kg m-2`` as an area weighted sum. With ``mdtol=0`` this will consistently
underestimate this total and with ``mdtol=1`` will consistently overestimate. This can
be somewhat mitigated with a choice of ``mdtol=0.5``, but you should still be aware of
potential inaccuracies. It should be noted that this choice of ``mdtol`` is highly
context dependent and there will likely be occasions where a choice of ``mdtol=0`` or
``mdtol=1`` is more suitable. The important thing is to *know your data, know what*
*you're doing with your data and know how your regridder fits in this process*.

.. todo::

    add worked example

.. _regridder smoothness:

Data Gradient/Smoothness
------------------------
Alternatively, rather than conserving global properties, it may be more important to
approximate each individual point of data as accurately as possible. In this case, it
may be more appropriate to use a *linear* regridder such as :class:`~iris.analysis.Linear`
or :class:`~iris-esmf-regrid:esmf_regrid.schemes.ESMFBilinear`.

The linear method calculates each target point as the weighted average of the four
surrounding source points. This average is weighted according to how close this target
point is to the surrounding points. Notably, the value assigned to a target point varys
*continuously* with its position (as opposed to nearest neighbour regridding).

Such regridders work best when the data in question can be considered
as a collection of measurements made at *points on a smoothly varying field*. The
difference in behaviour between linear and conservative regridders can be seen most
clearly when there is a large difference between the source and target grid resolution.

Suppose you were regridding from a high resolution to a low resolution, if you were
regridding using a *conservative* method, each result point would be the average of many
result points. On the other hand, if you were using a *linear* method then the result
would only be the average the 4 nearest source points. This means that, while
*conservative* methods will give you a better idea of the *totality* of the source data,
*linear* methods will give you a better idea of the source data at a *particular point*.

Conversely, suppose you were regridding from a low resolution to a high resolution. For
other regridding methods (conservative and nearest), most of the target points covered by
a given source point would have the same value and there would be a steep difference between
target points near the cell boundary. For linear regridding however, the resulting data
will vary smoothly.

.. todo::

    add worked example

Consistency
-----------
As noted above, each regridding method has its own unique effect on the data. While this can
be manageable when contained within context of a particular workflow, you should take care
not to compare data which has been regrid with different regridding methods as the artefacts
of that regridding method may dominate the underlying differences.

.. todo::

    add worked example

It should also be noted that some implementations of the *same method* (e.g.,
:class:`~iris.analysis.Nearest` and :class:`~iris.analysis.UnstructuredNearest`) may
differ slightly and so may yield slightly different results when applied to equivalent
data. However this difference will be significantly less than the difference between
regridders based on different methods.

.. _regridder performance:

Performance
-----------
Regridding can be an expensive operation, but there are ways to work with regridders to
mitigate this cost. For most regridders, the regridding process can be broken down into
two steps:

- *Preparing* the regridder by comparing the source and target grids and generating weights.
- *Performing* the regridding by applying those weights to the source data.

Generally, the *prepare* step is the more expensive of the two. It is better to avoid
repeating this step unnecessarily. This can be done by *reusing* a regridder, as described
in the :ref:`user guide <caching_a_regridder>`.

.. todo::

    add benchmarks - note the iris and iris-esmf-regrid version
