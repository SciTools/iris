.. _lenient maths:

Lenient Cube Maths
******************

This section provides an overview of lenient cube maths. In particular, it explains
what lenient maths involves, clarifies how it differs from normal or strict cube
maths, and demonstrates how you can exercise fine control over whether your cube
maths operations are lenient or strict.

Note that, lenient cube maths is the default behaviour of Iris from version
``3.0.0``.


Introduction
============

Lenient maths stands somewhat on the shoulders of giants. If you've not already
done so, you may want to recap the material discussed in the following sections,

- :ref:`cube maths`,
- :ref:`metadata`,
- :ref:`lenient metadata`

In addition to this, cube maths leans heavily on the :mod:`~iris.common.resolve`
module, which provides the necessary infrastructure required by Iris to analyse
and combine each :class:`~iris.cube.Cube` operand involved in a maths operation
into the resultant :class:`~iris.cube.Cube`. It may be worth while investing
some time to understand how the :class:`~iris.common.resolve.Resolve` class
underpins cube maths, and consider how it may be used in general to combine
or resolve cubes together.

Given these prerequisites, recall that :ref:`lenient behaviour <lenient behaviour>`
introduced and discussed the concept of lenient metadata; a more pragmatic and
forgiving approach to :ref:`comparing <lenient equality>`,
:ref:`combining <lenient combination>` and understanding the
:ref:`differences <lenient difference>` between your metadata
(:numref:`metadata members table`). The lenient metadata philosophy introduced
there is extended to cube maths, with the view to also preserving as much common
coordinate (:numref:`metadata classes table`) information, as well as common
metadata, between the participating :class:`~iris.cube.Cube` operands as possible.

Let's consolidate our understanding of lenient and strict cube maths through
a practical worked example, which we'll explore together next.


.. _lenient example:

Lenient Example
===============

.. testsetup:: lenient-example

   import iris
   from iris.common import LENIENT
   experiment = iris.load_cube(iris.sample_data_path("hybrid_height.nc"), "air_potential_temperature")
   control = experiment[0]
   control.remove_aux_factory(control.aux_factory())
   for coord in ["sigma", "forecast_reference_time", "forecast_period", "atmosphere_hybrid_height_coordinate", "surface_altitude"]:
       control.remove_coord(coord)
   control.attributes["Conventions"] = "CF-1.7"
   experiment.attributes["experiment-id"] = "RT3 50"

Consider the following :class:`~iris.cube.Cube` of ``air_potential_temperature``,
which has an `atmosphere hybrid height parametric vertical coordinate`_, and
represents the output of an low-resolution global atmospheric ``experiment``,

.. doctest:: lenient-example

    >>> print(experiment)
    air_potential_temperature / (K)             (model_level_number: 15; grid_latitude: 100; grid_longitude: 100)
        Dimension coordinates:
            model_level_number                                     x                  -                    -
            grid_latitude                                          -                  x                    -
            grid_longitude                                         -                  -                    x
        Auxiliary coordinates:
            atmosphere_hybrid_height_coordinate                    x                  -                    -
            sigma                                                  x                  -                    -
            surface_altitude                                       -                  x                    x
        Derived coordinates:
            altitude                                               x                  x                    x
        Scalar coordinates:
            forecast_period                     0.0 hours
            forecast_reference_time             2009-09-09 17:10:00
            time                                2009-09-09 17:10:00
        Attributes:
            Conventions                         'CF-1.5'
            STASH                               m01s00i004
            experiment-id                       'RT3 50'
            source                              'Data from Met Office Unified Model 7.04'

Consider also the following :class:`~iris.cube.Cube`, which has the same global
spatial extent, and acts as a ``control``,

.. doctest:: lenient-example

    >>> print(control)
    air_potential_temperature / (K)     (grid_latitude: 100; grid_longitude: 100)
        Dimension coordinates:
            grid_latitude                             x                    -
            grid_longitude                            -                    x
        Scalar coordinates:
            model_level_number          1
            time                        2009-09-09 17:10:00
        Attributes:
            Conventions                 'CF-1.7'
            STASH                       m01s00i004
            source                      'Data from Met Office Unified Model 7.04'

Now let's subtract these cubes in order to calculate a simple ``difference``,

.. doctest:: lenient-example

    >>> difference = experiment - control
    >>> print(difference)
    unknown / (K)                               (model_level_number: 15; grid_latitude: 100; grid_longitude: 100)
        Dimension coordinates:
            model_level_number                                     x                  -                    -
            grid_latitude                                          -                  x                    -
            grid_longitude                                         -                  -                    x
        Auxiliary coordinates:
            atmosphere_hybrid_height_coordinate                    x                  -                    -
            sigma                                                  x                  -                    -
            surface_altitude                                       -                  x                    x
        Derived coordinates:
            altitude                                               x                  x                    x
        Scalar coordinates:
            forecast_period                     0.0 hours
            forecast_reference_time             2009-09-09 17:10:00
            time                                2009-09-09 17:10:00
        Attributes:
            experiment-id                       'RT3 50'
            source                              'Data from Met Office Unified Model 7.04'

Note that, cube maths automatically takes care of broadcasting the
dimensionality of the ``control`` up to that of the ``experiment``, in order to
calculate the ``difference``. This is performed only after ensuring that both
the **dimension coordinates** ``grid_latitude`` and ``grid_longitude`` are first
:ref:`leniently equivalent <lenient equality>`.

As expected, the resultant ``difference`` contains the
:class:`~iris.aux_factory.HybridHeightFactory` and all it's associated **auxiliary
coordinates**. However, the **scalar coordinates** have been leniently combined to
preserve as much coordinate information as possible, and the ``attributes``
dictionaries have also been leniently combined. In addition, see what further
:ref:`rationalisation <sanitise metadata>` is always performed by cube maths on
the resultant metadata and coordinates.

Also, note that the ``model_level_number`` **scalar coordinate** from the
``control`` has be superseded by the similarly named **dimension coordinate**
from the ``experiment`` in the resultant ``difference``.

Now let's compare and contrast this lenient result with the strict alternative.
But before we do so, let's first clarify how to control the behaviour of cube maths.


Control the Behaviour
=====================

As stated earlier, lenient cube maths is the default behaviour from Iris ``3.0.0``.
However, this behaviour may be controlled via the thread-safe ``LENIENT["maths"]``
runtime option,

.. doctest:: lenient-example

    >>> from iris.common import LENIENT
    >>> print(LENIENT)
    Lenient(maths=True)

Which may be set and applied globally thereafter for Iris within the current
thread of execution,

.. doctest:: lenient-example

    >>> LENIENT["maths"] = False  # doctest: +SKIP
    >>> print(LENIENT)  # doctest: +SKIP
    Lenient(maths=False)

Or alternatively, temporarily alter the behaviour of cube maths only within the
scope of the ``LENIENT`` `context manager`_,

.. doctest:: lenient-example

    >>> print(LENIENT)
    Lenient(maths=True)
    >>> with LENIENT.context(maths=False):
    ...     print(LENIENT)
    ...
    Lenient(maths=False)
    >>> print(LENIENT)
    Lenient(maths=True)


Strict Example
==============

Now that we know how to control the underlying behaviour of cube maths,
let's return to our :ref:`lenient example <lenient example>`, but this
time perform **strict** cube maths instead,

.. doctest:: lenient-example

    >>> with LENIENT.context(maths=False):
    ...     difference = experiment - control
    ...
    >>> print(difference)
    unknown / (K)                               (model_level_number: 15; grid_latitude: 100; grid_longitude: 100)
        Dimension coordinates:
            model_level_number                                     x                  -                    -
            grid_latitude                                          -                  x                    -
            grid_longitude                                         -                  -                    x
        Auxiliary coordinates:
            atmosphere_hybrid_height_coordinate                    x                  -                    -
            sigma                                                  x                  -                    -
            surface_altitude                                       -                  x                    x
        Derived coordinates:
            altitude                                               x                  x                    x
        Scalar coordinates:
            time                                2009-09-09 17:10:00
        Attributes:
            source                              'Data from Met Office Unified Model 7.04'

Although the numerical result of this strict cube maths operation is identical,
it is not as rich in metadata as the :ref:`lenient alternative <lenient example>`.
In particular, it does not contain the ``forecast_period`` and ``forecast_reference_time``
**scalar coordinates**, or the ``experiment-id`` in the ``attributes`` dictionary.

This is because strict cube maths, in general, will only return common metadata
and common coordinates that are :ref:`strictly equivalent <strict equality>`.


Finer Detail
============

In general, if you want to preserve as much metadata and coordinate information as
possible during cube maths, then opt to use the default lenient behaviour. Otherwise,
favour the strict alternative if you require to enforce precise metadata and
coordinate commonality.

The following information may also help you decide whether lenient cube maths best
suits your use case,

- lenient behaviour uses :ref:`lenient equality <lenient equality>` to match the
  metadata of coordinates, which is more tolerant to certain metadata differences,
- lenient behaviour uses :ref:`lenient combination <lenient combination>` to create
  the metadata of coordinates on the resultant :class:`~iris.cube.Cube`,
- lenient behaviour will attempt to cover each dimension with a :class:`~iris.coords.DimCoord`
  in the resultant :class:`~iris.cube.Cube`, even though only one :class:`~iris.cube.Cube`
  operand may describe that dimension,
- lenient behaviour will attempt to include **auxiliary coordinates** in the
  resultant :class:`~iris.cube.Cube` that exist on only one :class:`~iris.cube.Cube`
  operand,
- lenient behaviour will attempt to include **scalar coordinates** in the
  resultant :class:`~iris.cube.Cube` that exist on only one :class:`~iris.cube.Cube`
  operand,
- lenient behaviour will add a coordinate to the resultant :class:`~iris.cube.Cube`
  with **bounds**, even if only one of the associated matching coordinates from the
  :class:`~iris.cube.Cube` operands has **bounds**,
- strict and lenient behaviour both require that the **points** and **bounds** of
  matching coordinates from :class:`~iris.cube.Cube` operands must be strictly
  equivalent. However, mismatching **bounds** of **scalar coordinates** are ignored
  i.e., a scalar coordinate that is common to both :class:`~iris.cube.Cube` operands, with
  equivalent **points** but different **bounds**, will be added to the resultant
  :class:`~iris.cube.Cube` with but with **no bounds**

.. _sanitise metadata:

Additionally, cube maths will always perform the following rationalisation of the
resultant :class:`~iris.cube.Cube`,

- clear the ``standard_name``, ``long_name`` and ``var_name``, defaulting the
  :meth:`~iris.common.mixin.CFVariableMixin.name` to ``unknown``,
- clear the :attr:`~iris.cube.Cube.cell_methods`,
- clear the :meth:`~iris.cube.Cube.cell_measures`,
- clear the :meth:`~iris.cube.Cube.ancillary_variables`,
- clear the ``STASH`` key from the :attr:`~iris.cube.Cube.attributes` dictionary,
- assign the appropriate :attr:`~iris.common.mixin.CFVariableMixin.units`


.. _atmosphere hybrid height parametric vertical coordinate: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#atmosphere-hybrid-height-coordinate
.. _context manager: https://docs.python.org/3/library/contextlib.html
