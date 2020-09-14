========
Metadata
========

So far we've introduced several classes in Iris that care about your ``data``,
and also your ``metadata`` i.e., `data about data`_. Primarily, these classes
are the :class:`~iris.cube.Cube`, the :class:`~iris.coords.AuxCoord`, and the
:class:`~iris.coords.DimCoord`, all of which should be familiar to you now.

As discussed in :ref:`iris_data_structures`, Iris draws heavily from the
`NetCDF CF Metadata Conventions`_ as a source for its data model, thus building
on the widely recognised and understood terminology defined within those
`CF Conventions`_ by the scientific community.

Iris models several other classes of `CF Conventions`_ metadata apart from
those mentioned above. Such as the :class:`~iris.coords.AncillaryVariable`
(see `Ancillary Data`_ and `Flags`_), the :class:`~iris.coords.CellMeasure`
(see `Cell Measures`_), and also the :class:`~iris.aux_factory.AuxCoordFactory`
(see `Parametric Vertical Coordinate`_).

.. hint::

    If there are any `CF Conventions`_ metadata missing from Iris that you
    care about, then please let us know by raising a `GitHub Issue`_ on
    `SciTools/iris`_


Common Metadata
---------------

What each of these **different** Iris `CF Conventions`_ classes all have in
**common** is that ``metadata`` is used to define them and give them meaning.

.. _metadata members:
.. table:: - Iris classes that model `CF Conventions`_ metadata
   :widths: auto
   :align: center

   =================== ===================== ============ =================== =============== ======== ============
   Metadata members    ``AncillaryVariable`` ``AuxCoord`` ``AuxCoordFactory`` ``CellMeasure`` ``Cube`` ``DimCoord``
   =================== ===================== ============ =================== =============== ======== ============
   ``standard name``   ✔                     ✔            ✔                   ✔               ✔        ✔
   ``long name``       ✔                     ✔            ✔                   ✔               ✔        ✔
   ``var name``        ✔                     ✔            ✔                   ✔               ✔        ✔
   ``units``           ✔                     ✔            ✔                   ✔               ✔        ✔
   ``attributes``      ✔                     ✔            ✔                   ✔               ✔        ✔
   ``coord_system``                          ✔            ✔                                            ✔
   ``climatological``                        ✔            ✔                                            ✔
   ``measure``                                                                ✔
   ``cell_methods``                                                                           ✔
   ``circular``                                                                                        ✔
   =================== ===================== ============ =================== =============== ======== ============

:numref:`metadata members` shows for each Iris `CF Conventions`_ class the
collective of individual ``metadata`` members used to define it. Almost all
of these members reference specific `CF Conventions`_ terms. However, some
of these members, such as :attr:`~iris.coords.DimCoord.var_name` and
:attr:`~iris.coords.DimCoord.circular` are Iris specific terms.

For example, the collective ``metadata`` used to define an
:class:`~iris.coords.AncillaryVariable` are the ``standard_name``, ``long_name``,
``var_name``, ``units``, and ``attributes`` members. Note that, these are the
actual `data attribute`_ names of the actual ``metadata`` members on the actual
Iris class.

As :numref:`metadata members` highlights, **specific** metadata is used to
define and represent each **specific** Iris `CF Conventions`_ class. This means
that this **specific** metadata can then be used to easily **identify**,
**compare** and **differentiate** between individual class instances.


Common Metadata API
-------------------

.. testsetup::

    import iris
    cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))

As of Iris ``3.0.0``, a unified treatment of ``metadata`` has been applied
across each Iris class in :numref:`metadata members` to allow users to easily
manage and manipulate their ``metadata`` in a consistent way.

This is achieved through the ``metadata`` property, which allows you to
manipulate the associated underlying ``metadata`` members as a collective.
For example, given the following :class:`~iris.cube.Cube`:

    >>> print(cube)
    air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)
         Dimension coordinates:
              time                           x              -              -
              latitude                       -              x              -
              longitude                      -              -              x
         Auxiliary coordinates:
              forecast_period                x              -              -
         Scalar coordinates:
              forecast_reference_time: 1859-09-01 06:00:00
              height: 1.5 m
         Attributes:
              Conventions: CF-1.5
              Model scenario: A1B
              STASH: m01s03i236
              source: Data from Met Office Unified Model 6.05
         Cell methods:
              mean: time (6 hour)

We can easily get all of the associated ``metadata`` of the :class:`~iris.cube.Cube`:

    >>> cube.metadata
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))

We can also inspect the ``metadata`` of the ``longitude``
:class:`~iris.coords.DimCoord` attached to the :class:`~iris.cube.Cube`:

    >>> cube.coord("longitude").metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Or use the ``metadata`` property again, but this time on the ``forecast_period``
:class:`~iris.coords.AuxCoord` attached to the :class:`~iris.cube.Cube`:

    >>> cube.coord("forecast_period").metadata
    CoordMetadata(standard_name='forecast_period', long_name=None, var_name='forecast_period', units=Unit('hours'), attributes={}, coord_system=None, climatological=False)

The ``metadata`` property will return an appropriate `namedtuple`_ metadata class
for each Iris `CF Conventions`_ class container. :numref:`metadata classes` namely:

.. _metadata classes:
.. table:: - Iris namedtuple metadata classes
   :widths: auto
   :align: center

   ===================== ========================================================
   Container class       Namedtuple metadata class
   ===================== ========================================================
   ``AncillaryVariable`` :class:`~iris.common.metadata.AncillaryVariableMetadata`
   ``AuxCoord``          :class:`~iris.common.metadata.CoordMetadata`
   ``AuxCoordFactory``   :class:`~iris.common.metadata.CoordMetadata`
   ``CellMeasure``       :class:`~iris.common.metadata.CellMeasureMetadata`
   ``Cube``              :class:`~iris.common.metadata.CubeMetadata`
   ``DimCoord``          :class:`~iris.common.metadata.DimCoordMetadata`
   ===================== ========================================================

As per the behaviour of a `namedtuple`_, the metadata classes in
:numref:`metadata classes` create tuple-like instances i.e., they provide a
**snapshot** of the associated metadata member **values**, which are **not**
settable, but they **may** be mutable depending on the data-type e.g.,

    >>> longitude = cube.coord("longitude")
    >>> metadata = longitude.metadata
    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

    >>> # The metadata member value is the instance member value...
    >>> metadata.attributes is longitude.attributes
    True
    >>> metadata.circular is longitude.circular
    True

    >>> # Metadata members with dictionaries are mutable...
    >>> longitude.attributes["grinning face"] = "😀"
    >>> longitude.attributes
    {'grinning face': '😀'}
    >>> metadata.attributes
    {'grinning face': '😀'}
    >>> metadata.attributes["grinning face"] = "😱"
    >>> longitude.attributes
    {'grinning face': '😱'}

    >>> # Metadata members with simple values are not mutable...
    >>> longitude.circular
    False
    >>> longitude.circular = True
    >>> longitude.circular
    True
    >>> metadata.circular
    False
    >>> metadata.circular is longitude.circular
    False

    >>> # The metadata property re-creates a new "snapshot" instance per invocation...
    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '😱'}, coord_system=GeogCS(6371229.0), climatological=False, circular=True)


Namedtuple metadata behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The metadata classes in :numref:`metadata classes` inherit the behaviour of a `namedtuple`_, e.g.,

    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '😱'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

    >>> # Create a new instance with the provided values...
    >>> values = (1, 2, 3, 4, 5, 6, 7, 8)
    >>> metadata._make(values)
    DimCoordMetadata(standard_name=1, long_name=2, var_name=3, units=4, attributes=5, coord_system=6, climatological=7, circular=8)

    >>> # Return a new dictionary which maps member names to their corresponding values...
    >>> metadata._asdict()
    OrderedDict([('standard_name', 'longitude'), ('long_name', None), ('var_name', 'longitude'), ('units', Unit('degrees')), ('attributes', {'grinning face': '😱'}), ('coord_system', GeogCS(6371229.0)), ('climatological', False), ('circular', False)])

    >>> # Return a new instance replacing the specified members with new values...
    >>> metadata._replace(standard_name=1, units=4)
    DimCoordMetadata(standard_name=1, long_name=None, var_name='longitude', units=4, attributes={'grinning face': '😱'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

    >>> # Returns a tuple of strings listing the member names...
    >>> metadata._fields
    ('standard_name', 'long_name', 'var_name', 'units', 'attributes', 'coord_system', 'climatological', 'circular')


Richer metadata behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~


Metadata equality
+++++++++++++++++


Metadata combination
++++++++++++++++++++


Metadata difference
+++++++++++++++++++


Metadata assignment
+++++++++++++++++++


Metadata conversion
+++++++++++++++++++





Lenient metadata behaviour
~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _data about data: https://en.wikipedia.org/wiki/Metadata
.. _data attribute: https://docs.python.org/3/tutorial/classes.html#instance-objects
.. _Ancillary Data: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#ancillary-data
.. _CF Conventions: https://cfconventions.org/
.. _Cell Measures: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-measures
.. _Flags: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
.. _GitHub Issue: https://github.com/SciTools/iris/issues/new/choose
.. _namedtuple: https://docs.python.org/3/library/collections.html#collections.namedtuple
.. _NetCDF CF Metadata Conventions: https://cfconventions.org/
.. _Parametric Vertical Coordinate: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-vertical-coordinate
.. _SciTools/iris: https://github.com/SciTools/iris#-----