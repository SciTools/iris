.. _further topics:
.. _metadata:

Metadata
********

This section provides a detailed overview of how your metadata is managed
within Iris. In particular, it discusses what metadata is, where it fits
into Iris, and more importantly how you can create, access, manipulate,
and analyse your metadata.

All the finer details covered here may not be entirely relevant to your use
case, but it's here if you ever need it. In fact, you may want to skip
straight ahead to :ref:`richer metadata`, and take it from there.


Introduction
============

As discussed in :ref:`iris_data_structures`, Iris draws heavily from the
`NetCDF CF Metadata Conventions`_ as a source for its data model, thus building
on the widely recognised and understood terminology defined within those
`CF Conventions`_ by the scientific community.

In :ref:`iris_data_structures` we introduced several fundamental classes in Iris
that care about your ``data``, and also your ``metadata`` i.e., `data about data`_.
These are the :class:`~iris.cube.Cube`, the :class:`~iris.coords.AuxCoord`, and the
:class:`~iris.coords.DimCoord`, all of which should be familiar to you now. In
addition to these, Iris models several other classes of `CF Conventions`_
metadata. Namely,

- the :class:`~iris.coords.AncillaryVariable`, see `Ancillary Data`_ and `Flags`_,
- the :class:`~iris.coords.CellMeasure`, see `Cell Measures`_,
- the :class:`~iris.aux_factory.AuxCoordFactory`, see `Parametric Vertical Coordinate`_

Collectively, the aforementioned classes will be known here as the Iris
`CF Conventions`_ classes.

.. hint::

    If there are any `CF Conventions`_ metadata missing from Iris that you
    care about, then please let us know by raising a :issue:`GitHub Issue<new/choose>`
    on `SciTools/iris`_


Common Metadata
===============

Each of the Iris `CF Conventions`_ classes use **metadata** to define them and
give them meaning.

The **metadata** used to define an Iris `CF Conventions`_ class is composed of
individual **metadata members**, almost all of which reference specific
`CF Conventions`_ terms. The individual metadata members used to define each of
the Iris `CF Conventions`_ classes are shown in :numref:`metadata members table`.

As :numref:`metadata members table` highlights, **specific** metadata is used to
define and represent each Iris `CF Conventions`_ class. This means that metadata
alone, can be used to easily **identify**, **compare** and **differentiate**
between individual class instances.

For example, the collective metadata used to define an
:class:`~iris.coords.AncillaryVariable` are the ``standard_name``, ``long_name``,
``var_name``, ``units``, and ``attributes`` members. Note that, these are the
actual `data attribute`_ names of the metadata members on the Iris class.


.. _metadata members table:
.. table:: Iris classes that model `CF Conventions`_ metadata
   :widths: auto
   :align: center

   =================== ======================================= ============================== ========================================== ================================= ======================== ==============================
   Metadata Members    :class:`~iris.coords.AncillaryVariable` :class:`~iris.coords.AuxCoord` :class:`~iris.aux_factory.AuxCoordFactory` :class:`~iris.coords.CellMeasure` :class:`~iris.cube.Cube` :class:`~iris.coords.DimCoord`
   =================== ======================================= ============================== ========================================== ================================= ======================== ==============================
   ``standard_name``   ✔                                       ✔                              ✔                                          ✔                                 ✔                        ✔
   ``long_name``       ✔                                       ✔                              ✔                                          ✔                                 ✔                        ✔
   ``var_name``        ✔                                       ✔                              ✔                                          ✔                                 ✔                        ✔
   ``units``           ✔                                       ✔                              ✔                                          ✔                                 ✔                        ✔
   ``attributes``      ✔                                       ✔                              ✔                                          ✔                                 ✔                        ✔
   ``coord_system``                                            ✔                              ✔                                                                                                      ✔
   ``climatological``                                          ✔                              ✔                                                                                                      ✔
   ``measure``                                                                                                                            ✔
   ``cell_methods``                                                                                                                                                          ✔
   ``circular``                                                                                                                                                                                       ✔
   =================== ======================================= ============================== ========================================== ================================= ======================== ==============================

.. note::

   The :attr:`~iris.coords.DimCoord.var_name` and :attr:`~iris.coords.DimCoord.circular`
   metadata members are Iris specific terms, rather than recognised `CF Conventions`_
   terms.


Common Metadata API
===================

.. testsetup::

    import iris
    cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))

As of Iris ``3.0.0``, a unified treatment of metadata has been applied
across each Iris class (:numref:`metadata members table`) to allow users
to easily manage and manipulate their metadata in a consistent way.

This is achieved through the ``metadata`` property, which allows you to
manipulate the associated underlying metadata members as a collective.
For example, given the following :class:`~iris.cube.Cube`,

    >>> print(cube)
    air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)
        Dimension coordinates:
            time                             x              -              -
            latitude                         -              x              -
            longitude                        -              -              x
        Auxiliary coordinates:
            forecast_period                  x              -              -
        Scalar coordinates:
            forecast_reference_time     1859-09-01 06:00:00
            height                      1.5 m
        Cell methods:
            mean                        time (6 hour)
        Attributes:
            Conventions                 'CF-1.5'
            Model scenario              'A1B'
            STASH                       m01s03i236
            source                      'Data from Met Office Unified Model 6.05'

We can easily get all of the associated metadata of the :class:`~iris.cube.Cube`
using the ``metadata`` property:

    >>> cube.metadata
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))

We can also inspect the ``metadata`` of the ``longitude``
:class:`~iris.coords.DimCoord` attached to the :class:`~iris.cube.Cube` in the same way:

    >>> cube.coord("longitude").metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Or use the ``metadata`` property again, but this time on the ``forecast_period``
:class:`~iris.coords.AuxCoord` attached to the :class:`~iris.cube.Cube`:

    >>> cube.coord("forecast_period").metadata
    CoordMetadata(standard_name='forecast_period', long_name=None, var_name='forecast_period', units=Unit('hours'), attributes={}, coord_system=None, climatological=False)

Note that, the ``metadata`` property is available on each of the Iris `CF Conventions`_
class containers referenced in :numref:`metadata members table`, and thus provides
a **common** and **consistent** approach to managing your metadata, which we'll
now explore a little more fully.


Metadata Classes
----------------

The ``metadata`` property will return an appropriate `namedtuple`_ metadata class
for each Iris `CF Conventions`_ class container. The metadata class returned by
each container class is shown in :numref:`metadata classes table` below,

.. _metadata classes table:
.. table:: - Iris namedtuple metadata classes
   :widths: auto
   :align: center

   ========================================== ========================================================
   Container Class                            Metadata Class
   ========================================== ========================================================
   :class:`~iris.coords.AncillaryVariable`    :class:`~iris.common.metadata.AncillaryVariableMetadata`
   :class:`~iris.coords.AuxCoord`             :class:`~iris.common.metadata.CoordMetadata`
   :class:`~iris.aux_factory.AuxCoordFactory` :class:`~iris.common.metadata.CoordMetadata`
   :class:`~iris.coords.CellMeasure`          :class:`~iris.common.metadata.CellMeasureMetadata`
   :class:`~iris.cube.Cube`                   :class:`~iris.common.metadata.CubeMetadata`
   :class:`~iris.coords.DimCoord`             :class:`~iris.common.metadata.DimCoordMetadata`
   ========================================== ========================================================

Akin to the behaviour of a `namedtuple`_, the metadata classes in
:numref:`metadata classes table` create **tuple-like** instances i.e., they provide a
**snapshot** of the associated metadata member **values**, which are **not
settable**, but they **may be mutable** depending on the data-type of the member.
For example, given the following ``metadata`` of a :class:`~iris.coords.DimCoord`,

    >>> longitude = cube.coord("longitude")
    >>> metadata = longitude.metadata
    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

The ``metadata`` member value **is** the same as the container class member value,

    >>> metadata.attributes is longitude.attributes
    True
    >>> metadata.circular is longitude.circular
    True

Like a `namedtuple`_, the ``metadata`` member is **not settable**,

    >>> metadata.attributes = {"grinning face": "🙂"}
    Traceback (most recent call last):
    AttributeError: can't set attribute

However, for a `dict`_ member, it **is mutable**,

    >>> metadata.attributes
    {}
    >>> longitude.attributes["grinning face"] = "🙂"
    >>> metadata.attributes
    {'grinning face': '🙂'}
    >>> metadata.attributes["grinning face"] = "🙃"
    >>> longitude.attributes
    {'grinning face': '🙃'}

But ``metadata`` members with simple values are **not** mutable,

    >>> metadata.circular
    False
    >>> longitude.circular = True
    >>> metadata.circular
    False

And of course, they're also **not** settable,

    >>> metadata.circular = True
    Traceback (most recent call last):
    AttributeError: can't set attribute

Note that, the ``metadata`` property re-creates a **new** instance per invocation,
with a **snapshot** of the container class metadata values at that point in time,

    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=True)

Skip ahead to :ref:`metadata assignment <metadata assignment>` for a fuller
discussion on options how to **set** and **get** metadata on the instance of
an Iris `CF Conventions`_ container class (:numref:`metadata classes table`).


Metadata Class Behaviour
------------------------

As mentioned previously, the metadata classes in :numref:`metadata classes table`
inherit the behaviour of a `namedtuple`_, and so act and feel like a `namedtuple`_,
just as you might expect. For example, given the following ``metadata``,

    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

We can use the `namedtuple._make`_ method to create a **new**
:class:`~iris.common.metadata.DimCoordMetadata` instance from an existing sequence
or iterable. The number and order of the values used in the iterable must match that
of the associated `namedtuple._fields`_, which is discussed later,

    >>> values = (1, 2, 3, 4, 5, 6, 7, 8)
    >>> metadata._make(values)
    DimCoordMetadata(standard_name=1, long_name=2, var_name=3, units=4, attributes=5, coord_system=6, climatological=7, circular=8)

Note that, `namedtuple._make`_ is a class method, and so it is possible to
create a **new** instance directly from the metadata class itself,

    >>> from iris.common import DimCoordMetadata
    >>> DimCoordMetadata._make(values)
    DimCoordMetadata(standard_name=1, long_name=2, var_name=3, units=4, attributes=5, coord_system=6, climatological=7, circular=8)

It is also possible to easily convert ``metadata`` to an `dict`_
using the `namedtuple._asdict`_ method. This can be particularly handy when a
standard Python built-in container is required to represent your ``metadata``,

    >>> metadata._asdict()
    {'standard_name': 'longitude', 'long_name': None, 'var_name': 'longitude', 'units': Unit('degrees'), 'attributes': {'grinning face': '🙃'}, 'coord_system': GeogCS(6371229.0), 'climatological': False, 'circular': False}

Using the `namedtuple._replace`_ method allows you to create a new metadata
class instance, but replacing specified members with **new** associated values,

    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)
    >>> metadata._replace(standard_name=None, units=None)
    DimCoordMetadata(standard_name=None, long_name=None, var_name='longitude', units=None, attributes={'grinning face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Another very useful method from the `namedtuple`_ toolkit is `namedtuple._fields`_.
This method returns a tuple of strings listing the ``metadata`` members, in a
fixed order. This allows you to easily iterate over the metadata class members,
for what ever purpose you may require, e.g.,

    >>> metadata._fields
    ('standard_name', 'long_name', 'var_name', 'units', 'attributes', 'coord_system', 'climatological', 'circular')

    >>> tuple([getattr(metadata, member) for member in metadata._fields])
    ('longitude', None, 'longitude', Unit('degrees'), {'grinning face': '🙃'}, GeogCS(6371229.0), False, False)

    >>> tuple([getattr(metadata, member) for member in metadata._fields if member.endswith("name")])
    ('longitude', None, 'longitude')

Note that, `namedtuple._fields`_ is also a class method, so you don't need
an instance to determine the members of a metadata class, e.g.,

    >>> from iris.common import CubeMetadata
    >>> CubeMetadata._fields
    ('standard_name', 'long_name', 'var_name', 'units', 'attributes', 'cell_methods')

Aside from the benefit of metadata classes inheriting behaviour and state
from `namedtuple`_, further additional rich behaviour is also available,
which we explore next.


.. _richer metadata:

Richer Metadata Behaviour
-------------------------

.. testsetup:: richer-metadata

    import iris
    import numpy as np
    from iris.common import CoordMetadata
    cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
    longitude = cube.coord("longitude")

The metadata classes from :numref:`metadata classes table` support additional
behaviour above and beyond that of the  standard Python `namedtuple`_, which
allows you to easily **compare**, **combine**, **convert** and understand the
**difference** between your ``metadata`` instances.


.. _metadata equality:

Metadata Equality
^^^^^^^^^^^^^^^^^

The metadata classes support both **equality** (``__eq__``) and **inequality**
(``__ne__``), but no other `rich comparison`_ operators are implemented.
This is simply because there is no obvious ordering to any collective of metadata
members, as defined in :numref:`metadata members table`.

For example, given the following :class:`~iris.coords.DimCoord`,

.. doctest:: richer-metadata

    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

We can compare ``metadata`` using the ``==`` operator, as you may naturally
expect,

.. doctest:: richer-metadata

    >>> longitude.metadata == longitude.metadata
    True

Or alternatively, using the ``equal`` method instead,

.. doctest:: richer-metadata

    >>> longitude.metadata.equal(longitude.metadata)
    True

Note that, the ``==`` operator (``__eq__``) and the ``equal`` method are
both functionally equivalent. However, the ``equal`` method also provides
a means to enable **lenient** equality, as discussed in :ref:`lenient equality`.


.. _strict equality:

Strict Equality
"""""""""""""""

By default, metadata class equality will perform a **strict** comparison between
each associated ``metadata`` member. If **any** ``metadata`` member has a
different value, then the result of the operation will be ``False``. For example,

.. doctest:: richer-metadata

    >>> other = longitude.metadata._replace(standard_name=None)
    >>> other
    DimCoordMetadata(standard_name=None, long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)
    >>> longitude.metadata == other
    False

.. doctest:: richer-metadata

    >>> longitude.attributes = {"grinning face": "🙂"}
    >>> other = longitude.metadata._replace(attributes={"grinning face":  "🙃"})
    >>> other
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)
    >>> longitude.metadata == other
    False

One further point worth highlighting is it is possible for `NumPy`_ scalars
and arrays to appear in the ``attributes`` `dict`_ of some Iris metadata class
instances. Normally, this would cause issues. For example,

.. doctest:: richer-metadata

    >>> simply = {"one": np.int32(1), "two": np.array([1.0, 2.0])}
    >>> simply
    {'one': 1, 'two': array([1., 2.])}
    >>> fruity = {"one": np.int32(1), "two": np.array([1.0, 2.0])}
    >>> fruity
    {'one': 1, 'two': array([1., 2.])}
    >>> simply == fruity
    Traceback (most recent call last):
    ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()

However, metadata class equality is rich enough to handle this eventuality,

.. doctest:: richer-metadata

    >>> metadata1 = cube.metadata._replace(attributes=simply)
    >>> metadata2 = cube.metadata._replace(attributes=fruity)
    >>> metadata1
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'one': 1, 'two': array([1., 2.])}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))
    >>> metadata2
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'one': 1, 'two': array([1., 2.])}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))

.. doctest:: richer-metadata

    >>> metadata1 == metadata2
    True

.. doctest:: richer-metadata

    >>> metadata1
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'one': 1, 'two': array([1., 2.])}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))
    >>> metadata2 = cube.metadata._replace(attributes={"one": np.int32(1), "two": np.array([1000.0, 2000.0])})
    >>> metadata2
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'one': 1, 'two': array([1000., 2000.])}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))
    >>> metadata1 == metadata2
    False


.. _compare like:

Comparing Like With Like
""""""""""""""""""""""""

So far in our journey through metadata class equality, we have only considered
cases where the operands are instances of the **same** type. It is possible to
compare instances of **different** metadata classes, but the result will always
be ``False``,

.. doctest:: richer-metadata

    >>> cube.metadata == longitude.metadata
    False

The reason different metadata classes cannot be compared is simply because each
metadata class contains **different** members, as shown in
:numref:`metadata members table`. However, there is an exception to the rule...


.. _exception rule:

Exception to the Rule
~~~~~~~~~~~~~~~~~~~~~

In general, **different** metadata classes cannot be compared, however support
is provided for comparing :class:`~iris.common.metadata.CoordMetadata` and
:class:`~iris.common.metadata.DimCoordMetadata` metadata classes. For example,
consider the following :class:`~iris.common.metadata.DimCoordMetadata`,

.. doctest:: richer-metadata

    >>> latitude = cube.coord("latitude")
    >>> latitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Next we create a new :class:`~iris.common.metadata.CoordMetadata` instance from
the :class:`~iris.common.metadata.DimCoordMetadata` instance,

.. doctest:: richer-metadata

    >>> kwargs = latitude.metadata._asdict()
    >>> del kwargs["circular"]
    >>> metadata = CoordMetadata(**kwargs)
    >>> metadata
    CoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False)

.. hint::

    Alternatively, use the ``from_metadata`` class method instead, see
    :ref:`metadata conversion`.

Comparing the instances confirms that equality is indeed supported between
:class:`~iris.common.metadata.DimCoordMetadata` and :class:`~iris.common.metadata.CoordMetadata`
classes,

.. doctest:: richer-metadata

    >>> latitude.metadata == metadata
    True

The reason for this behaviour is primarily historical. The ``circular``
member has **never** been used by the ``__eq__`` operator when comparing an
:class:`~iris.coords.AuxCoord` and a :class:`~iris.coords.DimCoord`. Therefore
for consistency, this behaviour is also extended to ``__eq__`` for the associated
container metadata classes.

However, note that the ``circular`` member **is used** by the ``__eq__`` operator
when comparing one :class:`~iris.coords.DimCoord` to another. This also applies
when comparing :class:`~iris.common.metadata.DimCoordMetadata`.

This exception to the rule for :ref:`equality <metadata equality>` also applies
to the :ref:`difference <metadata difference>` and :ref:`combine <metadata combine>`
methods of metadata classes.


.. _metadata difference:

Metadata Difference
^^^^^^^^^^^^^^^^^^^

Being able to compare metadata is valuable, especially when we have the
convenience of being able to do this easily with metadata classes. However,
when the result of comparing two metadata instances is ``False``, it begs
the question, "**what's the difference?**"

Well, this is where we pull the ``difference`` method out of the metadata
toolbox. First, let's create some ``metadata`` to compare,

.. doctest:: richer-metadata

    >>> longitude = cube.coord("longitude")
    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '🙂'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now, we replace some members of the :class:`~iris.common.metadata.DimCoordMetadata` with
different values,

.. doctest:: richer-metadata

    >>> from cf_units import Unit
    >>> metadata = longitude.metadata._replace(long_name="lon", var_name="lon", units=Unit("radians"))
    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name='lon', var_name='lon', units=Unit('radians'), attributes={'grinning face': '🙂'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

First, confirm that the ``metadata`` is different,

.. doctest:: richer-metadata

    >>> longitude.metadata != metadata
    True

As expected, the ``metadata`` is different. Now, let's answer the question,
"**what's the difference?**",

.. doctest:: richer-metadata

    >>> longitude.metadata.difference(metadata)
    DimCoordMetadata(standard_name=None, long_name=(None, 'lon'), var_name=('longitude', 'lon'), units=(Unit('degrees'), Unit('radians')), attributes=None, coord_system=None, climatological=None, circular=None)

The ``difference`` method returns a :class:`~iris.common.metadata.DimCoordMetadata` instance, when
there is **at least** one ``metadata`` member with a different value, where,

- ``None`` means that there was **no** difference for the member,
- a `tuple`_ contains the two different associated values for the member

Given our example, only the ``long_name``, ``var_name`` and ``units`` members
have different values, as expected. Note that, the ``difference`` method **is
not** commutative. The order of the tuple member values is the same order
of the metadata class instances being compared, e.g., changing the
``difference`` instance order is reflected in the result,

.. doctest:: richer-metadata

    >>> metadata.difference(longitude.metadata)
    DimCoordMetadata(standard_name=None, long_name=('lon', None), var_name=('lon', 'longitude'), units=(Unit('radians'), Unit('degrees')), attributes=None, coord_system=None, climatological=None, circular=None)

Also, when the ``metadata`` being compared **is identical**, then ``None``
is simply returned,

.. doctest:: richer-metadata

    >>> metadata.difference(metadata) is None
    True

It's worth highlighting that for the ``attributes`` `dict`_ member, only
those keys with **different values** or **missing keys** will be returned by the
``difference`` method. For example, let's customise the ``attributes`` member of
the following :class:`~iris.common.metadata.DimCoordMetadata`,

.. doctest:: richer-metadata

    >>> attributes = {"grinning face": "😀", "neutral face": "😐"}
    >>> longitude.attributes = attributes
    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '😀', 'neutral face': '😐'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Then create another :class:`~iris.common.metadata.DimCoordMetadata` with a different
``attributes`` `dict`_, namely,

- the ``grinning face`` key has the **same value**,
- the ``neutral face`` key has a **different value**,
- the ``upside-down face`` key is **new**

.. doctest:: richer-metadata

    >>> attributes = {"grinning face": "😀", "neutral face": "😜", "upside-down face": "🙃"}
    >>> metadata = longitude.metadata._replace(attributes=attributes)
    >>> metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={'grinning face': '😀', 'neutral face': '😜', 'upside-down face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now, let's compare the two above instances and see what ``attributes`` member differences we get,

.. doctest:: richer-metadata

    >>> longitude.metadata.difference(metadata)  # doctest: +SKIP
    DimCoordMetadata(standard_name=None, long_name=None, var_name=None, units=None, attributes=({'neutral face': '😐'}, {'neutral face': '😜', 'upside-down face': '🙃'}), coord_system=None, climatological=None, circular=None)


.. _diff like:

Diffing Like With Like
""""""""""""""""""""""

As discussed in :ref:`compare like`, it only makes sense to determine the
``difference`` between **similar** metadata class instances. However, note that
the :ref:`exception to the rule <exception rule>` still applies here i.e.,
support is provided between :class:`~iris.common.metadata.CoordMetadata` and
:class:`~iris.common.metadata.DimCoordMetadata` metadata classes.

For example, given the following :class:`~iris.coords.AuxCoord` and
:class:`~iris.coords.DimCoord`,

.. doctest:: richer-metadata

    >>> forecast_period = cube.coord("forecast_period")
    >>> latitude = cube.coord("latitude")

We can inspect their associated ``metadata``,

.. doctest:: richer-metadata

    >>> forecast_period.metadata
    CoordMetadata(standard_name='forecast_period', long_name=None, var_name='forecast_period', units=Unit('hours'), attributes={}, coord_system=None, climatological=False)
    >>> latitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Before comparing them to determine the values of metadata members that are different,

.. doctest:: richer-metadata

    >>> forecast_period.metadata.difference(latitude.metadata)
    CoordMetadata(standard_name=('forecast_period', 'latitude'), long_name=None, var_name=('forecast_period', 'latitude'), units=(Unit('hours'), Unit('degrees')), attributes=None, coord_system=(None, GeogCS(6371229.0)), climatological=None)

.. doctest:: richer-metadata

    >>> latitude.metadata.difference(forecast_period.metadata)
    DimCoordMetadata(standard_name=('latitude', 'forecast_period'), long_name=None, var_name=('latitude', 'forecast_period'), units=(Unit('degrees'), Unit('hours')), attributes=None, coord_system=(GeogCS(6371229.0), None), climatological=None, circular=(False, None))

In general, however, comparing **different** metadata classes will result in a
``TypeError`` being raised,

.. doctest:: richer-metadata

    >>> cube.metadata.difference(longitude.metadata)
    Traceback (most recent call last):
    TypeError: Cannot differ 'CubeMetadata' with <class 'iris.common.metadata.DimCoordMetadata'>.


.. _metadata combine:

Metadata Combination
^^^^^^^^^^^^^^^^^^^^

.. testsetup:: metadata-combine

   import iris
   cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
   longitude = cube.coord("longitude")

So far we've seen how to :ref:`compare metadata <metadata equality>`, and also how
to determine the :ref:`difference between metadata <metadata difference>`. Now we
take the next step, and explore how to combine metadata together using the ``combine``
metadata class method.

For example, consider the following :class:`~iris.common.metadata.CubeMetadata`,

.. doctest:: metadata-combine

    >>> cube.metadata  # doctest: +SKIP
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))

We can perform the **identity function** by comparing the metadata with itself,

.. doctest:: metadata-combine

    >>> metadata = cube.metadata.combine(cube.metadata)
    >>> cube.metadata == metadata
    True

As you might expect, combining identical metadata returns metadata that is
also identical.

The ``combine`` method will always return **a new** metadata class instance,
where each metadata member is either ``None`` or populated with a **common value**.
Let's clarify this, by combining our above :class:`~iris.common.metadata.CubeMetadata`
with another instance that's identical apart from its ``standard_name`` member,
which is replaced with a **different value**,

.. doctest:: metadata-combine

    >>> metadata = cube.metadata._replace(standard_name="air_pressure_at_sea_level")
    >>> metadata != cube.metadata
    True
    >>> metadata.combine(cube.metadata)  # doctest: +SKIP
    CubeMetadata(standard_name=None, long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'STASH': STASH(model=1, section=3, item=236), 'source': 'Data from Met Office Unified Model 6.05', 'Model scenario': 'A1B', 'Conventions': 'CF-1.5'}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))

The ``combine`` method combines metadata by performing a **strict** comparison
between each of the associated metadata member values,

- if the values are **different**, then the combined result is ``None``
- otherwise, the combined result is the **common value**

Let's reinforce this behaviour, but this time by combining metadata where the
``attributes`` `dict`_ member is different, where,

- the ``STASH`` and ``source`` keys are **missing**,
- the ``Model scenario`` key has the **same value**,
- the ``Conventions`` key has a **different value**,
- the ``grinning face`` key is **new**

.. doctest:: metadata-combine

    >>> attributes = {"Model scenario": "A1B", "Conventions": "CF-1.8", "grinning face": "🙂" }
    >>> metadata = cube.metadata._replace(attributes=attributes)
    >>> metadata != cube.metadata
    True
    >>> metadata.combine(cube.metadata).attributes
    {'Model scenario': 'A1B'}

The combined result for the ``attributes`` member only contains those
**common keys** with **common values**.

Note that, the ``combine`` method is **commutative**,

.. doctest:: metadata-combine

    >>> cube.metadata.combine(metadata) == metadata.combine(cube.metadata)
    True

Although, this is only the case when combining instances of the **same**
metadata class. This is explored in a little further detail next.


.. _combine like:

Combine Like With Like
""""""""""""""""""""""

Akin to the :ref:`equal <metadata equality>` and
:ref:`difference <metadata difference>` methods, only instances of **similar**
metadata classes can be combined, otherwise a ``TypeError`` is raised,

.. doctest:: metadata-combine

    >>> cube.metadata.combine(longitude.metadata)
    Traceback (most recent call last):
    TypeError: Cannot combine 'CubeMetadata' with <class 'iris.common.metadata.DimCoordMetadata'>.

Again, however, the :ref:`exception to the rule <exception rule>` also applies
here i.e., support is provided between :class:`~iris.common.metadata.CoordMetadata` and
:class:`~iris.common.metadata.DimCoordMetadata` metadata classes.

For example, we can ``combine`` the metadata of the following
:class:`~iris.coords.AuxCoord` and :class:`~iris.coords.DimCoord`,

.. doctest:: metadata-combine

    >>> forecast_period = cube.coord("forecast_period")
    >>> longitude = cube.coord("longitude")

First, let's see their associated metadata,

.. doctest:: metadata-combine

    >>> forecast_period.metadata
    CoordMetadata(standard_name='forecast_period', long_name=None, var_name='forecast_period', units=Unit('hours'), attributes={}, coord_system=None, climatological=False)
    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Before combining their metadata together,

.. doctest:: metadata-combine

    >>> forecast_period.metadata.combine(longitude.metadata)
    CoordMetadata(standard_name=None, long_name=None, var_name=None, units=None, attributes={}, coord_system=None, climatological=False)
    >>> longitude.metadata.combine(forecast_period.metadata)
    DimCoordMetadata(standard_name=None, long_name=None, var_name=None, units=None, attributes={}, coord_system=None, climatological=False, circular=None)

However, note that commutativity in this case cannot be honoured, for obvious reasons.


.. _metadata conversion:

Metadata Conversion
^^^^^^^^^^^^^^^^^^^

.. testsetup:: metadata-convert

   import iris
   from iris.common import DimCoordMetadata
   cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
   longitude = cube.coord("longitude")

In general, the :ref:`equal <metadata equality>`, :ref:`difference <metadata difference>`,
and :ref:`combine <metadata combine>` methods only support operations on instances
of the same metadata class (see :ref:`exception to the rule <exception rule>`).

However, metadata may be converted from one metadata class to another using
the ``from_metadata`` class method. For example, given the following
:class:`~iris.common.metadata.CubeMetadata`,

.. doctest:: metadata-convert

    >>> cube.metadata  # doctest: +SKIP
    CubeMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, cell_methods=(CellMethod(method='mean', coord_names=('time',), intervals=('6 hour',), comments=()),))

We can easily convert it to a :class:`~iris.common.metadata.DimCoordMetadata` instance
using ``from_metadata``,

.. doctest:: metadata-convert

    >>> DimCoordMetadata.from_metadata(cube.metadata)  # doctest: +SKIP
    DimCoordMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, coord_system=None, climatological=None, circular=None)

By examining :numref:`metadata members table`, we can see that the
:class:`~iris.cube.Cube` and :class:`~iris.coords.DimCoord` container
classes share the following common metadata members,

- ``standard_name``,
- ``long_name``,
- ``var_name``,
- ``units``,
- ``attributes``

As such, all of these metadata members of the resultant
:class:`~iris.common.metadata.DimCoordMetadata` instance are populated from the associated
:class:`~iris.common.metadata.CubeMetadata` instance members. However, a
:class:`~iris.common.metadata.CubeMetadata` class does not contain the following
:class:`~iris.common.metadata.DimCoordMetadata` members,

- ``coords_system``,
- ``climatological``,
- ``circular``

Thus these particular metadata members are set to ``None`` in the resultant
:class:`~iris.common.metadata.DimCoordMetadata` instance.

Note that, the ``from_metadata`` method is also available on a metadata
class instance,

.. doctest:: metadata-convert

    >>> longitude.metadata.from_metadata(cube.metadata)
    DimCoordMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, coord_system=None, climatological=None, circular=None)


.. _metadata assignment:

Metadata Assignment
^^^^^^^^^^^^^^^^^^^

.. testsetup:: metadata-assign

   import iris
   cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
   longitude = cube.coord("longitude")
   original = longitude.copy()
   latitude = cube.coord("latitude")

The ``metadata`` property available on each Iris `CF Conventions`_ container
class (:numref:`metadata classes table`) can not only be used **to get**
the metadata of an instance, but also **to set** the metadata on an instance.

For example, given the following :class:`~iris.common.metadata.DimCoordMetadata` of the
``longitude`` coordinate,

.. doctest:: metadata-assign

    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

We can assign to it directly using the :class:`~iris.common.metadata.DimCoordMetadata` of the ``latitude``
coordinate,

.. doctest:: metadata-assign

    >>> latitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)
    >>> longitude.metadata = latitude.metadata
    >>> longitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)


Assign by Iterable
""""""""""""""""""

It is also possible to assign to the ``metadata`` property of an Iris
`CF Conventions`_ container with an iterable containing the **correct
number** of associated member values, e.g.,

.. doctest:: metadata-assign

    >>> values = [getattr(latitude, member) for member in latitude.metadata._fields]
    >>> longitude.metadata = values
    >>> longitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)


Assign by Namedtuple
""""""""""""""""""""

A `namedtuple`_ may also be used to assign to the ``metadata`` property of an
Iris `CF Conventions`_ container. For example, let's first create a custom
namedtuple class,

.. doctest:: metadata-assign

    >>> from collections import namedtuple
    >>> Metadata = namedtuple("Metadata", ["standard_name", "long_name", "var_name", "units", "attributes", "coord_system", "climatological", "circular"])

Now create an instance of this custom namedtuple class, and populate it,

.. doctest:: metadata-assign

    >>> metadata = Metadata(*values)
    >>> metadata
    Metadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now we can use the custom namedtuple instance to assign directly to the metadata
of the ``longitude`` coordinate,

.. doctest:: metadata-assign

    >>> longitude.metadata = metadata
    >>> longitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)


Assign by Mapping
"""""""""""""""""

It is also possible to assign to the ``metadata`` property using a `mapping`_,
such as a `dict`_,

.. doctest:: metadata-assign

    >>> mapping = latitude.metadata._asdict()
    >>> mapping
    {'standard_name': 'latitude', 'long_name': None, 'var_name': 'latitude', 'units': Unit('degrees'), 'attributes': {}, 'coord_system': GeogCS(6371229.0), 'climatological': False, 'circular': False}
    >>> longitude.metadata = mapping
    >>> longitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Support is also provided for assigning a **partial** mapping, for example,

.. testcode:: metadata-assign
   :hide:

   longitude = original

.. doctest:: metadata-assign

    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)
    >>> longitude.metadata = dict(var_name="lat", units="radians", circular=True)
    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='lat', units=Unit('radians'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=True)

Indeed, it's also possible to assign to the ``metadata`` property with a
**different** metadata class instance,

.. testcode:: metadata-assign
   :hide:

   longitude.metadata = dict(var_name="longitude", units="degrees", circular=False)

.. doctest:: metadata-assign

    >>> longitude.metadata
    DimCoordMetadata(standard_name='longitude', long_name=None, var_name='longitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)
    >>> longitude.metadata = cube.metadata
    >>> longitude.metadata  # doctest: +SKIP
    DimCoordMetadata(standard_name='air_temperature', long_name=None, var_name='air_temperature', units=Unit('K'), attributes={'Conventions': 'CF-1.5', 'STASH': STASH(model=1, section=3, item=236), 'Model scenario': 'A1B', 'source': 'Data from Met Office Unified Model 6.05'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Note that, only **common** metadata members will be assigned new associated
values. All other metadata members will be left unaltered.


.. _data about data: https://en.wikipedia.org/wiki/Metadata
.. _data attribute: https://docs.python.org/3/tutorial/classes.html#instance-objects
.. _dict: https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
.. _Ancillary Data: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#ancillary-data
.. _CF Conventions: https://cfconventions.org/
.. _Cell Measures: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-measures
.. _Flags: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags
.. _mapping: https://docs.python.org/3/glossary.html#term-mapping
.. _namedtuple: https://docs.python.org/3/library/collections.html#collections.namedtuple
.. _namedtuple._make: https://docs.python.org/3/library/collections.html#collections.somenamedtuple._make
.. _namedtuple._asdict: https://docs.python.org/3/library/collections.html#collections.somenamedtuple._asdict
.. _namedtuple._replace: https://docs.python.org/3/library/collections.html#collections.somenamedtuple._replace
.. _namedtuple._fields: https://docs.python.org/3/library/collections.html#collections.somenamedtuple._fields
.. _NetCDF: https://www.unidata.ucar.edu/software/netcdf/
.. _NetCDF CF Metadata Conventions: https://cfconventions.org/
.. _NumPy: https://github.com/numpy/numpy
.. _Parametric Vertical Coordinate: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-vertical-coordinate
.. _rich comparison: https://www.python.org/dev/peps/pep-0207/
.. _SciTools/iris: https://github.com/SciTools/iris
.. _tuple: https://docs.python.org/3/library/stdtypes.html#tuples
