.. _lenient metadata:

Lenient Metadata
****************

This section discusses lenient metadata; what it is, what it means, and how you
can perform **lenient** rather than **strict** operations with your metadata.


Introduction
============

As discussed in :ref:`metadata`, a rich, common metadata API is available within
Iris that supports metadata :ref:`equality <metadata equality>`,
:ref:`difference <metadata difference>`, :ref:`combination <metadata combine>`,
and also :ref:`conversion <metadata conversion>`.

The common metadata API is implemented through the ``metadata`` property
on each of the Iris `CF Conventions`_ class containers
(:numref:`metadata classes table`), and provides a common gateway for users to
easily manage and manipulate their metadata in a consistent and unified way.

This is primarily all thanks to the metadata classes (:numref:`metadata classes table`)
that support the necessary state and behaviour required by the common metadata
API. Namely, it is the ``equal`` (``__eq__``), ``difference`` and ``combine``
methods that provide this rich metadata behaviour, all of which are explored
more fully in :ref:`metadata`.


Strict Behaviour
================

.. testsetup:: strict-behaviour

    import iris
    cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
    latitude = cube.coord("latitude")

The feature that is common between the ``equal``, ``difference`` and
``combine`` metadata class methods, is that they all perform **strict**
metadata member comparisons **by default**.

The **strict** behaviour implemented by these methods can be summarised
as follows, where ``X`` and ``Y`` are any objects that are non-identical,

.. _strict equality table:
.. table:: - :ref:`Strict equality <metadata equality>`
   :widths: auto
   :align: center

   ======== ======== =========
   Left     Right    ``equal``
   ======== ======== =========
   ``X``    ``Y``    ``False``
   ``Y``    ``X``    ``False``
   ``X``    ``X``    ``True``
   ``X``    ``None`` ``False``
   ``None`` ``X``    ``False``
   ======== ======== =========

.. _strict difference table:
.. table:: - :ref:`Strict difference <metadata difference>`
   :widths: auto
   :align: center

   ======== ======== =================
   Left     Right    ``difference``
   ======== ======== =================
   ``X``    ``Y``    (``X``, ``Y``)
   ``Y``    ``X``    (``Y``, ``X``)
   ``X``    ``X``    ``None``
   ``X``    ``None`` (``X``, ``None``)
   ``None`` ``X``    (``None``, ``X``)
   ======== ======== =================

.. _strict combine table:
.. table:: - :ref:`Strict combination <metadata combine>`
   :widths: auto
   :align: center

   ======== ======== ===========
   Left     Right    ``combine``
   ======== ======== ===========
   ``X``    ``Y``    ``None``
   ``Y``    ``X``    ``None``
   ``X``    ``X``    ``X``
   ``X``    ``None`` ``None``
   ``None`` ``X``    ``None``
   ======== ======== ===========

.. _strict example:

This type of **strict** behaviour does offer obvious benefit and value. However,
it can be unnecessarily restrictive. For example, consider the metadata of the
following ``latitude`` coordinate,

.. doctest:: strict-behaviour

    >>> latitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now, let's create a doctored version of this metadata with a different ``var_name``,

.. doctest:: strict-behaviour

    >>> metadata = latitude.metadata._replace(var_name=None)
    >>> metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name=None, units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Clearly, these metadata are different,

.. doctest:: strict-behaviour

    >>> metadata != latitude.metadata
    True
    >>> metadata.difference(latitude.metadata)
    DimCoordMetadata(standard_name=None, long_name=None, var_name=(None, 'latitude'), units=None, attributes=None, coord_system=None, climatological=None, circular=None)

And yet, they both have the same ``name``, which some may find slightly confusing
(see :meth:`~iris.common.metadata.BaseMetadata.name` for clarification)

.. doctest:: strict-behaviour

    >>> metadata.name()
    'latitude'
    >>> latitude.name()
    'latitude'

Resolving this metadata inequality can only be overcome by ensuring that each
metadata member precisely matches.

If your workflow demands such metadata rigour, then the default strict behaviour
of the common metadata API will satisfy your needs. Typically though, such
strictness is not necessary, and as of Iris ``3.0.0`` an alternative more
practical behaviour is available.


.. _lenient behaviour:

Lenient Behaviour
=================

.. testsetup:: lenient-behaviour

    import iris
    cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
    latitude = cube.coord("latitude")

Lenient metadata aims to offer a practical, common sense alternative to the
strict rigour of the default Iris metadata behaviour. It is intended to be
complementary, and suitable for those users with a more relaxed requirement
regarding their metadata.

The lenient behaviour that is implemented as an alternative to the
:ref:`strict equality <strict equality table>`, :ref:`strict difference <strict difference table>`,
and :ref:`strict combination <strict combine table>` can be summarised
as follows,

.. _lenient equality table:
.. table:: - Lenient equality
   :widths: auto
   :align: center

   ======== ======== =========
   Left     Right    ``equal``
   ======== ======== =========
   ``X``    ``Y``    ``False``
   ``Y``    ``X``    ``False``
   ``X``    ``X``    ``True``
   ``X``    ``None`` ``True``
   ``None`` ``X``    ``True``
   ======== ======== =========

.. _lenient difference table:
.. table:: - Lenient difference
   :widths: auto
   :align: center

   ======== ======== =================
   Left     Right    ``difference``
   ======== ======== =================
   ``X``    ``Y``    (``X``, ``Y``)
   ``Y``    ``X``    (``Y``, ``X``)
   ``X``    ``X``    ``None``
   ``X``    ``None`` ``None``
   ``None`` ``X``    ``None``
   ======== ======== =================

.. _lenient combine table:
.. table:: - Lenient combination
   :widths: auto
   :align: center

   ======== ======== ===========
   Left     Right    ``combine``
   ======== ======== ===========
   ``X``    ``Y``    ``None``
   ``Y``    ``X``    ``None``
   ``X``    ``X``    ``X``
   ``X``    ``None`` ``X``
   ``None`` ``X``    ``X``
   ======== ======== ===========

Lenient behaviour is enabled for the ``equal``, ``difference``, and ``combine``
metadata class methods via the ``lenient`` keyword argument, which is ``False``
by default. Let's first explore some examples of lenient equality, difference
and combination, before going on to clarify which metadata members adopt
lenient behaviour for each of the metadata classes.


.. _lenient equality:

Lenient Equality
----------------

Lenient equality is enabled using the ``lenient`` keyword argument, therefore
we are forced to use the ``equal`` method rather than the ``==`` operator
(``__eq__``). Otherwise, the ``equal`` method and ``==`` operator are both
functionally equivalent.

For example, consider the :ref:`previous strict example <strict example>`,
where two separate ``latitude`` coordinates are compared, each with different
``var_name`` members,

.. doctest:: strict-behaviour

    >>> metadata.equal(latitude.metadata, lenient=True)
    True

Unlike strict comparison, lenient comparison is a little more forgiving. In
this case, leniently comparing **something** with **nothing** (``None``) will
always be ``True``; it's the graceful compromise to the strict alternative.

So let's take the opportunity to reinforce this a little further before moving on,
by leniently comparing different ``attributes`` dictionaries; a constant source
of strict contention.

Firstly, populate the metadata of our ``latitude`` coordinate appropriately,

.. doctest:: lenient-behaviour

    >>> attributes = {"grinning face": "😀", "neutral face": "😐"}
    >>> latitude.attributes = attributes
    >>> latitude.metadata  # doctest: +SKIP
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={'grinning face': '😀', 'neutral face': '😐'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Then create another :class:`~iris.common.metadata.DimCoordMetadata` with a different
``attributes`` `dict`_, namely,

- the ``grinning face`` key is **missing**,
- the ``neutral face`` key has the **same value**, and
- the ``upside-down face`` key is **new**

.. doctest:: lenient-behaviour

    >>> attributes = {"neutral face": "😐", "upside-down face": "🙃"}
    >>> metadata = latitude.metadata._replace(attributes=attributes)
    >>> metadata  # doctest: +SKIP
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={'neutral face': '😐', 'upside-down face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now, compare our metadata,

.. doctest:: lenient-behaviour

    >>> metadata.equal(latitude.metadata)
    False
    >>> metadata.equal(latitude.metadata, lenient=True)
    True

Again, lenient equality (:numref:`lenient equality table`) offers a more
forgiving and practical alternative to strict behaviour.


.. _lenient difference:

Lenient Difference
------------------

Similar to :ref:`lenient equality`, the lenient ``difference`` method
(:numref:`lenient difference table`) considers there to be no difference between
comparing **something** with **nothing** (``None``). This working assumption is
not naively applied to all metadata members, but rather a more pragmatic approach
is adopted, as discussed later in :ref:`lenient members`.

Again, lenient behaviour for the ``difference`` metadata class method is enabled
by the ``lenient`` keyword argument. For example, consider again the
:ref:`previous strict example <strict example>` involving our ``latitude``
coordinate,

.. doctest:: strict-behaviour

    >>> metadata.difference(latitude.metadata)
    DimCoordMetadata(standard_name=None, long_name=None, var_name=(None, 'latitude'), units=None, attributes=None, coord_system=None, climatological=None, circular=None)
    >>> metadata.difference(latitude.metadata, lenient=True) is None
    True

And revisiting our slightly altered ``attributes`` member comparison example,
brings home the benefits of the lenient difference behaviour. So, given our
``latitude`` coordinate with its populated ``attributes`` dictionary,

.. doctest:: lenient-behaviour

    >>> latitude.attributes  # doctest: +SKIP
    {'grinning face': '😀', 'neutral face': '😐'}

We create another :class:`~iris.common.metadata.DimCoordMetadata` with a dissimilar
``attributes`` member, namely,

- the ``grinning face`` key is **missing**,
- the ``neutral face`` key has a **different value**, and
- the ``upside-down face`` key is **new**

.. doctest:: lenient-behaviour

    >>> attributes = {"neutral face": "😜", "upside-down face": "🙃"}
    >>> metadata = latitude.metadata._replace(attributes=attributes)
    >>> metadata  # doctest: +SKIP
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={'neutral face': '😜', 'upside-down face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now comparing the strict and lenient behaviour for the ``difference`` method,
highlights the change in how such dissimilar metadata is treated gracefully,

.. doctest:: lenient-behaviour

    >>> metadata.difference(latitude.metadata).attributes  # doctest: +SKIP
    {'upside-down face': '🙃', 'neutral face': '😜'}, {'neutral face': '😐', 'grinning face': '😀'}
    >>> metadata.difference(latitude.metadata, lenient=True).attributes  # doctest: +SKIP
    {'neutral face': '😜'}, {'neutral face': '😐'}


.. _lenient combination:

Lenient Combination
-------------------

The behaviour of the lenient ``combine`` metadata class method is outlined
in :numref:`lenient combine table`, and as with :ref:`lenient equality` and
:ref:`lenient difference` is enabled through the ``lenient`` keyword argument.

The difference in behaviour between **lenient** and
:ref:`strict combination <strict combine table>` is centred around the lenient
handling of combining **something** with **nothing** (``None``) to return
**something**. Whereas strict
combination will only return a result from combining identical objects.

Again, this is best demonstrated through a simple example of attempting to combine
partially overlapping ``attributes`` member dictionaries. For example, given the
following ``attributes`` dictionary of our favoured ``latitude`` coordinate,

.. doctest:: lenient-behaviour

    >>> latitude.attributes  # doctest: +SKIP
    {'grinning face': '😀', 'neutral face': '😐'}

We create another :class:`~iris.common.metadata.DimCoordMetadata` with overlapping
keys and values, namely,

- the ``grinning face`` key is **missing**,
- the ``neutral face`` key has the **same value**, and
- the ``upside-down face`` key is **new**

.. doctest:: lenient-behaviour

    >>> attributes = {"neutral face": "😐", "upside-down face": "🙃"}
    >>> metadata = latitude.metadata._replace(attributes=attributes)
    >>> metadata  # doctest: +SKIP
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={'neutral face': '😐', 'upside-down face': '🙃'}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Comparing the strict and lenient behaviour of ``combine`` side-by-side
highlights the difference in behaviour, and the advantages of lenient combination
for more inclusive, richer metadata,

.. doctest:: lenient-behaviour

    >>> metadata.combine(latitude.metadata).attributes
    {'neutral face': '😐'}
    >>> metadata.combine(latitude.metadata, lenient=True).attributes  # doctest: +SKIP
    {'neutral face': '😐', 'upside-down face': '🙃', 'grinning face': '😀'}


.. _lenient members:

Lenient Members
---------------

:ref:`lenient behaviour` is not applied regardlessly across all metadata members
participating in a lenient ``equal``, ``difference`` or ``combine`` operation.
Rather, a more pragmatic application is employed based on the `CF Conventions`_
definition of the member, and whether being lenient would result in erroneous
behaviour or interpretation.

.. _lenient members table:
.. table:: - Lenient member participation
   :widths: auto
   :align: center

   ============================================================================================= ================== ============
   Metadata Class                                                                                Member             Behaviour
   ============================================================================================= ================== ============
   All metadata classes†                                                                         ``standard_name``  ``lenient``‡
   All metadata classes†                                                                         ``long_name``      ``lenient``‡
   All metadata classes†                                                                         ``var_name``       ``lenient``‡
   All metadata classes†                                                                         ``units``          ``strict``
   All metadata classes†                                                                         ``attributes``     ``lenient``
   :class:`~iris.common.metadata.CellMeasureMetadata`                                            ``measure``        ``strict``
   :class:`~iris.common.metadata.CoordMetadata`, :class:`~iris.common.metadata.DimCoordMetadata` ``coord_system``   ``strict``
   :class:`~iris.common.metadata.CoordMetadata`, :class:`~iris.common.metadata.DimCoordMetadata` ``climatological`` ``strict``
   :class:`~iris.common.metadata.CubeMetadata`                                                   ``cell_methods``   ``strict``
   :class:`~iris.common.metadata.DimCoordMetadata`                                               ``circular``       ``strict`` §
   ============================================================================================= ================== ============

| **Key**
| † - Applies to all metadata classes including :class:`~iris.common.metadata.AncillaryVariableMetadata`, which has no other specialised members
| ‡ - See :ref:`special lenient name` for ``standard_name``, ``long_name``, and ``var_name``
| § - The ``circular`` is ignored for operations between :class:`~iris.common.metadata.CoordMetadata` and :class:`~iris.common.metadata.DimCoordMetadata`

In summary, only ``standard_name``, ``long_name``, ``var_name`` and the ``attributes``
members are treated leniently. All other members are considered to represent
fundamental metadata that cannot, by their nature, be consider equivalent to
metadata that is missing or ``None``. For example, a :class:`~iris.cube.Cube`
with ``units`` of ``ms-1`` cannot be considered equivalent to another
:class:`~iris.cube.Cube` with ``units`` of ``unknown``; this would be a false
and dangerous scientific assumption to make.

Similar arguments can be made for the ``measure``, ``coord_system``, ``climatological``,
``cell_methods``, and ``circular`` members, all of which are treated with
strict behaviour, regardlessly.


.. _special lenient name:

Special Lenient Name Behaviour
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``standard_name``, ``long_name`` and ``var_name`` have a closer association
with each other compared to all other metadata members, as they all
underpin the functionality provided by the :meth:`~iris.common.mixin.CFVariableMixin.name`
method. It is imperative that the :meth:`~iris.common.mixin.CFVariableMixin.name`
derived from metadata remains constant for strict and lenient equality alike.

As such, these metadata members have an additional layer of behaviour enforced
during :ref:`lenient equality` in order to ensure that the identity or name of
metadata does not change due to a side-effect of lenient comparison.

For example, if simple :ref:`lenient equality <lenient equality table>`
behaviour was applied to the ``standard_name``, ``long_name`` and ``var_name``,
the following would be considered **not** equal,

.. table::
   :widths: auto
   :align: center

   ================= ============ ============
   Member            Left         Right
   ================= ============ ============
   ``standard_name`` ``None``     ``latitude``
   ``long_name``     ``latitude`` ``None``
   ``var_name``      ``lat``      ``latitude``
   ================= ============ ============

Both the **Left** and **Right** metadata would have the same
:meth:`~iris.common.mixin.CFVariableMixin.name` by definition i.e., ``latitude``.
However, lenient equality would fail due to the difference in ``var_name``.

To account for this, lenient equality is performed by two simple consecutive steps:

- ensure that the result returned by the :meth:`~iris.common.mixin.CFVariableMixin.name`
  method is the same for the metadata being compared, then
- only perform :ref:`lenient equality <lenient equality table>` between the
  ``standard_name`` and ``long_name`` i.e., the ``var_name`` member is **not**
  compared explicitly, as its value may have been accounted for through
  :meth:`~iris.common.mixin.CFVariableMixin.name` equality


.. _dict: https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
.. _CF Conventions: https://cfconventions.org/
