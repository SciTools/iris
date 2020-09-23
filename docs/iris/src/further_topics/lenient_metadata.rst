.. _lenient metadata:

****************
Lenient metadata
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


Strict behaviour
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
   left     right    **equal**
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
   left     right    **difference**
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
   left     right    **combine**
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

Regardlessly, at the end of the day we may not actually care that the
``var_name`` is different. However, Iris relentlessly forces us to deal
with such a difference; sometimes this can be challenging to overcome.

If your workflow demands such metadata rigour, then the default strict behaviour
of the common metadata API will satisfy your needs. Typically though, such
strictness is not necessary, and as of Iris ``3.0.0`` an alternative more
practical behaviour is available.


Lenient behaviour
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
   left     right    **equal**
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
   left     right    **difference**
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
   left     right    **combine**
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

Lenient equality
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

Then create another :class:`~iris.common.DimCoordMetadata` with a different
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
forgiving and practical alternative to the strict alternative.


.. _lenient difference:

Lenient difference
------------------

*In semper in ex ac consectetur. Mauris vulputate malesuada bibendum. Aliquam ac nisl ultricies, porta felis nec, ultrices ante. Fusce placerat fermentum rhoncus. Fusce porta ut ligula quis tristique. Sed cursus blandit felis eu mollis. Aliquam scelerisque purus et pellentesque tempor. Vestibulum ac aliquam sapien. Proin vel mi quis turpis vulputate sodales non vel orci. Etiam nec hendrerit lectus. Vestibulum aliquet eleifend metus, et efficitur ante sagittis non. Sed tincidunt consectetur nibh, ac sodales elit ultrices eu. Duis blandit elementum libero quis maximus. Proin odio ipsum, congue et bibendum non, vehicula eget dui. Sed nec cursus leo. Nunc in massa eget nisi luctus eleifend id bibendum ante.*


.. _lenient combination:

Lenient combination
-------------------

*In semper in ex ac consectetur. Mauris vulputate malesuada bibendum. Aliquam ac nisl ultricies, porta felis nec, ultrices ante. Fusce placerat fermentum rhoncus. Fusce porta ut ligula quis tristique. Sed cursus blandit felis eu mollis. Aliquam scelerisque purus et pellentesque tempor. Vestibulum ac aliquam sapien. Proin vel mi quis turpis vulputate sodales non vel orci. Etiam nec hendrerit lectus. Vestibulum aliquet eleifend metus, et efficitur ante sagittis non. Sed tincidunt consectetur nibh, ac sodales elit ultrices eu. Duis blandit elementum libero quis maximus. Proin odio ipsum, congue et bibendum non, vehicula eget dui. Sed nec cursus leo. Nunc in massa eget nisi luctus eleifend id bibendum ante.*


.. _lenient members:

Lenient members
---------------

*In semper in ex ac consectetur. Mauris vulputate malesuada bibendum. Aliquam ac nisl ultricies, porta felis nec, ultrices ante. Fusce placerat fermentum rhoncus. Fusce porta ut ligula quis tristique. Sed cursus blandit felis eu mollis. Aliquam scelerisque purus et pellentesque tempor. Vestibulum ac aliquam sapien. Proin vel mi quis turpis vulputate sodales non vel orci. Etiam nec hendrerit lectus. Vestibulum aliquet eleifend metus, et efficitur ante sagittis non. Sed tincidunt consectetur nibh, ac sodales elit ultrices eu. Duis blandit elementum libero quis maximus. Proin odio ipsum, congue et bibendum non, vehicula eget dui. Sed nec cursus leo. Nunc in massa eget nisi luctus eleifend id bibendum ante.*


.. _dict: https://docs.python.org/3/library/stdtypes.html#mapping-types-dict
.. _CF Conventions: https://cfconventions.org/