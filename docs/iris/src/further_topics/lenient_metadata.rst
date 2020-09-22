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

This type of **strict** behaviour does offer obvious benefit and value. However,
it can be unnecessarily restrictive. For example, consider the metadata of the
following ``latitude`` coordinate,

.. doctest:: strict-behaviour

    >>> latitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now, let's create a doctored version of this metadata with a different ``var_name``,

.. doctest:: strict-behaviour

    >>> metadata = latitude.metadata._replace(var_name="lat")
    >>> metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='lat', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Clearly, these metadata are different,

.. doctest:: strict-behaviour

    >>> metadata != latitude.metadata
    True
    >>> metadata.difference(latitude.metadata)
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name=('lat', 'latitude'), units=None, attributes=None, coord_system=None, climatological=None, circular=None)

And yet, they both have the same ``name``, which is slightly confusing
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

*Mauris facilisis imperdiet mi, quis pellentesque urna vulputate id. Ut mi neque, condimentum a augue non, tempor mollis ipsum. Ut nec leo maximus nisi facilisis auctor. In hac habitasse platea dictumst. Nullam et posuere nulla, eget commodo ligula. Nulla eleifend euismod odio, sed vulputate ipsum ornare sit amet. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Nulla luctus, mauris pretium rhoncus ultrices, urna justo pharetra urna, sit amet porta risus diam vitae dui. Pellentesque a leo ligula. Curabitur sit amet augue id elit pretium condimentum quis a nisi. Duis egestas faucibus velit, non blandit augue finibus non. Proin vel tempor dolor, non volutpat tortor. Vestibulum sollicitudin eu elit vel placerat. Sed vestibulum purus lectus, vel feugiat est venenatis sed. In ultrices pharetra elit.*



.. _CF Conventions: https://cfconventions.org/