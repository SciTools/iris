.. _lenient metadata:

****************
Lenient metadata
****************

As discussed in :ref:`metadata`, a rich, common metadata API is available within
Iris that supports metadata :ref:`equality <metadata equality>`,
:ref:`difference <metadata difference>`, :ref:`combination <metadata combine>`,
and also :ref:`conversion <metadata conversion>`.

The common metadata API is implemented through the ``metadata`` property
on each of the Iris `CF Conventions`_ class containers in
:numref:`metadata classes table`, and provides a common gateway for users to
easily manage and manipulate their metadata in a consistent and unified way.

This is primarily all thanks to the metadata classes (:numref:`metadata classes table`)
that support the necessary state and behaviour required by the common metadata
API. Namely, it is the ``equal`` (``__eq__``), ``difference`` and
``combine`` methods that provide this rich metadata behaviour, all of which are
explored more fully in :ref:`metadata`.

Strict behaviour
================

.. testsetup:: strict-behaviour

    import iris
    cube = iris.load_cube(iris.sample_data_path("A1B_north_america.nc"))
    latitude = cube.coord("latitude")

In particular, the feature that is common between the ``equal``, ``difference``
and ``combine`` metadata class methods, is that they all perform **strict**
metadata member comparisons **by default**.

This **strict** behaviour of these methods can be summarised as follows,
where ``X`` and ``Y`` are any object that are non-identical,

.. _strict equality table:
.. table:: - Strict equality
   :widths: auto
   :align: center

   ======== ======== ==========
   left     right    **__eq__**
   ======== ======== ==========
   ``X``    ``Y``    ``False``
   ``Y``    ``X``    ``False``
   ``X``    ``X``    ``True``
   ``X``    ``None`` ``False``
   ``None`` ``X``    ``False``
   ======== ======== ==========

.. _strict difference table:
.. table:: - Strict difference
   :widths: auto
   :align: center

   ======== ======== =================
   left     right    **__eq__**
   ======== ======== =================
   ``X``    ``Y``    (``X``, ``Y``)
   ``Y``    ``X``    (``Y``, ``X``)
   ``X``    ``X``    ``None``
   ``X``    ``None`` (``X``, ``None``)
   ``None`` ``X``    (``None``, ``X``)
   ======== ======== =================

.. _strict combine table:
.. table:: - Strict combination
   :widths: auto
   :align: center

   ======== ======== ==========
   left     right    **__eq__**
   ======== ======== ==========
   ``X``    ``Y``    ``None``
   ``Y``    ``X``    ``None``
   ``X``    ``X``    ``X``
   ``X``    ``None`` ``None``
   ``None`` ``X``    ``None``
   ======== ======== ==========

This type of **strict** behaviour does offer clear benefit and value. However,
it can be unnecessarily restrictive and unforgiving. For example, consider the
metadata of the following ``latitude`` coordinate,

.. doctest:: strict-behaviour

    >>> latitude.metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='latitude', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Now, let's create a doctored version of this metadata that has a different ``var_name``,

.. doctest:: strict-behaviour

    >>> metadata = latitude.metadata._replace(var_name="lat")
    >>> metadata
    DimCoordMetadata(standard_name='latitude', long_name=None, var_name='lat', units=Unit('degrees'), attributes={}, coord_system=GeogCS(6371229.0), climatological=False, circular=False)

Clearly, these metadata are indeed different,

.. doctest:: strict-behaviour

    >>> metadata == latitude.metadata
    False
    >>> metadata.difference(latitude.metadata)
    DimCoordMetadata(standard_name=None, long_name=None, var_name=('lat', 'latitude'), units=None, attributes=None, coord_system=None, climatological=None, circular=None)

If your workflow demands such metadata rigour, then this default behaviour of
the common metadata API will easily satisfy your needs. However, such strict,
unforgiving behaviour can be a small and yet very inconvenient barrier to
repeatedly overcome. In particular, this situation may be exacerbated when
using Iris to analyse data from several disparate data sources and/or file
formats.




*Mauris facilisis imperdiet mi, quis pellentesque urna vulputate id. Ut mi neque, condimentum a augue non, tempor mollis ipsum. Ut nec leo maximus nisi facilisis auctor. In hac habitasse platea dictumst. Nullam et posuere nulla, eget commodo ligula. Nulla eleifend euismod odio, sed vulputate ipsum ornare sit amet. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Nulla luctus, mauris pretium rhoncus ultrices, urna justo pharetra urna, sit amet porta risus diam vitae dui. Pellentesque a leo ligula. Curabitur sit amet augue id elit pretium condimentum quis a nisi. Duis egestas faucibus velit, non blandit augue finibus non. Proin vel tempor dolor, non volutpat tortor. Vestibulum sollicitudin eu elit vel placerat. Sed vestibulum purus lectus, vel feugiat est venenatis sed. In ultrices pharetra elit.*



.. _CF Conventions: https://cfconventions.org/