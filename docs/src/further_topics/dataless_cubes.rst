.. _dataless-cubes:

==============
Dataless Cubes
==============
It is possible for a cube to exist without a data payload.
In this case ``cube.data`` is ``None``, instead of containing an array (real or lazy) as
usual.

This can be useful when the cube is used purely as a placeholder for metadata, e.g. to
represent a combination of coordinates.

Most notably, dataless cubes can be used as the target "grid cube" for most regridding
schemes, since in that case the cube's coordinates are all that the method uses.
See also :meth:`iris.util.make_gridcube`.


Properties of dataless cubes
----------------------------

* ``cube.shape`` is unchanged
* ``cube.data`` == ``None``
* ``cube.dtype`` == ``None``
* ``cube.core_data()`` == ``cube.lazy_data()`` == ``None``
* ``cube.is_dataless()`` == ``True``
* ``cube.has_lazy_data()`` == ``False``


Cube creation
-------------
You can create a dataless cube with the :meth:`~iris.cube.Cube` constructor
(i.e. ``__init__`` call), by specifying the ``shape`` keyword in place of ``data``.
If both are specified, an error is raised (even if data and shape are compatible).


Data assignment
---------------
You can make an existing cube dataless, by setting ``cube.data = None``.
The data array is simply discarded.

Likewise, you can add data by assigning any data array of the correct shape, which
turns it into a 'normal' cube.

Note that ``cube.dtype`` always matches ``cube.data.dtype``.  A dataless cube has a
dtype of ``None``.


Cube copy
---------
The syntax that allows you to replace data on copying,
e.g. ``cube2 = cube.copy(new_data)``, has been extended to accept the special value
:data:`iris.DATALESS`.

So, ``cube2 = cube.copy(iris.DATALESS)`` makes ``cube2`` a
dataless copy of ``cube``.
This is equivalent to ``cube2 = cube.copy(); cube2.data = None``.


Save and Load
-------------
The netcdf file interface can save and re-load dataless cubes correctly.
See: :ref:`save_load_dataless`.

.. _dataless_merge:

Merging
-------
Merging is fully supported for dataless cubes, including combining them with "normal"
cubes.

* in all cases, the result has the same shape and metadata as if the same cubes had
  data.
* Merging multiple dataless cubes produces a dataless result.
* Merging dataless and non-dataless cubes results in a partially 'missing' data array,
  i.e. the relevant sections are filled with masked data.
* Laziness is also preserved.


Operations NOT supported
-------------------------
Dataless cubes are relatively new, and only partly integrated with Iris cube operations
generally.

The following are some of the notable features which do *not* support dataless cubes,
at least as yet :

* plotting

* cube arithmetic

* statistics

* concatenation

* :meth:`iris.cube.CubeList.realise_data`

* various :class:`~iris.cube.Cube` methods, including at least:

  * :meth:`~iris.cube.Cube.convert_units`

  * :meth:`~iris.cube.Cube.subset`

  * :meth:`~iris.cube.Cube.intersection`

  * :meth:`~iris.cube.Cube.slices`

  * :meth:`~iris.cube.Cube.interpolate`

  * :meth:`~iris.cube.Cube.regrid`
    Note: in this case the target ``grid`` can be dataless, but not the source
    (``self``) cube.
