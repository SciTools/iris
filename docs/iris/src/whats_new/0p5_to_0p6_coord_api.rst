.. _new_coord_api:

================================
The new Coord API in release 0.6
================================

In order to support some forthcoming Iris features, a new coordinate API has been designed to replace the old.
Dictionary access to coordinates from a cube (via the ``cube.coords`` property), has been replaced by the 
``cube.coord`` and ``cube.coords`` methods.

Examples of the new API
-----------------------

::

	# Add a coord
	level_coord = ExplicitCoord("level", unit="1", points=[10, 20, 30])
	cube.add_coord(level_coord)
	
	# Retrieve the "level" coord (or fail if there is not exactly one found)
	brian_coord = cube.coord("level")

	# Remove the "level" coord
	cube.remove_coord("level")

	# Retrieve a list of non-dimensional coords (may be an empty list)
	coords = cube.coords(dimensions=[])


How to modify existing code
---------------------------

To update your Iris 0.5 code for the 0.6 API, the following guidelines will be of assistance.

Add a coordinate to a cube
""""""""""""""""""""""""""

Iris version 0.5 allowed coordinate addition with::

	cube.coords["level"] = level_coord

As of Iris 0.6 a new method, ``cube.add_coord`` should be used instead::

	cube.add_coord(level_coord)

Retrieving a single coordinate from a cube
""""""""""""""""""""""""""""""""""""""""""

Iris version 0.5 allowed coordinate retrieval with::
	
	level_coord = cube.coords["level"]

As of Iris 0.6 the new method, ``cube.coord`` should be used instead::

	level_coord = cube.coord("level")


Retrieving multiple coordinates from a cube
"""""""""""""""""""""""""""""""""""""""""""

In Iris version 0.5 it was possible to get all non-data describing coordinates with::
	
	my_coords = cube.query_coords(dimensions=[])

The new API's equivalent is through using ``cube.coords``::

	coords = cube.coords(dimensions=[])


Identifying coordinate data dimensions from a cube
""""""""""""""""""""""""""""""""""""""""""""""""""

In Iris version 0.5 it was possible to identify the data dimension of a coordinate with::

    data_dim = cube.axes[level_coord.axis_name]
    
The new API's equivalent is through using ``cube.coord_dims``, which returns a list of data dimensions::

    data_dims = cube.coord_dims(level_coord)


Further help
------------
Many of the changes should be a trivial search & replace for ``.coords[``, ``.axes`` and ``.query_coords``. 
If you have any complications in converting your code, please do not hesitate to contact the Iris team. 
