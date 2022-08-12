.. _glossary:

Glossary
=============

----

.. _cartopy_gl:

**Cartopy** -

    A python package for producing maps, and other geospatial data. Allows plotting on these maps, over a range of projections.

    Related: `MatPlotLib <matplotlib_gl_>`_

    More information: `CartoPy Site <https://scitools.org.uk/cartopy/docs/latest/>`_

`To top <glossary_>`_

----

.. _coords_gl:

**Coordinates** -

    A container for data points, comes in three main flavours.

    - **Dimensional Coordinates** -
        A coordinate that represents a singular data dimension. These are organised in ascending order of dimension. Can only contain numerical data.
    - **Auxiliary Coordinates** -
        A coordinate that can map to multiple data dimensions. Can contain any type of data.
    - **Scalar Coordinates** -
        A coordinate that is not mapped to any data dimension, instead representing the cube as a whole.

    Related: `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _cubes_gl:

**Cubes** -

    Cubes are the main method of storing data in Iris. A cube consists of:

    - `Coordinates <coords_gl_>`_
    - **Array of** `Phenomenon Data <phenomenon_gl_>`_
    - `Standard Name <standard_name_gl_>`_
    - `Long Name <long_name_gl_>`_
    - `Units <units_gl_>`_
    - **List of** `Cell methods <cell_method_gl_>`_
    - **List of** `Coordinate factories <coordinate_factory_gl_>`_


    Related: `NumPy <numpy_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _cell_method_gl:

**Cell Method** -

    A cell method represent past operations on a cube's data, such as a MEAN or SUM operation

    Related: `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _coordinate_factory_gl:

**Coordinate Factory** -

    A coordinate factory derives coordinates from the values of existing coordinates.

    Related: `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _dask_gl:

**Dask** -

    A collection of NumPy-esque arrays, stored in hard disk. When needed, the data is temporarily loaded into RAM, and operated on, in batches.

    Related: `Lazy Data <lazy_data_gl_>`_ \| `NumPy <numpy_gl_>`_

    More information: :doc:`real_and_lazy_data`

`To top <glossary_>`_

----

.. _lazy_data_gl:

**Lazy Data** -

    Data stored in hard drive, and then temporarily loaded into RAM in batches when needed. Allows of less memory usage and faster performance, thanks to parallel processing.

    Related: `Dask <dask_gl_>`_ | `Real Data <real_data_gl_>`_

    More information: :doc:`real_and_lazy_data`

`To top <glossary_>`_

----

.. _long_name_gl:

**Long Name** -

    A name describing a `phenomenon <phenomenon_gl_>`_, not limited to the the same restraints as `standard names <standard_name_gl_>`_.

    Related: `Standard Name <standard_name_gl_>`_ | `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _matplotlib_gl:

**MatPlotLib** -

    A python package for plotting and projecting data in a wide variety of formats.

    Related: `CartoPy <cartopy_gl_>`_ | `NumPy <numpy_gl_>`_

    More information: `MatPlotLib <https://scitools.org.uk/cartopy/docs/latest/>`_

`To top <glossary_>`_

----

.. _metadata_gl:

**Meta Data** -

    The data which is used to describe phenomenon data e.g. longitude.

    Related: `Phenomenon <phenomenon_gl_>`_ | `Cubes <cubes_gl_>`_

    More information: :doc:`../further_topics/metadata`

`To top <glossary_>`_

----

.. _numpy_gl:

**NumPy** -

    A mathematical Python library, predominantly based around multi-dimensional .

    Related: `Dask <dask_gl_>`_ | `Cubes <cubes_gl_>`_

    More information: `NumPy.org <https://numpy.org/>`_

`To top <glossary_>`_

----

.. _phenomenon_gl:

**Phenomenon** -

    The primary data which is measured, usually within a cube, e.g. air temperature.

    Related: `Meta Data <metadata_gl_>`_ | `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _real_data_gl:

**Real Data** -

    Data that has been loaded into RAM, as opposed to sitting on the hard drive.

    Related: `Lazy Data <lazy_data_gl_>`_

    More information: :doc:`real_and_lazy_data`

`To top <glossary_>`_

----

.. _standard_name_gl:

**Standard Name** -

    A name describing a `phenomenon <phenomenon_gl_>`_, keeping within bounds of `CF Standardisation <http://cfconventions.org/standard-names.html>`_.

    Related: `Long Name <long_name_gl_>`_ | `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----

.. _units_gl:

**Units** -

    The unit with which the phenomenon is measured.

    Related: `Cubes <cubes_gl_>`_

    More information: :doc:`iris_cubes`

`To top <glossary_>`_

----