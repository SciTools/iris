.. _glossary:

Glossary
=============

.. glossary::

   Cartopy
        A python package for producing maps, and other geospatial data.
        Allows plotting on these maps, over a range of projections.

        | **Related:** :term:`MatPlotLib`
        | **More information:** `CartoPy Site <https://scitools.org.uk/cartopy/docs/latest/>`_
        |

   Coordinates
        A container for data points, comes in three main flavours.

        - Dimensional Coordinates -
            A coordinate that represents a singular data dimension.
            These are organised in ascending order of dimension. Can only
            contain numerical data.
        - Auxiliary Coordinates -
            A coordinate that can map to multiple data dimensions. Can
            contain any type of data.
        - Scalar Coordinates -
                A coordinate that is not mapped to any data dimension, instead
                representing the cube as a whole.

        | **Related:** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Cubes
        Cubes are the main method of storing data in Iris. A cube consists of:

        - :term:`Coordinates`
        - Array of :term:`Phenomenon` Data
        - :term:`Standard Name`
        - :term:`Long Name`
        - :term:`Units`
        - List of :term:`Cell Method`
        - List of :term:`Coordinate Factory`

        | **Related:** :term:`NumPy`
        | **More information:** :doc:`iris_cubes`
        |

   Cell Method
        A cell method represent past operations on a cube's data, such as a
        MEAN or SUM operation

        | **Related:** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Coordinate Factory
        A coordinate factory derives coordinates from the values of existing
        coordinates.

        | **Related:** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Dask
        A collection of NumPy-esque arrays, stored in hard disk. When needed,
        the data is temporarily loaded into RAM, and operated on, in batches.

        | **Related:** :term:`Lazy Data` **//** :term:`NumPy`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Lazy Data
        Data stored in hard drive, and then temporarily loaded into RAM in
        batches when needed. Allows of less memory usage and faster performance,
        thanks to parallel processing.

        | **Related:** :term:`Dask` // :term:`Real Data`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Long Name
        A name describing a :term:`phenomenon`, not limited to the
        the same restraints as :term:`standard name`.

        | **Related:** :term:`Standard Name` **||** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   MatPlotLib
        A python package for plotting and projecting data in a wide variety
        of formats.

        | **Related:** :term:`CartoPy` || :term:`NumPy`
        | **More information:** `MatPlotLib <https://scitools.org.uk/cartopy/docs/latest/>`_
        |

   Meta Data
        The data which is used to describe phenomenon data e.g. longitude.

        | **Related:** :term:`Phenomenon` **//** :term:`Cubes`
        | **More information:** :doc:`../further_topics/metadata`
        |

   NumPy
        A mathematical Python library, predominantly based around
        multi-dimensional arrays.

        | **Related:** :term:`Dask`  **//** :term:`Cubes`
        | **More information:** `NumPy.org <https://numpy.org/>`_
        |

   Phenomenon
        The primary data which is measured, usually within a cube, e.g.
        air temperature.

        | **Related:** :term:`Meta Data` **//** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Real Data
        Data that has been loaded into RAM, as opposed to sitting
        on the hard drive.

        | **Related:** :term:`Lazy Data`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Standard Name
        A name describing a :term:`phenomenon`, keeping within
        bounds of `CF Standardisation <http://cfconventions.org/standard-names.html>`_.

        | **Related:** :term:`Long Name` **//** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Units
        The unit with which the phenomenon is measured.

        | **Related:** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

    ----


   `To top <glossary_>`_

