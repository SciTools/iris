.. include:: ../common_links.inc

.. _glossary:

Glossary
=============

.. glossary::

   Cartopy
        A python package for producing maps, and other geospatial data.
        Allows plotting on these maps, over a range of projections.

        | **Related:** :term:`Matplotlib`
        | **More information:** `CartoPy Site <https://scitools.org.uk/cartopy/docs/latest/>`_
        |

   CF Conventions
        Rules for storing meteorological Climate and Forecast data in
        :term:`NetCDF Format` files, defining a standard metadata format to
        describe what the data is.
        This also forms the data model which iris is based on.

        | **Related:** :term:`NetCDF Format`
        | **More information:** `CF Conventions <http://cfconventions.org/>`_
        |

   Coordinate
        A container for data points, comes in three main flavours.

        - Dimensional Coordinate -
            A coordinate that describes a single data dimension of a cube.
            They can only contain numerical values, in a sorted order (ascending
            or descending).
        - Auxiliary Coordinate -
            A coordinate that can map to multiple data dimensions. Can
            contain any type of data.
        - Scalar Coordinate -
                A coordinate that is not mapped to any data dimension, instead
                representing the cube as a whole.

        | **Related:** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Cube
        Cubes are the main method of storing data in Iris. A cube can consist of:

        - Array of :term:`Phenomenon` Data (Required)
        - :term:`Coordinates <Coordinate>`
        - :term:`Standard Name`
        - :term:`Long Name`
        - :term:`Unit`
        - :term:`Cell Methods <Cell Method>`
        - :term:`Coordinate Factories <Coordinate Factory>`

        | **Related:** :term:`NumPy`
        | **More information:** :doc:`iris_cubes`
        |

   Cell Method
        A cell method represents that a cube's data has been derived from
        a past statistical operation, such as a
        MEAN or SUM operation.

        | **Related:** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Coordinate Factory
        A coordinate factory derives coordinates (sometimes referred to as
        derived coordinates) from the values of existing coordinates.
        E.g. A hybrid height factory might use "height above sea level"
        and "height at ground level" coordinate data to calculate a
        "height above ground level" coordinate.

        | **Related:** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |


   Dask
        A data analytics python library. Iris predominantly uses Dask Arrays;
        a collection of NumPy-esque arrays. The data is operated in batches,
        so that not all data is in RAM at once.

        | **Related:** :term:`Lazy Data` **|** :term:`NumPy`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Fields File (FF) Format
        A meteorological file format, the output of the Unified Model.

        | **Related:**  :term:`GRIB Format`
         **|** :term:`Post Processing (PP) Format` **|** :term:`NetCDF Format`
        | **More information:** `Unified Model <https://www.metoffice.gov.uk/research/approach/modelling-systems/unified-model/index>`_
        |

   GRIB Format
        A WMO-standard meteorological file format.

        | **Related:** :term:`Fields File (FF) Format`
         **|** :term:`Post Processing (PP) Format` **|** :term:`NetCDF Format`
        | **More information:** `GRIB 1 User Guide <https://old.wmo.int/extranet/pages/prog/www/WMOCodes/Guides/GRIB/GRIB1-Contents.html>`_
         **|** `GRIB 2 User Guide.pdf <https://old.wmo.int/extranet/pages/prog/www/WMOCodes/Guides/GRIB/GRIB2_062006.pdf>`_
        |

   Lazy Data
        Data stored in hard drive, and then temporarily loaded into RAM in
        batches when needed. Allows of less memory usage and faster performance,
        thanks to parallel processing.

        | **Related:** :term:`Dask` **|** :term:`Real Data`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Long Name
        A name describing a :term:`phenomenon`, not limited to the
        the same restraints as :term:`standard name`.

        | **Related:** :term:`Standard Name` **|** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Matplotlib
        A python package for plotting and projecting data in a wide variety
        of formats.

        | **Related:** :term:`CartoPy` **|** :term:`NumPy`
        | **More information:**  `matplotlib`_
        |

   Metadata
        The information which describes a phenomenon.
        Within Iris specifically, all information which
        distinguishes one phenomenon from another,
        e.g. :term:`units <Unit>` or :term:`Cell Methods <Cell Method>`

        | **Related:** :term:`Phenomenon` **|** :term:`Cube`
        | **More information:** :doc:`../further_topics/metadata`
        |

   NetCDF Format
        A flexible file format for storing multi-dimensional array-like data.
        When Iris loads this format, it also especially recognises and interprets data
        encoded according to the :term:`CF Conventions`.

        __ `NetCDF4`_

        | **Related:** :term:`Fields File (FF) Format`
         **|** :term:`GRIB Format` **|** :term:`Post Processing (PP) Format`
        | **More information:** `NetCDF-4 Python Git`__
        |

   NumPy
        A mathematical Python library, predominantly based around
        multi-dimensional arrays.

        | **Related:** :term:`Dask`  **|** :term:`Cube`
         **|** :term:`Xarray`
        | **More information:** `NumPy.org <https://numpy.org/>`_
        |

   Phenomenon
        The primary data which is measured, usually within a cube, e.g.
        air temperature.

        | **Related:** :term:`Metadata`
         **|** :term:`Standard Name` **|** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Post Processing (PP) Format
        A meteorological file format, created from a post processed
        :term:`Fields File (FF) Format`.

        | **Related:** :term:`GRIB Format` **|** :term:`NetCDF Format`
        | **More information:** `PP Wikipedia Page <https://en.wikipedia.org/wiki/PP-format>`_
        |

   Real Data
        Data that has been loaded into RAM, as opposed to sitting
        on the hard drive.

        | **Related:** :term:`Lazy Data` **|** :term:`NumPy`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Standard Name
        A name describing a :term:`phenomenon`,  one from a fixed list
        defined at `CF Standard Names <http://cfconventions.org/standard-names.html>`_.

        | **Related:** :term:`Long Name` **|** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Unit
        The unit with which the :term:`phenomenon` is measured e.g. m / sec.

        | **Related:** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Xarray
        A python library for sophisticated labelled multi-dimensional operations.
        Has a broader scope than Iris - it is not focused on meteorological data.

        | **Related:** :term:`NumPy`
        | **More information:** `Xarray Documentation <https://docs.xarray.dev/en/stable/index.html>`_
        |

----

`To top <glossary_>`_
