.. _glossary:

Glossary
=============

Table Of Contents
_________________
    | - :term:`Cartopy`
    | - :term:`Coordinates`
    | - :term:`Cubes`
    | - :term:`Cell Method`
    | - :term:`Coordinate Factory`
    | - :term:`Dask`
    | - :term:`Fields File (FF) Format`
    | - :term:`GRIB Format`
    | - :term:`Lazy Data`
    | - :term:`Long Name`
    | - :term:`Matplotlib`
    | - :term:`Metadata`
    | - :term:`Post Processing (PP) Format`
    | - :term:`NumPy`
    | - :term:`Phenomenon`
    | - :term:`NetCDF Format`
    | - :term:`Real Data`
    | - :term:`Standard Name`
    | - :term:`Units`
    | - :term:`Xarray`


.. glossary::

   Cartopy
        A python package for producing maps, and other geospatial data.
        Allows plotting on these maps, over a range of projections.

        | **Related:** :term:`Matplotlib`
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
        Cubes are the main method of storing data in Iris. A cube can consist of:

        - Array of :term:`Phenomenon` Data (Required)
        - :term:`Coordinates`
        - :term:`Standard Name`
        - :term:`Long Name`
        - :term:`Units`
        - :term:`Cell Method`
        - :term:`Coordinate Factory`

        | **Related:** :term:`NumPy`
        | **More information:** :doc:`iris_cubes`
        |

   Cell Method
        A cell method represent past operations on a cube's data, such as a
        MEAN or SUM operation.

        | **Related:** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Coordinate Factory
        A coordinate factory derives coordinates (sometimes referred to as
        derived coordinates) from the values of existing coordinates.
        E.g. A hybrid height factory might use "height above sea level"
        and "height at ground level" coordinate data to calculate a
        "height above ground level" coordinate.

        | **Related:** :term:`Cubes`
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

        | **Related:** :term:`Standard Name` **|** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Matplotlib
        A python package for plotting and projecting data in a wide variety
        of formats.

        | **Related:** :term:`CartoPy` **|** :term:`NumPy`
        | **More information:** `Matplotlib <https://scitools.org.uk/cartopy/docs/latest/>`_
        |

   Metadata
        The data which is used to describe phenomenon data e.g. longitude.

        | **Related:** :term:`Phenomenon` **|** :term:`Cubes`
        | **More information:** :doc:`../further_topics/metadata`
        |

   NetCDF Format
        A meteorological file format; this is the data model
        iris is based on. Follows `CF Conventions <http://cfconventions.org/>`_.

        | **Related:** :term:`Fields File (FF) Format`
         **|** :term:`GRIB Format` **|** :term:`Post Processing (PP) Format`
        | **More information:** `NetCD-4 Python Git <https://github.com/Unidata/netcdf4-python>`_
        |

   NumPy
        A mathematical Python library, predominantly based around
        multi-dimensional arrays.

        | **Related:** :term:`Dask`  **|** :term:`Cubes`
         **|** :term:`Xarray`
        | **More information:** `NumPy.org <https://numpy.org/>`_
        |

   Phenomenon
        The primary data which is measured, usually within a cube, e.g.
        air temperature.

        | **Related:** :term:`Metadata` **|** :term:`Cubes`
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

        | **Related:** :term:`Lazy Data`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Standard Name
        A name describing a :term:`phenomenon`, keeping within
        bounds of `CF Standardisation <http://cfconventions.org/standard-names.html>`_.

        | **Related:** :term:`Long Name` **|** :term:`Cubes`
        | **More information:** :doc:`iris_cubes`
        |

   Units
        The unit with which the phenomenon is measured.

        | **Related:** :term:`Cubes`
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

