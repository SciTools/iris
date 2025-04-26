.. include:: ../common_links.inc

.. _glossary:

Glossary
=============

.. glossary::

   Aggregation
        The process of summarising data across one or more dimensions, for example by computing the mean, sum, or maximum across a given coordinate. In Iris, aggregation operations help reduce data complexity and are often used before visualisation or analysis.

        | **Related:** :term:`Cube` **|** :term:`Cell Method`
        | **More information:** :doc:`iris_cubes`
        |

   Cartopy
        A Python package for producing maps and working with geospatial data. Cartopy enables plotting of data over a range of map projections, and integrates well with Matplotlib.

        | **Related:** :term:`Matplotlib` **|** :term:`Projection`
        | **More information:** `CartoPy Site <https://scitools.org.uk/cartopy/docs/latest/>`_
        |

   CF Conventions
        A set of rules for storing Climate and Forecast (CF) metadata in :term:`NetCDF Format` files. These conventions define a standard metadata model that describes the meaning of the data and underpin Iris's internal data structure.

        | **Related:** :term:`NetCDF Format` **|** :term:`Metadata`
        | **More information:** `CF Conventions <https://cfconventions.org/>`_
        |

   Cell Method
        Describes the statistical operations that have been performed to derive the data values within a :term:`Cube`, such as averaging (MEAN), summing (SUM), or finding minima or maxima.

        | **Related:** :term:`Cube` **|** :term:`Aggregation`
        | **More information:** :doc:`iris_cubes`
        |

   Constraint
        A way of selecting subsets of data from a :term:`Cube` or :term:`CubeList`, based on metadata conditions such as names, attributes, or coordinate values.

        | **Related:** :term:`Cube` **|** :term:`CubeList`
        | **More information:** :doc:`../userguide/constraints`
        |

   Coordinate
        A container for data points that describe properties along one or more dimensions. Iris distinguishes three types:

        - Dimensional Coordinate  
            Describes a single data dimension, contains numerical values ordered either ascending or descending.
        - Auxiliary Coordinate  
            Provides additional information, can map to multiple dimensions and may contain non-numerical data.
        - Scalar Coordinate  
            Applies to an entire dataset and is not associated with any particular dimension.

        | **Related:** :term:`Cube` **|** :term:`Coordinate System`
        | **More information:** :doc:`iris_cubes`
        |

   Coordinate Factory
        A mechanism for deriving new coordinates from existing ones. For example, a factory might compute "height above ground level" using "height above sea level" and "surface height" coordinates.

        | **Related:** :term:`Cube` **|** :term:`Coordinate`
        | **More information:** :doc:`iris_cubes`
        |

   Coordinate System
        Metadata that describes the spatial reference system associated with coordinates. This ensures consistency in geospatial analyses and projections, especially when working with Cartopy.

        | **Related:** :term:`Coordinate` **|** :term:`Projection`
        | **More information:** :doc:`../userguide/coord_systems`
        |

   Cube
        The primary data structure in Iris for holding scientific data. A Cube contains:

        - An array of :term:`Phenomenon` data (required)
        - One or more :term:`Coordinates <Coordinate>`
        - :term:`Standard Name` and/or :term:`Long Name`
        - :term:`Unit`
        - :term:`Cell Methods <Cell Method>`
        - :term:`Coordinate Factories <Coordinate Factory>`

        | **Related:** :term:`CubeList` **|** :term:`NumPy`
        | **More information:** :doc:`iris_cubes`
        |

   CubeList
        A collection of multiple :term:`Cubes`, often used for batch operations, loading datasets, or storing results of splitting or merging processes.

        | **Related:** :term:`Cube`
        | **More information:** :doc:`../userguide/cubelist`
        |

   Dask
        A Python library for parallel computing and handling large datasets. Iris uses Dask Arrays to process data lazily, allowing operations on data that are too large to fit entirely in memory.

        | **Related:** :term:`Lazy Data` **|** :term:`NumPy`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Fields File (FF) Format
        A meteorological file format, typically output from the Unified Model. It often undergoes conversion or post-processing before scientific analysis.

        | **Related:** :term:`GRIB Format` **|** :term:`Post Processing (PP) Format` **|** :term:`NetCDF Format`
        | **More information:** `Unified Model <https://www.metoffice.gov.uk/research/approach/modelling-systems/unified-model/index>`_
        |

   GRIB Format
        A standard meteorological file format defined by the WMO for storing gridded data, supporting both GRIB1 and GRIB2 versions.

        | **Related:** :term:`Fields File (FF) Format` **|** :term:`Post Processing (PP) Format` **|** :term:`NetCDF Format`
        | **More information:** `GRIB 1 User Guide <https://old.wmo.int/extranet/pages/prog/www/WMOCodes/Guides/GRIB/GRIB1-Contents.html>`_ **|** `GRIB 2 User Guide <https://old.wmo.int/extranet/pages/prog/www/WMOCodes/Guides/GRIB/GRIB2_062006.pdf>`_
        |

   Lazy Data
        Data that is not immediately loaded into memory but accessed in small parts when required. This enables working with large datasets efficiently.

        | **Related:** :term:`Dask` **|** :term:`Real Data`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Long Name
        A human-readable description of a :term:`Phenomenon`, which is not restricted by the fixed controlled vocabulary required for a :term:`Standard Name`.

        | **Related:** :term:`Standard Name` **|** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Matplotlib
        A popular Python package for creating static, animated, and interactive plots. Used within Iris workflows to visualise data.

        | **Related:** :term:`Cartopy` **|** :term:`NumPy`
        | **More information:** `matplotlib <https://matplotlib.org/>`_
        |

   Metadata
        Information that describes the data, including its physical meaning, units, cell methods, and coordinate descriptions. In Iris, metadata ensures that different :term:`Phenomena` can be properly compared and interpreted.

        | **Related:** :term:`Phenomenon` **|** :term:`Cube`
        | **More information:** :doc:`../further_topics/metadata`
        |

   NetCDF Format
        A widely used, flexible file format for storing array-oriented scientific data. When reading NetCDF files, Iris can automatically interpret metadata encoded according to the :term:`CF Conventions`.

        | **Related:** :term:`Fields File (FF) Format` **|** :term:`GRIB Format` **|** :term:`Post Processing (PP) Format`
        | **More information:** `NetCDF-4 Python Git <https://github.com/Unidata/netcdf4-python>`_
        |

   NumPy
        A fundamental Python library for numerical computations, particularly efficient handling of multi-dimensional arrays, used throughout Iris for data management.

        | **Related:** :term:`Dask` **|** :term:`Cube` **|** :term:`Xarray`
        | **More information:** `NumPy.org <https://numpy.org/>`_
        |

   Phenomenon
        The underlying scientific quantity measured or simulated, for example air temperature, humidity, or wind speed, which Iris stores within a :term:`Cube`.

        | **Related:** :term:`Metadata` **|** :term:`Standard Name` **|** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Post Processing (PP) Format
        A meteorological file format, typically resulting from the post-processing of :term:`Fields File (FF) Format` data for analysis and visualisation.

        | **Related:** :term:`GRIB Format` **|** :term:`NetCDF Format`
        | **More information:** `PP Wikipedia Page <https://en.wikipedia.org/wiki/PP-format>`_
        |

   Projection
        A mathematical mapping from the curved surface of the Earth onto a flat plane, crucial for accurate geospatial data visualisation. Iris uses Cartopy for applying projections.

        | **Related:** :term:`Cartopy` **|** :term:`Coordinate System`
        | **More information:** :doc:`../userguide/coord_systems`
        |

   Real Data
        Data that has been fully loaded into RAM, allowing immediate operations without deferred computation.

        | **Related:** :term:`Lazy Data` **|** :term:`NumPy`
        | **More information:** :doc:`real_and_lazy_data`
        |

   Regridding
        The process of interpolating data from one grid to another, often required when datasets have different spatial resolutions or coordinate systems.

        | **Related:** :term:`Cube` **|** :term:`Coordinate`
        | **More information:** :doc:`../userguide/regridding_and_collapse`
        |

   Standard Name
        A controlled vocabulary name describing a :term:`Phenomenon`, following the CF Conventions standard names list to promote consistency and interoperability.

        | **Related:** :term:`Long Name` **|** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Unit
        The measurement unit associated with the :term:`Phenomenon`, for example metres per second (m/s) or degrees Celsius (°C).

        | **Related:** :term:`Cube`
        | **More information:** :doc:`iris_cubes`
        |

   Xarray
        A Python library offering labelled multi-dimensional arrays and datasets. Xarray is more general-purpose than Iris and is often used in complementary workflows.

        | **Related:** :term:`NumPy`
        | **More information:** `Xarray Documentation <https://docs.xarray.dev/en/stable/index.html>`_
        |
----

`To top <glossary_>`_
