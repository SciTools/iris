.. _cube-statistics:

===============
Cube Statistics
===============

.. seealso::

    Relevant gallery example:
    :ref:`sphx_glr_generated_gallery_general_plot_zonal_means.py` (Collapsing)

.. _cube-statistics-collapsing:

Collapsing Entire Data Dimensions
---------------------------------

.. testsetup::

    import iris
    filename = iris.sample_data_path('uk_hires.pp')
    cube = iris.load_cube(filename, 'air_potential_temperature')

    import iris.analysis.cartography
    cube.coord('grid_latitude').guess_bounds()
    cube.coord('grid_longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)


In the :doc:`subsetting_a_cube` section we saw how to extract a subset of a
cube in order to reduce either its dimensionality or its resolution.
Instead of simply extracting a sub-region of the data,
we can produce statistical functions of the data values
across a particular dimension,
such as a 'mean over time' or 'minimum over latitude'.

.. _cube-statistics_forecast_printout:

For instance, suppose we have a cube:

    >>> import iris
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cube = iris.load_cube(filename, 'air_potential_temperature')
    >>> print(cube)
    air_potential_temperature / (K)     (time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
        Dimension coordinates:
            time                             x                      -                 -                    -
            model_level_number               -                      x                 -                    -
            grid_latitude                    -                      -                 x                    -
            grid_longitude                   -                      -                 -                    x
        Auxiliary coordinates:
            forecast_period                  x                      -                 -                    -
            level_height                     -                      x                 -                    -
            sigma                            -                      x                 -                    -
            surface_altitude                 -                      -                 x                    x
        Derived coordinates:
            altitude                         -                      x                 x                    x
        Scalar coordinates:
            forecast_reference_time     2009-11-19 04:00:00
        Attributes:
            STASH                       m01s00i004
            source                      'Data from Met Office Unified Model'
            um_version                  '7.3'


In this case we have a 4 dimensional cube;
to mean the vertical (z) dimension down to a single valued extent
we can pass the coordinate name and the aggregation definition to the
:meth:`Cube.collapsed() <iris.cube.Cube.collapsed>` method:

    >>> import iris.analysis
    >>> vertical_mean = cube.collapsed('model_level_number', iris.analysis.MEAN)
    >>> print(vertical_mean)
    air_potential_temperature / (K)     (time: 3; grid_latitude: 204; grid_longitude: 187)
        Dimension coordinates:
            time                             x                 -                    -
            grid_latitude                    -                 x                    -
            grid_longitude                   -                 -                    x
        Auxiliary coordinates:
            forecast_period                  x                 -                    -
            surface_altitude                 -                 x                    x
        Derived coordinates:
            altitude                         -                 x                    x
        Scalar coordinates:
            forecast_reference_time     2009-11-19 04:00:00
            level_height                696.6666 m, bound=(0.0, 1393.3333) m
            model_level_number          10, bound=(1, 19)
            sigma                       0.92292976, bound=(0.8458596, 1.0)
        Cell methods:
            mean                        model_level_number
        Attributes:
            STASH                       m01s00i004
            source                      'Data from Met Office Unified Model'
            um_version                  '7.3'


Similarly other analysis operators such as ``MAX``, ``MIN`` and ``STD_DEV``
can be used instead of ``MEAN``, see :mod:`iris.analysis` for a full list
of currently supported operators.

For an example of using this functionality, the
:ref:`sphx_glr_generated_gallery_meteorology_plot_hovmoller.py`
example found
in the gallery takes a zonal mean of an ``XYT`` cube by using the
``collapsed`` method with ``latitude`` and ``iris.analysis.MEAN`` as arguments.

.. _cube-statistics-collapsing-average:

Area Averaging
^^^^^^^^^^^^^^

Some operators support additional keywords to the ``cube.collapsed`` method.
For example, :func:`iris.analysis.MEAN <iris.analysis.MEAN>` supports
a weights keyword which can be combined with
:func:`iris.analysis.cartography.area_weights` to calculate an area average.

Let's use the same data as was loaded in the previous example.
Since ``grid_latitude`` and ``grid_longitude`` were both point coordinates
we must guess bound positions for them
in order to calculate the area of the grid boxes::

    import iris.analysis.cartography
    cube.coord('grid_latitude').guess_bounds()
    cube.coord('grid_longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)

These areas can now be passed to the ``collapsed`` method as weights:

.. doctest::

    >>> new_cube = cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN, weights=grid_areas)
    >>> print(new_cube)
    air_potential_temperature / (K)     (time: 3; model_level_number: 7)
        Dimension coordinates:
            time                             x                      -
            model_level_number               -                      x
        Auxiliary coordinates:
            forecast_period                  x                      -
            level_height                     -                      x
            sigma                            -                      x
        Derived coordinates:
            altitude                         -                      x
        Scalar coordinates:
            forecast_reference_time     2009-11-19 04:00:00
            grid_latitude               1.5145501 degrees, bound=(0.14430022, 2.8848) degrees
            grid_longitude              358.74948 degrees, bound=(357.494, 360.00497) degrees
            surface_altitude            399.625 m, bound=(-14.0, 813.25) m
        Cell methods:
            mean                        grid_longitude, grid_latitude
        Attributes:
            STASH                       m01s00i004
            source                      'Data from Met Office Unified Model'
            um_version                  '7.3'

Several examples of area averaging exist in the gallery which may be of interest,
including an example on taking a :ref:`global area-weighted mean
<sphx_glr_generated_gallery_meteorology_plot_COP_1d.py>`.

.. _cube-statistics-aggregated-by:

Partially Reducing Data Dimensions
----------------------------------

Instead of completely collapsing a dimension, other methods can be applied
to reduce or filter the number of data points of a particular dimension.


Aggregation of Grouped Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :meth:`Cube.aggregated_by <iris.cube.Cube.aggregated_by>` operation
combines data for all points with the same value of a given coordinate.
To do this, you need a coordinate whose points take on only a limited set
of different values -- the *number* of these then determines the size of the
reduced dimension.
The :mod:`iris.coord_categorisation` module can be used to make such
'categorical' coordinates out of ordinary ones: The most common use is
to aggregate data over regular *time intervals*,
such as by calendar month or day of the week.

For example, let's create two new coordinates on the cube
to represent the climatological seasons and the season year respectively::

    import iris
    import iris.coord_categorisation

    filename = iris.sample_data_path('ostia_monthly.nc')
    cube = iris.load_cube(filename, 'surface_temperature')

    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='season_year')



.. note::

    The 'season year' is not the same as year number, because (e.g.) the months
    Dec11, Jan12 + Feb12 all belong to 'DJF-12'.
    See :meth:`iris.coord_categorisation.add_season_year`.


.. testsetup:: aggregation

    import datetime
    import iris

    filename = iris.sample_data_path('ostia_monthly.nc')
    cube = iris.load_cube(filename, 'surface_temperature')

    import iris.coord_categorisation
    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='season_year')

    annual_seasonal_mean = cube.aggregated_by(
         ['clim_season', 'season_year'],
         iris.analysis.MEAN)


Printing this cube now shows that two extra coordinates exist on the cube:

.. doctest:: aggregation

    >>> print(cube)
    surface_temperature / (K)           (time: 54; latitude: 18; longitude: 432)
        Dimension coordinates:
            time                             x             -              -
            latitude                         -             x              -
            longitude                        -             -              x
        Auxiliary coordinates:
            clim_season                      x             -              -
            forecast_reference_time          x             -              -
            season_year                      x             -              -
        Scalar coordinates:
            forecast_period             0 hours
        Cell methods:
            mean                        month, year
        Attributes:
            Conventions                 'CF-1.5'
            STASH                       m01s00i024


These two coordinates can now be used to aggregate by season and climate-year:

.. doctest:: aggregation

    >>> annual_seasonal_mean = cube.aggregated_by(
    ...     ['clim_season', 'season_year'],
    ...     iris.analysis.MEAN)
    >>> print(repr(annual_seasonal_mean))
    <iris 'Cube' of surface_temperature / (K) (time: 19; latitude: 18; longitude: 432)>

The primary change in the cube is that the cube's data has been
reduced in the 'time' dimension by aggregation (taking means, in this case).
This has collected together all data points with the same values of season and
season-year.
The results are now indexed by the 19 different possible values of season and
season-year in a new, reduced 'time' dimension.

We can see this by printing the first 10 values of season+year
from the original cube:  These points are individual months,
so adjacent ones are often in the same season:

.. doctest:: aggregation
    :options: +NORMALIZE_WHITESPACE

    >>> for season, year in zip(cube.coord('clim_season')[:10].points,
    ...                         cube.coord('season_year')[:10].points):
    ...     print(season + ' ' + str(year))
    mam 2006
    mam 2006
    jja 2006
    jja 2006
    jja 2006
    son 2006
    son 2006
    son 2006
    djf 2007
    djf 2007

Compare this with the first 10 values of the new cube's coordinates:
All the points now have distinct season+year values:

.. doctest:: aggregation
    :options: +NORMALIZE_WHITESPACE

    >>> for season, year in zip(
    ...         annual_seasonal_mean.coord('clim_season')[:10].points,
    ...         annual_seasonal_mean.coord('season_year')[:10].points):
    ...     print(season + ' ' + str(year))
    mam 2006
    jja 2006
    son 2006
    djf 2007
    mam 2007
    jja 2007
    son 2007
    djf 2008
    mam 2008
    jja 2008

Because the original data started in April 2006 we have some incomplete seasons
(e.g. there were only two months worth of data for 'mam-2006').
In this case we can fix this by removing all of the resultant 'times' which
do not cover a three month period (note: judged here as > 3*28 days):

.. doctest:: aggregation

    >>> tdelta_3mth = datetime.timedelta(hours=3*28*24.0)
    >>> spans_three_months = lambda t: (t.bound[1] - t.bound[0]) > tdelta_3mth
    >>> three_months_bound = iris.Constraint(time=spans_three_months)
    >>> full_season_means = annual_seasonal_mean.extract(three_months_bound)
    >>> full_season_means
    <iris 'Cube' of surface_temperature / (K) (time: 17; latitude: 18; longitude: 432)>

The final result now represents the seasonal mean temperature for 17 seasons
from jja-2006 to jja-2010:

.. doctest:: aggregation
    :options: +NORMALIZE_WHITESPACE

    >>> for season, year in zip(full_season_means.coord('clim_season').points,
    ...                         full_season_means.coord('season_year').points):
    ...     print(season + ' ' + str(year))
    jja 2006
    son 2006
    djf 2007
    mam 2007
    jja 2007
    son 2007
    djf 2008
    mam 2008
    jja 2008
    son 2008
    djf 2009
    mam 2009
    jja 2009
    son 2009
    djf 2010
    mam 2010
    jja 2010

