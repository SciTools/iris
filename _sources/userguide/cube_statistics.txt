===============
Cube statistics
===============


Collapsing entire data dimensions
---------------------------------

.. testsetup::

    import iris
    filename = iris.sample_data_path('uk_hires.pp')
    cube = iris.load_strict(filename, 'air_potential_temperature')

    import iris.analysis.cartography
    cube.coord('grid_latitude').guess_bounds()
    cube.coord('grid_longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)


In the section :doc:`reducing_a_cube` we saw how to extract a subset of a cube in order to reduce either its dimensionality or its resolution. 
Instead of downsampling the data, a similar goal can be achieved using statistical operations over *all* of the data. Suppose we have a cube:

    >>> import iris
    >>> filename = iris.sample_data_path('uk_hires.pp')
    >>> cube = iris.load_strict(filename, 'air_potential_temperature')
    >>> print cube
    air_potential_temperature           (time: 3; model_level_number: 7; grid_latitude: 204; grid_longitude: 187)
         Dimension coordinates:
              time                           x                      -                 -                    -
              model_level_number             -                      x                 -                    -
              grid_latitude                  -                      -                 x                    -
              grid_longitude                 -                      -                 -                    x
         Auxiliary coordinates:
              forecast_period                x                      -                 -                    -
              level_height                   -                      x                 -                    -
              sigma                          -                      x                 -                    -
              surface_altitude               -                      -                 x                    x
         Derived coordinates:
              altitude                       -                      x                 x                    x
         Scalar coordinates:
              source: Data from Met Office Unified Model 7.03
         Attributes:
              STASH: m01s00i004




In this case we have a 4 dimensional cube; to mean the vertical (z) dimension down to a single valued extent we can pass the coordinate
name and the aggregation definition to the :meth:`Cube.collapsed() <iris.cube.Cube.collapsed>` method:

    >>> import iris.analysis
    >>> vertical_mean = cube.collapsed('model_level_number', iris.analysis.MEAN)
    >>> print vertical_mean
    air_potential_temperature           (time: 3; grid_latitude: 204; grid_longitude: 187)
         Dimension coordinates:
              time                           x                 -                    -
              grid_latitude                  -                 x                    -
              grid_longitude                 -                 -                    x
         Auxiliary coordinates:
              forecast_period                x                 -                    -
              surface_altitude               -                 x                    x
         Derived coordinates:
              altitude                       -                 x                    x
         Scalar coordinates:
              level_height: Cell(point=696.66663, bound=(0.0, 1393.3333)) m
              model_level_number: Cell(point=10, bound=(1, 19))
              sigma: Cell(point=0.92292976, bound=(0.84585959, 1.0))
              source: Data from Met Office Unified Model 7.03
         Attributes:
              STASH: m01s00i004
              history: Mean of air_potential_temperature over model_level_number
         Cell methods:
              mean: model_level_number


Similarly other analysis operators such as ``MAX``, ``MIN`` and ``STD_DEV`` can be used instead of ``MEAN``, 
see :mod:`iris.analysis` for a full list of currently supported operators.


Area averaging
^^^^^^^^^^^^^^

Some operators support additional keywords to the ``cube.collapsed`` method. For example, :func:`iris.analysis.MEAN <iris.analysis.MEAN>` 
supports a weights keyword which can be combined with :func:`iris.analysis.cartography.area_weights` to calculate an area average.

Let's use the same data as was loaded in the previous example. Since ``grid_latitude`` and ``grid_longitude`` were both
point coordinates we must guess bound positions for them in order to calculate the area of the grid boxes::

    import iris.analysis.cartography
    cube.coord('grid_latitude').guess_bounds()
    cube.coord('grid_longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)

These areas can now be passed to the ``collapsed`` method as weights:

.. doctest::

    >>> new_cube = cube.collapsed(['grid_longitude', 'grid_latitude'], iris.analysis.MEAN, weights=grid_areas)
    >>> print new_cube
    air_potential_temperature           (time: 3; model_level_number: 7)
         Dimension coordinates:
              time                           x                      -
              model_level_number             -                      x
         Auxiliary coordinates:
              forecast_period                x                      -
              level_height                   -                      x
              sigma                          -                      x
         Derived coordinates:
              altitude                       -                      x
         Scalar coordinates:
              grid_latitude: Cell(point=1.5145501, bound=(0.14430022, 2.8848)) degrees
              grid_longitude: Cell(point=358.74948, bound=(357.49399, 360.00497)) degrees
              source: Data from Met Office Unified Model 7.03
              surface_altitude: Cell(point=399.625, bound=(-14.0, 813.25)) m
         Attributes:
              STASH: m01s00i004
              history: Mean of air_potential_temperature over grid_longitude, grid_latitude
         Cell methods:
              mean: grid_longitude, grid_latitude





Partially collapsing data dimensions
------------------------------------

Instead of completely collapsing a dimension, other methods can be applied to reduce or filter the number of data points of a particular dimension. 



Aggregation of grouped data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

An aggregation on a *group* of coordinate values can be achieved with :meth:`Cube.aggregated_by <iris.cube.Cube.aggregated_by>`, 
which can be combined with the :mod:`iris.coord_categorisation` module to group the coordinate in the first place.

First, let's create two coordinates on a cube which represent the climatological seasons and the season year respectively::

    import iris
    import iris.coord_categorisation

    filename = iris.sample_data_path('ostia_monthly.nc')
    cube = iris.load_strict(filename, 'surface_temperature')

    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='season_year')


.. testsetup:: aggregation

    import iris

    filename = iris.sample_data_path('ostia_monthly.nc')
    cube = iris.load_strict(filename, 'surface_temperature')

    import iris.coord_categorisation
    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='season_year')

    annual_seasonal_mean = cube.aggregated_by(['clim_season', 'season_year'], iris.analysis.MEAN)

    
Printing this cube now shows that two extra coordinates exist on the cube:

.. doctest:: aggregation

    >>> print cube
    surface_temperature                 (time: 54; latitude: 18; longitude: 432)
         Dimension coordinates:
              time                           x             -              -
              latitude                       -             x              -
              longitude                      -             -              x
         Auxiliary coordinates:
              clim_season                    x             -              -
              forecast_reference_time        x             -              -
              season_year                    x             -              -
         Scalar coordinates:
              forecast_period: 0 hours
         Attributes:
              Conventions: CF-1.5
              STASH: m01s00i024
              history: Mean of surface_temperature aggregated over month, year
         Cell methods:
              mean: month, year


These two coordinates can now be used as *groups* over which to do an aggregation:

.. doctest:: aggregation

    >>> annual_seasonal_mean = cube.aggregated_by(['clim_season', 'season_year'], iris.analysis.MEAN)
    >>> print repr(annual_seasonal_mean)
    <iris 'Cube' of surface_temperature (*ANONYMOUS*: 19; latitude: 18; longitude: 432)>
    
The primary change in the cube is that the cube's data has shrunk on the t axis as a result of the meaning aggregation. 
We have now collapsed all repeating copies of season (DJF etc.) and year to represent a single position in the t axis.
We can see this by printing the first 10 values of the original coordinates:

.. doctest:: aggregation

    >>> print cube.coord('clim_season')[:10].points
    ['mam' 'mam' 'jja' 'jja' 'jja' 'son' 'son' 'son' 'djf' 'djf']
    >>> print cube.coord('season_year')[:10].points
    [2006 2006 2006 2006 2006 2006 2006 2006 2007 2007]

And then comparing with the first 10 values of the new cube's coordinates:

.. doctest:: aggregation

    >>> print annual_seasonal_mean.coord('clim_season')[:10].points
    ['mam' 'jja' 'son' 'djf' 'mam' 'jja' 'son' 'djf' 'mam' 'jja']
    >>> print annual_seasonal_mean.coord('season_year')[:10].points
    [2006 2006 2006 2007 2007 2007 2007 2008 2008 2008]


Because the original data started in April 2006 we have some incomplete seasons
(e.g. there were only two months worth of data for ``mam 2006``).
In this case we can fix this by removing all of the resultant ``times`` which do not cover a
three month period (n.b. 3 months = 3 * 30 * 24 = 2160 hours):

.. doctest:: aggregation

    >>> spans_three_months = lambda time: (time.bound[1] - time.bound[0]) == 2160
    >>> three_months_bound = iris.Constraint(time=spans_three_months)
    >>> print annual_seasonal_mean.extract(three_months_bound)
    surface_temperature                 (*ANONYMOUS*: 3; latitude: 18; longitude: 432)
         Dimension coordinates:
              latitude                              -            x              -
              longitude                             -            -              x
         Auxiliary coordinates:
              clim_season                           x            -              -
              forecast_reference_time               x            -              -
              season_year                           x            -              -
              time                                  x            -              -
         Scalar coordinates:
              forecast_period: 0 hours
         Attributes:
              Conventions: CF-1.5
              STASH: m01s00i024
              history: Mean of surface_temperature aggregated over month, year
    Mean of surface_temperature...
         Cell methods:
              mean: month, year
              mean: clim_season, season_year



The final result now represents the seasonal mean temperature for 63 seasons starting from ``March April May 1990``.
