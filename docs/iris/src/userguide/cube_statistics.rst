===============
Cube statistics
===============


Collapsing entire data dimensions
---------------------------------

.. testsetup::

    import iris
    filename = iris.sample_data_path('PP', 'globClim1', 'theta.pp')
    cube = iris.load_strict(filename)

    import iris.analysis.cartography
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)


In the section :doc:`reducing_a_cube` we saw how to extract a subset of a cube in order to reduce either its dimensionality or its resolution. 
Instead of downsampling the data, a similar goal can be achieved using statistical operations over *all* of the data. Suppose we have a cube:

    >>> import iris
    >>> filename = iris.sample_data_path('PP', 'globClim1', 'theta.pp')
    >>> cube = iris.load_strict(filename)
    >>> print cube
    air_potential_temperature           (level_height: 38; latitude: 145; longitude: 192)
         Dimension coordinates:
              level_height                           x             -               -
              latitude                               -             x               -
              longitude                              -             -               x
         Auxiliary coordinates:
              model_level_number                     x             -               -
              sigma                                  x             -               -
         Scalar coordinates:
              forecast_period: 26280 hours
              forecast_reference_time: 306816.0 hours since 1970-01-01 00:00:00
              source: Data from Met Office Unified Model 6.06
              time: Cell(point=332724.0, bound=(332352.0, 333096.0)) hours since 1970-01-01 00:00:00
         Attributes:
              STASH: m01s00i004
         Cell methods:
              mean: time (1 hour)


In this case we have a 3 dimensional cube; to mean the z axis down to a single value extent we can pass the coordinate 
name and the aggregation definition to the :meth:`Cube.collapsed() <iris.cube.Cube.collapsed>` method:

    >>> import iris.analysis
    >>> vertical_mean = cube.collapsed('model_level_number', iris.analysis.MEAN)
    >>> print vertical_mean
    air_potential_temperature           (latitude: 145; longitude: 192)
         Dimension coordinates:
              latitude                           x               -
              longitude                          -               x
         Scalar coordinates:
              forecast_period: 26280 hours
              forecast_reference_time: 306816.0 hours since 1970-01-01 00:00:00
              level_height: Cell(point=21213.951, bound=(0.0, 42427.902)) m
              model_level_number: Cell(point=19, bound=(1, 38))
              sigma: Cell(point=0.5, bound=(0.0, 1.0))
              source: Data from Met Office Unified Model 6.06
              time: Cell(point=332724.0, bound=(332352.0, 333096.0)) hours since 1970-01-01 00:00:00
         Attributes:
              STASH: m01s00i004
              history: Mean of air_potential_temperature over model_level_number
         Cell methods:
              mean: time (1 hour)
              mean: model_level_number


Similarly other analysis operators such as ``MAX``, ``MIN`` and ``STD_DEV`` can be used instead of ``MEAN``, 
see :mod:`iris.analysis` for a full list of currently supported operators.


Area averaging
^^^^^^^^^^^^^^

Some operators support additional keywords to the ``cube.collapsed`` method. For example, :func:`iris.analysis.MEAN <iris.analysis.MEAN>` 
supports a weights keyword which can be combined with :func:`iris.analysis.cartography.area_weights` to calculate an area average.

Let's use the same data as was loaded in the previous example. Since ``latitude`` and ``longitude`` were both 
point coordinates we must guess bound positions for them in order to calculate the area of the grid boxes::

    import iris.analysis.cartography
    cube.coord('latitude').guess_bounds()
    cube.coord('longitude').guess_bounds()
    grid_areas = iris.analysis.cartography.area_weights(cube)

These areas can now be passed to the ``collapsed`` method as weights:

.. doctest::

    >>> new_cube = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)
    >>> print new_cube
    air_potential_temperature           (level_height: 38)
         Dimension coordinates:
              level_height                           x
         Auxiliary coordinates:
              model_level_number                     x
              sigma                                  x
         Scalar coordinates:
              forecast_period: 26280 hours
              forecast_reference_time: 306816.0 hours since 1970-01-01 00:00:00
              latitude: Cell(point=0.0, bound=(-90.0, 90.0)) degrees
              longitude: Cell(point=180.0, bound=(0.0, 360.0)) degrees
              source: Data from Met Office Unified Model 6.06
              time: Cell(point=332724.0, bound=(332352.0, 333096.0)) hours since 1970-01-01 00:00:00
         Attributes:
              STASH: m01s00i004
              history: Mean of air_potential_temperature over longitude, latitude
         Cell methods:
              mean: time (1 hour)
              mean: longitude, latitude


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

    filename = iris.sample_data_path('PP', 'decadal_subset', 'ajnuqa.pm*.pp')
    cube = iris.load_strict(filename, 'air_temperature')

    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='season_year')


.. testsetup:: aggregation

    import iris

    filename = iris.sample_data_path('PP', 'decadal_subset', 'ajnuqa.pm*.pp')
    cube = iris.load_strict(filename, 'air_temperature')

    import iris.coord_categorisation
    iris.coord_categorisation.add_season(cube, 'time', name='clim_season')
    iris.coord_categorisation.add_season_year(cube, 'time', name='season_year')

    annual_seasonal_mean = cube.aggregated_by(['clim_season', 'season_year'], iris.analysis.MEAN)

    
Printing this cube now shows that two extra coordinates exist on the cube:

.. doctest:: aggregation

    >>> print cube
    air_temperature                     (forecast_period: 192; latitude: 145; longitude: 192)
         Dimension coordinates:
              forecast_period                           x              -               -
              latitude                                  -              x               -
              longitude                                 -              -               x
         Auxiliary coordinates:
              clim_season                               x              -               -
              season_year                               x              -               -
              time                                      x              -               -
         Scalar coordinates:
              forecast_reference_time: -959040.0 hours since 1970-01-01 00:00:00
              height: 1.5 m
              source: Data from Met Office Unified Model 6.06
         Attributes:
              STASH: m01s03i236
         Cell methods:
              mean: time (1 hour)


These two coordinates can now be used as *groups* over which to do an aggregation:

.. doctest:: aggregation

    >>> annual_seasonal_mean = cube.aggregated_by(['clim_season', 'season_year'], iris.analysis.MEAN)
    >>> print repr(annual_seasonal_mean)
    <iris 'Cube' of air_temperature (*ANONYMOUS*: 65; latitude: 145; longitude: 192)>
    
The primary change in the cube is that the cube's data has shrunk on the t axis as a result of the meaning aggregation. 
We have now collapsed all repeating copies of season (DJF etc.) and year to represent a single position in the t axis.
We can see this by printing the first 10 values of the original coordinates:

.. doctest:: aggregation

    >>> print cube.coord('clim_season')[:10].points
    ['djf' 'djf' 'mam' 'mam' 'mam' 'jja' 'jja' 'jja' 'son' 'son']
    >>> print cube.coord('season_year')[:10].points
    [1990 1990 1990 1990 1990 1990 1990 1990 1990 1990]

And then comparing with the first 10 values of the new cube's coordinates:

.. doctest:: aggregation

    >>> print annual_seasonal_mean.coord('clim_season')[:10].points
    ['djf' 'mam' 'jja' 'son' 'djf' 'mam' 'jja' 'son' 'djf' 'mam']
    >>> print annual_seasonal_mean.coord('season_year')[:10].points
    [1990 1990 1990 1990 1991 1991 1991 1991 1992 1992]


Because the original data started in January 1990 and ends in December we have some incomplete seasons 
(e.g. there were only two months worth of data for djf 1990). 
In this case we can fix this by removing all of the resultant ``times`` which do not cover a
three month period (n.b. 3 months = 3 * 30 * 24 = 2160 hours):

.. doctest:: aggregation

    >>> spans_three_months = lambda time: (time.bound[1] - time.bound[0]) == 2160
    >>> three_months_bound = iris.Constraint(time=spans_three_months)
    >>> print annual_seasonal_mean.extract(three_months_bound)
    air_temperature                     (*ANONYMOUS*: 63; latitude: 145; longitude: 192)
         Dimension coordinates:
              latitude                              -             x               -
              longitude                             -             -               x
         Auxiliary coordinates:
              clim_season                           x             -               -
              forecast_period                       x             -               -
              season_year                           x             -               -
              time                                  x             -               -
         Scalar coordinates:
              forecast_reference_time: -959040.0 hours since 1970-01-01 00:00:00
              height: 1.5 m
              source: Data from Met Office Unified Model 6.06
         Attributes:
              STASH: m01s03i236
              history: Mean of air_temperature aggregated over clim_season, season_year
         Cell methods:
              mean: time (1 hour)
              mean: clim_season, season_year


The final result now represents the seasonal mean temperature for 63 seasons starting from ``March April May 1990``.
