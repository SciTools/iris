.. _interpolation_and_regridding:


.. testsetup:: *

  import numpy as np
  import iris

=================================
Cube regridding and interpolation
=================================

Iris builds upon interpolation schemes implemented by scipy, and other packages,
to add powerful cube-aware regrid and interpolation functionality exposed through
simple cube methods.

.. _interpolation:

Interpolation
-------------

Interpolation can be achieved on a cube with the :meth:`~iris.cube.Cube.interpolate`
method, with the first argument being the points to interpolate, and the second being
the interpolation scheme to use. The result is a new interpolated cube.

Sample points can be defined as an iterable of ``(coord/coord name, value(s))`` pairs, e.g. ``[('latitude', 51.48), ('longitude', 0)]``.
The values for coordinates which correspond to date/times may optionally
be supplied as datetime.datetime or netcdftime.datetime instances,
e.g. ``[('time', datetime.datetime(2009, 11, 19, 10, 30))]``).

Whilst more interpolation schemes will become available, the only interpolation scheme
currently implementing Iris' interpolate interface is :class:`iris.analysis.Linear`.

Taking the air temperature cube we've seen previously:

    >>> air_temp = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> print air_temp
    air_temperature / (K)               (latitude: 73; longitude: 96)
         Dimension coordinates:
              latitude                           x              -
              longitude                          -              x
         Scalar coordinates:
              forecast_period: 6477 hours, bound=(-28083.0, 6477.0) hours
              forecast_reference_time: 1998-03-01 03:00:00
              pressure: 1000.0 hPa
              time: 1998-12-01 00:00:00, bound=(1994-12-01 00:00:00, 1998-12-01 00:00:00)
         Attributes:
              STASH: m01s16i203
              source: Data from Met Office Unified Model
         Cell methods:
              mean within years: time
              mean over years: time

We can interpolate specific values from the coordinates of the cube:

    >>> sample_points = [('latitude', 51.48), ('longitude', 0)]
    >>> print air_temp.interpolate(sample_points, iris.analysis.Linear())
    air_temperature / (K)               (scalar cube)
         Scalar coordinates:
              forecast_period: 6477 hours, bound=(-28083.0, 6477.0) hours
              forecast_reference_time: 1998-03-01 03:00:00
              latitude: 51.48 degrees
              longitude: 0 degrees
              pressure: 1000.0 hPa
              time: 1998-12-01 00:00:00, bound=(1994-12-01 00:00:00, 1998-12-01 00:00:00)
         Attributes:
              STASH: m01s16i203
              source: Data from Met Office Unified Model
         Cell methods:
              mean within years: time
              mean over years: time

As we can see, the resulting cube is scalar and has longitude and latitude coordinates with
the values defined in our sample points.

It isn't necessary to specify sample points for each dimension - any dimensions which aren't
specified are preserved:

    >>> result = air_temp.interpolate([('longitude', 0)], iris.analysis.Linear())

    >>> print 'Original:', air_temp.summary(shorten=True)
    Original: air_temperature / (K)               (latitude: 73; longitude: 96)
    >>> print 'Interpolated:', result.summary(shorten=True)
    Interpolated: air_temperature / (K)               (latitude: 73)

The sample points needn't be a scalar value and may be an array of values instead.
When multiple coordinates are provided with arrays instead of scalars, the coordinates
on the resulting cube will be orthogonal:

    >>> sample_points = [('longitude', np.linspace(-11, 2, 14)),
    ...                  ('latitude',  np.linspace(48, 60, 13))]
    >>> result = air_temp.interpolate(sample_points, iris.analysis.Linear())
    >>> print result.summary(shorten=True)
    air_temperature / (K)               (latitude: 13; longitude: 14)


Interpolating non horizontal coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interpolation in Iris is not limited to horizontal-spatial coordinates - any
coordinate satisfying the prerequisites of the chosen scheme may be interpolated
over.

For instance, the :class:`iris.analysis.Linear` scheme requires 1D numeric,
monotonic, coordinates. Supposing we have a single column cube such as
the one defined below:

    >>> column = iris.load_cube(iris.sample_data_path('hybrid_height.nc'))[:, 0, 0]
    >>> print column.summary(shorten=True)
    air_potential_temperature / (K)     (model_level_number: 15)

This cube has a "hybrid-height" vertical coordinate system, meaning that the vertical
coordinate is unevenly spaced in altitude:

    >>> print column.coord('altitude').points
    [  418.7    434.57   456.79   485.37   520.29   561.58   609.21   663.21
       723.58   790.31   863.41   942.88  1028.74  1120.98  1219.61]

We could regularise the vertical coordinate by defining 10 equally spaced altitude
sample points between 400 and 1250:

    >>> sample_points = [('altitude', np.linspace(400, 1250, 10))]
    >>> new_column = column.interpolate(sample_points, iris.analysis.Linear())
    >>> print new_column.summary(shorten=True)
    air_potential_temperature / (K)     (model_level_number: 10)

To see what is going on, let's look at the original data, the interpolation line, and
the new data in a plot: 

.. plot:: userguide/regridding_plots/interpolate_column.py

As we can see with the red diamonds on the extremes of the altitude values, we have
extrapolated data beyond the range of the original data. In some cases this is desirable
functionality, and for others it is not - for instance, this column defined
a surface altitude value of 414m, so extrapolating "air potential temperature" at 400m
in this case makes little physical sense.

Fortunately we can control the extrapolation mode when defining the interpolation scheme
with the ``extrapolation_mode`` keyword.  For :class:`iris.analysis.Linear` the
``extrapolation_mode`` must be one of ``linear``, ``error``, ``nan``, ``mask`` or
``nanmask``. To mask the values which lie beyond the range of the original data, using
the ``mask`` extrapolation mode is just a matter of constructing the appropriate scheme
and passing it through to the :meth:`~iris.cube.Cube.interpolate` method:

    >>> scheme = iris.analysis.Linear(extrapolation_mode='mask')
    >>> new_column = column.interpolate(sample_points, scheme)

The result will be a cube of the number of points passed through to interpolate, with the
values requiring extrapolation being masked.

.. _regridding:


Regridding
----------

Regridding is conceptually a very similar process to interpolation in Iris. 
The primary difference is that interpolation is based on sample points, while
regridding is based on the **horizontal** grid of *another cube*.

Regridding a cube is achieved with the :meth:`cube.regrid() <iris.cube.Cube.regrid>` method.
This method expects two arguments: 
 #. *another cube* that defines the target grid onto which the cube should be regridded, and
 #. the regridding scheme to use.

.. note::

    Regridding is a common operation needed to allow comparisons of data on different grids.
    The powerful mapping functionality provided by cartopy, however, means that regridding
    is often not necessary if performed just for visualisation purposes.

Let's load two cubes that have different grids and coordinate systems:

    >>> global_air_temp = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> rotated_psl = iris.load_cube(iris.sample_data_path('rotated_pole.nc'))

We can visually confirm that they are on different grids by plotting the two cubes:

.. plot:: userguide/regridding_plots/regridding_plot.py

Let's regrid the ``global_air_temp`` cube onto a rotated pole grid
using a linear regridding scheme. To achieve this we pass the ``rotated_psl``
cube to the regridder to supply the target grid to regrid the ``global_air_temp``
cube onto:

    >>> rotated_air_temp = global_air_temp.regrid(rotated_psl, iris.analysis.Linear())

.. plot:: userguide/regridding_plots/regridded_to_rotated.py

We could regrid the pressure values onto the global grid, but this will involve
some form of extrapolation. As with interpolation, we can control the extrapolation
mode when defining the regridding scheme.

For the available regridding schemes in Iris, the ``extrapolation_mode`` keyword
must be one of:
 * ``extrapolate`` --
    * for :class:`~iris.analysis.Linear` the extrapolation points will be calculated by extending the gradient of the closest two points.
    * for :class:`~iris.analysis.Nearest` the extrapolation points will take their value from the nearest source point.
 * ``nan`` -- the extrapolation points will be be set to NaN.
 * ``error`` -- a ValueError exception will be raised, notifying an attempt to extrapolate.
 * ``mask`` -- the extrapolation points will always be masked, even if the source data is not a MaskedArray.
 * ``nanmask`` -- if the source data is a MaskedArray the extrapolation points will be masked. Otherwise they will be set to NaN.

The ``rotated_psl`` cube is defined on a limited area rotated pole grid. If we regridded
the ``rotated_psl`` cube onto the global grid as defined by the ``global_air_temp`` cube
any linearly extrapolated values would quickly become dominant and highly inaccurate.
We can control this behaviour by defining the ``extrapolation_mode`` in the constructor
of the regridding scheme to mask values that lie outside of the domain of the rotated
pole grid:

    >>> scheme = iris.analysis.Linear(extrapolation_mode='mask')
    >>> global_psl = rotated_psl.regrid(global_air_temp, scheme)

.. plot:: userguide/regridding_plots/regridded_to_global.py

Notice that although we can still see the approximate shape of the rotated pole grid, the
cells have now become rectangular in a plate carrée (equirectangular) projection.
The spatial grid of the resulting cube is really global, with a large proportion of the
data being masked.

Area-weighted regridding
^^^^^^^^^^^^^^^^^^^^^^^^

It is often the case that a point-based regridding scheme (such as
:class:`iris.analysis.Linear` or :class:`iris.analysis.Nearest`) is not
appropriate when you need to conserve quantities when regridding. The
:class:`iris.analysis.AreaWeighted` scheme is less general than
:class:`~iris.analysis.Linear` or :class:`~iris.analysis.Nearest`, but is a
conservative regridding scheme, meaning that the area-weighted total is
approximately preserved across grids.

With the :class:`~iris.analysis.AreaWeighted` regridding scheme, each target grid-box's
data is computed as a weighted mean of all grid-boxes from the source grid. The weighting
for any given target grid-box is the area of the intersection with each of the
source grid-boxes. This scheme performs well when regridding from a high
resolution source grid to a lower resolution target grid, since all source data
points will be accounted for in the target grid.

Let's demonstrate this with the global air temperature cube we saw previously,
along with a limited area cube containing total concentration of volcanic ash:

    >>> global_air_temp = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> print global_air_temp.summary(shorten=True)
    air_temperature / (K)               (latitude: 73; longitude: 96)
    >>>
    >>> regional_ash = iris.load_cube(iris.sample_data_path('NAME_output.txt'))
    >>> regional_ash = regional_ash.collapsed('flight_level', iris.analysis.SUM)
    >>> print regional_ash.summary(shorten=True)
    VOLCANIC_ASH_AIR_CONCENTRATION / (g/m3) (latitude: 214; longitude: 584)

One of the key limitations of the :class:`~iris.analysis.AreaWeighted`
regridding scheme is that the two input grids must be defined in the same
coordinate system as each other. Both input grids must also contain monotonic,
bounded, 1D spatial coordinates.

.. note::

    The :class:`~iris.analysis.AreaWeighted` regridding scheme requires spatial
    areas, therefore the longitude and latitude coordinates must be bounded.
    If the longitude and latitude bounds are not defined in the cube we can
    guess the bounds based on the coordinates' point values:

        >>> global_air_temp.coord('longitude').guess_bounds()
        >>> global_air_temp.coord('latitude').guess_bounds()

Using NumPy's masked array module we can mask any data that falls below a meaningful
concentration:

    >>> regional_ash.data = np.ma.masked_less(regional_ash.data, 5e-6)

Finally, we can regrid the data using the :class:`~iris.analysis.AreaWeighted`
regridding scheme:

    >>> scheme = iris.analysis.AreaWeighted(mdtol=0.5)
    >>> global_ash = regional_ash.regrid(global_air_temp, scheme)
    >>> print global_ash.summary(shorten=True)
    VOLCANIC_ASH_AIR_CONCENTRATION / (g/m3) (latitude: 73; longitude: 96)

Note that the :class:`~iris.analysis.AreaWeighted` regridding scheme allows us
to define a missing data tolerance (``mdtol``), which specifies the tolerated
fraction of masked data in any given target grid-box. If the fraction of masked
data within a target grid-box exceeds this value, the data in this target
grid-box will be masked in the result.

The fraction of masked data is calculated based on the area of masked source
grid-boxes that overlaps with each target grid-box. Defining an ``mdtol`` in the
:class:`~iris.analysis.AreaWeighted` regridding scheme allows fine control
of masked data tolerance. It is worth remembering that defining an ``mdtol`` of
anything other than 1 will prevent the scheme from being fully conservative, as
some data will be disregarded if it lies close to masked data.

To visualise the above regrid, let's plot the original data, along with 3 distinct
``mdtol`` values to compare the result:

.. plot:: userguide/regridding_plots/regridded_to_global_area_weighted.py


Caching a regridder
^^^^^^^^^^^^^^^^^^^

If you need to regrid multiple cubes with a common source grid onto a common
target grid you can 'cache' a regridder to be used for each of these regrids.
This can shorten the execution time of your code as the most computationally
intensive part of a regrid is setting up the regridder.

To cache a regridder you must set up a regridder scheme and call the
scheme's regridder method. The regridder method takes as arguments:
 #. a cube (that is to be regridded) defining the source grid, and
 #. a cube defining the target grid to regrid the source cube to.

For example:

    >>> global_air_temp = iris.load_cube(iris.sample_data_path('air_temp.pp'))
    >>> rotated_psl = iris.load_cube(iris.sample_data_path('rotated_pole.nc'))
    >>> regridder = iris.analysis.Nearest().regridder(global_air_temp, rotated_psl)

When this cached regridder is called you must pass it a cube on the same grid
as the source grid cube (in this case ``global_air_temp``) that is to be
regridded to the target grid. For example::

    >>> for cube in list_of_cubes_on_source_grid:
    ...     result = regridder(cube)

In each case ``result`` will be the input cube regridded to the grid defined by
the target grid cube (in this case ``rotated_psl``) that we used to define the
cached regridder.
