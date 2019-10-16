======================
Basic cube mathematics
======================


The section :doc:`navigating_a_cube` highlighted that 
every cube has a data attribute; 
this attribute can then be manipulated directly::

   cube.data -= 273.15

The problem with manipulating the data directly is that other metadata may
become inconsistent; in this case the units of the cube are no longer what was
intended. This example could be rectified by changing the units attribute::

   cube.units = 'celsius'

.. note::

    :meth:`iris.cube.Cube.convert_units` can be used to automatically convert a
    cube's data and update its units attribute.
    So, the two steps above can be achieved by::

        cube.convert_units('celsius')

In order to reduce the amount of metadata which becomes inconsistent,
fundamental arithmetic operations such as addition, subtraction, division
and multiplication can be applied directly to any cube.

Calculating the difference between two cubes
--------------------------------------------

Let's load some air temperature which runs from 1860 to 2100::

    filename = iris.sample_data_path('E1_north_america.nc')
    air_temp = iris.load_cube(filename, 'air_temperature')

We can now get the first and last time slices using indexing 
(see :ref:`subsetting_a_cube` for a reminder)::

    t_first = air_temp[0, :, :]
    t_last = air_temp[-1, :, :]

.. testsetup::

    filename = iris.sample_data_path('E1_north_america.nc')
    air_temp = iris.load_cube(filename, 'air_temperature')
    t_first = air_temp[0, :, :]
    t_last = air_temp[-1, :, :]

And finally we can subtract the two. 
The result is a cube of the same size as the original two time slices, 
but with the data representing their difference:

    >>> print(t_last - t_first)
    unknown / (K)                       (latitude: 37; longitude: 49)
         Dimension coordinates:
              latitude                           x              -
              longitude                          -              x
         Scalar coordinates:
              forecast_reference_time: 1859-09-01 06:00:00
              height: 1.5 m


.. note::

    Notice that the coordinates "time" and "forecast_period" have been removed 
    from the resultant cube; 
    this is because these coordinates differed between the two input cubes.


.. _cube-maths_anomaly:

Calculating a cube anomaly
--------------------------

In section :doc:`cube_statistics` we discussed how the dimensionality of a cube
can be reduced using the :meth:`Cube.collapsed <iris.cube.Cube.collapsed>` method
to calculate a statistic over a dimension.

Let's use that method to calculate a mean of our air temperature time-series,
which we'll then use to calculate a time mean anomaly and highlight the powerful
benefits of cube broadcasting.

First, let's remind ourselves of the shape of our air temperature time-series
cube::

    >>> print(air_temp.summary(True))
    air_temperature / (K)               (time: 240; latitude: 37; longitude: 49)

Now, we'll calculate the time-series mean using the
:meth:`Cube.collapsed <iris.cube.Cube.collapsed>` method::

    >>> air_temp_mean = air_temp.collapsed('time', iris.analysis.MEAN)
    >>> print(air_temp_mean.summary(True))
    air_temperature / (K)               (latitude: 37; longitude: 49)

As expected the *time* dimension has been collapsed, reducing the
dimensionality of the resultant *air_temp_mean* cube. This time-series mean can
now be used to calculate the time mean anomaly against the original
time-series::

    >>> anomaly = air_temp - air_temp_mean
    >>> print(anomaly.summary(True))
    unknown / (K)                       (time: 240; latitude: 37; longitude: 49)

Notice that the calculation of the *anomaly* involves subtracting a
*2d* cube from a *3d* cube to yield a *3d* result. This is only possible
because cube broadcasting is performed during cube arithmetic operations.

Cube broadcasting follows similar broadcasting rules as
`NumPy <http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_, but
the additional richness of Iris coordinate meta-data provides an enhanced
capability beyond the basic broadcasting behaviour of NumPy.

As the coordinate meta-data of a cube uniquely describes each dimension, it is
possible to leverage this knowledge to identify the similar dimensions involved
in a cube arithmetic operation. This essentially means that we are no longer
restricted to performing arithmetic on cubes with identical shapes.

This extended broadcasting behaviour is highlighted in the following
examples. The first of these shows that it is possible to involve the
transpose of the air temperature time-series in an arithmetic operation with
itself.

Let's first create the transpose of the air temperature time-series::

    >>> air_temp_T = air_temp.copy()
    >>> air_temp_T.transpose()
    >>> print(air_temp_T.summary(True))
    air_temperature / (K)               (longitude: 49; latitude: 37; time: 240)

Now add the transpose to the original time-series::

    >>> result = air_temp + air_temp_T
    >>> print(result.summary(True))
    unknown / (K)                       (time: 240; latitude: 37; longitude: 49)

Notice that the *result* is the same dimensionality and shape as *air_temp*.
Let's check that the arithmetic operation has calculated a result that
we would intuitively expect::

    >>> result == 2 * air_temp
    True

Let's extend this example slightly, by taking a slice from the middle
*latitude* dimension of the transpose cube::

    >>> air_temp_T_slice = air_temp_T[:, 0, :]
    >>> print(air_temp_T_slice.summary(True))
    air_temperature / (K)               (longitude: 49; time: 240)

Compared to our original time-series, the *air_temp_T_slice* cube has one
less dimension *and* it's shape if different. However, this doesn't prevent
us from performing cube arithmetic with it, thanks to the extended cube
broadcasting behaviour::

    >>> result = air_temp - air_temp_T_slice
    >>> print(result.summary(True))
    unknown / (K)                       (time: 240; latitude: 37; longitude: 49)

Combining multiple phenomena to form a new one
----------------------------------------------

Combining cubes of potential-temperature and pressure we can calculate 
the associated temperature using the equation:

.. math::
   
    T = \theta (\frac{p}{p_0}) ^ {(287.05 / 1005)}

Where :math:`p` is pressure, :math:`\theta` is potential temperature, 
:math:`p_0` is the potential temperature reference pressure 
and :math:`T` is temperature.

First, let's load pressure and potential temperature cubes::

    filename = iris.sample_data_path('colpex.pp')
    phenomenon_names = ['air_potential_temperature', 'air_pressure']
    pot_temperature, pressure = iris.load_cubes(filename, phenomenon_names)

In order to calculate :math:`\frac{p}{p_0}` we can define a coordinate which 
represents the standard reference pressure of 1000 hPa::

    import iris.coords
    p0 = iris.coords.AuxCoord(1000.0,
                              long_name='reference_pressure',
                              units='hPa')

We must ensure that the units of ``pressure`` and ``p0`` are the same,
so convert the newly created coordinate using
the :meth:`iris.coords.Coord.convert_units` method::

    p0.convert_units(pressure.units)

Now we can combine all of this information to calculate the air temperature 
using the equation above::

    temperature = pot_temperature * ( (pressure / p0) ** (287.05 / 1005) )

Finally, the cube we have created needs to be given a suitable name::

    temperature.rename('air_temperature')

The result could now be plotted using the guidance provided in the 
:doc:`plotting_a_cube` section.

.. only:: html

    A very similar example to this can be found in 
    :doc:`/examples/Meteorology/deriving_phenomena`.

.. only:: latex

    A very similar example to this can be found in the examples section, 
    with the title "Deriving Exner Pressure and Air Temperature".

