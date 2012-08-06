======================
Basic cube mathematics
======================


The section :doc:`navigating_a_cube` highlighted that every cube has a data attribute; this attribute can then be manipulated directly::

   cube.data -= 273.16

The problem with manipulating the data directly is that other metadata may become inconsistent; in this case the unit of 
the cube is no longer what was intended, therefore this example could be rectified by setting the unit::

   cube.units = 'C'

In order to reduce the amount of metadata which becomes inconsistent, fundamental arithmetic operations such as addition, 
subtraction, division, and multiplication can be applied directly to any cube.

Calculating the difference between two cubes
--------------------------------------------

Let's load some data which represents air pressure on the first model level of a single model run::

    filename = iris.sample_data_path('PP', 'COLPEX', 'air_potential_and_air_pressure.pp')
    air_press_lev1 = iris.Constraint('air_pressure', model_level_number=1)
    air_press = iris.load_strict(filename, air_press_lev1)

We can now get the first and last time slices using indexing (see :doc:`reducing_a_cube` for a reminder)::

    t_first = air_press[0, :, :]
    t_last = air_press[-1, :, :]

.. testsetup::

    filename = iris.sample_data_path('PP', 'COLPEX', 'air_potential_and_air_pressure.pp')
    cube = iris.load_strict(filename, iris.Constraint('air_pressure', model_level_number=1))
    t_first = cube[0, :, :]
    t_last = cube[1, :, :]

And finally we can subtract the two. The result is a cube of the same size as the original two time slices, but with the 
data representing their difference:

    >>> print t_last - t_first
    unknown                             (grid_latitude: 412; grid_longitude: 412)
         Dimension coordinates:
              grid_latitude                           x                    -
              grid_longitude                          -                    x
         Scalar coordinates:
              level_height: Cell(point=5.0, bound=(0.0, 13.333332)) m
              model_level_number: 1
              sigma: Cell(point=0.9994238, bound=(1.0, 0.99846387))
              source: Data from Met Office Unified Model 7.04
         Attributes:
              history: air_pressure - air_pressure (ignoring forecast_period, time)

.. note::
    Notice that the coordinates "time" and "forecast_period" have been removed from the resultant cube; this 
    is because these coordinates differed between the two input cubes. For more control on whether or not coordinates 
    should be automatically ignored :func:`iris.analysis.maths.subtract` can be used instead.


Combining multiple phenomenon to calculate another
--------------------------------------------------

Combining cubes of potential temperature and pressure we can calculate the associated temperature using the equation:

.. math::
   
    T = \theta (\frac{p}{p_0}) ^ {(287.05 / 1005)}

Where :math:`p` is pressure, :math:`\theta` is potential temperature, :math:`p_0` is the potential temperature 
reference pressure and :math:`T` is temperature.

First, let's load pressure and potential temperature cubes::

    filename = iris.sample_data_path('PP', 'COLPEX', 'air_potential_and_air_pressure.pp')
    phenomenon_names = ['air_potential_temperature', 'air_pressure']
    pot_temperature, pressure = iris.load_strict(filename, phenomenon_names)

In order to calculate :math:`\frac{p}{p_0}` we can define a coordinate which represents the standard reference pressure of 1000 hPa::

    import iris.coords
    p0 = iris.coords.ExplicitCoord('reference_pressure', 'hPa', points=1000.)

We must ensure that the units of ``pressure`` and ``p0`` are the same, so convert the newly created coordinate using 
the :meth:`iris.coords.Coord.unit_converted` method::

    p0 = p0.unit_converted(pressure.units)

Now we can combine all of this information to calculate the air temperature using the equation above::

    temperature = pot_temperature * ( (pressure / p0) ** (287.05 / 1005) )

Finally, the cube we have created needs to have its standard name and units set correctly::

    temperature.standard_name = 'air_temperature'
    temperature.units = 'kelvin'

The result could now be plotted using the guidance provided in the :doc:`plotting_a_cube` section.

.. htmlonly::
    A very similar example to this can be found in :doc:`/examples/graphics/deriving_phenomena`.

.. latexonly::
    A very similar example to this can be found in the examples section, with the title "Deriving Exner Pressure and Air Temperature".

