.. _controlling_merge:

=================================
Controlling Merge and Concatenate
=================================


Sometimes it is not possible to appropriately combine a CubeList using merge and concatenate on their own. In such cases
it is possible to achieve much more control over cube combination by using the :func:`~iris.util.new_axis` utility.
Consider the following set of cubes:

    >>> file_1 = iris.sample_data_path("time_varying_hybrid_height", "*_2160-12.pp")
    >>> file_2 = iris.sample_data_path("time_varying_hybrid_height", "*_2161-01.pp")
    >>> cubes = iris.load([file_1, file_2], "x_wind")
    >>> print(cubes[0])
    x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)
        Dimension coordinates:
            model_level_number                             x            -               -
            latitude                                       -            x               -
            longitude                                      -            -               x
        Auxiliary coordinates:
            level_height                                   x            -               -
            sigma                                          x            -               -
            surface_altitude                               -            x               x
        Derived coordinates:
            altitude                                       x            x               x
        Scalar coordinates:
            forecast_period             1338840.0 hours, bound=(1338480.0, 1339200.0) hours
            forecast_reference_time     2006-01-01 00:00:00
            time                        2160-12-16 00:00:00, bound=(2160-12-01 00:00:00, 2161-01-01 00:00:00)
        Cell methods:
            0                           time: mean (interval: 1 hour)
        Attributes:
            STASH                       m01s00i002
            source                      'Data from Met Office Unified Model'
            um_version                  '12.1'
    >>> print(cubes[1])
    x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)
        Dimension coordinates:
            model_level_number                             x            -               -
            latitude                                       -            x               -
            longitude                                      -            -               x
        Auxiliary coordinates:
            level_height                                   x            -               -
            sigma                                          x            -               -
            surface_altitude                               -            x               x
        Derived coordinates:
            altitude                                       x            x               x
        Scalar coordinates:
            forecast_period             1339560.0 hours, bound=(1339200.0, 1339920.0) hours
            forecast_reference_time     2006-01-01 00:00:00
            time                        2161-01-16 00:00:00, bound=(2161-01-01 00:00:00, 2161-02-01 00:00:00)
        Cell methods:
            0                           time: mean (interval: 1 hour)
        Attributes:
            STASH                       m01s00i002
            source                      'Data from Met Office Unified Model'
            um_version                  '12.1'

These two cubes have different time points (i.e. scalar time value).  So we would normally be able to merge them,
creating a time dimension.  However, in this case we can not combine them with :meth:`~iris.cube.Cube.merge`
due to the fact that their ``surface_altitude`` coordinate also varies over time:

    >>> cubes.merge_cube()
    Traceback (most recent call last):
    ...
    iris.exceptions.MergeError: failed to merge into a single cube.
      Coordinates in cube.aux_coords (non-scalar) differ: surface_altitude.

Since surface altitude is preventing merging, we want to find a way of combining these cubes while also *explicitly*
combining the ``surface_altitude`` coordinate so that it also varies along the time dimension. We can do this by first
adding a dimension to the cube *and* the ``surface_altitude`` coordinate using :func:`~iris.util.new_axis`, and then
concatenating those cubes together. We can attempt this as follows:

    >>> from iris.util import new_axis
    >>> from iris.cube import CubeList
    >>> processed_cubes = CubeList([new_axis(cube, scalar_coord="time", expand_extras=["surface_altitude"]) for cube in cubes])
    >>> processed_cubes.concatenate_cube()
    Traceback (most recent call last):
    ...
    iris.exceptions.ConcatenateError: failed to concatenate into a single cube.
      Scalar coordinates values or metadata differ: forecast_period != forecast_period

This error alerts us to the fact that the ``forecast_period`` coordinate is also varying over time. To get concatenation
to work, we will have to expand the dimensions of this coordinate to include "time", by passing it also to the
``expand_extras`` keyword.

    >>> processed_cubes = CubeList(
    ... [new_axis(cube, scalar_coord="time", expand_extras=["surface_altitude", "forecast_period"]) for cube in cubes]
    ... )
    >>> result = processed_cubes.concatenate_cube()
    >>> print(result)
    x_wind / (m s-1)                    (time: 2; model_level_number: 5; latitude: 144; longitude: 192)
        Dimension coordinates:
            time                             x                      -            -               -
            model_level_number               -                      x            -               -
            latitude                         -                      -            x               -
            longitude                        -                      -            -               x
        Auxiliary coordinates:
            forecast_period                  x                      -            -               -
            surface_altitude                 x                      -            x               x
            level_height                     -                      x            -               -
            sigma                            -                      x            -               -
        Derived coordinates:
            altitude                         x                      x            x               x
        Scalar coordinates:
            forecast_reference_time     2006-01-01 00:00:00
        Cell methods:
            0                           time: mean (interval: 1 hour)
        Attributes:
            STASH                       m01s00i002
            source                      'Data from Met Office Unified Model'
            um_version                  '12.1'

.. note::
    Since the derived coordinate ``altitude`` derives from ``surface_altitude``, adding ``time`` to the dimensions of
    ``surface_altitude`` also means it is added to the dimensions of ``altitude``. So in the combined cube, both of
    these coordinates vary along the ``time`` dimension.

Controlling over multiple dimensions
------------------------------------

We now consider a more complex case. Instead of loading 2 files across different time steps we now load 15 such files.
Each of these files covers a month's time step, however, the ``surface_altitude`` coordinate changes only once per year.
The files span 3 years so there are 3 different ``surface_altitude`` coordinates.

    >>> filename = iris.sample_data_path('time_varying_hybrid_height', '*.pp')
    >>> cubes = iris.load(filename, constraints="x_wind")
    >>> print(cubes)
    0: x_wind / (m s-1)                    (time: 2; model_level_number: 5; latitude: 144; longitude: 192)
    1: x_wind / (m s-1)                    (time: 12; model_level_number: 5; latitude: 144; longitude: 192)
    2: x_wind / (m s-1)                    (model_level_number: 5; latitude: 144; longitude: 192)

When :func:`iris.load` attempts to merge these cubes, it creates a cube for every unique ``surface_altitude`` coordinate.
Note that since there is only one time point associated with the last cube, the "time" coordinate has not been promoted
to a dimension. The ``surface_altitude`` in each of the above cubes is 2D, however, since some of these coordinates
already have a time dimension, it is not possible to use :func:`~iris.util.new_axis` as above to promote
``surface_altitude`` as we have done above.

In order to fully control the merge process we instead use :func:`iris.load_raw`:

    >>> raw_cubes = iris.load_raw(filename, constraints="x_wind")
    >>> print(raw_cubes)
    0: x_wind / (m s-1)                    (latitude: 144; longitude: 192)
    1: x_wind / (m s-1)                    (latitude: 144; longitude: 192)
    ...
    73: x_wind / (m s-1)                    (latitude: 144; longitude: 192)
    74: x_wind / (m s-1)                    (latitude: 144; longitude: 192)

The raw cubes also separate cubes along the ``model_level_number`` dimension. In this instance, we will need to
merge/concatenate along two different dimensions. Specifically, we can merge by promoting the ``model_level_number`` to
a dimension, since ``surface_altitude`` does  not vary along this dimension, and we can concatenate along the ``time``
dimension as before. We expand the ``time`` dimension first, as before:

    >>> processed_raw_cubes = CubeList(
    ... [new_axis(cube, scalar_coord="time", expand_extras=["surface_altitude", "forecast_period"]) for cube in raw_cubes]
    ... )
    >>> print(processed_raw_cubes)
    0: x_wind / (m s-1)                    (time: 1; latitude: 144; longitude: 192)
    1: x_wind / (m s-1)                    (time: 1; latitude: 144; longitude: 192)
    ...
    73: x_wind / (m s-1)                    (time: 1; latitude: 144; longitude: 192)
    74: x_wind / (m s-1)                    (time: 1; latitude: 144; longitude: 192)

Then we merge, promoting the different ``model_level_number`` scalar coordinates to a dimension coordinate.
Note, however, that merging these cubes does *not* affect the ``time`` dimension, since merging only
applies to scalar coordinates, not dimension coordinates of length 1.

    >>> merged_cubes = processed_raw_cubes.merge()
    >>> print(merged_cubes)
    0: x_wind / (m s-1)                    (model_level_number: 5; time: 1; latitude: 144; longitude: 192)
    1: x_wind / (m s-1)                    (model_level_number: 5; time: 1; latitude: 144; longitude: 192)
    ...
    13: x_wind / (m s-1)                    (model_level_number: 5; time: 1; latitude: 144; longitude: 192)
    14: x_wind / (m s-1)                    (model_level_number: 5; time: 1; latitude: 144; longitude: 192)

Once merged, we can now concatenate all these cubes into a single result cube, which is what we wanted:

    >>> result = merged_cubes.concatenate_cube()
    >>> print(result)
    x_wind / (m s-1)                    (model_level_number: 5; time: 15; latitude: 144; longitude: 192)
        Dimension coordinates:
            model_level_number                             x        -             -               -
            time                                           -        x             -               -
            latitude                                       -        -             x               -
            longitude                                      -        -             -               x
        Auxiliary coordinates:
            level_height                                   x        -             -               -
            sigma                                          x        -             -               -
            forecast_period                                -        x             -               -
            surface_altitude                               -        x             x               x
        Derived coordinates:
            altitude                                       x        x             x               x
        Scalar coordinates:
            forecast_reference_time     2006-01-01 00:00:00
        Cell methods:
            0                           time: mean (interval: 1 hour)
        Attributes:
            STASH                       m01s00i002
            source                      'Data from Met Office Unified Model'
            um_version                  '12.1'

See Also
--------
* :func:`iris.combine_cubes` can perform similar operations automatically
* :data:`iris.LOAD_POLICY` controls the application of :func:`~iris.combine_cubes`
  during the load operations, i.e. :func:`~iris.load`, :func:`~iris.load_cube` and
  :func:`~iris.load_cubes`.
