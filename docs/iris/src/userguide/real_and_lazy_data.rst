.. _real_and_lazy_data:

==================
Real and Lazy Data
==================

What is Real and Lazy Data?
---------------------------

Every Iris cube contains an n-dimensional  data array, which could be real or
lazy.

Real data is contained in an array which has a shape, a data type, some other
useful information and many data points, each of which use up a small
allocation of memory.  This generally takes the form of a numpy array.

Lazy data is contained in a conceptual array which retains the information
about its real counterpart but has no actual data points, so its memory
allocation is much smaller.  This will be in the form of a dask array.

Arrays in Iris can be converted flexibly(?) between their real and lazy states,
although there are some limits to this process.  The advantage of using lazy
data is that it has a small memory footprint, so certain operations
(such as...?) can be much faster.  However, in order to perform other
operations (such as calculations on actual data values) the real data must be
realized.

* When a cube is loaded, are the data arrays always lazy to begin with? *

You can check whether the data array on your cube is lazy using the Iris
function 'has_lazy_data'.  For example:

>>> import iris
>>> filename = iris.sample_data_path('uk_hires.pp')
>>> cube = iris.load_cube(filename, 'air_potential_temperature')
>>> cube.has_lazy_data()
True
>>> _ = cube.data
>>> cube.has_lazy_data()
False

Assigning the data array to a variable causes the real data to be realized, at
which point the array ceases to be lazy.  Any action which requires the use of
actual data values (such as cube maths) will have this effect, although data
realization is always deferred until the last possible moment:

>>> my_cube = iris.load_cube(iris.sample_data_path('rotated_pole.nc'))
>>> my_cube.has_lazy_data()
True
>>> my_cube += 5
>>> my_cube.has_lazy_data()
True
>>> my_cube.data
>>> my_cube.has_lazy_data()
False

Core data refers to the current state of the cube's data, be it real or
lazy.  This can be used if you wish to refer to the data array but are
indifferent to its current state.  If the cube's data is lazy, it will not be
realized when you reference the core data attribute (?):

>>> cube = iris.load_cube(iris.sample_data_path('air_temp.pp'))
>>> cube.has_lazy_data()
True
>>> the_data = cube.core_data
>>> cube.has_lazy_data()
True
>>> real_data = cube.data
>>> cube.has_lazy_data()
False


Changing a Cube's Data
----------------------


