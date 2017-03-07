Iris Dask Interface
*******************

Iris uses `dask <http://dask.pydata.org>`_ to manage lazy data interfaces and processing graphs.
The key principles that define this interface are:

* A call to :attr:`cube.data` will always load all of the data.

* Once this has happened:

  * :attr:`cube.data` is a mutable NumPy masked array or ``ndarray``, and
  * ``cube._numpy_array`` is a private NumPy masked array, accessible via :attr:`cube.data`, which may strip off the mask and return a reference to the bare ``ndarray``.

* You can use :attr:`cube.data` to set the data. This accepts:

  * a NumPy array (including masked array), which is assigned to ``cube._numpy_array``, or
  * a dask array, which is assigned to ``cube._dask_array``, while ``cube._numpy_array`` is set to None.

* ``cube._dask_array`` may be None, otherwise it is expected to be a dask array:

  * this may wrap a proxy to a file collection, or
  * this may wrap the NumPy array in ``cube._numpy_array``.

* All dask arrays wrap array-like objects where missing data are represented by ``nan`` values:

  * Masked arrays derived from these dask arrays create their mask using the locations of ``nan`` values.
  * Where dask-wrapped arrays of ``int`` require masks, these arrays will first be cast to ``float``.

* In order to support this mask conversion, cubes have a ``fill_value`` defined as part of their metadata, which may be ``None``.

* Array copying is kept to an absolute minimum:

  * array references should always be passed, not new arrays created, unless an explicit copy operation is requested.

* To test for the presence of a dask array of any sort, we use :func:`iris._lazy_data.is_lazy_data`. This is implemented as ``hasattr(data, 'compute')``.
