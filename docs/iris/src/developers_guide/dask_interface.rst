Iris Dask Interface
*******************

Iris uses dask (http://dask.pydata.org) to manage lazy data interfaces and processing graphs.  The key principles which define this interface are:

* A call to `cube.data` will always load all of the data.
  * Once this has happened:
    * `cube.data` is a mutable numpy masked array or ndarray;
    * `cube._numpy_array` is a private numpy masked array, accessible via `cube.data`, which may strip off the mask and return a reference to the bare ndarray.
* `cube.data` may be used to set the data, this accepts:
  * a numpy array (including masked array), which is assigned to `cube._numpy_array`;
  * a dask array, which is assigned to `cube._dask_array` an `cube._numpy_array` is set to None.
* `cube._dask_array` may be None, otherwise it is expected to be a dask graph:
  * this may wrap a proxy to a file collection;
  * this may wrap the numpy array in `cube._numpy_array`.
* All dask graphs wrap array-like object where missing data is represented by `nan`:
  * masked arrays derived from these arrays shall create their mask using the nan location;
  * where dask wrapped `int` arrays require masks, these will first be cast to `float`.
* In order to support this mask conversion, cube's have a `fill_value` as part of their metadata, which may be None.
* Array copying is kept to an absolute minimum:
  * array references should always be passed, not new arrays created, unless an explicit copy operation is requested.
* To test for the presence of a dask array of any sort, we use:
  * `iris._lazy_data.is_lazy_data` which is implemented as `hasattr(data, 'compute')`.
