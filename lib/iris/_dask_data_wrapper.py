import dask.array as da
import numpy as np


def from_proxy(proxy):
    return _DataWrapper(da.from_array(proxy, 1), True)


def from_lazy_array(array):
    return _DataWrapper(array, True)


def from_array(array, lazy=False):
    return _DataWrapper(array, lazy)


class _DataWrapper(object):
    def __init__(self, data, lazy):
        if lazy:
            self._real_data = None
            self._lazy_data = data
        else:
            self._real_data = data
            self._lazy_data = None

    def has_lazy_data(self):
        return self._real_data is None

    @property
    def shape(self):
        return self.real_or_lazy_data.shape

    @property
    def dtype(self):
        return self.real_or_lazy_data.dtype

    def lazy_data(self):
        return da.from_array(self._real_data, 1) \
            if self._real_data is not None else self._lazy_data

    @property
    def real_or_lazy_data(self):
        return self._real_data if self._real_data is not None else \
               self._lazy_data

    @property
    def data(self):
        if self.has_lazy_data():
            self._real_data = self._lazy_data.compute()
            del self._lazy_data
        return self._real_data

    def __getitem__(self, indices):
        return from_array(self.real_or_lazy_data[indices],
                          self.has_lazy_data())

    def __add__(self, other):
        lazy = self.has_lazy_data() or other.has_lazy_data()
        return from_array(
                self.real_or_lazy_data + other.real_or_lazy_data, lazy)

    def transpose(self, axis=None):
        return from_array(self.real_or_lazy_data.transpose(),
                          self.has_lazy_data())

    def concatenate(self, wrappers, axis=0):
        arrays = [wrapper.real_or_lazy_data for wrapper in wrappers]
        lazy = np.any([wrapper.has_lazy_data() for wrapper in wrappers])
        return from_array(da.concatenate(arrays, axis), lazy)
