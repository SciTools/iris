import dask

def from_proxy(proxy):
    return DataWrapper(dask.from_array(proxy, np.ones(proxy.ndim))), True)

def from_array(array, lazy=False):
    return DataWrapper(array, lazy)

class _DataWrapper(object):
    def __init__(self, data, lazy):
        if lazy:
            self._real_data = None
            self._lazy_data = data
        else
            self._real_data = data
            self._lazy_data = None

    def has_lazy_data(self):
        return self._real_data is None

    @property
    def lazy_data(self):
        return da.from_array(self._real_data, np.ones(self._real_data.ndim) \
            if self._real_data is not None else self._lazy_data

    @property
    def real_or_lazy_data(self):
        return self._real_data if self._real_data is not None else \
               self._lazy_data

    @property
    def data(self):
        if self.has_lazy_data():
            self._real_data = self._lazy_data.compute()
            # TODO: Delete self._lazy_data here?
        return self._real_data

    def __getitem__(self, indices):
        return from_array(self.lazy_data[indices], self.has_lazy_data())

    def __add__(self, other):
        lazy = self.has_lazy_data() or other.has_lazy_data():
        return from_array(
                self.real_or_lazy_data + other.real_or_lazy_data, lazy)
