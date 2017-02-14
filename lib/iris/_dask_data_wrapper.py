import dask

def from_proxy(proxy):
    return DataWrapper(dask.from_array(proxy, np.ones(proxy.ndim))))

def from_lazy_array(array, lazy=True):
    return DataWrapper(lazy_data=lazy_array)

def from_array(array):
    return DataWrapper(real_data=array), lazy=False)

class DataWrapper(object):
    def __init__(self, real_data=None, lazy_data=None):
        self._real_data = real_data
        self._lazy_data = lazy_data

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
        return self._real_data

    def __getitem__(self, indices):
        return self.real_or_lazy_data[indices]

    def __add__(self, other):
        if self.has_lazy_data() or other.has_lazy_data():
            return from_lazy_array(
                self.real_or_lazy_data + other.real_or_lazy_data)
        else:
            return from_array(self.data + other.data)
