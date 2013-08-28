# (C) British Crown Copyright 2010 - 2013, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""
Provides support for virtual cube data and deferred loading.

"""

from copy import deepcopy
import types

import numpy as np
import numpy.ma as ma

import iris.util


class _HashableSlice(iris.util._OrderedHashable):
    """Represents a :class:`slice` in a hashable way."""
    _names = ('start', 'stop', 'step')
    
    @staticmethod
    def from_slice(slice_object):
        """
        Generate a :class:`iris.fileformats.manager._HashableSlice` from a :class:`slice` object.
        
        Args:
        
        * slice_object (:class:`slice`):
            Slice object to be converted into a :class:`iris.fileformats.manager._HashableSlice`.
            
        Returns:
            :class:`iris.fileformats.manager._HashableSlice`.
        
        """
        return _HashableSlice(slice_object.start, slice_object.stop, slice_object.step)
    
    def indices(self, length):
        """
        Calculate start, stop and stride for current slice given the length of the sequence.
        
        Args:
        
        * length (int):
           Length of the sequence for which this slice is to calculate over.
           
        * Returns:
           Tuple of start, stop and stride.
        
        """
        return self.as_slice().indices(length)
    
    def as_slice(self):
        """
        Convert the :class:`iris.fileformats.manager._HashableSlice` into a :class:`slice`.
        
        Returns:
            :class:`slice`.
        
        """
        return slice(self.start, self.stop, self.step)


class DataManager(iris.util._OrderedHashable):
    """
    Holds the context that allows a corresponding array of DataProxy objects to be
    converted into a real data array.
    
    """

    _names = ('_orig_data_shape', 'data_type', 'mdi', 'deferred_slices')

    def __init__(self, data_shape, data_type, mdi, deferred_slices=()):
        self._init(data_shape, data_type, mdi, deferred_slices)

    #: The data shape of the array in file; may differ from the result
    #: of :py:ref:`load` if there are pending slices.
    _orig_data_shape = None

    #: Tuple of keys tuples as would be used in a __getitem__ context.
    deferred_slices = None
    
    def pre_slice_array_shape(self, proxy_array):
        """
        Given the associated proxy_array, calculate the shape of the resultant
        data without loading it.
        
        .. note::

            This may differ from the result of :meth:`load` if there are
            pending post load slices in :attr:`deferred_slices`.
            
        """
        return proxy_array.shape + self._orig_data_shape
        
    def _post_slice_data_shape(self):
        """The shape of the data manager data, after deferred slicing."""
        orig_shape = self._orig_data_shape
        resultant_shape = []
        
        for deferred_keys in self.deferred_slices:
            # For each of the slices, identify which will leave a dimension intact, 
            # and store each intact dimension's length in a list
            for i, key in enumerate(deferred_keys):
                if isinstance(key, _HashableSlice):
                    len_this_dim = orig_shape[i]
                    (start, stop, step) = key.indices(len_this_dim)
                    count = len(range(start, stop, step))
                    resultant_shape.append(count)
                elif isinstance(key, tuple):
                    if key and isinstance(key[0], (bool, np.bool_)):
                        resultant_shape.append(sum(key))
                    else:
                        resultant_shape.append(len(key))
                elif isinstance(key, int):
                    pass
                else:
                    raise TypeError('Unexpected type for key in DataManager. Got %s.' % type(key))
            
            orig_shape = tuple(resultant_shape)
            resultant_shape = []
                    
        return orig_shape
        
    def shape(self, proxy_array):
        """The shape of the data array given the associated proxy array, including effects of deferred slicing."""
        return proxy_array.shape + self._post_slice_data_shape()
    
    def getitem(self, proxy_array, keys):
        """The equivalent method to python's __getitem__ but with added proxy array capability."""
        # Find out how many dimensions the data array would have if it were loaded now
        self_ndim = len(self.shape(proxy_array))
        
        full_slice = iris.util._build_full_slice_given_keys(keys, self_ndim)
        
        # slice the proxy array according to the full slice provided
        # Add the Ellipsis object to the end of the slice to handle the special case where the full slice is
        # a tuple of a single tuple i.e. ( (0, 2, 3), ) which in numpy should be represented as ( (0, 2, 3), :)
        # NB: assumes that the proxy array is always the first dimensions
        new_proxy_array = proxy_array[full_slice[0:proxy_array.ndim] + (Ellipsis, )]

        # catch the situation where exactly one element from the proxy_array is requested:
        # (A MaskedConstant is an instance of a numpy array, so check for this specifically too) 
        if (not isinstance(new_proxy_array, np.ndarray)) or (isinstance(new_proxy_array, ma.core.MaskedConstant)):
            new_proxy_array = np.array(new_proxy_array)
        
        # get the ndim of the data manager array
        ds_ndim = len(self._post_slice_data_shape())
        # Just pull out the keys which apply to the data manager array
        if ds_ndim == 0:
            deferred_slice = full_slice[0:0]
        else:
            deferred_slice = full_slice[-ds_ndim:]
        
        hashable_conversion = {
                             types.SliceType: _HashableSlice.from_slice,
                             np.ndarray: tuple,
                             } 
        new_deferred_slice = tuple([hashable_conversion.get(type(index), lambda index: index)(index)
                                    for index in deferred_slice])
        
        # Apply the slice to a new data manager (to be deferred)
        defered_slices = self.deferred_slices + (new_deferred_slice, )
        new_data_manager = self.new_data_manager(defered_slices)

        return new_proxy_array, new_data_manager

    def new_data_manager(self, deferred_slices=Ellipsis):
        """
        Creates a new data manager instance with the given deferred slice.

        """
        return self.__class__(data_shape=deepcopy(self._orig_data_shape),
                              data_type=deepcopy(self.data_type),
                              mdi=deepcopy(self.mdi),
                              deferred_slices=deferred_slices)

    def _deferred_slice_merge(self):
        """Determine the single slice that is equivalent to all the accumulated deferred slices."""

        # The merged slice will always have the same number of index items 
        # as the original data dimensionality.
        merged_slice = [slice(None)] * len(self._orig_data_shape) 
        # Maintain the overall deferred slice shape as we merge in each 
        # of the deferred slices.
        deferred_shape = list(self._orig_data_shape)
        full_slice = slice(None)

        # The deferred slices tuple consists of one or more sub-tuples each of which may
        # contain a mixture of a _HashableSlice object, tuple of scalar indexes,
        # or a single scalar index. The dimensionality of each sub-tuple will be no 
        # greater than the original data dimensionality. The dimensionality of a sub-tuple
        # will be less than the original data dimensionality only if a previously merged
        # sub-tuple collapsed a dimension via a single scalar index.
        
        # Process each deferred slice sub-tuple.
        for deferred_slice in self.deferred_slices:
            # Identify those dimensions in the merged slice that have not been collapsed.
            # A collapsed dimension is one that is represented by a single scalar index. 
            mapping = [i for i, value in enumerate(merged_slice) if not isinstance(value, int)]
        
            # Process each index item in the deferred slice sub-tuple.
            for i, index_item in enumerate(deferred_slice):
                # First re-map deferred slice dimensions to account for any pre-merged 
                # dimensions that have already been collapsed.
                i = mapping[i]
                
                # Translate a hashable slice into a slice.
                if isinstance(index_item, _HashableSlice):
                    index_item = index_item.as_slice()

                # Process the index item only if it will change the 
                # corresponding merged slice index item.
                if index_item != full_slice:
                    if isinstance(merged_slice[i], slice):
                        # A slice object is not iterable, so it is not possible 
                        # to index or slice a slice object. Therefore translate
                        # the slice into an explicit tuple.
                        merged_slice_item = tuple(range(*merged_slice[i].indices(deferred_shape[i])))
                    else:
                        # The merged slice item must be a tuple.
                        merged_slice_item = merged_slice[i]
        
                    # Sample the merged slice item with the index item.
                    if isinstance(index_item, tuple):
                        if index_item and isinstance(index_item[0], (bool, np.bool_)):
                            index_item = np.where(index_item)[0]

                        # Sample for each tuple item.
                        merged_slice[i] = tuple([merged_slice_item[index]
                                                 for index in index_item])
                    else:
                        # Sample with a slice or single scalar index.
                        merged_slice[i] = merged_slice_item[index_item]
                
                    # Maintain the overall deferred slice shape as we merge.
                    if isinstance(merged_slice[i], tuple):
                        # New dimension depth is the length of the tuple.
                        deferred_shape[i] = len(merged_slice[i])
                    elif isinstance(merged_slice[i], int):
                        # New dimension depth has been collapsed by single scalar index.
                        deferred_shape[i] = 0
                    else:
                        # New dimension depth is the length of the sliced dimension.
                        deferred_shape[i] = len(range(deferred_shape[i])[merged_slice[i]])
        
        return tuple(merged_slice)

    def load(self, proxy_array):
        """Returns the real data array that corresponds to the given array of proxies."""
        
        deferred_slice = self._deferred_slice_merge()
        array_shape = self.shape(proxy_array)
        
        # Create fully masked data (all missing)
        try:
            raw_data = np.empty(array_shape,
                                dtype=self.data_type.newbyteorder('='))
            mask = np.ones(array_shape, dtype=np.bool)
            data = ma.MaskedArray(raw_data, mask=mask,
                                     fill_value=self.mdi)
        except ValueError:
            raise DataManager.ArrayTooBigForAddressSpace(
                    'Cannot create an array of shape %r as it will not fit in'
                    ' memory. Consider using indexing to select a subset of'
                    ' the Cube.'.format(array_shape))

        for index, proxy in np.ndenumerate(proxy_array):
            if proxy not in [None, 0]:  # 0 can come from slicing masked proxy; np.array(masked_constant).
                payload = proxy.load(self._orig_data_shape, self.data_type, self.mdi, deferred_slice)

                # Explicitly set the data fill value when no mdi value has been specified
                # in order to override default masked array fill value behaviour.
                if self.mdi is None and ma.isMaskedArray(payload):
                    data.fill_value = payload.fill_value

                data[index] = payload

        # we can turn the masked array into a normal array if it's full.
        if ma.count_masked(data) == 0:
            data = data.filled() 

        # take a copy of the data as it may be discontiguous (i.e. when numpy "fancy" indexing has taken place)
        if not data.flags['C_CONTIGUOUS']:
            data = data.copy()

        return data
    
    # nested exception definition inside DataManager
    class ArrayTooBigForAddressSpace(Exception):
        """Raised when numpy cannot possibly allocate an array as it is too big for the address space."""
        pass


