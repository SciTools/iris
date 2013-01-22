# (C) British Crown Copyright 2010 - 2012, Met Office
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


# TODO: Is this a mixin or a base class?

import warnings

import iris.std_names


class LimitedAttributeDict(dict):
    _forbidden_keys = ('standard_name', 'long_name', 'units', 'bounds', 'axis', 
                       'calendar', 'leap_month', 'leap_year','month_lengths',
                       'coordinates', 'grid_mapping', 'climatology', 'cell_methods', 'formula_terms', 
                       'compress', 'missing_value', 'add_offset', 'scale_factor', 
                       'valid_max', 'valid_min', 'valid_range',
                       '_FillValue')
                        
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        # Check validity of keys
        for key in self.iterkeys():
            if key in self._forbidden_keys:
                raise ValueError('%r is not a permitted attribute' % key)

    def __eq__(self, other):
        # Extend equality to allow for NumPy arrays.
        match = self.viewkeys() == other.viewkeys()
        if match:
            for key, value in self.iteritems():
                match = value == other[key]
                try:
                    match = bool(match)
                except ValueError:
                    match = match.all()
                if not match:
                    break
        return match

    def __setitem__(self, key, value):
        if key in self._forbidden_keys:
            raise ValueError('%r is not a permitted attribute' % key)
        dict.__setitem__(self, key, value)
    
    def update(self, other, **kwargs):
        # Gather incoming keys
        keys = []
        if hasattr(other, "keys"):
            keys += other.keys()
        else:
            keys += [k for k,v in other]
        
        keys += kwargs.keys()
        
        # Check validity of keys
        for key in keys:
            if key in self._forbidden_keys:
                raise ValueError('%r is not a permitted attribute' % key)                

        dict.update(self, other, **kwargs)


class CFVariableMixin(object):
    def name(self, default='unknown'):
        """
        Returns a human-readable name.

        First it tries :attr:`standard_name`, then it tries the 'long_name'
        attributes, before falling back to the value of `default` (which
        itself defaults to 'unknown').

        """
        return self.standard_name or self.long_name or default

    def rename(self, name):
        """
        Changes the human-readable name.

        If 'name' is a valid standard name it will assign it to
        :attr:`standard_name`, otherwise it will assign it to
        :attr:`long_name`.

        """
        try:
            self.standard_name = name
            self.long_name = None
        except ValueError:
            self.standard_name = None
            self.long_name = unicode(name)

    # TODO: Decide if this exists!
#    @property
#    def long_name(self):
#        return self._attributes.get('long_name')
#
#    @long_name.setter
#    def long_name(self, value):
#        self._attributes['long_name'] = str(value)

    @property
    def standard_name(self):
        """The standard name for the Cube's data."""
        return self._standard_name

    @standard_name.setter
    def standard_name(self, name):
        if name is None or name in iris.std_names.STD_NAMES:
            self._standard_name = name
        else:
            raise ValueError('%r is not a valid standard_name' % name)

    @property
    def units(self):
        """The :mod:`~iris.unit.Unit` instance of the object."""
        return getattr(self, '_units', None)

    @units.setter
    def units(self, unit):
        unit = iris.unit.as_unit(unit)
        # Allow assignment if the current units attrinute is None/unknown,
        # or assigning None/unknown to clear the units. Also allow
        # assignment of an equal unit e.g. 'degC' to 'Celsius'.
        if (unit.unknown or self.units is None or self.units.unknown or
                self.units == unit):
            self._units = unit
        else:
            raise iris.exceptions.ExistingUnitsError()

    def replace_units(self, unit):
        """Changes the object's units without modifying its values."""
        unit = iris.unit.as_unit(unit)
        # Clear any existing units.
        self.units = None
        # Assign replacement value.
        self.units = unit

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = LimitedAttributeDict(attributes or {}) 
