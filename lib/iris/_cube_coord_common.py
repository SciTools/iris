# (C) British Crown Copyright 2010 - 2015, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# TODO: Is this a mixin or a base class?

import string

import cf_units

import iris.std_names


class LimitedAttributeDict(dict):
    _forbidden_keys = ('standard_name', 'long_name', 'units', 'bounds', 'axis',
                       'calendar', 'leap_month', 'leap_year', 'month_lengths',
                       'coordinates', 'grid_mapping', 'climatology',
                       'cell_methods', 'formula_terms', 'compress',
                       'missing_value', 'add_offset', 'scale_factor',
                       'valid_max', 'valid_min', 'valid_range', '_FillValue')

    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        # Check validity of keys
        for key in six.iterkeys(self):
            if key in self._forbidden_keys:
                raise ValueError('%r is not a permitted attribute' % key)

    def __eq__(self, other):
        # Extend equality to allow for NumPy arrays.
        match = set(self.keys()) == set(other.keys())
        if match:
            for key, value in six.iteritems(self):
                match = value == other[key]
                try:
                    match = bool(match)
                except ValueError:
                    match = match.all()
                if not match:
                    break
        return match

    def __ne__(self, other):
        return not self == other

    def __setitem__(self, key, value):
        if key in self._forbidden_keys:
            raise ValueError('%r is not a permitted attribute' % key)
        dict.__setitem__(self, key, value)

    def update(self, other, **kwargs):
        # Gather incoming keys
        keys = []
        if hasattr(other, "keys"):
            keys += list(other.keys())
        else:
            keys += [k for k, v in other]

        keys += list(kwargs.keys())

        # Check validity of keys
        for key in keys:
            if key in self._forbidden_keys:
                raise ValueError('%r is not a permitted attribute' % key)

        dict.update(self, other, **kwargs)


class CFVariableMixin(object):
    def name(self, default='unknown'):
        """
        Returns a human-readable name.

        First it tries :attr:`standard_name`, then 'long_name', then 'var_name'
        before falling back to the value of `default` (which itself defaults to
        'unknown').

        """
        return self.standard_name or self.long_name or self.var_name or default

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
            self.long_name = six.text_type(name)

        # Always clear var_name when renaming.
        self.var_name = None

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
        """The :mod:`~cf_units.Unit` instance of the object."""
        return self._units

    @units.setter
    def units(self, unit):
        self._units = cf_units.as_unit(unit)

    @property
    def var_name(self):
        """The CF variable name for the object."""
        return self._var_name

    @var_name.setter
    def var_name(self, name):
        if name is not None:
            if not name:
                raise ValueError('An empty string is not a valid CF variable '
                                 'name.')
            elif set(name).intersection(string.whitespace):
                raise ValueError('{!r} is not a valid CF variable name because'
                                 ' it contains whitespace.'.format(name))
        self._var_name = name

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = LimitedAttributeDict(attributes or {})
