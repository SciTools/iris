# (C) British Crown Copyright 2010 - 2020, Met Office
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


from collections import namedtuple
import re
import string

import cf_units

import iris.std_names


# https://www.unidata.ucar.edu/software/netcdf/docs/netcdf_data_set_components.html#object_name
_TOKEN_PARSE = re.compile(r'''^[a-zA-Z0-9][\w\.\+\-@]*$''')


class Names(
    namedtuple("Names", ["standard_name", "long_name", "var_name", "STASH"])
):
    """
    Immutable container for name metadata.

    Args:

    * standard_name:
        A string representing the CF Conventions and Metadata standard name, or
        None.
    * long_name:
        A string representing the CF Conventions and Metadata long name, or
        None
    * var_name:
        A string representing the associated NetCDF variable name, or None.
    * STASH:
        A string representing the `~iris.fileformats.pp.STASH` code, or None.

    """

    __slots__ = ()


def get_valid_standard_name(name):
    # Standard names are optionally followed by a standard name
    # modifier, separated by one or more blank spaces

    if name is not None:
        name_is_valid = False
        # Supported standard name modifiers. Ref: [CF] Appendix C.
        valid_std_name_modifiers = ['detection_minimum',
                                    'number_of_observations',
                                    'standard_error',
                                    'status_flag']

        valid_name_pattern = re.compile(r'''^([a-zA-Z_]+)( *)([a-zA-Z_]*)$''')
        name_groups = valid_name_pattern.match(name)

        if name_groups:
            std_name, whitespace, std_name_modifier = name_groups.groups()
            if (std_name in iris.std_names.STD_NAMES) and (
                bool(whitespace) == (std_name_modifier in
                                     valid_std_name_modifiers)):
                name_is_valid = True

        if name_is_valid is False:
            raise ValueError('{!r} is not a valid standard_name'.format(
                    name))

    return name


class LimitedAttributeDict(dict):
    _forbidden_keys = ('standard_name', 'long_name', 'units', 'bounds', 'axis',
                       'calendar', 'leap_month', 'leap_year', 'month_lengths',
                       'coordinates', 'grid_mapping', 'climatology',
                       'cell_methods', 'formula_terms', 'compress',
                       'add_offset', 'scale_factor',
                       '_FillValue')

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

    _DEFAULT_NAME = 'unknown'  # the name default string

    @staticmethod
    def token(name):
        '''
        Determine whether the provided name is a valid NetCDF name and thus
        safe to represent a single parsable token.

        Args:

        * name:
            The string name to verify

        Returns:
            The provided name if valid, otherwise None.

        '''
        if name is not None:
            result = _TOKEN_PARSE.match(name)
            name = result if result is None else name
        return name

    def name(self, default=None, token=False):
        """
        Returns a human-readable name.

        First it tries :attr:`standard_name`, then 'long_name', then
        'var_name', then the STASH attribute before falling back to
        the value of `default` (which itself defaults to 'unknown').

        Kwargs:

        * default:
            The value of the default name.
        * token:
            If true, ensure that the name returned satisfies the criteria for
            the characters required by a valid NetCDF name. If it is not
            possible to return a valid name, then a ValueError exception is
            raised.

        Returns:
            String.

        """
        def _check(item):
            return self.token(item) if token else item

        default = self._DEFAULT_NAME if default is None else default

        result = (_check(self.standard_name) or _check(self.long_name) or
                  _check(self.var_name) or
                  _check(str(self.attributes.get('STASH', ''))) or
                  _check(default))

        if token and result is None:
            emsg = 'Cannot retrieve a valid name token from {!r}'
            raise ValueError(emsg.format(self))

        return result

    @property
    def names(self):
        """
        A tuple containing all of the metadata names. This includes the
        standard name, long name, NetCDF variable name, and attributes
        STASH name.

        """
        standard_name = self.standard_name
        long_name = self.long_name
        var_name = self.var_name
        stash_name = self.attributes.get("STASH")
        if stash_name is not None:
            stash_name = str(stash_name)
        return Names(standard_name, long_name, var_name, stash_name)

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
        self._standard_name = get_valid_standard_name(name)

    @property
    def units(self):
        """The :mod:`~cf_units.Unit` instance of the object."""
        return self._units

    @units.setter
    def units(self, unit):
        self._units = cf_units.as_unit(unit)

    @property
    def var_name(self):
        """The netCDF variable name for the object."""
        return self._var_name

    @var_name.setter
    def var_name(self, name):
        if name is not None:
            result = self.token(name)
            if result is None or not name:
                emsg = '{!r} is not a valid NetCDF variable name.'
                raise ValueError(emsg.format(name))
        self._var_name = name

    @property
    def attributes(self):
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        self._attributes = LimitedAttributeDict(attributes or {})
