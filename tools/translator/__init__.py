# (C) British Crown Copyright 2014, Met Office
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
Provides the framework to support the encoding of metarelate mapping
translations.

"""

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
import re
import warnings

from metarelate.fuseki import FusekiServer
import metarelate

# Restrict the tokens exported from this module.
__all__ = ['Mapping', 'CFFieldcodeMapping',
           'FieldcodeCFMapping', 'StashCFMapping',
           'GRIB1LocalParamCFMapping', 'CFGRIB1LocalParamMapping',
           'GRIB1LocalParamCFConstrainedMapping',
           'CFConstrainedGRIB1LocalParamMapping',
           'GRIB2ParamCFMapping', 'CFGRIB2ParamMapping']

CFName = namedtuple('CFName', 'standard_name long_name units')
DimensionCoordinate = namedtuple('DimensionCoordinate',
                                 'standard_name units points')
G1LocalParam = namedtuple('G1LocalParam', 'edition t2version centre iParam')
G2Param = namedtuple('G2Param', 'edition discipline category number')


class Mapping(object):
    """
    Abstract base class to support the encoding of specific metarelate
    mapping translations.

    """
    __metaclass__ = ABCMeta

    def __init__(self, mappings):
        """
        Filter the given sequence of mappings for those member
        :class:`metarelate.Mapping` translations containing a source
        :class`metarelate.Component` with a matching
        :attribute:`Mapping.source_scheme` and a target
        :class:`metarelate.Component` with a matching
        :attribute:`Mapping.target_scheme`.

        Also see :method:`Mapping.valid_mapping` for further matching
        criterion for candidate metarelate mapping translations.

        Args:
        * mappings:
            Iterator of :class:`metarelate.Mapping` instances.

        """
        temp = []
        # Filter the mappings for the required type of translations.
        for mapping in mappings:
            source = mapping.source
            target = mapping.target
            if source.com_type == self.source_scheme and \
                    target.com_type == self.target_scheme and \
                    self.valid_mapping(mapping):
                temp.append(mapping)
        self.mappings = temp
        if len(self) == 0:
            msg = '{!r} contains no mappings.'
            warnings.warn(msg.format(self.__class__.__name__))

    def _sort_lines(self, payload):
        """
        Return a sorted list of strings,
        sort on dict key
        """
        return payload

    def lines(self, fuseki_process):
        """
        Provides an iterator generating the encoded string representation
        of each member of this metarelate mapping translation.

        Returns:
            An iterator of string.

        """
        msg = '\tGenerating phenomenon translation {!r}.'
        print msg.format(self.mapping_name)
        lines = ['\n%s = {\n' % self.mapping_name]

        for mapping in self.mappings:
            try:
                self.encode(mapping, fuseki_process)
            except Exception, e:
                import pdb; pdb.set_trace()
                self.encode(mapping, fuseki_process)
        payload = [self.encode(mapping, fuseki_process) for mapping in self.mappings]
        ## now sort the payload
        payload.sort(key=self._key)
        lines.extend(payload)
        lines.append('    }\n')
        return iter(lines)

    def __len__(self):
        return len(self.mappings)

    def _key(self, line):
        """Abstract method to provide the sort key of the mappings order."""
        return line

    @abstractproperty
    def mapping_name(self):
        """
        Abstract property that specifies the name of the dictionary
        to contain the encoding of this metarelate mapping translation.

        """

    @abstractproperty
    def source_scheme(self):
        """
        Abstract property that specifies the name of the scheme for
        the source :class:`metarelate.Component` defining this metarelate
        mapping translation.

        """

    @abstractproperty
    def target_scheme(self):
        """
        Abstract property that specifies the name of the scheme for
        the target :class:`metarelate.Component` defining this metarelate
        mapping translation.

        """

    @abstractmethod
    def valid_mapping(self, mapping):
        """
        Abstract method that determines whether the provided
        :class:`metarelate.Mapping` is a translation from the required
        source :class:`metarelate.Component` to the required target
        :class:`metarelate.Component`.

        """

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {}
        targetid = {}
        return sourceid, targetid

    def encode(self, mapping, fuseki_process):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping source to target.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation.

        Returns:
            String.

        """
        sourcemsg, targetmsg = self.msg_strings()
        sourceid, targetid = self.get_initial_id_nones()
        for prop in mapping.source.properties:
            sourceid.update(prop.get_identifiers(fuseki_process))
        for prop in mapping.target.properties:
            targetid.update(prop.get_identifiers(fuseki_process))
        return '{}: {}'.format(sourcemsg.format(**sourceid), 
                               targetmsg.format(**targetid))

    def is_cf(self, comp):
        """
        Determines whether the provided component from a mapping
        represents a simple CF component of the given kind.

        Args:
        * component:
            A :class:`metarelate.Component` or
            :class:`metarelate.Component` instance.

        Returns:
            Boolean.

        """
        kind='<http://def.scitools.org.uk/cfdatamodel/Field>'
        result = False
        result = hasattr(comp, 'com_type') and \
            comp.com_type == kind and \
            hasattr(comp, 'units') and \
            len(comp) in [1, 2]
        return result

    def is_cf_constrained(self, comp):
        """
        Determines whether the provided component from a mapping
        represents a compound CF component for a phenomenon and
        one dimension coordinate.

        Args:
        * component:
            A :class:`metarelate.Component` instance.

        Returns:
            Boolean.

        """
        ftype = '<http://def.scitools.org.uk/cfdatamodel/Field>'
        result = False
        cffield = hasattr(comp, 'com_type') and comp.com_type == ftype and \
                  hasattr(comp, 'units') and (hasattr(comp, 'standard_name') or\
                                              hasattr(comp, 'long_name'))
        dctype = '<http://def.scitools.org.uk/cfdatamodel/DimCoord>'
        dimcoord = hasattr(comp, 'dim_coord') and \
                   isinstance(comp.dim_coord, metarelate.ComponentProperty) and \
                   comp.dim_coord.component.com_type == dctype
        result = cffield and dimcoord
        return result

    def is_fieldcode(self, component):
        """
        Determines whether the provided concept from a mapping
        represents a simple UM concept for a field-code.

        Args:
        * concept:
            A :class:`metarelate.Component` instance.

        Returns:
            Boolean.

        """
        result = False
        result = hasattr(component, 'lbfc') and len(component) == 1
        return result

    def is_grib1_local_param(self, component):
        """
        Determines whether the provided component from a mapping
        represents a simple GRIB edition 1 component for a local
        parameter.

        Args:
        * component:
            A :class:`metarelate.Component` instance.

        Returns:
            Boolean.

        """
        gpd = '<http://codes.wmo.int/def/grib1/parameter>'
        result = len(component) == 1 and hasattr(component, gpd)
        return result

    def is_grib2_param(self, component):
        """
        Determines whether the provided component from a mapping
        represents a simple GRIB edition 2 component for a parameter.

        Args:
        * component:
            A :class:`metarelate.Component` instance.

        Returns:
            Boolean.

        """
        
        gpd = '<http://codes.wmo.int/def/grib2/parameter>'
        result = len(component) == 1 and hasattr(component, gpd)
        return result

    def is_stash(self, component):
        """
        Determines whether the provided concept for a mapping
        represents a simple UM concept for a stash-code.

        Args:
        * concept:
            A :class:`metarelate.Component` instance.

        Returns:
            Boolean.

        """
        result = False
        result = hasattr(component, 'stash') and len(component) == 1
        return result


def _cfn(line):
    match = re.match('^    CFName\((.+), (.+), (.+)\):.+,', line)
    if match is None:
        raise ValueError('encoding not sortable')
    sn, ln, u = match.groups()
    if sn == 'None':
        sn = None
    if ln == 'None':
        ln = None
    return [sn, ln, u]

class CFFieldcodeMapping(Mapping):
    """
    Represents a container for CF phenomenon to UM field-code metarelate
    mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from CF standard name, long name,
    and units to UM field-code.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return _cfn(line)

    def msg_strings(self):
        return ('    CFName({standard_name!r}, {long_name!r}, '
                '{units!r})',
                '{lbfc},\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {'standard_name': None, 'long_name': None}
        targetid = {}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'CF_TO_LBFC'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://reference.metoffice.gov.uk/um/f3/UMField>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        CF to UM field-code translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_cf(mapping.source) and self.is_fieldcode(mapping.target)
        
        


class FieldcodeCFMapping(Mapping):
    """
    Represents a container for UM field-code to CF phenomenon metarelate
    mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from UM field-code to
    CF standard name, long name, and units.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return int(line.split(':')[0].strip())

    def msg_strings(self):
        return ('    {lbfc}',
                'CFName({standard_name!r}, {long_name!r}, {units!r}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {}
        targetid = {'standard_name': None, 'long_name': None}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'LBFC_TO_CF'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://reference.metoffice.gov.uk/um/f3/UMField>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        UM field-code to CF translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_fieldcode(mapping.source) and self.is_cf(mapping.target)


class StashCFMapping(Mapping):
    """
    Represents a container for UM stash-code to CF phenomenon metarelate
    mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from UM stash-code to CF
    standard name, long name, and units.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return line.split(':')[0].strip()

    def msg_strings(self):
        return('    {stash!r}',
               'CFName({standard_name!r}, '
               '{long_name!r}, {units!r}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {}
        targetid = {'standard_name': None, 'long_name': None}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'STASH_TO_CF'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://reference.metoffice.gov.uk/um/f3/UMField>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        UM stash-code to CF translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_stash(mapping.source) and self.is_cf(mapping.target)


class GRIB1LocalParamCFMapping(Mapping):
    """
    Represents a container for GRIB (edition 1) local parameter to
    CF phenomenon metarelate mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from GRIB1 edition, table II version,
    centre and indicator of parameter to CF standard name, long name and units.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        match = re.match('^    G1LocalParam\(([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\):.*', line)
        if match is None:
            raise ValueError('encoding not sortable')
        return [int(i) for i in match.groups()]

    def msg_strings(self):
        return ('    G1LocalParam({editionNumber}, {table2version}, '
                '{centre}, {indicatorOfParameter})',
                'CFName({standard_name!r}, '
                '{long_name!r}, {units!r}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {}
        targetid = {'standard_name': None, 'long_name': None}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'GRIB1_LOCAL_TO_CF'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://codes.wmo.int/def/codeform/GRIB-message>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        GRIB1 local parameter to CF phenomenon translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_grib1_local_param(mapping.source) and \
            self.is_cf(mapping.target)


class CFGRIB1LocalParamMapping(Mapping):
    """
    Represents a container for CF phenomenon to GRIB (edition 1) local
    parameter metarelate mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from CF standard name, long name
    and units to GRIB1 edition, table II version, centre and indicator of
    parameter.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return _cfn(line)

    def msg_strings(self):
        return ('    CFName({standard_name!r}, {long_name!r}, '
                '{units!r})',
                'G1LocalParam({editionNumber}, {table2version}, '
                '{centre}, {indicatorOfParameter}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {'standard_name': None, 'long_name': None}
        targetid = {}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'CF_TO_GRIB1_LOCAL'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://codes.wmo.int/def/codeform/GRIB-message>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        CF phenomenon to GRIB1 local parameter translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_cf(mapping.source) and \
            self.is_grib1_local_param(mapping.target)


class GRIB1LocalParamCFConstrainedMapping(Mapping):
    """
    Represents a container for GRIB (edition 1) local parameter to
    CF phenomenon and dimension coordinate constraint metarelate mapping
    translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from GRIB1 edition, table II version,
    centre and indicator of parameter to CF phenomenon standard name, long name
    and units, and CF dimension coordinate standard name, units and points.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return line.split(':')[0].strip()

    def msg_strings(self):
        return ('    G1LocalParam({editionNumber}, {table2version}, '
                '{centre}, {indicatorOfParameter})',
                '(CFName({standard_name!r}, '
                '{long_name!r}, {units!r}), '
                'DimensionCoordinate({dim_coord[standard_name]!r}, '
                '{dim_coord[units]!r}, {dim_coord[points]})),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {}
        targetid = {'standard_name': None, 'long_name': None}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'GRIB1_LOCAL_TO_CF_CONSTRAINED'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://codes.wmo.int/def/codeform/GRIB-message>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        GRIB1 local parameter to CF phenomenon and dimension coordinate
        translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_grib1_local_param(mapping.source) and \
            self.is_cf_constrained(mapping.target)


class CFConstrainedGRIB1LocalParamMapping(Mapping):
    """
    Represents a container for CF phenomenon and dimension coordinate
    constraint to GRIB (edition 1) local parameter metarelate mapping
    translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from CF phenomenon standard name,
    long name and units, and CF dimension coordinate standard name, units and
    points to GRIB1 edition, table II version, centre and indicator of
    parameter.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return line.split(':')[0].strip()

    def msg_strings(self):
        return ('    (CFName({standard_name!r}, '
                '{long_name!r}, {units!r}), '
                'DimensionCoordinate({dim_coord[standard_name]!r}, '
                '{dim_coord[units]!r}, {dim_coord[points]}))',
                'G1LocalParam({editionNumber}, {table2version}, '
                '{centre}, {indicatorOfParameter}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {'standard_name': None, 'long_name': None}
        targetid = {}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'CF_CONSTRAINED_TO_GRIB1_LOCAL'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://codes.wmo.int/def/codeform/GRIB-message>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        CF phenomenon and dimension coordinate to GRIB1 local parameter
        translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_cf_constrained(mapping.source) and \
            self.is_grib1_local_param(mapping.target)


class GRIB2ParamCFMapping(Mapping):
    """
    Represents a container for GRIB (edition 2) parameter to CF phenomenon
    metarelate mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from GRIB2 edition, discipline,
    parameter category and indicator of parameter to CF standard name,
    long name and units.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        match = re.match('^    G2Param\(([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\):.*', line)
        if match is None:
            raise ValueError('encoding not sortable')
        return [int(i) for i in match.groups()]

    def msg_strings(self):
        return ('    G2Param({editionNumber}, {discipline}, '
                '{parameterCategory}, {parameterNumber})',
                'CFName({standard_name!r}, {long_name!r}, '
                '{units!r}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {}
        targetid = {'standard_name': None, 'long_name': None}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'GRIB2_TO_CF'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://codes.wmo.int/def/codeform/GRIB-message>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        GRIB2 parameter to CF phenomenon translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_grib2_param(mapping.source) and \
            self.is_cf(mapping.target)


class CFGRIB2ParamMapping(Mapping):
    """
    Represents a container for CF phenomenon to GRIB (edition 2) parameter
    metarelate mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from CF standard name, long name
    and units to GRIB2 edition, discipline, parameter category and indicator
    of parameter.

    """
    def _key(self, line):
        """Provides the sort key of the mappings order."""
        return _cfn(line)

    def msg_strings(self):
        return ('    CFName({standard_name!r}, {long_name!r}, '
                '{units!r})',
                'G2Param({editionNumber}, {discipline}, '
                '{parameterCategory}, {parameterNumber}),\n')

    def get_initial_id_nones(self):
        """
        Return the identifier items which may be None, and are needed
        for a msg_string 
        """
        sourceid = {'standard_name': None, 'long_name': None}
        targetid = {}
        return sourceid, targetid

    @property
    def mapping_name(self):
        """
        Property that specifies the name of the dictionary to contain the
        encoding of this metarelate mapping translation.

        """
        return 'CF_TO_GRIB2'

    @property
    def source_scheme(self):
        """
        Property that specifies the name of the scheme for the source
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://def.scitools.org.uk/cfdatamodel/Field>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return '<http://codes.wmo.int/def/codeform/GRIB-message>'

    def valid_mapping(self, mapping):
        """
        Determine whether the provided :class:`metarelate.Mapping` represents a
        CF phenomenon to GRIB2 parameter translation.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance.

        Returns:
            Boolean.

        """
        return self.is_cf(mapping.source) and \
            self.is_grib2_param(mapping.target)
