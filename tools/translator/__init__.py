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
import warnings

from metarelate.fuseki import FusekiServer


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
        self.mappings = sorted(temp, key=self._key)
        if len(self) == 0:
            msg = '{!r} contains no mappings.'
            warnings.warn(msg.format(self.__class__.__name__))

    def lines(self):
        """
        Provides an iterator generating the encoded string representation
        of each member of this metarelate mapping translation.

        Returns:
            An iterator of string.

        """
        msg = '\tGenerating phenomenon translation {!r}.'
        print msg.format(self.mapping_name)
        lines = ['\n%s = {\n' % self.mapping_name]
        payload = [self.encode(mapping) for mapping in self.mappings]
        lines.extend(payload)
        lines.append('    }\n')
        return iter(lines)

    def __len__(self):
        return len(self.mappings)

    @abstractmethod
    def _key(self, mapping):
        """Abstract method to provide the sort key of the mappings order."""

    @abstractmethod
    def encode(self, mapping):
        """
        Abstract method to return the chosen encoded representation
        of a metarelate mapping translation.

        """

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

    # def _available(self, prop):
    #     """Determine whether a fully populated property is available."""
    #     return prop is not None# and prop.complete

    # def cf_constrained_notation(self, concept):
    #     """
    #     Given a CF component from a mapping, the skos notation for
    #     the associated CF coordinate and phenomenon are returned.

    #     See :meth:`Mapping.cf_coordinate_notation` and
    #     :meth:`Mapping.cf_phenomenon_notation`.

    #     Args:
    #     * concept:
    #         A :class:`metarelate.Component` instance.

    #     Returns:
    #         Tuple containing the :class:`DimensionCoordinate` and
    #         :class:`CFName` namedtuples.

    #     """
    #     coordinate = phenomenon = None
    #     for component in concept.components:
    #         if component.type == 'dimensionCoordinate':
    #             coordinate = self.cf_coordinate_notation(component)
    #         if component.type == 'field':
    #             phenomenon = self.cf_phenomenon_notation(component)
    #     return coordinate, phenomenon

    # def cf_coordinate_notation(self, component):
    #     """
    #     Given a CF component from a mapping, the skos notation for
    #     the associated CF standard name, units and points are returned.

    #     Args:
    #     * component:
    #         A :class:`metarelate.Component` instance.

    #     Returns:
    #         Tuple containing the CF standard name, units and points
    #         skos notation.

    #     """
    #     units = component.units.value.notation
    #     points = int(component.points.value.notation)
    #     standard_name = component.standard_name
    #     if self._available(standard_name):
    #         standard_name = standard_name.value.notation
    #     return DimensionCoordinate(standard_name, units, points)

    def cf_phenomenon_notation(self, component):
        """
        Given a CF component from a mapping, the skos notation for
        the associated CF standard name, long name and units of the
        phenomenon are returned.

        Args:
        * component:
            A :class:`metarelate.Component` or
            :class:`metarelate.Component` instance.

        Returns:
            Tuple containing the CF standard name, long name and units
            skos notation.

        """
        units = component.units.notation
        if isinstance(units, unicode):
            units = str(units)
        #standard_name = component.standard_name
        standard_name = None
        long_name = None
        if hasattr(component, 'standard_name'):
            standard_name = component.standard_name.notation
        elif hasattr(component, 'long_name'):
            long_name = component.long_name.notation
        return CFName(standard_name, long_name, units)

    # def grib1_notation(self, concept):
    #     """
    #     Given a GRIB (edition 1) concept from a mapping, the skos notation
    #     for the associated GRIB edition, table II version, centre and
    #     indicator of parameter are returned.

    #     Args:
    #     * concept:
    #         A :class:`metarelate.Component` instance.

    #     Returns:
    #         Tuple containing the GRIB1 edition, version, centre and
    #         indicator skos notation.

    #     """
    #     edition = int(concept.editionNumber.value.notation)
    #     version = int(concept.table2Version.value.notation)
    #     centre = int(concept.centre.value.notation)
    #     indicator = int(concept.indicatorOfParameter.value.notation)
    #     return G1LocalParam(edition, version, centre, indicator)

    def grib2_notation(self, component):
        """
        Given a GRIB (edition 2) parameter component from a mapping, the skos
        notation for the associated GRIB edition, discipline, parameter
        category and parameter number are returned.

        Args:
        * component:
            A :class:`metarelate.Component` instance.

        Returns:
            Tuple containing the GRIB2 edition, discipline, category and
            number skos notation.

        """
        gpd = '<http://codes.wmo.int/def/grib2/parameter>'
        if not (len(component) == 1 and hasattr(component, gpd)):
            raise ValueError('component is not a GRIB2 parameter')
        import requests

        pref = ('prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>'
                'prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#>'
                'prefix skos: <http://www.w3.org/2004/02/skos/core#>')
        puri = component.__getattr__(gpd).rdfobject.data
        qstr = ('SELECT ?key ?val '
                'WHERE {'
                '    { SELECT ?param ?key ?val '
                'WHERE {'
                '      ?param ?p ?o .'
                '    FILTER(?param = %s)'
                '          ?p rdfs:range ?range .'
                '        ?range skos:notation ?key .'
                '        ?o skos:notation ?val . '
                '  } } UNION'
                '  { SELECT ?param ?key ?val WHERE {'
                '           ?param rdf:type ?ptype .'
                '        ?ptype skos:notation ?key .'
                '        ?param skos:notation ?val .'
                '    FILTER(?param = %s)'
                '    } }'
                '}' % (puri, puri))

        base = 'http://codes.wmo.int/system/query?'
        r = requests.get(base, params={'query':pref+qstr, 'output':'json'})
        notation = {}
        for b in r.json()['results']['bindings']:
            notation[str(b['key']['value'])] = str(b['val']['value'])

        # edition = int(component.editionNumber.value.notation)
        # discipline = int(component.discipline.value.notation)
        # category = int(component.parameterCategory.value.notation)
        # number = int(component.parameterNumber.value.notation)
        return G2Param(int(notation['editionNumber']),
                       int(notation['discipline']),
                       int(notation['parameterCategory']),
                       int(notation['parameterNumber']))

    def is_cf(self, comp, kind='<http://def.scitools.org.uk/cfmodel/Field>'):
        """
        Determines whether the provided component from a mapping
        represents a simple CF component of the given kind.

        Args:
        * component:
            A :class:`metarelate.Component` or
            :class:`metarelate.Component` instance.

        Kwargs:
        * kind:
            The type of CF :class:`metarelate.Component` or
            :class:`metarelate.Component`. Defaults to 'field'.

        Returns:
            Boolean.

        """
        result = False
        result = hasattr(comp, 'com_type') and \
            comp.com_type == kind and \
            hasattr(comp, 'units') and \
            len(comp) in [1, 2]
        return result

    # def is_cf_constrained(self, concept):
    #     """
    #     Determines whether the provided concept from a mapping
    #     represents a compound CF concept for a phenomenon and
    #     a dimension coordinate constraint.

    #     Args:
    #     * concept:
    #         A :class:`metarelate.Component` instance.

    #     Returns:
    #         Boolean.

    #     """
    #     result = False
    #     if len(concept) == 2:
    #         constraint = phenomenon = False
    #         for component in concept.components:
    #             if self.is_cf(component, kind='dimensionCoordinate') and \
    #                     self._available(component.points):
    #                 constraint = True
    #             if self.is_cf(component):
    #                 phenomenon = True
    #         result = constraint and phenomenon
    #     return result

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

    # def is_grib1_local_param(self, concept):
    #     """
    #     Determines whether the provided concept from a mapping
    #     represents a simple GRIB edition 1 concept for a local
    #     parameter.

    #     Args:
    #     * concept:
    #         A :class:`metarelate.Component` instance.

    #     Returns:
    #         Boolean.

    #     """
    #     result = False
    #     if concept.simple:
    #         result = self._available(concept.editionNumber) and \
    #             self._available(concept.table2Version) and \
    #             self._available(concept.centre) and \
    #             self._available(concept.indicatorOfParameter)
    #     return result

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
        
        # result = False
        # import pdb; pdb.set_trace()
        gpd = '<http://codes.wmo.int/def/grib2/parameter>'
        result = len(component) == 1 and hasattr(component, gpd)
        # if component.simple:
        #     result = self._available(component.editionNumber) and \
        #         self._available(component.discipline) and \
        #         self._available(component.parameterCategory) and \
        #         self._available(component.parameterNumber)
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


class CFFieldcodeMapping(Mapping):
    """
    Represents a container for CF phenomenon to UM field-code metarelate
    mapping translations.

    Encoding support is provided to generate the Python dictionary source
    code representation of these mappings from CF standard name, long name,
    and units to UM field-code.

    """
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        return self.cf_phenomenon_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping CF standard name, long name, and units
        to UM field-code.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from CF to UM field-code.

        Returns:
            String.

        """
        msg = '    ' \
            'CFName({standard_name!r}, {long_name!r}, {units!r}): {lbfc},\n'
        cf = self.cf_phenomenon_notation(mapping.source)
        lbfc = mapping.target.lbfc.notation
        return msg.format(lbfc=lbfc, **cf._asdict())

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
        return '<http://def.scitools.org.uk/cfmodel/Field>'

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
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        return int(mapping.source.lbfc.notation)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping UM field-code to CF standard name,
        long name, and units.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from UM field-code to CF.

        Returns:
            String.

        """
        msg = '    ' \
            '{lbfc}: CFName({standard_name!r}, {long_name!r}, {units!r}),\n'
        lbfc = mapping.source.lbfc.notation
        cf = self.cf_phenomenon_notation(mapping.target)
        return msg.format(lbfc=lbfc, **cf._asdict())

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
        return '<http://def.scitools.org.uk/cfmodel/Field>'

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
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        # return mapping.source.stash.value.notation
        return mapping.source.stash.notation

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping UM stash-code to CF standard name,
        long name, and units.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from UM stash-code to CF.

        Returns:
            String.

        """
        msg = '    ' \
            '{stash!r}: CFName({standard_name!r}, {long_name!r}, {units!r}),\n'
        stash = mapping.source.stash.notation
        cf = self.cf_phenomenon_notation(mapping.target)
        return msg.format(stash=stash, **cf._asdict())

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
        #return 'um'
        return '<http://reference.metoffice.gov.uk/um/f3/UMField>'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        #return 'cf'
        return '<http://def.scitools.org.uk/cfmodel/Field>'

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
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        return self.grib1_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping GRIB1 local parameter to CF phenomenon.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from GRIB1 local parameter to CF phenomenon.

        Returns:
            String.

        """
        msg = '    ' \
            'G1LocalParam({grib.edition}, {grib.t2version}, {grib.centre}, ' \
            '{grib.iParam}): ' \
            'CFName({cf.standard_name!r}, {cf.long_name!r}, {cf.units!r}),\n'
        grib = self.grib1_notation(mapping.source)
        cf = self.cf_phenomenon_notation(mapping.target)
        return msg.format(grib=grib, cf=cf)

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
        return 'grib'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return 'cf'

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
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        return self.cf_phenomenon_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping CF phenomenon to GRIB1 local parameter.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from CF phenomenon to GRIB1 local parameter.

        Returns:
            String.

        """
        msg = '    ' \
            'CFName({cf.standard_name!r}, {cf.long_name!r}, {cf.units!r}): ' \
            'G1LocalParam({grib.edition}, {grib.t2version}, {grib.centre}, ' \
            '{grib.iParam}),\n'
        cf = self.cf_phenomenon_notation(mapping.source)
        grib = self.grib1_notation(mapping.target)
        return msg.format(cf=cf, grib=grib)

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
        return 'cf'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return 'grib'

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
    def _key(self, mapping):
        """Provides the sort key of the mapping order."""
        return self.grib1_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping GRIB1 local parameter to CF phenomenon
        and dimension coordinate.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from GRIB1 local parameter to CF phenomenon and dimension
            coordinate.

        Returns:
            String.

        """
        msg = '    ' \
            'G1LocalParam({grib.edition}, {grib.t2version}, {grib.centre}, ' \
            '{grib.iParam}): ' \
            '(CFName({phenomenon.standard_name!r}, ' \
            '{phenomenon.long_name!r}, {phenomenon.units!r}), ' \
            'DimensionCoordinate({coordinate.standard_name!r}, ' \
            '{coordinate.units!r}, ({coordinate.points},))),\n'
        grib = self.grib1_notation(mapping.source)
        coordinate, phenomenon = self.cf_constrained_notation(mapping.target)
        return msg.format(grib=grib, phenomenon=phenomenon,
                          coordinate=coordinate)

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
        return 'grib'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return 'cf'

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
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        return self.cf_constrained_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping CF phenomenon and dimension coordinate
        to GRIB1 local parameter.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from CF phenomenon and dimension coordinate to GRIB1 local
            parameter.

        Returns:
            String.

        """
        msg = '    ' \
            '(CFName({phenomenon.standard_name!r}, ' \
            '{phenomenon.long_name!r}, {phenomenon.units!r}), ' \
            'DimensionCoordinate({coordinate.standard_name!r}, ' \
            '{coordinate.units!r}, ({coordinate.points},))): ' \
            'G1LocalParam({grib.edition}, {grib.t2version}, {grib.centre}, ' \
            '{grib.iParam}),\n'
        coordinate, phenomenon = self.cf_constrained_notation(mapping.source)
        grib = self.grib1_notation(mapping.target)
        return msg.format(phenomenon=phenomenon, coordinate=coordinate,
                          grib=grib)

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
        return 'cf'

    @property
    def target_scheme(self):
        """
        Property that specifies the name of the scheme for the target
        :class:`metarelate.Component` defining this metarelate mapping
        translation.

        """
        return 'grib'

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
    def _key(self, mapping):
        """Provides the sort key of the mapping order."""
        return self.grib2_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represent an
        entry in a dictionary mapping GRIB2 parameter to CF phenomenon.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from GRIB2 parameter to CF phenomenon.

        Returns:
            String.

        """
        msg = '    ' \
            'G2Param({grib.edition}, {grib.discipline}, {grib.category}, ' \
            '{grib.number}): ' \
            'CFName({cf.standard_name!r}, {cf.long_name!r}, {cf.units!r}),\n'
        grib = self.grib2_notation(mapping.source)
        cf = self.cf_phenomenon_notation(mapping.target)
        return msg.format(grib=grib, cf=cf)

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
        return '<http://def.scitools.org.uk/cfmodel/Field>'

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
    def _key(self, mapping):
        """Provides the sort key of the mappings order."""
        return self.cf_phenomenon_notation(mapping.source)

    def encode(self, mapping):
        """
        Return a string of the Python source code required to represet an
        entry in a dictionary mapping CF phenomenon to GRIB2 parameter.

        Args:
        * mapping:
            A :class:`metarelate.Mapping` instance representing a translation
            from CF phenomenon to GRIB2 parameter.

        Returns:
            String.

        """
        msg = '    ' \
            'CFName({cf.standard_name!r}, {cf.long_name!r}, {cf.units!r}): ' \
            'G2Param({grib.edition}, {grib.discipline}, {grib.category}, ' \
            '{grib.number}),\n'
        cf = self.cf_phenomenon_notation(mapping.source)
        grib = self.grib2_notation(mapping.target)
        return msg.format(cf=cf, grib=grib)

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
        return '<http://def.scitools.org.uk/cfmodel/Field>'

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
