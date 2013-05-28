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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris. If not, see <http://www.gnu.org/licenses/>.

import metocean.queries as moq

OPEQ = '<http://www.openmath.org/cd/relation1.xhtml#eq>'

def make_concept(definition, fu_p):
    """
    Concept object factory
    selects the appropriate subclass for the inputs
    """

    built_concepts = []
    for concept_type in Concept.__subclasses__():
        if concept_type.type_match(definition, fu_p):
            built_concepts.append(concept_type(definition, fu_p))
    if len(built_concepts) != 1:
        if len(built_concepts) == 0:
            ec = 'no matching Concept type found \n{}'.format(definition)
            #raise ValueError(ec)
            built_concepts = [None]
        else:
            ec = 'multiple matching Concept types found\n {}'
            ec = ec.format(built_concepts)
            raise ValueError(ec)
            built_concepts = [None]
    return built_concepts[0]


class Concept(object):
    """
    a source or target concept
    """
    def __init__(self, definition, fu_p):
        return NotImplemented
    def notation(self, direction):
        ## direction should be 'test' or 'assign' only
        return NotImplemented
    def type_match(definition, fu_p):
        return NotImplemented
    

class StashConcept(Concept):
    """
    a concept which is a UM stash code only
    """
    def __init__(self, definition, fu_p):
        self.fformat = definition['mr:hasFormat']
        self.properties = definition['mr:hasProperty']
        self.id = definition['component']
        self.fu_p = fu_p
    def notation(self, direction=None):
        val = self.properties[0].get('rdf:value')
        stash = moq.get_label(self.fu_p, val)
        return stash
    @staticmethod
    def type_match(definition, fu_p):
        STASHC = '<http://reference.metoffice.gov.uk/def/um/stash/concept/'
        F3STASH = '<http://reference.metoffice.gov.uk/def/um/umdp/F3/stash>'
        fformat = '<http://www.metarelate.net/metOcean/format/um>'
        ff = definition['mr:hasFormat'] == fformat
        properties = definition.get('mr:hasProperty', [])
        components = definition.get('mr:hasComponent', [])
        if ff and len(properties) == 1 and len(components) == 0:
            val = properties[0].get('rdf:value')
            if val:
                stashval = val.startswith(STASHC)
            else:
                stashval = False
            name = properties[0].get('mr:name')
            if name:
                stashname = name == F3STASH
            else:
                stashname = False
            operator = properties[0].get('mr:operator')
            if operator:
                op_eq = operator = OPEQ
            else:
                op_eq = False
            if stashval and stashname and op_eq:
                stash = True
            else:
                stash = False
        else:
            stash = False
        return stash


class FieldcodeConcept(Concept):
    """
    a concept which is a UM field code only
    """
    def __init__(self, definition, fu_p):
        self.fformat = definition['mr:hasFormat']
        self.properties = definition['mr:hasProperty']
        self.id = definition['component']
        self.fu_p = fu_p
    def notation(self, direction=None):
        val = self.properties[0].get('rdf:value')
        fcode = moq.get_label(self.fu_p, val)
        return fcode
    @staticmethod
    def type_match(definition, fu_p):
        FIELDC = '<http://reference.metoffice.gov.uk/def/um/fieldcode/'
        F3FIELD = '<http://reference.metoffice.gov.uk/def/um/umdp/F3/lbfc>'
        fformat = '<http://www.metarelate.net/metOcean/format/um>'
        ff = definition['mr:hasFormat'] == fformat
        properties = definition.get('mr:hasProperty', [])
        components = definition.get('mr:hasComponent', [])
        if ff and len(properties) == 1 and len(components) == 0:
            val = properties[0].get('rdf:value')
            if val:
                fieldval = val.startswith(FIELDC)
            else:
                fieldval = False
            name = properties[0].get('mr:name')
            if name:
                fieldname = name == F3FIELD
            else:
                fieldname = False
            operator = properties[0].get('mr:operator')
            if operator:
                op_eq = operator = OPEQ
            else:
                op_eq = False
            if fieldval and fieldname and op_eq:
                fieldcode = True
            else:
                fieldcode = False
        else:
            fieldcode = False
        return fieldcode



class CFPhenomDefConcept(Concept):
    """
    a concept which is only defining a CF Field's base phenomenon
    """
    def __init__(self, definition, fu_p):
        self.fformat = definition['mr:hasFormat']
        self.properties = definition['mr:hasProperty']
        self.id = definition['component']
        self.fu_p = fu_p
    def notation(self, direction=None):
        cfsn = None
        lname = None
        units = None
        for p in self.properties:
            if moq.get_label(self.fu_p, p.get('mr:name')) == '"standard_name"':
                cfsn = moq.get_label(self.fu_p, p.get('rdf:value'))
                if cfsn.startswith('<'):
                    cfsn = None
            elif moq.get_label(self.fu_p, p.get('mr:name')) == '"long_name"':
                lname = p.get('rdf:value')
            elif moq.get_label(self.fu_p, p.get('mr:name')) == '"units"':
                units = moq.get_label(self.fu_p, p.get('rdf:value'))
        return cfsn, lname, units
    @staticmethod
    def type_match(definition, fu_p):
        fformat = '<http://www.metarelate.net/metOcean/format/cf>'
        ff = definition['mr:hasFormat'] == fformat
        properties = definition.get('mr:hasProperty', [])
        components = definition.get('mr:hasComponent', [])
        if ff and len(components) == 0:
            define = {}
            for prop in properties:
                op = prop.get('mr:operator')
                name = prop.get('mr:name', '')
                value = prop.get('rdf:value')
                if op and value and op == OPEQ:
                    # name_label = moq.get_label(fu_p, name)
                    # value_label = moq.get_label(fu_p, value)
                    # if not define.get(name_label):
                    #     define[name_label] = value_label
                    if not define.get(name):
                        define[name] = value
            # required = set(('"units"', '"type"'))
            # eitheror = set(('"standard_name"', '"long_name"'))
            required = set(('<http://def.cfconventions.org/datamodel/units>',
                            '<http://def.cfconventions.org/datamodel/type>'))
            eitheror = set(('<http://def.cfconventions.org/datamodel/standard_name>',
                            '<http://def.cfconventions.org/datamodel/long_name>'))
            if set(define.keys()).issuperset(required) and \
                set(define.keys()).issubset(required.union(eitheror)):
                phenom = True
            else:
                phenom = False
        else:
            phenom = False
        return phenom

