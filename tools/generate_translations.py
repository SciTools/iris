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

import itertools

import metocean.queries as moq
import metocean.fuseki as fuseki


header = '''# (C) British Crown Copyright 2010 - 2013, Met Office
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

# DO NOT EDIT: AUTO-GENERATED 


'''

def bodc_udunit_fix(units):
    """
    helper function to update syntax in use in BODC server to conform to udunits
    
    """
    units = units.strip('"')
    units = units.replace('Dmnless','1')
    units = units.replace('#','1')
    units = units.replace('deg','degree')
    units = units.replace('degreeree','degree')
    if units.startswith("'canonical_units': '/"):
        denom = units.split("'canonical_units': '/")[-1]
        units = "'canonical_units': '1/" + denom
    ## wrong, dB should be a recognised unit
    units = units.replace("'dB'", "'1'")
    return units


def main():
    """
    creates the Cf standard names dictionary in the Iris code base
    
    """
    with fuseki.FusekiServer(3333) as fu_p:
        #generate standard names dictionary
        sn_file = '../lib/iris/std_names.py'
        with open(sn_file, 'w') as snf:
            snf.write(header)
            snf.write('STD_NAMES = {')
            sn_graph = 'http://CF/cf-standard-name-table.ttl'
            for sn in moq.get_all_notation_note(fu_p, sn_graph):
                notation = sn['notation']
                units = sn['units']
                # non udunits compliant syntax used on NERC vocab server
                units = bodc_udunit_fix(units)
                snf.write('''%s: {%s},\n''' % (notation, units))
            snf.write('}')

if __name__ == '__main__':
    main()


