# (C) British Crown Copyright 2013, Met Office
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

import itertools
from time import time

import metocean.queries as moq
import metocean.fuseki as fuseki
import translator.mappings as mappings

header = """# (C) British Crown Copyright 2013, Met Office
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

header += '''
# DO NOT EDIT: AUTO-GENERATED 

'''


icol = 'import collections\n'

cf_tuple_def = '''
CFname = collections.namedtuple('CFname', ['standard_name', 'long_name',
                                           'unit'])
'''


end_dictionary = '''
}
'''

BUILT_FILES = {'../lib/iris/fileformats/um_cf_map.py': [icol, cf_tuple_def],}


def str_line_sort(st):
    sort_st = st.split('\n')
    sort_st.sort()
    st = '\n'.join(sort_st)
    return st

def dict_line_sort(st):
    sort_st = st.split('\n')[0:-2]
    try:
        sort_st.sort(key=lambda str: int(str.split(':')[0].strip().replace('"','')))
        st = '\n'.join(sort_st)
    except ValueError:
        st = str_line_sort(st)
    return st



iris_format = '<http://www.metarelate.net/metOcean/format/cf>'

formats = ['<http://www.metarelate.net/metOcean/format/um>']



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
    start_time = time()
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
        # generate translations
        format_maps = {}
        for fformat in formats:
            rtime = time()
            print fformat, ' retrieving: '
            format_maps[fformat] = {'import':{}, 'export':{}}
            ret_start_time = time()
            imports = fu_p.retrieve_mappings(fformat, iris_format)
            ret_time = time()
            print 'imp retrieve_mappings: {} s'.format(int(ret_time - ret_start_time))
            imp_maps = [mappings.make_mapping(amap, fu_p) for amap in imports]
            makem_time = time()
            print 'imp make_mappings: {} s'.format(int(makem_time-ret_time))
            imp_maps.sort(key=type)
            for g_type, group in itertools.groupby(imp_maps, key=type):
                format_maps[fformat]['import'][g_type.__name__] = list(group)
            ret_start_time = time()
            exports = fu_p.retrieve_mappings(iris_format, fformat)
            ret_time = time()
            print 'exp retrieve_mappings: {} s'.format(int(ret_time - ret_start_time))
            exp_maps = [mappings.make_mapping(amap, fu_p) for amap in exports]
            makem_time = time()
            print 'exp make_mappings: {} s'.format(int(makem_time-ret_time))
            exp_maps.sort(key=type)
            for g_type, group in itertools.groupby(exp_maps, key=type):
                format_maps[fformat]['export'][g_type.__name__] = list(group)
            ftime = time()
            
            print len(imports), ' imports, ', len(exports), 'exports'
            print str(int(ftime - rtime)), 's'
            rtime = ftime
        for afile in BUILT_FILES:
            f = open(afile, 'w')
            f.write(header)
            for extras in BUILT_FILES[afile]:
                f.write(extras)
            f.close()

        for fformat in formats:
            for direction in ['import', 'export']:
                ports = format_maps[fformat][direction]
                for map_set in ports:
                    print direction
                    print map_set
                    if map_set == 'NoneType':
                        ec = 'Some {} {} mappings not categorised'
                        ec = ec.format(fformat, direction)
                        print ec
                    else:
                        if ports[map_set][0].in_file not in BUILT_FILES:
                            ec = '{} writing to unmanaged file {}'
                            ec = ec.format(map_set, ports[map_set][0].in_file)
                            raise ValueError(ec)
                        map_str = ''
                        for port_mappings in ports[map_set]:
                            map_str += port_mappings.encode()
                        if ports[map_set][0].to_sort:
                            map_str = dict_line_sort(map_str)
                        if ports[map_set][0].container:
                            map_str = ports[map_set][0].container + map_str
                        if ports[map_set][0].closure:
                            map_str += ports[map_set][0].closure
                        with open(ports[map_set][0].in_file, 'a') as in_file:
                            in_file.write(map_str)
            fftime = time()
            print fformat, ' writing: ', str(int(fftime - ftime)), 's'
            ftime = fftime
    end_time = time()
    print 'total time: ', str(int(end_time - start_time)), 's'


if __name__ == '__main__':
    main()


