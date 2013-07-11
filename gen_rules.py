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

import re


def _write_rule(module_file, conditions, actions):
    module_file.write('\n')
    if len(conditions) == 1:
        module_file.write('    if {}:\n'.format(conditions[0]))
    else:
        module_file.write('    if \\\n')
        for condition in conditions[:-1]:
            module_file.write('            ({}) and \\\n'.format(condition))
        module_file.write('            ({}):\n'.format(conditions[-1]))
    for action in actions:
        if action.startswith('CoordAndDims(DimCoord('):
            match = re.match(r'CoordAndDims\((.*), ([0-1]+)\)$', action)
            if match:
                action = 'cube.add_dim_coord({})'.format(action[13:-1])
            else:
                action = 'cube.add_aux_coord({})'.format(action[13:-1])
        elif action.startswith('CoordAndDims(AuxCoord('):
            action = 'cube.add_aux_coord({})'.format(action[13:-1])
        elif action.startswith('CellMethod('):
            action = 'cube.add_cell_method({})'.format(action)
        elif action.startswith('CMCustomAttribute('):
            match = re.match(r'CMCustomAttribute\(([\'"0-9a-zA-Z_]+), (.+)\)$',
                             action)
            name = match.group(1)
            value = match.group(2)
            action = 'cube.attributes[{}] = {}'.format(name, value)
        elif action.startswith('CMAttribute('):
            match = re.match(r'CMAttribute\(([\'"0-9a-zA-Z_]+), (.+)\)$',
                             action)
            name = eval(match.group(1))
            value = match.group(2)
            action = 'cube.{} = {}'.format(name, value)
        elif action.startswith('Factory('):
            action = 'factories.append({})'.format(action)
        elif action.startswith('ReferenceTarget('):
            action = 'references.append({})'.format(action)
        else:
            print action
        module_file.write('        {}\n'.format(action))


def write_rules_module(rules_paths, module_path):
    # Define state constants
    IN_CONDITION = 1
    IN_ACTION = 2


    with open(module_path, 'w') as module_file:
        module_file.write('from iris.aux_factory import HybridHeightFactory\n')
        module_file.write('from iris.coords import AuxCoord, CellMethod,' \
                          ' DimCoord\n')
        module_file.write('from iris.fileformats.mosig_cf_map import' \
                          ' MOSIG_STASH_TO_CF\n')
        module_file.write('from iris.fileformats.rules import Factory,' \
                          ' Reference\n')
        module_file.write('from iris.fileformats.um_cf_map import' \
                          ' STASH_TO_CF\n')
        module_file.write('import iris.fileformats.pp\n')
        module_file.write('import iris.unit\n')
        module_file.write('\n\ndef convert(cube, field):\n')
        module_file.write('    f = field\n')
        module_file.write('    cm = cube\n')
        module_file.write('    factories = []\n')
        module_file.write('    references = []\n')
        for rules_path in rules_paths:
            with open(rules_path, 'r') as rules_file:
                conditions = []
                actions = []
                state = None
                for line in rules_file:
                    line = line.rstrip()
                    if line == "IF":
                        if conditions and actions:
                            _write_rule(module_file, conditions, actions)
                        conditions = []
                        actions = []
                        state = IN_CONDITION
                    elif line == "THEN":
                        state = IN_ACTION
                    elif len(line) == 0:
                        pass
                    elif line.strip().startswith('#'):
                        pass
                    elif state == IN_CONDITION:
                        conditions.append(line)
                    elif state == IN_ACTION:
                        actions.append(line)
                    else:
                        raise Exception('Rule file not read correctly at line: ' + line)
                if conditions and actions:
                    _write_rule(module_file, conditions, actions)
        module_file.write('\n    return factories, references\n')


if __name__ == '__main__':
    write_rules_module(
        ['/data/local/ithr/git/iris/lib/iris/etc/pp_rules.txt',
         '/data/local/ithr/git/iris/lib/iris/etc/pp_cross_reference_rules.txt',
        ],
        '/data/local/ithr/git/iris/lib/iris/fileformats/pp_rules.py')
