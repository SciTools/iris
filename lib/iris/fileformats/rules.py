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
"""
Processing of simple IF-THEN rules.

"""

import abc
import collections
import getpass
import logging
import logging.handlers as handlers
import operator
import os
import os.path
import platform
import sys
import types
import warnings

import numpy

import iris.config as config
import iris.cube
import iris.exceptions
import iris.unit


RuleResult = collections.namedtuple('RuleResult', ['cube', 'matching_rules', 'factories'])
Factory = collections.namedtuple('Factory', ['factory_class', 'args'])


# Controls the deferred import of all the symbols from iris.coords.
# This import all is used as the rules file does not use fully qualified class names.
_import_pending = True


# Dummy logging routine for when we don't want to do any logging.
def _dummy_log(format, filename, rules):
    pass


# Genuine logging routine
def _real_log(format, filename, rules):
    # Replace "\" with "\\", and "," with "\,"
    filename = filename.replace('\\', '\\\\').replace(',', '\\,')
    _rule_logger.info("%s,%s,%s" % (format, filename, ','.join([rule.id for rule in rules])))


# Debug logging routine (more informative that just object ids)
def _verbose_log(format, filename, rules):
    # Replace "\" with "\\", and "," with "\,"
    filename = filename.replace('\\', '\\\\').replace(',', '\\,')
    _rule_logger.info("\n\n-----\n\n%s,%s,%s" % (format, filename, '\n\n'.join([str(rule) for rule in rules])))


# Prepares a logger for file-based logging of rule usage
def _prepare_rule_logger(verbose=False):
    # Default to the dummy logger that does nothing
    logger = _dummy_log

    # Only do real logging if we've been told the directory to use ...
    log_dir = config.RULE_LOG_DIR
    if log_dir is not None:
        user = getpass.getuser()

        # .. and if we haven't been told to ignore the current invocation.
        ignore = False
        ignore_users = config.RULE_LOG_IGNORE
        if ignore_users is not None:
            ignore_users = ignore_users.split(',')
            ignore = user in ignore_users

        if not ignore:
            try:
                hostname = platform.node() or 'UNKNOWN'
                log_path = os.path.join(log_dir, '_'.join([hostname, user]))
                file_handler = handlers.RotatingFileHandler(log_path, maxBytes=1e7, backupCount=5)
                format = '%%(asctime)s,%s,%%(message)s' % getpass.getuser()
                file_handler.setFormatter(logging.Formatter(format, '%Y-%m-%d %H:%M:%S'))

                global _rule_logger
                _rule_logger = logging.getLogger('iris.fileformats.rules')
                _rule_logger.setLevel(logging.INFO)
                _rule_logger.addHandler(file_handler)
                _rule_logger.propagate = False

                if verbose:
                    logger = _verbose_log
                else:
                    logger = _real_log
                    
            except IOError:
                # If we can't create the log file for some reason then it's fine to just silently
                # ignore the error and fallback to using the dummy logging routine.
                pass

    return logger


# Defines the "log" function for this module
log = _prepare_rule_logger()


class DebugString(str):
    """
    Used by the rules for debug purposes

    """


class CMAttribute(object):
    """
    Used by the rules for defining attributes on the Cube in a consistent manner.
    
    """
    __slots__ = ('name', 'value')
    def __init__(self, name, value):
        self.name = name
        self.value = value


class CMCustomAttribute(object):
    """
    Used by the rules for defining custom attributes on the Cube in a consistent manner.
    
    """
    __slots__ = ('name', 'value')
    def __init__(self, name, value):
        self.name = name
        self.value = value


class CoordAndDims(object):
    """
    Used within rules to represent a mapping of coordinate to data dimensions. 
    
    """
    def __init__(self, coord, dims=None):
        self.coord = coord
        if dims is None:
            dims = []
        if not isinstance(dims, list):
            dims = [dims]
        self.dims = dims
        
    def add_coord(self, cube):
        added = False

        # Try to add to dim_coords?
        if isinstance(self.coord, iris.coords.DimCoord) and self.dims:
            if len(self.dims) > 1:
                raise Exception("Only 1 dim allowed for a DimCoord")
             
            # Does the cube already have a coord for this dim?
            already_taken = False
            for coord, coord_dim in cube._dim_coords_and_dims:
                if coord_dim == self.dims[0]:
                    already_taken = True
                    break
                    
            if not already_taken:
                cube.add_dim_coord(self.coord, self.dims[0])
                added = True

        # If we didn't add it to dim_coords, add it to aux_coords.                    
        if not added:
            cube.add_aux_coord(self.coord, self.dims)
        
    def __repr__(self):
        return "<CoordAndDims: %r, %r>" % (self.coord.name, self.dims)


class Reference(iris.util._OrderedHashable):
    _names = ('name',)
    """
    A named placeholder for inter-field references.

    """

    
# TODO: This function only uses data from a coord, and produces information only pertaining to a coord, so should it be in the coord.
def is_regular(coord):
    """Determine if the given coord is regular."""
    try:
        regular_step(coord)
    except iris.exceptions.CoordinateNotRegularError:
        return False
    except (TypeError, ValueError):
        return False
    return True


# TODO: This function only uses data from a coord, and produces information only pertaining to a coord, so should it be in the coord.
def regular_step(coord):
    """Return the regular step from a coord or fail."""
    if coord.ndim != 1:
        raise iris.exceptions.CoordinateMultiDimError("Expected 1D coord")
    if coord.shape[0] < 2:
        raise ValueError("Expected a non-scalar coord")

    diffs = coord.points[1:] - coord.points[:-1]
    avdiff = numpy.mean(diffs)
    if not numpy.allclose(diffs, avdiff, rtol=0.001):  # TODO: This value is set for test_analysis to pass... 
        raise iris.exceptions.CoordinateNotRegularError("Coord %s is not regular" % coord.name())
    return avdiff.astype(coord.points.dtype)


class Rule(object):
    """
    A collection of condition expressions and their associated action expressions.
    
    Example rule::

        IF
            f.lbuser[6] == 2
            f.lbuser[3] == 101
        THEN
            CMAttribute('standard_name', 'sea_water_potential_temperature')
            CMAttribute('units', 'Celsius')

    """
    def __init__(self, conditions, actions):
        """Create instance methods from our conditions and actions."""
        if not hasattr(conditions, '__iter__'):
            raise TypeError('Variable conditions should be iterable, got: '+ type(conditions))
        if not hasattr(actions, '__iter__'):
            raise TypeError('Variable actions should be iterable, got: '+ type(actions))
        
        self._conditions = conditions
        self._actions = actions
        self._exec_actions = []

        self.id = str(hash((tuple(self._conditions), tuple(self._actions))))

        for i, condition in enumerate(conditions):
            self._conditions[i] = condition

        # Create the conditions method.
        self._create_conditions_method()

        # Create the action methods.
        for i, action in enumerate(self._actions):
            if not action:
                action = 'None'
            self._create_action_method(i, action)

    def _create_conditions_method(self):
        # Bundle all the conditions into one big string.
        conditions = '(%s)' % ') and ('.join(self._conditions)
        if not conditions:
            conditions = 'None'
        # Create a method to evaluate the conditions.
        # NB. This creates the name '_exec_conditions' in the local
        # namespace, which is then used below.
        code = 'def _exec_conditions(self, field, f, pp, grib, cm): return %s'
        exec compile(code % conditions, '<string>', 'exec')
        # Make it a method of ours.
        self._exec_conditions = types.MethodType(_exec_conditions, self, type(self))
    
    @abc.abstractmethod
    def _create_action_method(self, i, action):
        pass
    
    @abc.abstractmethod
    def _process_action_result(self, obj, cube):
        pass    
    
    def __repr__(self):
        string = "IF\n"
        string += '\n'.join(self._conditions)
        string += "\nTHEN\n"
        string += '\n'.join(self._actions)
        return string

    def evaluates_true(self, cube, field):
        """Returns True if and only if all the conditions evaluate to True for the given field."""
        field = field
        f = field
        pp = field
        grib = field
        cm = cube
        
        try:
            result = self._exec_conditions(field, f, pp, grib, cm)
        except Exception, err:
            print >> sys.stderr, 'Condition failed to run conditions: %s : %s' % (self._conditions, err)
            raise err
                               
        return result

    def _matches_field(self, field):
        """Simple wrapper onto evaluates_true in the case where cube is None."""
        return self.evaluates_true(None, field)
        
    def run_actions(self, cube, field):
        """
        Adds to the given cube based on the return values of all the actions.
    
        """
        # Deferred import of all the symbols from iris.coords.
        # This import all is used as the rules file does not use fully qualified class names.
        global _import_pending
        if _import_pending:
            globals().update(iris.aux_factory.__dict__)
            globals().update(iris.coords.__dict__)
            globals().update(iris.coord_systems.__dict__)
            globals().update(iris.unit.__dict__)
            _import_pending = False
        
        # Define the variables which the eval command should be able to see
        f = field
        pp = field
        grib = field
        cm = cube
        
        factories = []
        for i, action in enumerate(self._actions):
            try:
                # Run this action.
                obj = self._exec_actions[i](field, f, pp, grib, cm)
                # Process the return value (if any), e.g a CM object or None.
                action_factory = self._process_action_result(obj, cube)
                if action_factory:
                    factories.append(action_factory)

            except iris.exceptions.CoordinateNotFoundError, err:
                print >> sys.stderr, 'Failed (msg:%(error)s) to find coordinate, perhaps consider running last: %(command)s' % {'command':action, 'error': err}
            except AttributeError, err:
                print >> sys.stderr, 'Failed to get value (%(error)s) to execute: %(command)s' % {'command':action, 'error': err}
            except Exception, err:
                print >> sys.stderr, 'Failed (msg:%(error)s) to run: %(command)s\nFrom the rule:%(me)r' % {'me':self, 'command':action, 'error': err}
                raise err
        return factories


class FunctionRule(Rule):
    """A Rule with values returned by its actions."""
    def _create_action_method(self, i, action):
        # CM loading style action. Returns an object, such as a coord.
        exec compile('def _exec_action_%d(self, field, f, pp, grib, cm): return %s' % (i, action), '<string>', 'exec')
        # Make it a method of ours.
        exec 'self._exec_action_%d = types.MethodType(_exec_action_%d, self, type(self))' % (i, i)
        # Add to our list of actions.
        exec 'self._exec_actions.append(self._exec_action_%d)' % i

    def _process_action_result(self, obj, cube):
        """Process the result of an action."""

        factory = None

        # NB. The names such as 'Coord' and 'CellMethod' are defined by
        # the "deferred import" performed by Rule.run_actions() above.
        if isinstance(obj, Coord):
            cube.add_coord(obj)

        elif isinstance(obj, CoordAndDims):
            obj.add_coord(cube)

        elif isinstance(obj, Factory):
            factory = obj

        #cell methods - not yet implemented
        elif isinstance(obj, CellMethod):
            cube.add_cell_method(obj)
            
        elif isinstance(obj, DebugString):
            print obj

        elif isinstance(obj, CMAttribute):
            # Temporary code to deal with invalid standard names from the translation table.
            # TODO: when name is "standard_name" force the value to be a real standard name
            if obj.name == 'standard_name' and obj.value is not None:
                cube.rename(obj.value)
            else:
                setattr(cube, obj.name, obj.value)
            
        elif isinstance(obj, CMCustomAttribute):
            cube.attributes[obj.name] = obj.value

        # The function returned nothing, like the pp save actions, "lbft = 3"
        elif obj is None:
            pass
        
        else:
            raise Exception("Object could not be added to cube. Unknown type: " + obj.__class__.__name__)

        return factory


class ObjectReturningRule(FunctionRule):
    """A rule which returns a list of objects when its actions are run.""" 
    def run_actions(self, cube, field):
        f = pp = grib = field
        cm = cube
        return [action(field, f, pp, grib, cm) for action in self._exec_actions]


class ProcedureRule(Rule):
    """A Rule with nothing returned by its actions."""
    def _create_action_method(self, i, action):
        # PP saving style action. No return value, e.g. "pp.lbft = 3".
        exec compile('def _exec_action_%d(self, field, f, pp, grib, cm): %s' % (i, action), '<string>', 'exec')
        # Make it a method of ours.
        exec 'self._exec_action_%d = types.MethodType(_exec_action_%d, self, type(self))' % (i, i)
        # Add to our list of actions.
        exec 'self._exec_actions.append(self._exec_action_%d)' % i

    def _process_action_result(self, obj, cube):
        # This should always be None, as our rules won't create anything.
        pass
            
    def conditional_warning(self, condition, warning):
        pass  # without this pass statement it alsp print, "  Args:" on a new line.
        if condition:
            warnings.warn(warning)


class RulesContainer(object):
    """
    A collection of :class:`Rule` instances, with the ability to read rule
    definitions from files and run the rules against given fields.
    
    """
    def __init__(self, filepath=None, rule_type=FunctionRule):
        """Create a new rule set, optionally adding rules from the specified file.
        
        The rule_type defaults to :class:`FunctionRule`,
        e.g for CM loading actions that return objects, such as *AuxCoord(...)*
        
        rule_type can also be set to :class:`ProcedureRule`
        e.g for PP saving actions that do not return anything, such as *pp.lbuser[3] = 16203* 
        """
        self._rules = []
        self.rule_type = rule_type
        if filepath is not None:
            self.import_rules(filepath)

    def import_rules(self, filepath):
        """Extend the rule collection with the rules defined in the specified file."""
        # Define state constants
        IN_CONDITION = 1
        IN_ACTION = 2
        
        rule_file = os.path.expanduser(filepath)
        file = open(rule_file, 'r')
        
        conditions = []
        actions = []
        state = None
        for line in file:
            line = line.rstrip()
            if line == "IF":
                if conditions and actions:
                    self._rules.append(self.rule_type(conditions, actions))
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
            self._rules.append(self.rule_type(conditions, actions))
        file.close()
            
    def result(self, field):
        """
        Return the :class:`iris.cube.Cube` resulting from running this
        set of rules with the given field.

        Args:
        
        * field:
            A field object relevant to the rule set.
        
        Returns: (cube, matching_rules)
        
        * cube - the resultant cube
        * matching_rules - a list of rules which matched

        """
        
        # If the field has a data manager, then put it on the cube, otherwise transfer the data to the cube
        if getattr(field, '_data_manager', None) is not None:
            data = field._data
            data_manager = field._data_manager
        else:
            data = field.data
            data_manager = None

        cube = iris.cube.Cube(data, data_manager=data_manager)
        
        verify_result = self.verify(cube, field)
        return verify_result
    
    def matching_rules(self, field):
        """
        Return a list of rules which match the given field.

        Returns: list of Rule instances
        
        """
        return filter(lambda rule: rule._matches_field(field), self._rules)
        
    def verify(self, cube, field):
        """
        Add to the given :class:`iris.cube.Cube` by running this set of
        rules with the given field.

        Args:
        
        * cube:
            An instance of :class:`iris.cube.Cube`.
        * field:
            A field object relevant to the rule set.
            
        Returns: (cube, matching_rules)
        
        * cube - the resultant cube
        * matching_rules - a list of rules which matched
        
        """
        matching_rules = []
        factories = []
        for rule in self._rules:
            if rule.evaluates_true(cube, field):
                matching_rules.append(rule)
                rule_factories = rule.run_actions(cube, field)
                if rule_factories:
                    factories.extend(rule_factories)
        return RuleResult(cube, matching_rules, factories)


def scalar_coord(cube, coord_name):
    """Try to find a single-valued coord with the given name."""
    found_coord = None
    for coord in cube.coords(name=coord_name):
        if coord.shape == (1,):
            found_coord = coord
            break
    return found_coord


def vector_coord(cube, coord_name):
    """Try to find a one-dimensional, multi-valued coord with the given name."""
    found_coord = None
    for coord in cube.coords(name=coord_name):
        if len(coord.shape) == 1 and coord.shape[0] > 1:
            found_coord = coord
            break
    return found_coord


def scalar_cell_method(cube, method, coord_name):
    """Try to find the given type of cell method over a single coord with the given name."""
    found_cell_method = None
    for cell_method in cube.cell_methods:
        if cell_method.method == method and len(cell_method.coord_names) == 1:
            name = cell_method.coord_names[0]
            coords = cube.coords(name=name)
            if len(coords) == 1:
                found_cell_method = cell_method
    return found_cell_method
