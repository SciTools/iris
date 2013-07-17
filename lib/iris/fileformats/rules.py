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
"""
Processing of simple IF-THEN rules.

"""

import abc
import collections
import copy
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

import numpy as np
import numpy.ma as ma

import iris.config as config
import iris.cube
import iris.exceptions
import iris.fileformats.mosig_cf_map
import iris.fileformats.um_cf_map
import iris.unit

RuleResult = collections.namedtuple('RuleResult', ['cube', 'matching_rules', 'factories'])
Factory = collections.namedtuple('Factory', ['factory_class', 'args'])
ReferenceTarget = collections.namedtuple('ReferenceTarget',
                                         ('name', 'transform'))


class ConcreteReferenceTarget(object):
    """Everything you need to make a real Cube for a named reference."""

    def __init__(self, name, transform=None):
        self.name = name
        """The name used to connect references with referencees."""
        self.transform = transform
        """An optional transformation to apply to the cubes."""
        self._src_cubes = iris.cube.CubeList()
        self._final_cube = None

    def add_cube(self, cube):
        self._src_cubes.append(cube)

    def as_cube(self):
        if self._final_cube is None:
            src_cubes = self._src_cubes
            if len(src_cubes) > 1:
                # Merge the reference cubes to allow for
                # time-varying surface pressure in hybrid-presure.
                src_cubes = src_cubes.merge()
                if len(src_cubes) > 1:
                    warnings.warn('Multiple reference cubes for {}'
                                  .format(self.name))
            src_cube = src_cubes[-1]

            if self.transform is None:
                self._final_cube = src_cube
            else:
                final_cube = src_cube.copy()
                attributes = self.transform(final_cube)
                for name, value in attributes.iteritems():
                    setattr(final_cube, name, value)
                self._final_cube = final_cube

        return self._final_cube


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

    def __repr__(self):
        return '<CMAttribute: name="{!s}", value={!r} >'.format(
            self.name, self.value)

    def __deepcopy__(self, memo):
        """
        Accelerate the full copy operation.

        For some reason tuple deepcopy is generally slow.

        """
        return CMAttribute(self.name, copy.deepcopy(self.value, memo))


class CMCustomAttribute(object):
    """
    Used by the rules for defining custom attributes on the Cube in a consistent manner.
    
    """
    __slots__ = ('name', 'value')
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return '<CMCustomAttribute: name="{!s}", value={!r} >'.format(
            self.name, self.value)

    def __deepcopy__(self, memo):
        """
        Accelerate the full copy operation.

        For some reason tuple deepcopy is generally slow.

        """
        return CMCustomAttribute(self.name, copy.deepcopy(self.value, memo))


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

    def __deepcopy__(self, memo):
        """
        Accelerate the full copy operation.

        For some reason tuple deepcopy is generally slow.

        """
        coord = copy.deepcopy(self.coord, memo)
        dims = self.dims[:]
        return CoordAndDims(coord, dims)

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
        return "<CoordAndDims: %r, %r>" % (self.coord.name(), self.dims)


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
    avdiff = np.mean(diffs)
    if not np.allclose(diffs, avdiff, rtol=0.001):  # TODO: This value is set for test_analysis to pass... 
        raise iris.exceptions.CoordinateNotRegularError("Coord %s is not regular" % coord.name())
    return avdiff.astype(coord.points.dtype)


def calculate_forecast_period(time, forecast_reference_time):
    """
    Return the forecast period in hours derived from time and
    forecast_reference_time scalar coordinates.

    """
    if time.points.size != 1:
        raise ValueError('Expected a time coordinate with a single '
                         'point. {!r} has {} points.'.format(time.name(),
                                                             time.points.size))

    if not time.has_bounds():
        raise ValueError('Expected a time coordinate with bounds.')

    if forecast_reference_time.points.size != 1:
        raise ValueError('Expected a forecast_reference_time coordinate '
                         'with a single point. {!r} has {} '
                         'points.'.format(forecast_reference_time.name(),
                                          forecast_reference_time.points.size))

    origin = time.units.origin.replace(time.units.origin.split()[0], 'hours')
    units = iris.unit.Unit(origin, calendar=time.units.calendar)

    # Determine start and eof of period in hours since a common epoch.
    end = time.units.convert(time.bounds[0, 1], units)
    start = forecast_reference_time.units.convert(
        forecast_reference_time.points[0], units)
    forecast_period = end - start

    return forecast_period


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

        # Reset any actions caches
        self.reset_action_caches()

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
        """Place the result of an action into the cube."""
        pass

    def exec_action(self, i, field, cube):
        """Run the code of a rule action."""
        # Define the variables which the eval command should be able to see
        f = field
        pp = field
        grib = field
        cm = cube
        # Execute the actual action code + return the result
        obj = self._exec_actions[i](field, f, pp, grib, cm)
        return obj

    def get_action_result(self, i, field, cube):
        """Get the result of a rule action."""
        # N.B. now overloaded by FunctionRule, to provide result caching.
        return self.exec_action(i, field, cube)

    def reset_action_caches(self):
        # Used by FunctionRules, otherwise no action
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
            globals().update(iris.fileformats.mosig_cf_map.__dict__)
            globals().update(iris.fileformats.um_cf_map.__dict__)
            globals().update(iris.unit.__dict__)
            _import_pending = False
        
        
        factories = []
        for i, action in enumerate(self._actions):
            try:
                # Run this action.
                obj = self.get_action_result(i, field, cube)
                # Process the return value (if any), e.g a CM object or None.
                action_factory = self._process_action_result(obj, cube)
                if action_factory:
                    factories.append(action_factory)

            except iris.exceptions.CoordinateNotFoundError, err:
                print >> sys.stderr, 'Failed (msg:%(error)s) to find coordinate, perhaps consider running last: %(command)s' % {'command':action, 'error': err}
            except AttributeError, err:
                print >> sys.stderr, 'Failed to get value (%(error)s) to execute: %(command)s' % {'command':action, 'error': err}
            except Exception, err:
                print >> sys.stderr, 'Failed (msg:%(error)s) to run:\n    %(command)s\nFrom the rule:\n%(me)r' % {'me':self, 'command':action, 'error': err}
                raise err

        return factories


class _ObjectAccessedWrapper(object):
    def __init__(self, target):
        self.target = target
        self.target_accessed = False

    def __getattr__(self, attname):
        self.target_accessed = True
        return getattr(self.target, attname)


def _value_as_hashable(value):
    """
    Make a basic attribute value hashable.

    Convert 1-d arrays to tuples, otherwise return unchanged.
    (N.B. no multidimensional arrays, for now).

    """
    if not isinstance(value, np.ndarray):
        return value
    # Handle a 1-D array (but nothing more complex).
    if value.ndim > 1:
        raise Exception(
            'pp element Array is > 1d, shape={}'.format(
                value.ndim))
    return tuple(value)


class FunctionRule(Rule):
    """A Rule with values returned by its actions."""
    def _create_action_method(self, i, action):
        # CM loading style action. Returns an object, such as a coord.
        exec compile('def _exec_action_%d(self, field, f, pp, grib, cm): return %s' % (i, action), '<string>', 'exec')
        # Make it a method of ours.
        exec 'self._exec_action_%d = types.MethodType(_exec_action_%d, self, type(self))' % (i, i)
        # Add to our list of actions.
        exec 'self._exec_actions.append(self._exec_action_%d)' % i

    def reset_action_caches(self):
        # Make an empty cache for each action, i.e. self._action_caches[i] = {}
        n_actions = len(self._actions)
        self._action_caches = {i_action: {} for i_action in range(n_actions)}

    def get_action_result(self, i, field, cube):
        # Overloaded form that caches results (aka memoising)
        if not hasattr(field, 'as_access_logging_field'):
            # Field does not provide access monitoring - no caching possible.
            return self.exec_action(i, field, cube)

        # 'Else' use results caching.
        # First, see if we have existing cached action results.
        action_cache = self._action_caches[i]
        action_keynames = action_cache.get('__field_keynames', None)
        if action_keynames is not None:
            # The attributes required by this action have been recorded.
            # Make a lookup key from the relevant field attribute values.
            result_keys = tuple(
                _value_as_hashable(getattr(field, keyname, None))
                for keyname in action_keynames)
            # Return cached result if we have a stored match.
            if result_keys in action_cache:
                result = action_cache[result_keys]
                # Make a deepcopy, to avoid unexpectedly cross-linked objects
                # TODO: **not** doing the deepcopy will potentially be *much*
                # faster, if merging can make that possible ?
                # Plus *another* big potential gain, if merge compare can then
                # use 'is' in place of '=='.
                return copy.deepcopy(result)

        # 'Else' the relevant field attributes are different from any field
        # previously cached :  Run this action + cache the result.

        # Create a wrapper to check whether the action accesses the cube.
        cube_wrapper = _ObjectAccessedWrapper(cube)
        # Run the action code + capture its field and cube accesses.
        field_wrapper = field.as_access_logging_field()
        result = self.exec_action(i, field_wrapper, cube_wrapper)
        if cube_wrapper.target_accessed:
            # Can't cache an action that reads anything from the cube, as cube
            # data does not go in the cache keys.
            # (N.B. at present, no function Rules do this anyway.)
            return result

        # Construct a name:value dictionary from the field attribute fetches.
        field_accesses = field_wrapper.access_log
        element_values = {}
        for attname, value in field_accesses:
            value = _value_as_hashable(value)
            if attname not in element_values:
                element_values[attname] = value
            else:
                # This attribute already seen : all values should be the same.
                if value != element_values[attname]:
                    all_vals = [val for name, val in field_accesses
                                if name == attname]
                    raise Exception('Rule action got multiple values for '
                                    'field.{} : {}'.format(attname, all_vals))

        # Make a sorted list of the field attributes used by the action.
        used_keys = sorted(element_values.keys())
        if action_keynames is None:
            # First occurrence sets the 'expected' caching names list.
            action_cache['__field_keynames'] = used_keys
        else:
            # Subsequent occurences must always match.
            if used_keys != action_keynames:
                print 'ERROR - inconsistent rule args'
                print 'rule: ', self
                print 'action : ', self._actions[i]
                raise Exception('Rule action arguments not consistent.'
                                '\n previously used field attributes : {}'
                                '\n now using : {}'.format(action_keynames,
                                                           used_keys))

        # Make the cache key : a tuple of (values in name order.
        result_keys = tuple(element_values[keyname]
                            for keyname in used_keys)
        # cache this result, and return it
        action_cache[result_keys] = result
        return result

    def _process_action_result(self, obj, cube):
        # (Overrides abstract Rule method)

        factory = None

        # NB. The names such as 'CoordAndDims' and 'CellMethod' are defined by
        # the "deferred import" performed by Rule.run_actions() above.
        if isinstance(obj, CoordAndDims):
            obj.add_coord(cube)

        #cell methods - not yet implemented
        elif isinstance(obj, CellMethod):
            cube.add_cell_method(obj)
            
        elif isinstance(obj, CMAttribute):
            # Temporary code to deal with invalid standard names from the translation table.
            # TODO: when name is "standard_name" force the value to be a real standard name
            if obj.name == 'standard_name' and obj.value is not None:
                cube.rename(obj.value)
            elif obj.name == 'units':
                # Graceful loading of units.
                try:
                    setattr(cube, obj.name, obj.value)
                except ValueError:
                    msg = 'Ignoring PP invalid units {!r}'.format(obj.value)
                    warnings.warn(msg)
                    cube.attributes['invalid_units'] = obj.value
                    cube.units = iris.unit._UNKNOWN_UNIT_STRING
            else:
                setattr(cube, obj.name, obj.value)
            
        elif isinstance(obj, CMCustomAttribute):
            cube.attributes[obj.name] = obj.value

        elif isinstance(obj, Factory):
            factory = obj

        elif isinstance(obj, DebugString):
            print obj

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

    def reset_action_caches(self):
        # Create a fresh cache for all our actions
        for i, rule in enumerate(self._rules):
            rule.reset_action_caches()

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


class _ReferenceError(Exception):
    """Signals an invalid/missing reference field."""
    pass


def _dereference_args(factory, reference_targets, regrid_cache, cube):
    """Converts all the arguments for a factory into concrete coordinates."""
    args = []
    for arg in factory.args:
        if isinstance(arg, Reference):
            if arg.name in reference_targets:
                src = reference_targets[arg.name].as_cube()
                # If necessary, regrid the reference cube to
                # match the grid of this cube.
                src = _ensure_aligned(regrid_cache, src, cube)
                if src is not None:
                    new_coord = iris.coords.AuxCoord(src.data,
                                                     src.standard_name,
                                                     src.long_name,
                                                     src.var_name,
                                                     src.units,
                                                     attributes=src.attributes)
                    dims = [cube.coord_dims(src_coord)[0]
                                for src_coord in src.dim_coords]
                    cube.add_aux_coord(new_coord, dims)
                    args.append(new_coord)
                else:
                    raise _ReferenceError('Unable to regrid reference for'
                                          ' {!r}'.format(arg.name))
            else:
                raise _ReferenceError("The file(s) {{filenames}} don't contain"
                                      " field(s) for {!r}.".format(arg.name))
        else:
            # If it wasn't a Reference, then arg is a dictionary
            # of keyword arguments for cube.coord(...).
            args.append(cube.coord(**arg))
    return args


def _regrid_to_target(src_cube, target_coords, target_cube):
    # Interpolate onto the target grid.
    sample_points = [(coord, coord.points) for coord in target_coords]
    result_cube = iris.analysis.interpolate.linear(src_cube, sample_points)

    # Any scalar coords on the target_cube will have become vector
    # coords on the resample src_cube (i.e. result_cube).
    # These unwanted vector coords need to be pushed back to scalars.
    index = [slice(None, None)] * result_cube.ndim
    for target_coord in target_coords:
        if not target_cube.coord_dims(target_coord):
            result_dim = result_cube.coord_dims(target_coord)[0]
            index[result_dim] = 0
    if not all(key == slice(None, None) for key in index):
        result_cube = result_cube[tuple(index)]
    return result_cube


def _ensure_aligned(regrid_cache, src_cube, target_cube):
    """
    Returns a version of `src_cube` suitable for use as an AuxCoord
    on `target_cube`, or None if no version can be made.

    """
    result_cube = None

    # Check that each of src_cube's dim_coords matches up with a single
    # coord on target_cube.
    try:
        target_coords = []
        for dim_coord in src_cube.dim_coords:
            target_coords.append(target_cube.coord(coord=dim_coord))
    except iris.exceptions.CoordinateNotFoundError:
        # One of the src_cube's dim_coords didn't exist on the
        # target_cube... so we can't regrid (i.e. just return None).
        pass
    else:
        # So we can use `iris.analysis.interpolate.linear()` later,
        # ensure each target coord is either a scalar or maps to a
        # single, distinct dimension.
        target_dims = [target_cube.coord_dims(coord) for coord in target_coords]
        target_dims = filter(None, target_dims)
        unique_dims = set()
        for dims in target_dims:
            unique_dims.update(dims)
        compatible = len(target_dims) == len(unique_dims)

        if compatible:
            cache_key = id(src_cube)
            if cache_key not in regrid_cache:
                regrid_cache[cache_key] = ([src_cube.dim_coords], [src_cube])
            grids, cubes = regrid_cache[cache_key]
            # 'grids' is a list of tuples of coordinates, so convert
            # the 'target_coords' list into a tuple to be consistent.
            target_coords = tuple(target_coords)
            try:
                # Look for this set of target coordinates in the cache.
                i = grids.index(target_coords)
                result_cube = cubes[i]
            except ValueError:
                # Not already cached, so do the hard work of interpolating.
                result_cube = _regrid_to_target(src_cube, target_coords,
                                                target_cube)
                # Add it to the cache.
                grids.append(target_coords)
                cubes.append(result_cube)

    return result_cube


Loader = collections.namedtuple('Loader',
                                ('field_generator', 'field_generator_kwargs',
                                 'load_rules', 'cross_ref_rules',
                                 'log_name'))


def load_cubes(filenames, user_callback, loader):
    concrete_reference_targets = {}
    results_needing_reference = []

    if isinstance(filenames, basestring):
        filenames = [filenames]

    # Initialise rules caching -- specific to each load operation.
    loader.load_rules.reset_action_caches()

    for filename in filenames:
        for field in loader.field_generator(filename, **loader.field_generator_kwargs):
            # Convert the field to a Cube, logging the rules that were used
            rules_result = loader.load_rules.result(field)
            cube = rules_result.cube
            log(loader.log_name, filename, rules_result.matching_rules)

            cube = iris.io.run_callback(user_callback, cube, field, filename)

            if cube is None:
                continue

            # Cross referencing
            rules = loader.cross_ref_rules.matching_rules(field)
            for rule in rules:
                reference, = rule.run_actions(cube, field)
                name = reference.name
                # Register this cube as a source cube for the named
                # reference.
                target = concrete_reference_targets.get(name)
                if target is None:
                    target = ConcreteReferenceTarget(name, reference.transform)
                    concrete_reference_targets[name] = target
                target.add_cube(cube)

            if rules_result.factories:
                results_needing_reference.append(rules_result)
            else:
                yield cube

    regrid_cache = {}
    for result in results_needing_reference:
        cube = result.cube
        for factory in result.factories:
            try:
                args = _dereference_args(factory, concrete_reference_targets,
                                         regrid_cache, cube)
            except _ReferenceError as e:
                msg = 'Unable to create instance of {factory}. ' + e.message
                factory_name = factory.factory_class.__name__
                warnings.warn(msg.format(filenames=filenames,
                                         factory=factory_name))
            else:
                aux_factory = factory.factory_class(*args)
                cube.add_aux_factory(aux_factory)
        yield cube
