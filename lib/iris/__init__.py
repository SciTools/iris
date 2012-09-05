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
A package for handling multi-dimensional data and associated metadata.

.. note ::
    The Iris documentation has further usage information, including
    a :ref:`user guide <user_guide_index>` which should be the first port of
    call for new users.

"""
import logging
import os
import warnings
import itertools

import iris.config
import iris.cube
import iris._constraints
import iris.fileformats
import iris.io


# Iris revision.
__version__ = '0.9-dev'

# Restrict the names imported when using "from iris import *"
__all__ = ['load', 'load_strict', 'save', 'Constraint', 'AttributeConstraint']


# When required, log the usage of Iris.
if iris.config.IMPORT_LOGGER:
    logging.getLogger(iris.config.IMPORT_LOGGER).info('iris %s' % __version__)


Constraint = iris._constraints.Constraint
AttributeConstraint = iris._constraints.AttributeConstraint


def _load_common(uris, constraints, strict=False, unique=False, callback=None, merge=True):
    """
    Provides a common interface for both load & load_strict.
    
    Args:
    
    * uris:
        An iterable of URIs to load
    * constraints:
        The constraints to pass through to :meth:`iris.cube.CubeList.extract`. May be None.
        
    Kwargs:
    
    * strict:
        Passed through to :meth:`iris.cube.CubeList.extract`.
    * unique:
        Passed through to :meth:`iris.cube.CubeList.merge` (if merge=True)
    *callback:
        A function following a specification defined in load & load_strict's documentation
    * merge:
        Whether or not to merge the resulting cubes.
        
    Returns - :class:`iris.cube.CubeList`

    """
    if isinstance(constraints, basestring) and os.path.exists(constraints):
        msg = 'The second argument %r appears to be a filename, but expected a standard_name or a Constraint.\n' \
            'If the constraint was genuine, then we recommend renaming %r.'
        msg = msg % (constraints, constraints)
        warnings.warn(msg, UserWarning, stacklevel=3)
    
    cubes = _load_cubes(uris, callback)

    if merge:
        merge_unique = bool(unique)
    else:
        merge_unique = None

    return iris.cube.CubeList._extract_and_merge(cubes, constraints, strict, merge_unique)


def _load_cubes(uris, callback):
    """Returns a generator of cubes given the URIs and a callback."""
    if isinstance(uris, basestring):
        uris = [uris] 
    
    # Group collections of uris by their iris handler
    # Create list of tuples relating schemes to part names
    uri_tuples = sorted(iris.io.decode_uri(uri) for uri in uris)
    
    for scheme, groups in (itertools.groupby(uri_tuples, key=lambda x: x[0])):
        part_names = [x[1] for x in groups]
    
        # Call each scheme handler with the approriate uris
        if scheme == 'file':
            for cube in iris.io.load_files(part_names, callback):
                yield cube
        else:
            raise ValueError('Iris cannot handle the URI scheme: %s' % scheme)

                      
def load_strict(uris, constraints=None, callback=None):
    """
    Loads fixed numbers of cubes.

    Loads from the given filename(s)/URI(s) to form one Cube for each constraint, or just
    one Cube if no constraints are specified.
    
    Constraints can be specified as standard names or with the :class:`iris.Constraint` class::
    
        # Load temperature data.
        load(uri, 'temperature')
        
    Format-specific translation behaviour can be modified by using:
        :func:`iris.fileformats.pp.add_load_rules`

        :func:`iris.fileformats.grib.add_load_rules`

    A callback facility is provided to allow metadata to be modified/augmented.
    
    For example::
    
        def callback(cube, field, filename):
            # Extract ID from filenames given as: <prefix>__<exp_id>
            experiment_id = filename.split('__')[1]
            experiment_coord = iris.coords.ExplicitCoord('experiment_id', '', points=[experiment_id])
            cube.add_coord(experiment_coord)

    Args:

    * uris:
        Either a single filename/URI expressed as a string, or an iterable of filename/URIs.

    Kwargs:

    * constraints:
        An iterable of constraints.
    * callback:
        A function to add metadata from the originating field and/or URI which obeys the following rules:
            1. Function signature must be: ``(cube, field, filename)``
            2. Must not return any value - any alterations to the cube must be made by reference
            3. If the cube is to be rejected the callback must raise an :class:`iris.exceptions.IgnoreCubeException`

    Returns:
        An :class:`iris.cube.CubeList` if multiple constraints were
        supplied, or a single :class:`iris.cube.Cube` otherwise.

    """
    # Load exactly one merged Cube from each constraint.
    cubes = _load_common(uris, constraints, strict=True, unique=True, callback=callback)
    return cubes


def load(uris, constraints=None, unique=False, callback=None, merge=True):
    """
    Loads cubes.
    
    Loads from the given filename(s)/URI(s) to form one or more Cubes for each constraint,
    or one or more Cubes if no constraints are specified.

    Constraints can be specified as standard names or with the :class:`iris.Constraint` class::
    
        # Load temperature data.
        load(uri, 'temperature')
            
    Format-specific translation behaviour can be modified by using:
        :func:`iris.fileformats.pp.add_load_rules`

        :func:`iris.fileformats.grib.add_load_rules`
    
    A callback facility is provided to allow metadata to be modified/augmented.
    
    For example::
    
        def callback(cube, field, filename):
            # Extract ID from filenames given as: <prefix>__<exp_id>
            experiment_id = filename.split('__')[1]
            experiment_coord = iris.coords.ExplicitCoord('experiment_id', '', points=[experiment_id])
            cube.add_coord(experiment_coord)
            
    Args:

    * uris:
        Either a single filename/URI expressed as a string, or an iterable of filenames/URIs.
    
    Kwargs:

    * constraints:
        An iterable of constraints.
    * unique:
        If set to True, raise an error if duplicate cubes are detected (ignored if merge is False)
    * callback:
         A function to add metadata from the originating field and/or URI which obeys the following rules:
            1. Function signature must be: ``(cube, field, filename)``
            2. Must not return any value - any alterations to the cube must be made by reference
            3. If the cube is to be rejected the callback must raise an :class:`iris.exceptions.IgnoreCubeException`
    * merge:
        If set to False, make no attempt to merge the resulting cubes. Note that if True (default) and 
        an iterable of constraints is provided, a separate merge is performed for each constraint.

    Returns:
        An :class:`iris.cube.CubeList`.

    """    
    # Load zero or more Cubes from each constraint.
    cubes = _load_common(uris, constraints, strict=False, unique=unique, callback=callback, merge=merge)
    return cubes


save = iris.io.save

def sample_data_path(*path_to_join):
    """Given the sample data resource, returns the full path to the file."""
    return os.path.join(iris.config.SAMPLE_DATA_DIR, *path_to_join)
