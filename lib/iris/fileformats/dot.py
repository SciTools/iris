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
Provides Creation and saving of DOT graphs for a :class:`iris.cube.Cube`.

"""

import os
import subprocess

import iris
import iris.util


_GRAPH_INDENT = ' ' * 4
_SUBGRAPH_INDENT = ' ' * 8


def save(cube, target):
    """Save a dot representation of the cube.
    
    Args:

        * cube   - A :class:`iris.cube.Cube`.
        * target - A filename or open file handle.

    See also :func:`iris.io.save`.
    
    """
    if isinstance(target, basestring):
        dot_file = open(target, "wt")
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" in target.mode:
            raise ValueError("Target is binary")
        dot_file = target
    else:
        raise ValueError("Can only save dot to filename or filehandle")
    
    dot_file.write(cube_text(cube))

    if isinstance(target, basestring):
        dot_file.close()
        

def save_png(source, target, launch=False):
    """
    Produces a "dot" instance diagram by calling dot and optionally launching the resulting image.

    Args:

        * source - A :class:`iris.cube.Cube`, or dot filename.
        * target - A filename or open file handle.
                   If passing a file handle, take care to open it for binary output.

    Kwargs:

        * launch - Display the image. Default is False.

    See also :func:`iris.io.save`.

    """
    # From cube or dot file?
    if isinstance(source, iris.cube.Cube):
        # Create dot file
        dot_file_path = iris.util.create_temp_filename(".dot")
        save(source, dot_file_path)
    elif isinstance(source, basestring):
        dot_file_path = source
    else:
        raise ValueError("Can only write dot png for a Cube or DOT file")
        
    # Create png data
    dot_exe = iris.config.get_option('System', 'dot_path', default='dot')
    # To filename or open file handle?
    if isinstance(target, basestring):
        subprocess.call([dot_exe, '-T', 'png', '-o', target, dot_file_path])
    elif hasattr(target, "write"):
        if hasattr(target, "mode") and "b" not in target.mode:
            raise ValueError("Target not binary")
        subprocess.call([dot_exe, '-T', 'png', dot_file_path], stdout=target)
    else:
        raise ValueError("Can only write dot png for a filename or writable")

    # Display?
    if launch:
        if os.name == 'mac':
            subprocess.call(('open', target))
        elif os.name == 'nt':
            subprocess.call(('start', target))
        elif os.name == 'posix':
            subprocess.call(('firefox', target))
        else:
            raise iris.exceptions.NotYetImplementedError('Unhandled operating system. The image has been created in %s' % target)

    # Remove the dot file if we created it
    if isinstance(source, iris.cube.Cube):
        os.remove(dot_file_path)
        

def cube_text(cube):
    """Return a DOT text representation a `iris.cube.Cube`.
    
    Args:
    
     * cube  -  The cube for which to create DOT text.
    
    """
    # We use r'' type string constructor as when we type \n in a string without the r'' constructor
    # we get back a new line character - this is not what DOT expects.
    # Therefore, newline characters should be created explicitly by having multi-lined strings.
    relationships = r''
    relationships_association = r''

    dimension_nodes = r'''
    subgraph clusterCubeDimensions {
        label="Cube data"
    '''

    # TODO: Separate dim_coords from aux_coords. 
    coord_nodes = r'''
    subgraph clusterCoords {
        label = "Coords"
'''

    coord_system_nodes = r'''
    subgraph clusterCoordSystems {
        label = "CoordSystems"
'''

    for i, size in enumerate(cube.shape):
        dimension_nodes += '\n' + _dot_node(_SUBGRAPH_INDENT, 'CubeDimension_' + str(i), str(i), [('len', size)])

    # Coords and their coord_systems
    coords = sorted(cube.coords(), key=lambda c: c.name())
    written_cs = []
    for i, coord in enumerate(coords):
        coord_label = 'Coord_' + str(i)
        coord_nodes += _coord_text(coord_label, coord)
        cs = coord.coord_system
        if cs:
            # Create the cs node - or find an identical, already written cs
            if cs not in written_cs:
                written_cs.append(cs)
                uid = written_cs.index(cs)
                coord_system_nodes += _coord_system_text(cs, uid)
            else:
                uid = written_cs.index(cs)
            relationships += '\n    "%s" -> "CoordSystem_%s_%s"' % (coord_label, coord.coord_system.__class__.__name__, uid)
        relationships += '\n    ":Cube" -> "%s"' % coord_label

        # Are there any relationships to data dimensions?
        dims = cube.coord_dims(coord=coord)
        for dim in dims:
            relationships_association += '\n    "%s" -> "CubeDimension_%s":w' % (coord_label, dim)


    dimension_nodes += '''
    }
    '''

    coord_nodes += '''
    }
    '''

    coord_system_nodes += '''
    }
    '''

    # return a string pulling everything together
    template = '''
digraph CubeGraph{

    rankdir = "LR"
    fontname = "Bitstream Vera Sans"
    fontsize = 8

    node [
        fontname = "Bitstream Vera Sans"
        fontsize = 8
        shape = "record"
    ]

#   Nodes
%(cube_node)s
    %(dimension_nodes)s
    %(coord_nodes)s
    %(coord_sys_nodes)s
    edge [
        arrowhead = "normal"
    ]

#   RELATIONSHIPS

#   Containment
    %(relationships)s
    edge [
        style="dashed"
        arrowhead = "onormal"
    ]

#   Association
    %(associations)s
}
    '''
    cube_attributes = [(name, value) for name, value in sorted(cube.attributes.iteritems(), key=lambda item: item[0])]
    cube_node = _dot_node(_GRAPH_INDENT, ':Cube', 'Cube', cube_attributes)
    res_string = template % {
                        'cube_node': cube_node,
                        'dimension_nodes': dimension_nodes,
                        'coord_nodes': coord_nodes,
                        'coord_sys_nodes': coord_system_nodes,
                        'relationships': relationships,
                        'associations': relationships_association
                    }
    return res_string


def _coord_text(label, coord):
    """
    Returns a string containing the dot representation for a single coordinate node.
     
    Args:
     
    * label
        The dot ID of the coordinate node.
    * coord
        The coordinate to convert.
     
    """
    # Which bits to write?
    # Note: This is is not very OO but we are achieving a separation of DOT from cdm by doing this.
    if isinstance(coord, iris.coords.DimCoord):
        _dot_attrs = ('standard_name', 'long_name', 'units', 'circular')
    elif isinstance(coord, iris.coords.AuxCoord):
        _dot_attrs = ('standard_name', 'long_name', 'units')
    else:
        raise ValueError("Unhandled coordinate type: " + str(type(coord)))
    attrs = [(name, getattr(coord, name)) for name in _dot_attrs]

    if coord.attributes:
        custom_attrs = sorted(coord.attributes.iteritems(), key=lambda item: item[0])
        attrs.extend(custom_attrs)

    node = _dot_node(_SUBGRAPH_INDENT, label, coord.__class__.__name__, attrs)
    return node


def _coord_system_text(cs, uid):
    """
    Returns a string containing the dot representation for a single coordinate system node.
     
    Args:
     
    * cs
        The coordinate system to convert.
    * uid
        The uid allows/distinguishes non-identical CoordSystems of the same type.
     
    """
    attrs = []
    for k, v in cs.__dict__.iteritems():
        if isinstance(v, iris.cube.Cube):
            attrs.append((k, 'defined'))
        else:
            attrs.append((k, v))
            
    attrs.sort(key=lambda attr: attr[0])

    label = "CoordSystem_%s_%s" % (cs.__class__.__name__, uid)
    node = _dot_node(_SUBGRAPH_INDENT, label, cs.__class__.__name__,  attrs)
    return node


def _dot_node(indent, id, name, attributes):
    """
    Returns a string containing the dot representation for a single node.
     
    Args:
     
     * id
        The ID of the node.
     * name
        The visual name of the node.
     * attributes
        An iterable of (name, value) attribute pairs.
     
    """
    attributes = r'\n'.join('%s: %s' % item for item in attributes)
    template = """%(indent)s"%(id)s" [
%(indent)s    label = "%(name)s|%(attributes)s"
%(indent)s]
"""
    node = template % {'id': id, 'name': name, 'attributes': attributes, 'indent': indent}
    return node
