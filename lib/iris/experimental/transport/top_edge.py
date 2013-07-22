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
"""
Functionality to support calculating mass transports along top edge T-cells
of constant latitude.

"""

from collections import Iterable
import cPickle
import glob
import math
import multiprocessing
import os
import subprocess
import sys

import numpy as np
import numpy.ma as ma
import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.geometry.polygon import LinearRing

import cartopy.crs as ccrs
import iris


# Segmented color map: binary and red.
_CMAP_TABLE = {'red':   [(0.0, 0.0, 1.0),
                         (0.8, 0.0, 1.0),
                         (1.0, 1.0, 1.0)
                         ],
               'green': [(0.0, 0.0, 1.0),
                         (0.8, 0.0, 0.0),
                         (1.0, 0.0, 0.0)
                         ],
               'blue':  [(0.0, 0.0, 1.0),
                         (0.8, 0.0, 0.0),
                         (1.0, 0.0, 0.0)
                         ]
               }

_CMAP = mpl_colors.LinearSegmentedColormap('_cmap', _CMAP_TABLE, 256)


def _worker(args):
    """
    Performs the actual processing of determining whether one or more lines
    of constant latitude intersect the bounded cell geometry.

    Used to generate a mask, therefore if a line intersects the cell
    geometry, then the result is False, otherwise True.

    Args:

    * args:
        Tuple containing a sequence of one or more constant lines of latitude,
        and the bounded cell :class:`shapely.geometry.polygon.LinearRing`.

    Returns:
        Boolean.

    """
    lines, ring = args
    result = True
    for line in lines:
        if ring.intersects(line):
            result = False
            break
    return result


def _distribute_work(data):
    """
    Distributes processing over all availalble cpu cores to determine
    which T-cells the one or more lines of constant latitude traverse.

    Args:

    * data:
        Pickled tuple containing one or more lines of constant
        latitude, and the :class:`shapely.geometry.polygon.LinearRing`
        for the bounds of each target T-cell.

    Returns:
        A boolean :class:`numpy.ndarray`

    """
    lines, rings = cPickle.loads(data)
    pool = multiprocessing.Pool()
    chunks = rings.size / multiprocessing.cpu_count()
    result = pool.map(_worker,
                      zip([lines] * rings.size, rings), chunks)
    result = np.array(result, dtype=np.bool)
    pool.close()
    pool.join()
    return result


def _remote_server():
    """
    Multi-processing remote host server, that distributes its processing
    over all available cpu cores.

    Process communication via the stdin/stdout file-handles.

    Returns:
        Pickled processing result via the stdout file-handle.

    """
    data = []
    # Receive input via stdin file-handle.
    for line in sys.stdin:
        data.append(line)
    data = ''.join(data)
    # Perform the processing and pickle result.
    result = _distribute_work(data)
    result = cPickle.dumps(result, cPickle.HIGHEST_PROTOCOL)
    # Return output via stdout file-handle.
    sys.stdout.write(result)


def _client(args):
    """
    Local host client that starts a dedicated remote host server
    and delegates work.

    Starts and communicates with the remote host server via a
    secure shell.

    Args:

    * args:
        Tuple containing the remote server hostname and pickled
        data payload to process.

    Returns:
        The processed remote server data payload.

    """
    host, data = args
#    sys.stderr.write('child-{}: starting ...\n'.format(host))
    if host == os.uname()[1]:
        # Distribute work over all the local host cores.
        result = _distribute_work(data)
    else:
        # Delegate work to remote host server.
        call_server = 'import iris.experimental.transport.top_edge as ' \
            'top_edge; top_edge._remote_server()'
        proc = subprocess.Popen(['ssh',
                                 '-o UserKnownHostsFile=/dev/null',
                                 '-o StrictHostKeyChecking=no',
                                 '{}@{}'.format(os.getlogin(), host),
                                 '/usr/local/sci/bin/python2.7',
                                 '-c',
                                 '"{}"'.format(call_server)],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        out, err = proc.communicate(data)
        result = cPickle.loads(out)
#    sys.stderr.write('child-{}: Done!\n'.format(host))
    return result


class TCell(object):
    def __init__(self, cube, hosts=None, cache_dir=None, ping=True):
        """
        Determine the T-cells participating in one or more lines
        of constant latitude.

        Args:

        * cube:
            The T-cell :class:`iris.cube.Cube`.

        Kwargs:

        * hosts:
            A sequence of one or more hostnames designated as
            potential processing resources.

        * cache_dir:
            Directory where intermediate processing results are cached.

        * ping:
            Determine the availability of the hosts before calling
            on them as a processing resource. Defaults to True.

        """
        self.cube = cube
        self._assert_cube()
        if isinstance(hosts, basestring):
            hosts = [hosts]
        self.hosts = self.remote_hosts(hosts, ping)
        if not self.hosts:
            self.hosts = [os.uname()[1]]
        self.cache_dir = cache_dir
        self._hash = self._generate_hash()
        if cache_dir is not None:
            self.cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            else:
                if not os.path.isdir(cache_dir):
                    raise ValueError('Invalid cache directory.')
        self._cache = set()
        self._populate_cache()

    def remote_hosts(self, hosts, ping=True):
        """
        Determine the availability of a farm of remote hosts.

        Always filters out the current hostname.

        Args:

        * hosts:
            A sequence of one or more hostnames designated as
            potential processing resources.

        Kwargs:

        * ping:
            Determine the availability of the hosts before calling
            on them as a processing resource. Defaults to True.

        Returns:
            A list of remote hostnames.

        """
        available = []
        if hosts is None:
            hosts = []
        if not isinstance(hosts, Iterable):
            hosts = [hosts]
        for host in hosts:
            # Don't include the current host.
            if host != os.uname()[1]:
                if ping:
                    try:
                        subprocess.check_output(['ping', '-c1', host])
                        available.append(host)
                    except subprocess.CalledProcessError:
                        # Host invalid or unavailable.
                        pass
                else:
                    available.append(host)
        return available

    def _assert_cube(self):
        """Sanity check the cube."""
        assert len(self.cube.coords('latitude')) == 1
        assert len(self.cube.coords('longitude')) == 1
        lat = self.cube.coord('latitude')
        lon = self.cube.coord('longitude')
        assert lat.shape == lon.shape
        assert self.cube.coord_dims(lat) == self.cube.coord_dims(lon)
        assert lat.has_bounds()
        assert lon.has_bounds()
        assert lat.bounds.shape == lon.bounds.shape
        assert lat.bounds.shape[-1] == 4
        assert lon.bounds.shape[-1] == 4

    def _generate_hash(self):
        """Attempt to generate a unique identifier for the cube."""
        lat = self.cube.coord('latitude').bounds.flatten()
        lon = self.cube.coord('longitude').bounds.flatten()
        _hash = '{}{}'.format(hash(tuple(lon)),
                              hash(tuple(lat)))
        return _hash

    def _populate_cache(self):
        """Populate the cache with pre-existing cache blobs."""
        if self.cache_dir is not None:
            for cache in glob.iglob(os.path.join(self.cache_dir, '*.tcell')):
                self._cache.add(cache)

    def _purge(self):
        """Purge the cache of T-cell intermediate results."""
        for cache in self._cache:
            try:
                os.remove(cache)
            except OSError:
                pass
        self._cache = set()

    @staticmethod
    def purge(cache_dir):
        """Purge the cache directory of cache files."""
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        if os.path.isdir(cache_dir):
            for cache in glob.iglob(os.path.join(cache_dir, '*.tcell')):
                try:
                    os.remove(cache)
                except OSError:
                    pass

    def cache_name(self, latitudes=None, indicies=None):
        """
        Generate a unique cache filename.

        Cache filename generate based on the T-cell :class:`iris.cube.Cube`
        and the constant lines of latitude or i-space indicies.

        Kwargs:

        * latitudes:
            A sequence of one or more constant lines of latitude.

        * indicies:
            A sequence of :class:`numpy.ndarray` indicies.

        Returns:
           Unique cache filename within the nominated cache directory.

        """
        name = None
        if self.cache_dir:
            ext = type(self).__name__.lower()
            if latitudes is None and indicies is None:
                _hash = '{}.{}'.format(self._hash, ext)
            elif latitudes is not None:
                _hash = '{}{}.mask.{}'.format(self._hash,
                                              hash(tuple(sorted(latitudes))),
                                              ext)
            else:
                hash_indicies = hash(tuple(map(tuple, indicies)))
                _hash = '{}{}.rings.{}'.format(self._hash,
                                               hash_indicies,
                                               ext)
            name = os.path.join(self.cache_dir, _hash)
        return name

    def _generate_cell_rings(self, indicies):
        """
        Generate a :class:`shapely.geometry.polygon.LinearRing` for the bounds
        of each target T-cell.

        Args:

        * indicies:
            A sequence containing the two :class:`numpy.ndarray` indicies
            of the target T-cells.

        Returns:
            A flattened :class:`numpy.ndarray` of T-cell geometries.

        """
        rings = None
        # Determine whether a ring cache can be loaded.
        if self.cache_dir is not None:
            cache_name = self.cache_name(indicies=indicies)
            if cache_name in self._cache:
#                sys.stderr.write('loading ring cache ... ')
                try:
                    with open(cache_name, 'r') as fh:
                        _, rings = cPickle.load(fh)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        # Generate the cell rings if no cache is available.
        if rings is None:
            lat = self.cube.coord('latitude')
            lon = self.cube.coord('longitude')
            rings = np.empty(indicies[0].size, dtype=np.object)

#            sys.stderr.write('Generating cell rings ... ')
            for index, (i, j) in enumerate(zip(*indicies)):
                cell = zip(lon.bounds[i, j].flatten(),
                           lat.bounds[i, j].flatten())
                rings[index] = LinearRing(cell)
#            sys.stderr.write('done!\n')

            # Determine whether to cache the rings.
            if self.cache_dir is not None:
#                sys.stderr.write('writing ring cache ... ')
                try:
                    with open(cache_name, 'w') as fh:
                        cPickle.dump((indicies, rings), fh,
                                     cPickle.HIGHEST_PROTOCOL)
                    self._cache.add(cache_name)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        return rings

    def _reduce(self, latitudes):
        """
        Limit the search space to those T-cells approximately around
        the one or more constant lines of latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Returns:
            A tuple pair contaning the final mask shape, and the i-j
            indicies of the target T-cells.

        """
        coord = self.cube.coord('latitude')
        shape = coord.shape

        # Discover the maximum bounded cell tolerance.
        bounds = coord.bounds
        diff = map(np.max, [np.diff(bounds, axis=0), np.diff(bounds, axis=1)])
        tolerance = np.round(np.max(diff), decimals=2)

        # Identify target T-cells.
        indicies = set()
        for latitude in latitudes:
            min_lat = latitude - tolerance
            max_lat = latitude + tolerance
            temp = np.where(bounds >= min_lat, bounds, np.inf)
            temp = np.where(temp <= max_lat, temp, np.inf)
            ii, jj, _ = np.where(temp != np.inf)
            # Determine unique pairs.
            for i, j in zip(ii, jj):
                indicies.add((i, j))

        ii = []
        jj = []
        for (i, j) in indicies:
            ii.append(i)
            jj.append(j)

        indicies = (np.array(ii), np.array(jj))
        return shape, indicies

    def latitude(self, latitudes):
        """
        Determine the T-cells participating in one or more lines of
        constant latitude.

        Args:

        * latitudes:
            One or more target latitudes, in Geodetic decimal degrees.

        Returns:
            A boolean :class:`numpy.ndarray` mask, where False represents
            a T-cell that participates in a target latitude.

        """
        mask = None
        if not isinstance(latitudes, Iterable):
            latitudes = [latitudes]

        # Determine whether the mask cache can be loaded.
        if self.cache_dir is not None:
            cache_name = self.cache_name(latitudes=latitudes)
            if cache_name in self._cache:
#                sys.stderr.write('loading mask cache ... ')
                try:
                    with open(cache_name, 'r') as fh:
                        _, mask = cPickle.load(fh)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        if mask is None:
            # Generate projected lines of constant latitude.
            lines = []
            projection = ccrs.PlateCarree()
            source = ccrs.PlateCarree()
            for latitude in latitudes:
                line = LineString([(-180., latitude), (180., latitude)])
                lines.append(projection.project_geometry(line, source))

            # Optimisation - reduce the bounded T-cell search space.
            final_shape, indicies = self._reduce(latitudes)
            mask = np.ones(final_shape, dtype=np.bool)

            # Generate the geometries for the gridded cells.
            rings = self._generate_cell_rings(indicies)

            payload = []
            step = int(math.ceil(rings.size / float(len(self.hosts))))
            for offset in range(0, rings.size, step):
                data = (lines, rings[offset:offset + step])
                payload.append(cPickle.dumps(data, cPickle.HIGHEST_PROTOCOL))

            if len(self.hosts) == 1 and self.hosts[0] == os.uname()[1]:
                result = _client((self.hosts[0], payload[0]))
            else:
                pool = multiprocessing.Pool()
                chunks = 1
                result = pool.map(_client, zip(self.hosts, payload), chunks)
                pool.close()
                pool.join()
                result = np.concatenate(result)

            ii, jj = indicies
            mask[ii, jj] = result
            del rings

            # Determine whether to cache the mask.
            if self.cache_dir is not None:
#                sys.stderr.write('writing mask cache ... ')
                try:
                    with open(cache_name, 'w') as fh:
                        cPickle.dump((latitudes, mask), fh,
                                     cPickle.HIGHEST_PROTOCOL)
                    self._cache.add(cache_name)
                except OSError:
                    self._purge()
#                sys.stderr.write('done!\n')

        return mask


def top_edge_path(mask):
    """
    Calculates a top-edge path containing one or more sub-paths.

    Args:

    * mask:
        The boolean :class:`numpy.ndarray` of T-cells participating
        in one or more lines of constant latitude, which are set to False.

    Returns:
        A list containing one or more top-edge sub-paths. Each sub-path is a
        list of (row, column) tuple pairs.

    """
    mask = np.logical_not(mask)
    path = []
    sub_path = []
    limit_j = mask.shape[0] - 1
    limit_i = mask.shape[1] - 1
    for i in xrange(mask.shape[1]):
        j_vals = mask[:, i].nonzero()[0]
        if not j_vals.size:
            if sub_path:
                path.append(sub_path)
                sub_path = []
            continue
        # Assume they are in order from nonzero()
        max_j = j_vals[-1]
        min_j = j_vals[0]
        if max_j != min_j + j_vals.size - 1:
            raise ValueError('Cannot handle path bifurcation.')
        max_j += 1
        if not sub_path:
            # Add new sub-path left-hand starting point.
            sub_path.append((max_j, i))
        elif sub_path[-1][0] != max_j:
            # Add left-hand vertical point/s up to new j.
            previous_j = sub_path[-1][0]
            if previous_j < max_j:
                # Step up new j left-hand points.
                previous_j += 1
                for j in range(previous_j, max_j):
                    sub_path.append((j, i))
            else:
                # Step down new j left-hand points.
                previous_j -= 1
                for j in range(previous_j, max_j, -1):
                    sub_path.append((j, i))
            sub_path.append((max_j, i))
        # Append right-hand vertex point.
        sub_path.append((max_j, i + 1))
    # Add last path (if any).
    if sub_path:
        path.append(sub_path)

    return path


def plot_ij(mask, path, **kwargs):
    """
    Convenience function to plot the T-cell mask and path
    in i-j space.

    Args:

    * mask:
        The boolean :class:`numpy.ndarray` of T-cells participating
        in one or more lines of constant latitude, which are set to False.

    * path:
        A list containing one or more top-edge sub-paths. Each sub-path is a
        list of (row, column) tuple pairs.

    """
    shape = mask.shape[-2:]
    nj, ni = shape
    ax = plt.axes()
    ax.set_xlim(0, ni)
    ax.set_ylim(0, nj)
    ax.set_xticks(np.arange(ni + 1))
    ax.set_yticks(np.arange(nj + 1))
    checkers = np.zeros(shape, dtype=np.float32)
    checkers[::2, ::2] = 0.1
    checkers[1::2, 1::2] = 0.1
    result = np.where(mask == 0)
    checkers[result[0], result[1]] = 1.0
    ax.pcolormesh(checkers, cmap=_CMAP)
    for sub_path in path:
        sub_path = np.asarray(sub_path)
        ax.plot(sub_path[:, 1], sub_path[:, 0], **kwargs)


def plot_ll(cube, mask, path, **kwargs):
    """
    Convenience funtion to plot the T-cell mask and path in lat-lon space.

    Args:

    * cube:
        The T-cell :class:`iris.cube.Cube`.

    * mask:
        The boolean :class:`numpy.ndarray` of T-cells participating
        in one or more lines of constant latitude, which are set to False.

    * path:
        A list containing one or more top-edge sub-paths. Each sub-path is a
        list of (row, column) i-j space tuple pairs.

    """
    shape = cube.shape[-2:]
    nj, ni = shape
    ax = plt.axes(aspect='equal')
    checkers = np.zeros(shape, dtype=np.float32)
    checkers[::2, ::2] = 0.1
    checkers[1::2, 1::2] = 0.1
    result = np.where(mask == 0)
    checkers[result[0], result[1]] = 1.0
    lat_blhc = cube.coord('latitude').bounds[..., 0]
    lon_blhc = cube.coord('longitude').bounds[..., 0]
    ax.pcolormesh(lon_blhc, lat_blhc, checkers, cmap=_CMAP)
    for sub_path in path:
        lats = []
        lons = []
        for i, (row, col) in enumerate(sub_path):
            try:
                lat = lat_blhc[row, col]
                lon = lon_blhc[row, col]
                if len(lons) and abs(lon - lons[-1]) > 90:
                    ax.plot(lons, lats, **kwargs)
                    lats = [lat]
                    lons = [lon]
                else:
                    lats.append(lat)
                    lons.append(lon)
            except IndexError:
                if len(lats):
                    ax.plot(lons, lats, **kwargs)
                    lats = []
                    lons = []
        if len(lats):
            ax.plot(lons, lats, **kwargs)


def generate_path(cube, latitudes, path_func=None, hosts=None, cache_dir=None):
    """
    Calculate the top-edge path for the one or more constant lines of latitude.

    Args:

    * cube:
        The T-cell :class:`iris.cube.Cube`.

    * latitudes:
        One or more target latitudes, in Geodetic decimal degrees.

    Kwargs:

    * path_func:
        The function responsible for generating the path list
        from a :class:`TCell` mask. Defaults to :func:`top_edge_path`.

    * hosts:
        A sequence of one or more hostnames designated as
        potential remote processing resources.

    * cache_dir:
        Directory where intermediate processing results are cached.

    Returns:
        A list containing one or more top-edge sub-paths. Each sub-path is a
        list of (row, column) tuple pairs.

    """
    tcell = TCell(cube, hosts=hosts, cache_dir=cache_dir)
    mask = tcell.latitude(latitudes)
    if path_func is None:
        path_func = top_edge_path
    return path_func(mask)
