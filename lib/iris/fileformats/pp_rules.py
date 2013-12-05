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

# DO NOT EDIT DIRECTLY
# Auto-generated from SciTools/iris-code-generators:tools/gen_rules.py

import warnings

import numpy as np

from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.rules import Factory, Reference, ReferenceTarget
from iris.fileformats.um_cf_map import LBFC_TO_CF, STASH_TO_CF
from iris.unit import Unit
import iris.fileformats.pp
import iris.unit


def convert(f):
    factories = []
    references = []
    standard_name = None
    long_name = None
    units = None
    attributes = {}
    cell_methods = []
    dim_coords_and_dims = []
    aux_coords_and_dims = []

    if \
            (f.lbtim.ia == 0) and \
            (f.lbtim.ib == 0) and \
            (f.lbtim.ic in [1, 2, 3]) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        aux_coords_and_dims.append((DimCoord(f.time_unit('hours').date2num(f.t1), standard_name='time', units=f.time_unit('hours')), None))

    if \
            (f.lbtim.ia == 0) and \
            (f.lbtim.ib == 1) and \
            (f.lbtim.ic in [1, 2, 3]) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        aux_coords_and_dims.append((DimCoord(f.time_unit('hours', f.t2).date2num(f.t1), standard_name='forecast_period', units='hours'), None))
        aux_coords_and_dims.append((DimCoord(f.time_unit('hours').date2num(f.t1), standard_name='time', units=f.time_unit('hours')), None))
        aux_coords_and_dims.append((DimCoord(f.time_unit('hours').date2num(f.t2), standard_name='forecast_reference_time', units=f.time_unit('hours')), None))

    if \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        aux_coords_and_dims.append((DimCoord(f.lbft, standard_name='forecast_period', units='hours'), None))
        aux_coords_and_dims.append((DimCoord((f.time_unit('hours').date2num(f.t1) + f.time_unit('hours').date2num(f.t2)) / 2.0, standard_name='time', units=f.time_unit('hours'), bounds=f.time_unit('hours').date2num([f.t1, f.t2])), None))
        aux_coords_and_dims.append((DimCoord(f.time_unit('hours').date2num(f.t2) - f.lbft, standard_name='forecast_reference_time', units=f.time_unit('hours')), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        aux_coords_and_dims.append((DimCoord(f.lbft, standard_name='forecast_period', units='hours'), None))
        aux_coords_and_dims.append((DimCoord((f.time_unit('hours').date2num(f.t1) + f.time_unit('hours').date2num(f.t2)) / 2.0, standard_name='time', units=f.time_unit('hours'), bounds=f.time_unit('hours').date2num([f.t1, f.t2])), None))
        aux_coords_and_dims.append((DimCoord(f.time_unit('hours').date2num(f.t2) - f.lbft, standard_name='forecast_reference_time', units=f.time_unit('hours')), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 12 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 3 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('djf', long_name='season', units='no_unit'), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 3 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 6 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('mam', long_name='season', units='no_unit'), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 6 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 9 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('jja', long_name='season', units='no_unit'), None))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 9 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 12 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        aux_coords_and_dims.append((AuxCoord('son', long_name='season', units='no_unit'), None))

    if \
            (f.bdx != 0.0) and \
            (f.bdx != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 1):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, standard_name=f._x_coord_name(), units='degrees', circular=(f.lbhem in [0, 4]), coord_system=f.coord_system()), 1))

    if \
            (f.bdx != 0.0) and \
            (f.bdx != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 2):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, standard_name=f._x_coord_name(), units='degrees', circular=(f.lbhem in [0, 4]), coord_system=f.coord_system(), with_bounds=True), 1))

    if \
            (f.bdy != 0.0) and \
            (f.bdy != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 1):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzy, f.bdy, f.lbrow, standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system()), 0))

    if \
            (f.bdy != 0.0) and \
            (f.bdy != f.bmdi) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 2):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzy, f.bdy, f.lbrow, standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system(), with_bounds=True), 0))

    if \
            (f.bdy == 0.0 or f.bdy == f.bmdi) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.iy == 10)):
        dim_coords_and_dims.append((DimCoord(f.y, standard_name=f._y_coord_name(), units='degrees', bounds=f.y_bounds, coord_system=f.coord_system()), 0))

    if \
            (f.bdx == 0.0 or f.bdx == f.bmdi) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix == 11)):
        dim_coords_and_dims.append((DimCoord(f.x, standard_name=f._x_coord_name(),  units='degrees', bounds=f.x_bounds, circular=(f.lbhem in [0, 4]), coord_system=f.coord_system()), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.iy == 4):
        dim_coords_and_dims.append((DimCoord(f.y, standard_name='depth', units='m', bounds=f.y_bounds, attributes={'positive': 'down'}), 0))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.ix == 10) and \
            (f.bdx != 0) and \
            (f.bdx != f.bmdi):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system()), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.iy == 1) and \
            (f.bdy == 0 or f.bdy == f.bmdi):
        dim_coords_and_dims.append((DimCoord(f.y, long_name='pressure', units='hPa', bounds=f.y_bounds), 0))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.ix == 1) and \
            (f.bdx == 0 or f.bdx == f.bmdi):
        dim_coords_and_dims.append((DimCoord(f.x, long_name='pressure', units='hPa', bounds=f.x_bounds), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.iy == 23):
        dim_coords_and_dims.append((DimCoord(f.y, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.y_bounds), 0))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.ix == 23):
        dim_coords_and_dims.append((DimCoord(f.x, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.x_bounds), 1))

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.ix == 13) and \
            (f.bdx != 0):
        dim_coords_and_dims.append((DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt, long_name='site_number', units='1'), 1))

    if \
            (len(f.lbcode) == 5) and \
            (13 in [f.lbcode.ix, f.lbcode.iy]) and \
            (11 not in [f.lbcode.ix, f.lbcode.iy]) and \
            (hasattr(f, 'lower_x_domain')) and \
            (hasattr(f, 'upper_x_domain')) and \
            (all(f.lower_x_domain != -1.e+30)) and \
            (all(f.upper_x_domain != -1.e+30)):
        aux_coords_and_dims.append((AuxCoord((f.lower_x_domain + f.upper_x_domain) / 2.0, standard_name=f._x_coord_name(), units='degrees', bounds=np.array([f.lower_x_domain, f.upper_x_domain]).T, coord_system=f.coord_system()), 1 if f.lbcode.ix == 13 else 0))

    if \
            (len(f.lbcode) == 5) and \
            (13 in [f.lbcode.ix, f.lbcode.iy]) and \
            (10 not in [f.lbcode.ix, f.lbcode.iy]) and \
            (hasattr(f, 'lower_y_domain')) and \
            (hasattr(f, 'upper_y_domain')) and \
            (all(f.lower_y_domain != -1.e+30)) and \
            (all(f.upper_y_domain != -1.e+30)):
        aux_coords_and_dims.append((AuxCoord((f.lower_y_domain + f.upper_y_domain) / 2.0, standard_name=f._y_coord_name(), units='degrees', bounds=np.array([f.lower_y_domain, f.upper_y_domain]).T, coord_system=f.coord_system()), 1 if f.lbcode.ix == 13 else 0))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cell_methods.append(CellMethod("mean", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 3):
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib not in [2, 3]):
        cell_methods.append(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cell_methods.append(CellMethod("minimum", coords="time"))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cell_methods.append(CellMethod("minimum", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib != 2):
        cell_methods.append(CellMethod("minimum", coords="time"))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cell_methods.append(CellMethod("maximum", coords="time"))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cell_methods.append(CellMethod("maximum", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib != 2):
        cell_methods.append(CellMethod("maximum", coords="time"))

    if f.lbproc not in [0, 128, 4096, 8192]:
        attributes["ukmo__process_flags"] = tuple(sorted([iris.fileformats.pp.lbproc_map[flag] for flag in f.lbproc.flags]))

    if \
            (f.lbvc == 1) and \
            (not (str(f.stash) in ['m01s03i236', 'm01s03i237', 'm01s03i245', 'm01s03i247', 'm01s03i250'])) and \
            (f.blev != -1):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='height', units='m', attributes={'positive': 'up'}), None))

    if str(f.stash) in ['m01s03i236', 'm01s03i237', 'm01s03i245', 'm01s03i247', 'm01s03i250']:
        aux_coords_and_dims.append((DimCoord(1.5, standard_name='height', units='m', attributes={'positive': 'up'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2):
        aux_coords_and_dims.append((DimCoord(f.lblev, standard_name='model_level_number', attributes={'positive': 'down'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2) and \
            (f.brsvd[0] == f.brlev):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='depth', units='m', attributes={'positive': 'down'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2) and \
            (f.brsvd[0] != f.brlev):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='depth', units='m', bounds=[f.brsvd[0], f.brlev], attributes={'positive': 'down'}), None))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 6):
        aux_coords_and_dims.append((DimCoord(f.blev, standard_name='model_level_number', attributes={'positive': 'down'}), None))

    if \
            (f.lbvc == 8) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and 1 not in [f.lbcode.ix, f.lbcode.iy])):
        aux_coords_and_dims.append((DimCoord(f.blev, long_name='pressure', units='hPa'), None))

    if f.lbvc == 65:
        aux_coords_and_dims.append((DimCoord(f.lblev, standard_name='model_level_number', attributes={'positive': 'up'}), None))
        aux_coords_and_dims.append((DimCoord(f.blev, long_name='level_height', units='m', bounds=[f.brlev, f.brsvd[0]], attributes={'positive': 'up'}), None))
        aux_coords_and_dims.append((AuxCoord(f.bhlev, long_name='sigma', bounds=[f.bhrlev, f.brsvd[1]]), None))
        factories.append(Factory(HybridHeightFactory, [{'long_name': 'level_height'}, {'long_name': 'sigma'}, Reference('orography')]))

    if f.lbrsvd[3] != 0:
        aux_coords_and_dims.append((DimCoord(f.lbrsvd[3], standard_name='realization'), None))

    if f.lbuser[4] != 0:
        aux_coords_and_dims.append((DimCoord(f.lbuser[4], long_name='pseudo_level', units='1'), None))

    if f.lbuser[6] == 1 and f.lbuser[3] == 5226:
        standard_name = "precipitation_amount"
        units = "kg m-2"

    if \
            (f.lbuser[6] == 2) and \
            (f.lbuser[3] == 101):
        standard_name = "sea_water_potential_temperature"
        units = "Celsius"

    if \
            ((f.lbsrce % 10000) == 1111) and \
            ((f.lbsrce / 10000) / 100.0 > 0):
        attributes['source'] = 'Data from Met Office Unified Model %4.2f' % ((f.lbsrce / 10000) / 100.0)

    if \
            ((f.lbsrce % 10000) == 1111) and \
            ((f.lbsrce / 10000) / 100.0 == 0):
        attributes['source'] = 'Data from Met Office Unified Model'

    if f.lbuser[6] != 0 or (f.lbuser[3] / 1000) != 0 or (f.lbuser[3] % 1000) != 0:
        attributes['STASH'] = f.stash

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 4205):
        standard_name = "mass_fraction_of_cloud_ice_in_air"
        units = "1"

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 4206):
        standard_name = "mass_fraction_of_cloud_liquid_water_in_air"
        units = "1"

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 30204):
        standard_name = "air_temperature"
        units = "K"

    if \
            (f.lbuser[6] == 4) and \
            (f.lbuser[3] == 6001):
        standard_name = "sea_surface_wave_significant_height"
        units = "m"

    if str(f.stash) in STASH_TO_CF:
        standard_name = STASH_TO_CF[str(f.stash)].standard_name
        units = STASH_TO_CF[str(f.stash)].units
        long_name = STASH_TO_CF[str(f.stash)].long_name

    if \
            (not f.stash.is_valid) and \
            (f.lbfc in LBFC_TO_CF):
        standard_name = LBFC_TO_CF[f.lbfc].standard_name
        units = LBFC_TO_CF[f.lbfc].units
        long_name = LBFC_TO_CF[f.lbfc].long_name

    if f.lbuser[3] == 33:
        references.append(ReferenceTarget('orography', None))

    return (factories, references, standard_name, long_name, units, attributes,
            cell_methods, dim_coords_and_dims, aux_coords_and_dims)
