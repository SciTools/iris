# (C) British Crown Copyright 2013 - 2014, Met Office
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

# Historically this was auto-generated from
# SciTools/iris-code-generators:tools/gen_rules.py

import warnings

import numpy as np

from iris.aux_factory import HybridHeightFactory, HybridPressureFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.rules import Factory, Reference, ReferenceTarget
from iris.fileformats.um_cf_map import LBFC_TO_CF, STASH_TO_CF
from iris.unit import Unit
import iris.fileformats.pp
import iris.unit


def _model_level_number(lblev):
    """
    Return model level number for an LBLEV value.

    Args:

    * lblev (int):
        PP field LBLEV value.

    Returns:
        Model level number (integer).

    """
    # See Word no. 33 (LBLEV) in section 4 of UM Model Docs (F3).
    SURFACE_AND_ZEROTH_RHO_LEVEL_LBLEV = 9999

    if lblev == SURFACE_AND_ZEROTH_RHO_LEVEL_LBLEV:
        model_level_number = 0
    else:
        model_level_number = lblev

    return model_level_number


def _convert_scalar_time_coords(lbcode, lbtim, epoch_hours_unit, t1, t2, lbft):
    """
    Encode scalar time values from PP headers as CM data components.

    Returns a list of coords_and_dims.

    """
    t1_epoch_hours = epoch_hours_unit.date2num(t1)
    t2_epoch_hours = epoch_hours_unit.date2num(t2)
    hours_from_t1_to_t2 = t2_epoch_hours - t1_epoch_hours
    hours_from_t2_to_t1 = t1_epoch_hours - t2_epoch_hours
    coords_and_dims = []

    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 0) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((DimCoord(t1_epoch_hours, standard_name='time', units=epoch_hours_unit), None))

    if \
            (lbtim.ia == 0) and \
            (lbtim.ib == 1) and \
            (lbtim.ic in [1, 2, 3, 4]) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((DimCoord(hours_from_t2_to_t1, standard_name='forecast_period', units='hours'), None))
        coords_and_dims.append((DimCoord(t1_epoch_hours, standard_name='time', units=epoch_hours_unit), None))
        coords_and_dims.append((DimCoord(t2_epoch_hours, standard_name='forecast_reference_time', units=epoch_hours_unit), None))

    if \
            (lbtim.ib == 2) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((
            DimCoord(standard_name='forecast_period', units='hours',
                     points=lbft - 0.5 * hours_from_t1_to_t2,
                     bounds=[lbft - hours_from_t1_to_t2, lbft]),
            None))
        coords_and_dims.append((
            DimCoord(standard_name='time', units=epoch_hours_unit,
                     points=0.5 * (t1_epoch_hours + t2_epoch_hours),
                     bounds=[t1_epoch_hours, t2_epoch_hours]),
            None))
        coords_and_dims.append((DimCoord(t2_epoch_hours - lbft, standard_name='forecast_reference_time', units=epoch_hours_unit), None))

    if \
            (lbtim.ib == 3) and \
            (lbtim.ic in [1, 2, 4]) and \
            ((len(lbcode) != 5) or (len(lbcode) == 5 and lbcode.ix not in [20, 21, 22, 23] and lbcode.iy not in [20, 21, 22, 23])):
        coords_and_dims.append((
            DimCoord(standard_name='forecast_period', units='hours',
                     points=lbft, bounds=[lbft - hours_from_t1_to_t2, lbft]),
            None))
        coords_and_dims.append((
            DimCoord(standard_name='time', units=epoch_hours_unit,
                     points=t2_epoch_hours, bounds=[t1_epoch_hours, t2_epoch_hours]),
            None))
        coords_and_dims.append((DimCoord(t2_epoch_hours - lbft, standard_name='forecast_reference_time', units=epoch_hours_unit), None))

    return coords_and_dims


_stashcode_implied_heights = {
    'm01s03i236': 1.5,
    'm01s03i237': 1.5,
    'm01s03i245': 1.5,
    'm01s03i247': 1.5,
    'm01s03i250': 1.5,
    'm01s03i225': 10.0,
    'm01s03i226': 10.0,
    'm01s03i463': 10.0}


def _convert_scalar_vertical_coords(lbcode, lbvc, blev, lblev, stash,
                                    bhlev, bhrlev, brsvd1, brsvd2, brlev):
    """
    Encode scalar vertical level values from PP headers as CM data components.

    Returns (<list of coords_and_dims>, <list of factories>)

    """
    factories = []
    coords_and_dims = []
    model_level_number = _model_level_number(lblev)

    if \
            (lbvc == 1) and \
            (not (str(stash) in _stashcode_implied_heights.keys())) and \
            (blev != -1):
        coords_and_dims.append(
            (DimCoord(blev, standard_name='height', units='m',
                      attributes={'positive': 'up'}),
             None))

    if str(stash) in _stashcode_implied_heights.keys():
        coords_and_dims.append(
            (DimCoord(_stashcode_implied_heights[str(stash)],
                      standard_name='height', units='m',
                      attributes={'positive': 'up'}),
             None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 2):
        coords_and_dims.append((DimCoord(model_level_number, standard_name='model_level_number', attributes={'positive': 'down'}), None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 2) and \
            (brsvd1 == brlev):
        coords_and_dims.append((DimCoord(blev, standard_name='depth', units='m', attributes={'positive': 'down'}), None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 2) and \
            (brsvd1 != brlev):
        coords_and_dims.append((DimCoord(blev, standard_name='depth', units='m', bounds=[brsvd1, brlev], attributes={'positive': 'down'}), None))

    # soil level
    if len(lbcode) != 5 and lbvc == 6:
        coords_and_dims.append((DimCoord(model_level_number, long_name='soil_model_level_number', attributes={'positive': 'down'}), None))

    if \
            (lbvc == 8) and \
            (len(lbcode) != 5 or (len(lbcode) == 5 and 1 not in [lbcode.ix, lbcode.iy])):
        coords_and_dims.append((DimCoord(blev, long_name='pressure', units='hPa'), None))

    if \
            (len(lbcode) != 5) and \
            (lbvc == 19):
        coords_and_dims.append((DimCoord(blev, standard_name='air_potential_temperature', units='K', attributes={'positive': 'up'}), None))

    # Hybrid pressure levels (--> scalar coordinates)
    if lbvc == 9:
        model_level_number = DimCoord(model_level_number,
                                      standard_name='model_level_number',
                                      attributes={'positive': 'up'})
        # The following match the hybrid height scheme, but data has the
        # blev and bhlev values the other way around.
        #level_pressure = DimCoord(blev,
        #                          long_name='level_pressure',
        #                          units='Pa',
        #                          bounds=[brlev, brsvd1])
        #sigma = AuxCoord(bhlev,
        #                 long_name='sigma',
        #                 bounds=[bhrlev, brsvd2])
        level_pressure = DimCoord(bhlev,
                                  long_name='level_pressure',
                                  units='Pa',
                                  bounds=[bhrlev, brsvd2])
        sigma = AuxCoord(blev,
                         long_name='sigma',
                         bounds=[brlev, brsvd1])
        coords_and_dims.extend([(model_level_number, None),
                                    (level_pressure, None),
                                    (sigma, None)])
        factories.append(Factory(HybridPressureFactory,
                                 [{'long_name': 'level_pressure'},
                                  {'long_name': 'sigma'},
                                  Reference('surface_air_pressure')]))

    # Hybrid height levels (--> scalar coordinates + factory)
    if lbvc == 65:
        coords_and_dims.append((DimCoord(model_level_number, standard_name='model_level_number', attributes={'positive': 'up'}), None))
        coords_and_dims.append((DimCoord(blev, long_name='level_height', units='m', bounds=[brlev, brsvd1], attributes={'positive': 'up'}), None))
        coords_and_dims.append((AuxCoord(bhlev, long_name='sigma', bounds=[bhrlev, brsvd2]), None))
        factories.append(Factory(HybridHeightFactory, [{'long_name': 'level_height'}, {'long_name': 'sigma'}, Reference('orography')]))

    return coords_and_dims, factories


def _convert_scalar_realization_coords(lbrsvd4):
    """
    Encode scalar 'realization' (aka ensemble) numbers as CM data.

    Returns a list of coords_and_dims.

    """
    # Realization (aka ensemble) (--> scalar coordinates)
    coords_and_dims = []
    if lbrsvd4 != 0:
        coords_and_dims.append(
            (DimCoord(lbrsvd4, standard_name='realization'), None))
    return coords_and_dims


def _convert_scalar_pseudo_level_coords(lbuser5):
    """
    Encode scalar pseudo-level values as CM data.

    Returns a list of coords_and_dims.

    """
    coords_and_dims = []
    if lbuser5 != 0:
        coords_and_dims.append(
            (DimCoord(lbuser5, long_name='pseudo_level', units='1'), None))
    return coords_and_dims


def convert(f):
    factories = []
    aux_coords_and_dims = []

    # "Normal" (non-cross-sectional) Time values (--> scalar coordinates)
    time_coords_and_dims = _convert_scalar_time_coords(
        lbcode=f.lbcode, lbtim=f.lbtim,
        epoch_hours_unit=f.time_unit('hours'),
        t1=f.t1, t2=f.t2, lbft=f.lbft)
    aux_coords_and_dims.extend(time_coords_and_dims)

    # "Normal" (non-cross-sectional) Vertical levels
    #    (--> scalar coordinates and factories)
    vertical_coords_and_dims, vertical_factories = \
        _convert_scalar_vertical_coords(
            lbcode=f.lbcode,
            lbvc=f.lbvc,
            blev=f.blev,
            lblev=f.lblev,
            stash=f.stash,
            bhlev=f.bhlev,
            bhrlev=f.bhrlev,
            brsvd1=f.brsvd[0],
            brsvd2=f.brsvd[1],
            brlev=f.brlev)
    aux_coords_and_dims.extend(vertical_coords_and_dims)
    factories.extend(vertical_factories)

    # Realization (aka ensemble) (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_realization_coords(
        lbrsvd4=f.lbrsvd[3]))

    # Pseudo-level coordinate (--> scalar coordinates)
    aux_coords_and_dims.extend(_convert_scalar_pseudo_level_coords(
        lbuser5=f.lbuser[4]))

    # All the other rules.
    references, standard_name, long_name, units, attributes, cell_methods, \
        dim_coords_and_dims, other_aux_coords_and_dims = _all_other_rules(f)
    aux_coords_and_dims.extend(other_aux_coords_and_dims)

    return (factories, references, standard_name, long_name, units, attributes,
            cell_methods, dim_coords_and_dims, aux_coords_and_dims)


def _all_other_rules(f):
    """
    This deals with all the other rules that have not been factored into any of
    the other convert_scalar_coordinate functions above.

    """
    references = []
    standard_name = None
    long_name = None
    units = None
    attributes = {}
    cell_methods = []
    dim_coords_and_dims = []
    aux_coords_and_dims = []

    # Season coordinates (--> scalar coordinates)
    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            (len(f.lbcode) != 5 or
             (len(f.lbcode) == 5 and
              (f.lbcode.ix not in [20, 21, 22, 23] and
               f.lbcode.iy not in [20, 21, 22, 23]))) and
            f.lbmon == 12 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 3 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('djf', long_name='season', units='no_unit'),
             None))

    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            ((len(f.lbcode) != 5) or
             (len(f.lbcode) == 5 and
              f.lbcode.ix not in [20, 21, 22, 23]
              and f.lbcode.iy not in [20, 21, 22, 23])) and
            f.lbmon == 3 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 6 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('mam', long_name='season', units='no_unit'),
             None))

    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            ((len(f.lbcode) != 5) or
             (len(f.lbcode) == 5 and
              f.lbcode.ix not in [20, 21, 22, 23] and
              f.lbcode.iy not in [20, 21, 22, 23])) and
            f.lbmon == 6 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 9 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('jja', long_name='season', units='no_unit'),
             None))

    if (f.lbtim.ib == 3 and f.lbtim.ic in [1, 2, 4] and
            ((len(f.lbcode) != 5) or
             (len(f.lbcode) == 5 and
              f.lbcode.ix not in [20, 21, 22, 23] and
              f.lbcode.iy not in [20, 21, 22, 23])) and
            f.lbmon == 9 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0 and
            f.lbmond == 12 and f.lbdatd == 1 and f.lbhrd == 0 and
            f.lbmind == 0):
        aux_coords_and_dims.append(
            (AuxCoord('son', long_name='season', units='no_unit'),
             None))

    # "Normal" (i.e. not cross-sectional) lats+lons (--> vector coordinates)
    if (f.bdx != 0.0 and f.bdx != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 1):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   standard_name=f._x_coord_name(),
                                   units='degrees',
                                   circular=(f.lbhem in [0, 4]),
                                   coord_system=f.coord_system()),
             1))

    if (f.bdx != 0.0 and f.bdx != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 2):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   standard_name=f._x_coord_name(),
                                   units='degrees',
                                   circular=(f.lbhem in [0, 4]),
                                   coord_system=f.coord_system(),
                                   with_bounds=True),
             1))

    if (f.bdy != 0.0 and f.bdy != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 1):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzy, f.bdy, f.lbrow,
                                   standard_name=f._y_coord_name(),
                                   units='degrees',
                                   coord_system=f.coord_system()),
             0))

    if (f.bdy != 0.0 and f.bdy != f.bmdi and len(f.lbcode) != 5 and
            f.lbcode[0] == 2):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzy, f.bdy, f.lbrow,
                                   standard_name=f._y_coord_name(),
                                   units='degrees',
                                   coord_system=f.coord_system(),
                                   with_bounds=True),
             0))

    if ((f.bdy == 0.0 or f.bdy == f.bmdi) and
            (len(f.lbcode) != 5 or
             (len(f.lbcode) == 5 and f.lbcode.iy == 10))):
        dim_coords_and_dims.append(
            (DimCoord(f.y, standard_name=f._y_coord_name(), units='degrees',
                      bounds=f.y_bounds, coord_system=f.coord_system()),
             0))

    if ((f.bdx == 0.0 or f.bdx == f.bmdi) and
            (len(f.lbcode) != 5 or
             (len(f.lbcode) == 5 and f.lbcode.ix == 11))):
        dim_coords_and_dims.append(
            (DimCoord(f.x, standard_name=f._x_coord_name(),  units='degrees',
                      bounds=f.x_bounds, circular=(f.lbhem in [0, 4]),
                      coord_system=f.coord_system()),
             1))

    # Cross-sectional vertical level types (--> vector coordinates)
    if (len(f.lbcode) == 5 and f.lbcode.iy == 2 and
            (f.bdy == 0 or f.bdy == f.bmdi)):
        dim_coords_and_dims.append(
            (DimCoord(f.y, standard_name='height', units='km',
                      bounds=f.y_bounds, attributes={'positive': 'up'}),
             0))

    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.iy == 4):
        dim_coords_and_dims.append(
            (DimCoord(f.y, standard_name='depth', units='m',
                      bounds=f.y_bounds, attributes={'positive': 'down'}),
             0))

    if (len(f.lbcode) == 5 and f.lbcode.ix == 10 and f.bdx != 0 and
            f.bdx != f.bmdi):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   standard_name=f._y_coord_name(),
                                   units='degrees',
                                   coord_system=f.coord_system()),
             1))

    if (len(f.lbcode) == 5 and
            f.lbcode.iy == 1 and
            (f.bdy == 0 or f.bdy == f.bmdi)):
        dim_coords_and_dims.append(
            (DimCoord(f.y, long_name='pressure', units='hPa',
                      bounds=f.y_bounds),
             0))

    if (len(f.lbcode) == 5 and f.lbcode.ix == 1 and
            (f.bdx == 0 or f.bdx == f.bmdi)):
        dim_coords_and_dims.append((DimCoord(f.x, long_name='pressure',
                                             units='hPa', bounds=f.x_bounds),
                                    1))

    # Cross-sectional time values (--> vector coordinates)
    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.iy == 23):
        dim_coords_and_dims.append(
            (DimCoord(
                f.y,
                standard_name='time',
                units=iris.unit.Unit('days since 0000-01-01 00:00:00',
                                     calendar=iris.unit.CALENDAR_360_DAY),
                bounds=f.y_bounds),
             0))

    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.ix == 23):
        dim_coords_and_dims.append(
            (DimCoord(
                f.x,
                standard_name='time',
                units=iris.unit.Unit('days since 0000-01-01 00:00:00',
                                     calendar=iris.unit.CALENDAR_360_DAY),
                bounds=f.x_bounds),
             1))

    # Site number (--> scalar coordinate)
    if (len(f.lbcode) == 5 and f.lbcode[-1] == 1 and f.lbcode.ix == 13 and
            f.bdx != 0):
        dim_coords_and_dims.append(
            (DimCoord.from_regular(f.bzx, f.bdx, f.lbnpt,
                                   long_name='site_number', units='1'),
             1))

    # Site number cross-sections (???)
    if (len(f.lbcode) == 5 and
            13 in [f.lbcode.ix, f.lbcode.iy] and
            11 not in [f.lbcode.ix, f.lbcode.iy] and
            hasattr(f, 'lower_x_domain') and
            hasattr(f, 'upper_x_domain') and
            all(f.lower_x_domain != -1.e+30) and
            all(f.upper_x_domain != -1.e+30)):
        aux_coords_and_dims.append(
            (AuxCoord((f.lower_x_domain + f.upper_x_domain) / 2.0,
                      standard_name=f._x_coord_name(), units='degrees',
                      bounds=np.array([f.lower_x_domain, f.upper_x_domain]).T,
                      coord_system=f.coord_system()),
             1 if f.lbcode.ix == 13 else 0))

    if (len(f.lbcode) == 5 and
            13 in [f.lbcode.ix, f.lbcode.iy] and
            10 not in [f.lbcode.ix, f.lbcode.iy] and
            hasattr(f, 'lower_y_domain') and
            hasattr(f, 'upper_y_domain') and
            all(f.lower_y_domain != -1.e+30) and
            all(f.upper_y_domain != -1.e+30)):
        aux_coords_and_dims.append(
            (AuxCoord((f.lower_y_domain + f.upper_y_domain) / 2.0,
                      standard_name=f._y_coord_name(), units='degrees',
                      bounds=np.array([f.lower_y_domain, f.upper_y_domain]).T,
                      coord_system=f.coord_system()),
             1 if f.lbcode.ix == 13 else 0))

    # LBPROC codings (--> cell methods + attributes)
    if f.lbproc == 128 and f.lbtim.ib == 2 and f.lbtim.ia == 0:
        cell_methods.append(CellMethod("mean", coords="time"))

    if f.lbproc == 128 and f.lbtim.ib == 2 and f.lbtim.ia != 0:
        cell_methods.append(CellMethod("mean", coords="time",
                                       intervals="%d hour" % f.lbtim.ia))

    if f.lbproc == 128 and f.lbtim.ib == 3:
        cell_methods.append(CellMethod("mean", coords="time"))

    if f.lbproc == 128 and f.lbtim.ib not in [2, 3]:
        cell_methods.append(CellMethod("mean", coords="time"))

    if f.lbproc == 4096 and f.lbtim.ib == 2 and f.lbtim.ia == 0:
        cell_methods.append(CellMethod("minimum", coords="time"))

    if f.lbproc == 4096 and f.lbtim.ib == 2 and f.lbtim.ia != 0:
        cell_methods.append(CellMethod("minimum", coords="time",
                                       intervals="%d hour" % f.lbtim.ia))

    if f.lbproc == 4096 and f.lbtim.ib != 2:
        cell_methods.append(CellMethod("minimum", coords="time"))

    if f.lbproc == 8192 and f.lbtim.ib == 2 and f.lbtim.ia == 0:
        cell_methods.append(CellMethod("maximum", coords="time"))

    if f.lbproc == 8192 and f.lbtim.ib == 2 and f.lbtim.ia != 0:
        cell_methods.append(CellMethod("maximum", coords="time",
                                       intervals="%d hour" % f.lbtim.ia))

    if f.lbproc == 8192 and f.lbtim.ib != 2:
        cell_methods.append(CellMethod("maximum", coords="time"))

    if f.lbproc not in [0, 128, 4096, 8192]:
        attributes["ukmo__process_flags"] = tuple(
            sorted([iris.fileformats.pp.lbproc_map[flag]
                    for flag in f.lbproc.flags]))

    if (f.lbsrce % 10000) == 1111:
        attributes['source'] = 'Data from Met Office Unified Model'
        # Also define MO-netCDF compliant UM version.
        um_major = (f.lbsrce / 10000) / 100
        if um_major != 0:
            um_minor = (f.lbsrce / 10000) % 100
            attributes['um_version'] = '{:d}.{:d}'.format(um_major, um_minor)

    if (f.lbuser[6] != 0 or
            (f.lbuser[3] / 1000) != 0 or
            (f.lbuser[3] % 1000) != 0):
        attributes['STASH'] = f.stash

    if str(f.stash) in STASH_TO_CF:
        standard_name = STASH_TO_CF[str(f.stash)].standard_name
        units = STASH_TO_CF[str(f.stash)].units
        long_name = STASH_TO_CF[str(f.stash)].long_name

    if (not f.stash.is_valid and f.lbfc in LBFC_TO_CF):
        standard_name = LBFC_TO_CF[f.lbfc].standard_name
        units = LBFC_TO_CF[f.lbfc].units
        long_name = LBFC_TO_CF[f.lbfc].long_name

    # Orography reference field (--> reference target)
    if f.lbuser[3] == 33:
        references.append(ReferenceTarget('orography', None))

    # Surface pressure reference field (--> reference target)
    if f.lbuser[3] == 409 or f.lbuser[3] == 1:
        references.append(ReferenceTarget('surface_air_pressure', None))

    return (references, standard_name, long_name, units, attributes,
            cell_methods, dim_coords_and_dims, aux_coords_and_dims)
