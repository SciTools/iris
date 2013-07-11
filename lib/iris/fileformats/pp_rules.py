from iris.aux_factory import HybridHeightFactory
from iris.coords import AuxCoord, CellMethod, DimCoord
from iris.fileformats.mosig_cf_map import MOSIG_STASH_TO_CF
from iris.fileformats.rules import Factory, Reference
from iris.fileformats.um_cf_map import STASH_TO_CF
import iris.fileformats.pp
import iris.unit


def convert(cube, field):
    f = field
    cm = cube
    factories = []
    references = []

    if \
            (f.lbtim.ia == 0) and \
            (f.lbtim.ib == 0) and \
            (f.lbtim.ic in [1, 2, 3]) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        cube.add_aux_coord(DimCoord(f.time_unit('hours').date2num(f.t1), standard_name='time', units=f.time_unit('hours')))

    if \
            (f.lbtim.ia == 0) and \
            (f.lbtim.ib == 1) and \
            (f.lbtim.ic in [1, 2, 3]) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        cube.add_aux_coord(DimCoord(f.time_unit('hours', f.t2).date2num(f.t1), standard_name='forecast_period', units='hours'))
        cube.add_aux_coord(DimCoord(f.time_unit('hours').date2num(f.t1), standard_name='time', units=f.time_unit('hours')))
        cube.add_aux_coord(DimCoord(f.time_unit('hours').date2num(f.t2), standard_name='forecast_reference_time', units=f.time_unit('hours')))

    if \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        cube.add_aux_coord(DimCoord(f.lbft, standard_name='forecast_period', units='hours'))
        cube.add_aux_coord(DimCoord((f.time_unit('hours').date2num(f.t1) + f.time_unit('hours').date2num(f.t2)) / 2.0, standard_name='time', units=f.time_unit('hours'), bounds=f.time_unit('hours').date2num([f.t1, f.t2])))
        cube.add_aux_coord(DimCoord(f.time_unit('hours').date2num(f.t2) - f.lbft, standard_name='forecast_reference_time', units=f.time_unit('hours')))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])):
        cube.add_aux_coord(DimCoord(f.lbft, standard_name='forecast_period', units='hours'))
        cube.add_aux_coord(DimCoord((f.time_unit('hours').date2num(f.t1) + f.time_unit('hours').date2num(f.t2)) / 2.0, standard_name='time', units=f.time_unit('hours'), bounds=f.time_unit('hours').date2num([f.t1, f.t2])))
        cube.add_aux_coord(DimCoord(f.time_unit('hours').date2num(f.t2) - f.lbft, standard_name='forecast_reference_time', units=f.time_unit('hours')))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 12 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 3 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        cube.add_aux_coord(AuxCoord('djf', long_name='season', units='no_unit'))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 3 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 6 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        cube.add_aux_coord(AuxCoord('mam', long_name='season', units='no_unit'))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 6 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 9 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        cube.add_aux_coord(AuxCoord('jja', long_name='season', units='no_unit'))

    if \
            (f.lbtim.ib == 3) and \
            (f.lbtim.ic in [1, 2]) and \
            ((len(f.lbcode) != 5) or (len(f.lbcode) == 5 and f.lbcode.ix not in [20, 21, 22, 23] and f.lbcode.iy not in [20, 21, 22, 23])) and \
            (f.lbmon == 9 and f.lbdat == 1 and f.lbhr == 0 and f.lbmin == 0) and \
            (f.lbmond == 12 and f.lbdatd == 1 and f.lbhrd == 0 and f.lbmind == 0):
        cube.add_aux_coord(AuxCoord('son', long_name='season', units='no_unit'))

    if \
            (f.bdx != 0.0) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 1):
        cube.add_dim_coord(DimCoord(f.regular_points("x"), standard_name=f._x_coord_name(), units='degrees', circular=(f.lbhem in [0, 4]), coord_system=f.coord_system()), 1)

    if \
            (f.bdx != 0.0) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 2):
        cube.add_dim_coord(DimCoord(f.regular_points("x"), standard_name=f._x_coord_name(), units='degrees', circular=(f.lbhem in [0, 4]), coord_system=f.coord_system(), bounds=f.regular_bounds("x")), 1)

    if \
            (f.bdy != 0.0) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 1):
        cube.add_dim_coord(DimCoord(f.regular_points("y"), standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system()), 0)

    if \
            (f.bdy != 0.0) and \
            (len(f.lbcode) != 5) and \
            (f.lbcode[0] == 2):
        cube.add_dim_coord(DimCoord(f.regular_points("y"), standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system(), bounds=f.regular_bounds("y")), 0)

    if \
            (f.bdy == 0.0) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.iy == 10)):
        cube.add_dim_coord(DimCoord(f.y, standard_name=f._y_coord_name(), units='degrees', bounds=f.y_bounds, coord_system=f.coord_system()), 0)

    if \
            (f.bdx == 0.0) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and f.lbcode.ix == 11)):
        cube.add_dim_coord(DimCoord(f.x, standard_name=f._x_coord_name(),  units='degrees', bounds=f.x_bounds, circular=(f.lbhem in [0, 4]), coord_system=f.coord_system()), 1)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.iy == 4):
        cube.add_dim_coord(DimCoord(f.y, standard_name='depth', units='m', bounds=f.y_bounds, attributes={'positive': 'down'}), 0)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.ix == 10) and \
            (f.bdx != 0):
        cube.add_dim_coord(DimCoord(f.regular_points("x"), standard_name=f._y_coord_name(), units='degrees', coord_system=f.coord_system()), 1)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.iy == 1) and \
            (f.bdy == 0):
        cube.add_dim_coord(DimCoord(f.y, long_name='pressure', units='hPa', bounds=f.y_bounds), 0)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode.ix == 1) and \
            (f.bdx == 0):
        cube.add_dim_coord(DimCoord(f.x, long_name='pressure', units='hPa', bounds=f.x_bounds), 1)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.iy == 23):
        cube.add_dim_coord(DimCoord(f.y, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.y_bounds), 0)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.ix == 23):
        cube.add_dim_coord(DimCoord(f.x, standard_name='time', units=iris.unit.Unit('days since 0000-01-01 00:00:00', calendar=iris.unit.CALENDAR_360_DAY), bounds=f.x_bounds), 1)

    if \
            (len(f.lbcode) == 5) and \
            (f.lbcode[-1] == 1) and \
            (f.lbcode.ix == 13) and \
            (f.bdx != 0):
        cube.add_dim_coord(DimCoord(f.regular_points("x"), long_name='site_number', units='1'), 1)

    if \
            (len(f.lbcode) == 5) and \
            (13 in [f.lbcode.ix, f.lbcode.iy]) and \
            (11 not in [f.lbcode.ix, f.lbcode.iy]) and \
            (hasattr(f, 'lower_x_domain')) and \
            (hasattr(f, 'upper_x_domain')) and \
            (all(f.lower_x_domain != -1.e+30)) and \
            (all(f.upper_x_domain != -1.e+30)):
        cube.add_aux_coord(AuxCoord((f.lower_x_domain + f.upper_x_domain) / 2.0, standard_name=f._x_coord_name(), units='degrees', bounds=np.array([f.lower_x_domain, f.upper_x_domain]).T, coord_system=f.coord_system()), 1 if f.lbcode.ix == 13 else 0)

    if \
            (len(f.lbcode) == 5) and \
            (13 in [f.lbcode.ix, f.lbcode.iy]) and \
            (10 not in [f.lbcode.ix, f.lbcode.iy]) and \
            (hasattr(f, 'lower_y_domain')) and \
            (hasattr(f, 'upper_y_domain')) and \
            (all(f.lower_y_domain != -1.e+30)) and \
            (all(f.upper_y_domain != -1.e+30)):
        cube.add_aux_coord(AuxCoord((f.lower_y_domain + f.upper_y_domain) / 2.0, standard_name=f._y_coord_name(), units='degrees', bounds=np.array([f.lower_y_domain, f.upper_y_domain]).T, coord_system=f.coord_system()), 1 if f.lbcode.ix == 13 else 0)

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cube.add_cell_method(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cube.add_cell_method(CellMethod("mean", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib == 3):
        cube.add_cell_method(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 128) and \
            (f.lbtim.ib not in [2, 3]):
        cube.add_cell_method(CellMethod("mean", coords="time"))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cube.add_cell_method(CellMethod("minimum", coords="time"))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cube.add_cell_method(CellMethod("minimum", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 4096) and \
            (f.lbtim.ib != 2):
        cube.add_cell_method(CellMethod("minimum", coords="time"))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia == 0):
        cube.add_cell_method(CellMethod("maximum", coords="time"))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib == 2) and \
            (f.lbtim.ia != 0):
        cube.add_cell_method(CellMethod("maximum", coords="time", intervals="%d hour" % f.lbtim.ia))

    if \
            (f.lbproc == 8192) and \
            (f.lbtim.ib != 2):
        cube.add_cell_method(CellMethod("maximum", coords="time"))

    if f.lbproc not in [0, 128, 4096, 8192]:
        cube.attributes["ukmo__process_flags"] = tuple(sorted([iris.fileformats.pp.lbproc_map[flag] for flag in f.lbproc.flags]))

    if \
            (f.lbvc == 1) and \
            (not (f.lbuser[6] == 1 and f.lbuser[3] == 3236)) and \
            (f.blev != -1):
        cube.add_aux_coord(DimCoord(f.blev, standard_name='height', units='m', attributes={'positive': 'up'}))

    if f.lbuser[6] == 1 and f.lbuser[3] == 3236:
        cube.add_aux_coord(DimCoord(1.5, standard_name='height', units='m', attributes={'positive': 'up'}))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2):
        cube.add_aux_coord(DimCoord(f.lblev, standard_name='model_level_number', attributes={'positive': 'down'}))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2) and \
            (f.brsvd[0] == f.brlev):
        cube.add_aux_coord(DimCoord(f.blev, standard_name='depth', units='m', attributes={'positive': 'down'}))

    if \
            (len(f.lbcode) != 5) and \
            (f.lbvc == 2) and \
            (f.brsvd[0] != f.brlev):
        cube.add_aux_coord(DimCoord(f.blev, standard_name='depth', units='m', bounds=[f.brsvd[0], f.brlev], attributes={'positive': 'down'}))

    if \
            (f.lbvc == 8) and \
            (len(f.lbcode) != 5 or (len(f.lbcode) == 5 and 1 not in [f.lbcode.ix, f.lbcode.iy])):
        cube.add_aux_coord(DimCoord(f.blev, long_name='pressure', units='hPa'))

    if f.lbvc == 65:
        cube.add_aux_coord(DimCoord(f.lblev, standard_name='model_level_number', attributes={'positive': 'up'}))
        cube.add_aux_coord(DimCoord(f.blev, long_name='level_height', units='m', bounds=[f.brlev, f.brsvd[0]], attributes={'positive': 'up'}))
        cube.add_aux_coord(AuxCoord(f.bhlev, long_name='sigma', bounds=[f.bhrlev, f.brsvd[1]]))
        factories.append(Factory(HybridHeightFactory, [{'long_name': 'level_height'}, {'long_name': 'sigma'}, Reference('orography')]))

    if f.lbrsvd[3] != 0:
        cube.add_aux_coord(DimCoord(f.lbrsvd[3], standard_name='realization'))

    if f.lbuser[4] != 0:
        cube.add_aux_coord(DimCoord(f.lbuser[4], long_name='pseudo_level', units='1'))

    if f.lbuser[6] == 1 and f.lbuser[3] == 5226:
        cube.standard_name = "precipitation_amount"
        cube.units = "kg m-2"

    if \
            (f.lbuser[6] == 2) and \
            (f.lbuser[3] == 101):
        cube.standard_name = "sea_water_potential_temperature"
        cube.units = "Celsius"

    if \
            ((f.lbsrce % 10000) == 1111) and \
            ((f.lbsrce / 10000) / 100.0 > 0):
        cube.attributes['source'] = 'Data from Met Office Unified Model %4.2f' % ((f.lbsrce / 10000) / 100.0)

    if \
            ((f.lbsrce % 10000) == 1111) and \
            ((f.lbsrce / 10000) / 100.0 == 0):
        cube.attributes['source'] = 'Data from Met Office Unified Model'

    if f.lbuser[6] != 0 or (f.lbuser[3] / 1000) != 0 or (f.lbuser[3] % 1000) != 0:
        cube.attributes['STASH'] = f.stash

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 4205):
        cube.standard_name = "mass_fraction_of_cloud_ice_in_air"
        cube.units = "1"

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 4206):
        cube.standard_name = "mass_fraction_of_cloud_liquid_water_in_air"
        cube.units = "1"

    if \
            (f.lbuser[6] == 1) and \
            (f.lbuser[3] == 30204):
        cube.standard_name = "air_temperature"
        cube.units = "K"

    if \
            (f.lbuser[6] == 4) and \
            (f.lbuser[3] == 6001):
        cube.standard_name = "sea_surface_wave_significant_height"
        cube.units = "m"

    if str(f.stash) in MOSIG_STASH_TO_CF:
        cube.standard_name = MOSIG_STASH_TO_CF[str(f.stash)].name
        cube.units = MOSIG_STASH_TO_CF[str(f.stash)].unit
        cube.long_name = None

    if str(f.stash) in STASH_TO_CF:
        cube.standard_name = STASH_TO_CF[str(f.stash)].cfname
        cube.units = STASH_TO_CF[str(f.stash)].unit
        cube.long_name = None

    if \
            (not f.stash.is_valid) and \
            (f.lbfc in LBFC_TO_CF):
        cube.standard_name = LBFC_TO_CF[f.lbfc].cfname
        cube.units = LBFC_TO_CF[f.lbfc].unit
        cube.long_name = None

    if f.lbuser[3] == 33:
        references.append(ReferenceTarget('orography', None))

    return factories, references
