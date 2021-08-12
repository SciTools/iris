# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
import sys


class Iris:
    warmup_time = 0
    number = 1
    repeat = 10

    def setup(self):
        self.before = set(sys.modules.keys())

    def teardown(self):
        after = set(sys.modules.keys())
        diff = after - self.before
        for module in diff:
            sys.modules.pop(module)

    def time_iris(self):
        import iris

    def time__concatenate(self):
        import iris._concatenate

    def time__constraints(self):
        import iris._constraints

    def time__data_manager(self):
        import iris._data_manager

    def time__deprecation(self):
        import iris._deprecation

    def time__lazy_data(self):
        import iris._lazy_data

    def time__merge(self):
        import iris._merge

    def time__representation(self):
        import iris._representation

    def time_analysis(self):
        import iris.analysis

    def time_analysis__area_weighted(self):
        import iris.analysis._area_weighted

    def time_analysis__grid_angles(self):
        import iris.analysis._grid_angles

    def time_analysis__interpolation(self):
        import iris.analysis._interpolation

    def time_analysis__regrid(self):
        import iris.analysis._regrid

    def time_analysis__scipy_interpolate(self):
        import iris.analysis._scipy_interpolate

    def time_analysis_calculus(self):
        import iris.analysis.calculus

    def time_analysis_cartography(self):
        import iris.analysis.cartography

    def time_analysis_geomerty(self):
        import iris.analysis.geometry

    def time_analysis_maths(self):
        import iris.analysis.maths

    def time_analysis_stats(self):
        import iris.analysis.stats

    def time_analysis_trajectory(self):
        import iris.analysis.trajectory

    def time_aux_factory(self):
        import iris.aux_factory

    def time_common(self):
        import iris.common

    def time_common_lenient(self):
        import iris.common.lenient

    def time_common_metadata(self):
        import iris.common.metadata

    def time_common_mixin(self):
        import iris.common.mixin

    def time_common_resolve(self):
        import iris.common.resolve

    def time_config(self):
        import iris.config

    def time_coord_categorisation(self):
        import iris.coord_categorisation

    def time_coord_systems(self):
        import iris.coord_systems

    def time_coords(self):
        import iris.coords

    def time_cube(self):
        import iris.cube

    def time_exceptions(self):
        import iris.exceptions

    def time_experimental(self):
        import iris.experimental

    def time_fileformats(self):
        import iris.fileformats

    def time_fileformats__ff(self):
        import iris.fileformats._ff

    def time_fileformats__ff_cross_references(self):
        import iris.fileformats._ff_cross_references

    def time_fileformats__pp_lbproc_pairs(self):
        import iris.fileformats._pp_lbproc_pairs

    def time_fileformats_structured_array_identification(self):
        import iris.fileformats._structured_array_identification

    def time_fileformats_abf(self):
        import iris.fileformats.abf

    def time_fileformats_cf(self):
        import iris.fileformats.cf

    def time_fileformats_dot(self):
        import iris.fileformats.dot

    def time_fileformats_name(self):
        import iris.fileformats.name

    def time_fileformats_name_loaders(self):
        import iris.fileformats.name_loaders

    def time_fileformats_netcdf(self):
        import iris.fileformats.netcdf

    def time_fileformats_nimrod(self):
        import iris.fileformats.nimrod

    def time_fileformats_nimrod_load_rules(self):
        import iris.fileformats.nimrod_load_rules

    def time_fileformats_pp(self):
        import iris.fileformats.pp

    def time_fileformats_pp_load_rules(self):
        import iris.fileformats.pp_load_rules

    def time_fileformats_pp_save_rules(self):
        import iris.fileformats.pp_save_rules

    def time_fileformats_rules(self):
        import iris.fileformats.rules

    def time_fileformats_um(self):
        import iris.fileformats.um

    def time_fileformats_um__fast_load(self):
        import iris.fileformats.um._fast_load

    def time_fileformats_um__fast_load_structured_fields(self):
        import iris.fileformats.um._fast_load_structured_fields

    def time_fileformats_um__ff_replacement(self):
        import iris.fileformats.um._ff_replacement

    def time_fileformats_um__optimal_array_structuring(self):
        import iris.fileformats.um._optimal_array_structuring

    def time_fileformats_um_cf_map(self):
        import iris.fileformats.um_cf_map

    def time_io(self):
        import iris.io

    def time_io_format_picker(self):
        import iris.io.format_picker

    def time_iterate(self):
        import iris.iterate

    def time_palette(self):
        import iris.palette

    def time_plot(self):
        import iris.plot

    def time_quickplot(self):
        import iris.quickplot

    def time_std_names(self):
        import iris.std_names

    def time_symbols(self):
        import iris.symbols

    def time_tests(self):
        import iris.tests

    def time_time(self):
        import iris.time

    def time_util(self):
        import iris.util

    # third-party imports

    def time_third_party_cartopy(self):
        import cartopy

    def time_third_party_cf_units(self):
        import cf_units

    def time_third_party_cftime(self):
        import cftime

    def time_third_party_matplotlib(self):
        import matplotlib

    def time_third_party_numpy(self):
        import numpy

    def time_third_party_scipy(self):
        import scipy
