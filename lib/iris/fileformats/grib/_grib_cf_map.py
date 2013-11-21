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
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris. If not, see <http://www.gnu.org/licenses/>.
"""
Provides Grib phenomenon translations in a form matching output from the
prototype Metarelate project.

At present, only these relations are actually used :
* grib1 --> cf
* grib2 --> cf
* cf --> grib2

Notes:
* The format here will surely change in future.
* This file is currently not pep8 compliant.
* Only the tables actually *used* have been manually modified, i.e.
GRIB1_LOCAL_TO_CF, GRIB2_TO_CF and CF_TO_GRIB2.  The other tables may now be
inconsistent with these changes.

This file is currently *not* actual Metarelate output, so these translations
are still controlled entirely by the Iris repository.
Therefore, this file is the place where any further changes and additions
should be made.

"""

from collections import namedtuple

# LIST OF CHANGES:
# Differences from the current Metarelate output.
# * CAPE had 'Kg', replaced with 'kg' (lowercase) as udunits requires it.
# * dewpoint added to both Grib1 and Grib2
# * "grib_physical _atmosphere_albedo" : removed space from name
# * albedo in Grib2 changed to a *percentage* (but still a fraction in Grib1)
# * air_temperature unit "kelvin" changed to "K", as udunits requires it.
# * 'soil_temperature' added
# * more grib2 translations :--
#   * "atmosphere_mass_content_of_water_vapor"  -->  [2, 0, 1, 64] / "kg m-2"
#     * ( Grib2 = "Total column integrated water vapour / kg m-2" )
#   * "land_binary_mask"  -->  [2, 2, 0, 0]
#     * same as "land_area_fraction", but obviously save-only
#   * "surface_altitude"  -->  [2, 2, 0, 7]
#     * ( Grib2 = "Model terrain height / m" )

# OUTSTANDING QUESTIONS:
# Remarks regarding the Metarelate output format
# * Why do G1LocalParam and G2Param contain an "edition" attribute ?
#     -- this seems already implied in the names
# * It would seem preferable to combine the tables GRIB1Local_TO_CF and
#     GRIB1_LOCAL_TO_CF_CONSTRAINED, if possible

CFName = namedtuple('CFName', 'standard_name long_name units')

DimensionCoordinate = namedtuple('DimensionCoordinate',
                                 'standard_name units points')

G1LocalParam = namedtuple('G1LocalParam', 'edition t2version centre iParam')
G2Param = namedtuple('G2Param', 'edition discipline category number')


GRIB1_LOCAL_TO_CF_CONSTRAINED = {
    G1LocalParam(1, 128, 98, 165): (CFName('x_wind', None, 'm s-1'), DimensionCoordinate('height', 'm', (10,))),
    G1LocalParam(1, 128, 98, 166): (CFName('y_wind', None, 'm s-1'), DimensionCoordinate('height', 'm', (10,))),
    G1LocalParam(1, 128, 98, 167): (CFName('air_temperature', None, 'K'), DimensionCoordinate('height', 'm', (2,))),
    G1LocalParam(1, 128, 98, 168): (CFName('dew_point_temperature', None, 'K'), DimensionCoordinate('height', 'm', (2,))),
    }

GRIB1_LOCAL_TO_CF = {
    G1LocalParam(1, 128, 98, 31): CFName('sea_ice_area_fraction', None, 1),
    G1LocalParam(1, 128, 98, 34): CFName('sea_surface_temperature', None, 'K'),
    G1LocalParam(1, 128, 98, 59): CFName('atmosphere_specific_convective_available_potential_energy', None, 'J kg-1'),
    G1LocalParam(1, 128, 98, 129): CFName('geopotential', None, 'm2 s-2'),
    G1LocalParam(1, 128, 98, 130): CFName('air_temperature', None, 'K'),
    G1LocalParam(1, 128, 98, 131): CFName('x_wind', None, 'm s-1'),
    G1LocalParam(1, 128, 98, 132): CFName('y_wind', None, 'm s-1'),
    G1LocalParam(1, 128, 98, 135): CFName('lagrangian_tendency_of_air_pressure', None, 'Pa s-1'),
    G1LocalParam(1, 128, 98, 141): CFName('thickness_of_snowfall_amount', None, 'm'),
    G1LocalParam(1, 128, 98, 151): CFName('air_pressure_at_sea_level', None, 'Pa'),
    G1LocalParam(1, 128, 98, 157): CFName('relative_humidity', None, '%'),
    G1LocalParam(1, 128, 98, 164): CFName('cloud_area_fraction', None, 1),
    G1LocalParam(1, 128, 98, 173): CFName('surface_roughness_length', None, 'm'),
    G1LocalParam(1, 128, 98, 174): CFName(None, 'grib_physical_atmosphere_albedo', 1),
    G1LocalParam(1, 128, 98, 186): CFName('low_type_cloud_area_fraction', None, 1),
    G1LocalParam(1, 128, 98, 187): CFName('medium_type_cloud_area_fraction', None, 1),
    G1LocalParam(1, 128, 98, 188): CFName('high_type_cloud_area_fraction', None, 1),
    G1LocalParam(1, 128, 98, 235): CFName(None, 'grib_skin_temperature', 'K'),
    }

GRIB2_TO_CF = {
    G2Param(2, 0, 0, 0): CFName('air_temperature', None, 'K'),
    G2Param(2, 0, 0, 2): CFName('air_potential_temperature', None, 'K'),
    G2Param(2, 0, 0, 17): CFName(None, 'grib_skin_temperature', 'K'),
    G2Param(2, 0, 1, 0): CFName('specific_humidity', None, 'kg kg-1'),
    G2Param(2, 0, 1, 1): CFName('relative_humidity', None, '%'),
    G2Param(2, 0, 1, 3): CFName(None, 'precipitable_water', 'kg m-2'),
    G2Param(2, 0, 1, 11): CFName('thickness_of_snowfall_amount', None, 'm'),
    G2Param(2, 0, 1, 13): CFName('liquid_water_content_of_surface_snow', None, 'kg m-2'),
    G2Param(2, 0, 1, 22): CFName(None, 'cloud_mixing_ratio', 'kg kg-1'),
    G2Param(2, 0, 1, 64): CFName('atmosphere_mass_content_of_water_vapor', None, 'kg m-2'),
    G2Param(2, 0, 2, 1): CFName('wind_speed', None, 'm s-1'),
    G2Param(2, 0, 2, 2): CFName('x_wind', None, 'm s-1'),
    G2Param(2, 0, 2, 3): CFName('y_wind', None, 'm s-1'),
    G2Param(2, 0, 2, 8): CFName('lagrangian_tendency_of_air_pressure', None, 'Pa s-1'),
    G2Param(2, 0, 2, 10): CFName('atmosphere_absolute_vorticity', None, 's-1'),
    G2Param(2, 0, 3, 0): CFName('air_pressure', None, 'Pa'),
    G2Param(2, 0, 3, 1): CFName('air_pressure_at_sea_level', None, 'Pa'),
    G2Param(2, 0, 3, 3): CFName(None, 'icao_standard_atmosphere_reference_height', 'm'),
    G2Param(2, 0, 3, 4): CFName('geopotential', None, 'm2 s-2'),
    G2Param(2, 0, 3, 5): CFName('geopotential_height', None, 'm'),
    G2Param(2, 0, 3, 9): CFName('geopotential_height_anomaly', None, 'm'),
    G2Param(2, 0, 6, 1): CFName('cloud_area_fraction', None, '%'),
    G2Param(2, 0, 6, 3): CFName('low_type_cloud_area_fraction', None, '%'),
    G2Param(2, 0, 6, 4): CFName('medium_type_cloud_area_fraction', None, '%'),
    G2Param(2, 0, 6, 5): CFName('high_type_cloud_area_fraction', None, '%'),
    G2Param(2, 0, 6, 6): CFName('atmosphere_mass_content_of_cloud_liquid_water', None, 'kg m-2'),
    G2Param(2, 0, 6, 7): CFName('cloud_area_fraction_in_atmosphere_layer', None, '%'),
    G2Param(2, 0, 7, 6): CFName('atmosphere_specific_convective_available_potential_energy', None, 'J kg-1'),
    G2Param(2, 0, 7, 7): CFName(None, 'convective_inhibition', 'J kg-1'),
    G2Param(2, 0, 7, 8): CFName(None, 'storm_relative_helicity', 'J kg-1'),
    G2Param(2, 0, 14, 0): CFName('atmosphere_mole_content_of_ozone', None, 'Dobson'),
    G2Param(2, 0, 19, 1): CFName(None, 'grib_physical_atmosphere_albedo', '%'),
    G2Param(2, 2, 0, 0): CFName('land_area_fraction', None, 1),
    G2Param(2, 2, 0, 1): CFName('surface_roughness_length', None, 'm'),
    G2Param(2, 2, 0, 2): CFName('soil_temperature', None, 'K'),
    G2Param(2, 2, 0, 7): CFName('surface_altitude', None, 'm'),
    G2Param(2, 10, 2, 0): CFName('sea_ice_area_fraction', None, 1),
    G2Param(2, 10, 3, 0): CFName('sea_surface_temperature', None, 'K'),
    }

CF_CONSTRAINED_TO_GRIB1_LOCAL = {
    (CFName('air_temperature', None, 'K'), DimensionCoordinate('height', 'm', (2,))): G1LocalParam(1, 128, 98, 167),
    (CFName('x_wind', None, 'm s-1'), DimensionCoordinate('height', 'm', (10,))): G1LocalParam(1, 128, 98, 165),
    (CFName('y_wind', None, 'm s-1'), DimensionCoordinate('height', 'm', (10,))): G1LocalParam(1, 128, 98, 166),
    }

CF_TO_GRIB1_LOCAL = {
    CFName(None, 'grib_physical_atmosphere_albedo', 1): G1LocalParam(1, 128, 98, 174),
    CFName(None, 'grib_skin_temperature', 'K'): G1LocalParam(1, 128, 98, 235),
    CFName('air_pressure_at_sea_level', None, 'Pa'): G1LocalParam(1, 128, 98, 151),
    CFName('air_temperature', None, 'K'): G1LocalParam(1, 128, 98, 130),
    CFName('atmosphere_specific_convective_available_potential_energy', None, 'J Kg-1'): G1LocalParam(1, 128, 98, 59),
    CFName('cloud_area_fraction', None, 1): G1LocalParam(1, 128, 98, 164),
    CFName('geopotential', None, 'm2 s-2'): G1LocalParam(1, 128, 98, 129),
    CFName('high_type_cloud_area_fraction', None, 1): G1LocalParam(1, 128, 98, 188),
    CFName('lagrangian_tendency_of_air_pressure', None, 'Pa s-1'): G1LocalParam(1, 128, 98, 135),
    CFName('low_type_cloud_area_fraction', None, 1): G1LocalParam(1, 128, 98, 186),
    CFName('medium_type_cloud_area_fraction', None, 1): G1LocalParam(1, 128, 98, 187),
    CFName('relative_humidity', None, '%'): G1LocalParam(1, 128, 98, 157),
    CFName('sea_ice_area_fraction', None, 1): G1LocalParam(1, 128, 98, 31),
    CFName('sea_surface_temperature', None, 'K'): G1LocalParam(1, 128, 98, 34),
    CFName('surface_roughness_length', None, 'm'): G1LocalParam(1, 128, 98, 173),
    CFName('thickness_of_snowfall_amount', None, 'm'): G1LocalParam(1, 128, 98, 141),
    CFName('x_wind', None, 'm s-1'): G1LocalParam(1, 128, 98, 131),
    CFName('y_wind', None, 'm s-1'): G1LocalParam(1, 128, 98, 132),
    }

CF_TO_GRIB2 = {
    CFName(None, 'cloud_mixing_ratio', 'kg kg-1'): G2Param(2, 0, 1, 22),
    CFName(None, 'convective_inhibition', 'J kg-1'): G2Param(2, 0, 7, 7),
    CFName(None, 'grib_physical_atmosphere_albedo', '%'): G2Param(2, 0, 19, 1),
    CFName(None, 'grib_skin_temperature', 'K'): G2Param(2, 0, 0, 17),
    CFName(None, 'icao_standard_atmosphere_reference_height', 'm'): G2Param(2, 0, 3, 3),
    CFName(None, 'precipitable_water', 'kg m-2'): G2Param(2, 0, 1, 3),
    CFName(None, 'storm_relative_helicity', 'J kg-1'): G2Param(2, 0, 7, 8),
    CFName('air_potential_temperature', None, 'K'): G2Param(2, 0, 0, 2),
    CFName('air_pressure', None, 'Pa'): G2Param(2, 0, 3, 0),
    CFName('air_pressure_at_sea_level', None, 'Pa'): G2Param(2, 0, 3, 1),
    CFName('air_temperature', None, 'K'): G2Param(2, 0, 0, 0),
    CFName('atmosphere_absolute_vorticity', None, 's-1'): G2Param(2, 0, 2, 10),
    CFName('atmosphere_mass_content_of_cloud_liquid_water', None, 'kg m-2'): G2Param(2, 0, 6, 6),
    CFName('atmosphere_mass_content_of_water_vapor', None, 'kg m-2'): G2Param(2, 0, 1, 64),
    CFName('atmosphere_mole_content_of_ozone', None, 'Dobson'): G2Param(2, 0, 14, 0),
    CFName('atmosphere_specific_convective_available_potential_energy', None, 'J kg-1'): G2Param(2, 0, 7, 6),
    CFName('cloud_area_fraction', None, '%'): G2Param(2, 0, 6, 1),
    CFName('cloud_area_fraction_in_atmosphere_layer', None, '%'): G2Param(2, 0, 6, 7),
    CFName('dew_point_temperature', None, 'K'): G2Param(2, 0, 0, 6),
    CFName('geopotential', None, 'm2 s-2'): G2Param(2, 0, 3, 4),
    CFName('geopotential_height', None, 'm'): G2Param(2, 0, 3, 5),
    CFName('geopotential_height_anomaly', None, 'm'): G2Param(2, 0, 3, 9),
    CFName('high_type_cloud_area_fraction', None, '%'): G2Param(2, 0, 6, 5),
    CFName('lagrangian_tendency_of_air_pressure', None, 'Pa s-1'): G2Param(2, 0, 2, 8),
    CFName('land_area_fraction', None, 1): G2Param(2, 2, 0, 0),
    CFName('land_binary_mask', None, 1): G2Param(2, 2, 0, 0),
    CFName('liquid_water_content_of_surface_snow', None, 'kg m-2'): G2Param(2, 0, 1, 13),
    CFName('low_type_cloud_area_fraction', None, '%'): G2Param(2, 0, 6, 3),
    CFName('medium_type_cloud_area_fraction', None, '%'): G2Param(2, 0, 6, 4),
    CFName('relative_humidity', None, '%'): G2Param(2, 0, 1, 1),
    CFName('sea_ice_area_fraction', None, 1): G2Param(2, 10, 2, 0),
    CFName('sea_surface_temperature', None, 'K'): G2Param(2, 10, 3, 0),
    CFName('soil_temperature', None, 'K'): G2Param(2, 2, 0, 2),
    CFName('specific_humidity', None, 'kg kg-1'): G2Param(2, 0, 1, 0),
    CFName('surface_altitude', None, 'm'): G2Param(2, 2, 0, 7),
    CFName('surface_roughness_length', None, 'm'): G2Param(2, 2, 0, 1),
    CFName('thickness_of_snowfall_amount', None, 'm'): G2Param(2, 0, 1, 11),
    CFName('wind_speed', None, 'm s-1'): G2Param(2, 0, 2, 1),
    CFName('x_wind', None, 'm s-1'): G2Param(2, 0, 2, 2),
    CFName('y_wind', None, 'm s-1'): G2Param(2, 0, 2, 3),
    }
