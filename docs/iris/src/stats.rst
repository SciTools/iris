
.. _github-stats:
Github stats
============

GitHub stats for 2012/06/18 - 2013/09/11 (tag: None)

These lists are automatically generated, and may be incomplete or contain duplicates.


We closed a total of 993 issues, 379 pull requests and 614 regular issues;
this is the full list (generated with the script 
:file:`matplotlib/tools/github_stats.py`):

Pull Requests (379):

* :ghpull:`750`: Fix GeoTIFF export
* :ghpull:`751`: BUG: Fixed typo in OSGB definition
* :ghpull:`754`: Transverse Mercator from netCDF
* :ghpull:`745`: Added flexibilty when saving integer types to netCDF3
* :ghpull:`748`: Respect major/minor axes when converting GeogCS to Cartopy.
* :ghpull:`731`: Ensure consistent memory layout for checksums.
* :ghpull:`740`: Added the magic lines to support UM ancillary files trunkated with the UM utility ieee.
* :ghpull:`743`: Handle directory creation race condition.
* :ghpull:`742`: Add netCDF export support for transverse Mercator.
* :ghpull:`735`: Linear doc example
* :ghpull:`734`: From reg fix
* :ghpull:`732`: Simplified ionosphere example.
* :ghpull:`733`: Modified cube dim_coords and aux_coords docstrings
* :ghpull:`728`: Cube extraction doc examples
* :ghpull:`729`: Added gdal and pandas to INSTALL
* :ghpull:`730`: Fix coding standards exclusions after 1.4.x merge.
* :ghpull:`727`: V1.4.x merge
* :ghpull:`726`: Makes TestLicenseHeaders error message better. Closes #505.
* :ghpull:`725`: Added the implied heights for several common PP STASH codes.
* :ghpull:`705`: Add new stashcodes for dewpoint and RH.
* :ghpull:`714`: ENH: Added support for checking netcdf file saving
* :ghpull:`711`: Modified attribute behaviour for netCDF save.
* :ghpull:`522`: y-axis inversion for vertical coords
* :ghpull:`694`: reordered phenomenon dictionaries
* :ghpull:`697`: Fix coding of maximum and minimum time-stats in Grib2 save.
* :ghpull:`719`: Added more complete tmerc to cartopy translation
* :ghpull:`707`: Nearest neighbour indexing no longer loads the data.
* :ghpull:`721`: BF - allow model level GRIB files to be loaded.
* :ghpull:`689`: Added support for loading regular gaussian grids from GRIB files.
* :ghpull:`702`: Make data checksum ignore masked values.
* :ghpull:`691`: BF - Correct Earth radius when saving GRIB2 files, fixes #690.
* :ghpull:`555`: Calculates correlation between cubes
* :ghpull:`619`: Varff
* :ghpull:`710`: Update trajectory.py
* :ghpull:`708`: Update interpolate.py
* :ghpull:`684`: Limit unit calendar argument
* :ghpull:`643`: Added support for bool array indexing on a cube.
* :ghpull:`663`: ENH: Added support to setup.py for quick testing
* :ghpull:`496`: working wrapper for animating in iris
* :ghpull:`701`: PEP 8 iris.fileformats.grib.grib_save_rules
* :ghpull:`553`: Add ability to pass data dimensions for slicing cube
* :ghpull:`671`: BUG: Fix broken test result TestAtlanticProfiles
* :ghpull:`698`: lamb/polar coord units
* :ghpull:`630`: abf for 32-bit arch
* :ghpull:`693`: DimCoord CML circular flag.
* :ghpull:`695`: Fix netCDF4 python package at version 1.0.2 for travis-ci
* :ghpull:`681`: Correct dim coord behaviour for aggregated by.
* :ghpull:`676`: Updated docs to use latest what's new.
* :ghpull:`610`: added 42-byte wmo header, requiring code tidy
* :ghpull:`625`: Percentile agg
* :ghpull:`672`: Cope with missing bounds
* :ghpull:`659`: Fast Cell construction during merge
* :ghpull:`668`: Added Grib2 translations for 'soil_temperature'.
* :ghpull:`637`: Fast add coords
* :ghpull:`642`: Switch to using DimCoord.from_regular
* :ghpull:`653`: Add new ocean profile plot example.
* :ghpull:`662`: PEP 8 iris.symbols
* :ghpull:`661`: PEP 8 iris.config
* :ghpull:`657`: Use id() based lookup to find aux_coords.
* :ghpull:`658`: Fix indent typo in BitwiseInt
* :ghpull:`631`: TestPPSave for 32-bit arch
* :ghpull:`632`: test_ff for 32 -bit arch
* :ghpull:`641`: ne logic
* :ghpull:`655`: MAINT: Refactoring of unit equality
* :ghpull:`635`: Global unit cache
* :ghpull:`654`: PEP 8 follow-up: fixup comments in iris.coord_categorisation
* :ghpull:`645`: PEP 8 iris.coord_categorisation
* :ghpull:`651`: ENH: Optimised unit equality special method
* :ghpull:`561`: Lambert conformal grib2 loading
* :ghpull:`649`: Support qplt.show in extest.
* :ghpull:`646`: PEP 8 iris.cube
* :ghpull:`644`: PEP 8 iris.aux_factory
* :ghpull:`650`: BUG: Fix pandas test for version update
* :ghpull:`633`: Cell is a named tuple
* :ghpull:`648`: Avoid version-specific formatting issues by checking data directly.
* :ghpull:`647`: Remove workaround from filtering example.
* :ghpull:`629`: PEP 8 iris.quickplot
* :ghpull:`640`: BUG: Updated copyright dates on docs
* :ghpull:`626`: Convert iris.analysis.SUM to a weighted aggregator.
* :ghpull:`547`: grib hind doc
* :ghpull:`617`: NF - add 1D scatter plot functions.
* :ghpull:`628`: PEP 8 iris.exceptions
* :ghpull:`627`: PEP 8 iris.palette and iris.pandas
* :ghpull:`634`: Update PPA to get a working libgeos-dev
* :ghpull:`622`: PEP 8 iris._merge
* :ghpull:`620`: PEP 8 iris._cube_coord_common
* :ghpull:`621`: PEP 8 iris.fileformats.ff
* :ghpull:`611`: PP/GRIB load rules as a function
* :ghpull:`618`: Convert post-item documentation to pre-item #: form
* :ghpull:`616`: Remove quickplot unit label for time reference coordinates. Closes #615.
* :ghpull:`462`: Added verbose functionality to is_compatible
* :ghpull:`580`: unit doc
* :ghpull:`614`: BUG: creating coordinate from another crs copy
* :ghpull:`613`: Add show to quickplot. Closes #607.
* :ghpull:`593`: Extend 1D plotting capabilities. Resolves #581.
* :ghpull:`587`: Optimise constructing ordered metadata (merge path)
* :ghpull:`578`: Corrected filtered values in SOI example.
* :ghpull:`602`: Added missing in_place keywords in analysis.math
* :ghpull:`601`: Added Polar stereographic plot
* :ghpull:`597`: Simple PEP8 fix.
* :ghpull:`594`: Consistant title
* :ghpull:`579`: name trajectory loading
* :ghpull:`588`: removed workaround to multiple coordinates been present when plotting.
* :ghpull:`586`: fix travis ref to new sample data
* :ghpull:`570`: Updated example to work with newer sample data.
* :ghpull:`566`: added packaged grib support
* :ghpull:`575`: PEP8 fixes for cartography module
* :ghpull:`571`: NF - Remove 1d restriction from cosine latitude weights
* :ghpull:`568`: Corrected rendering of code examples.
* :ghpull:`564`: Fix to prevent unnecessary rebuild of pyke rules.
* :ghpull:`558`: Doc tweaks for custom season categorisation
* :ghpull:`557`: Reset warnings in test_phenom_unknown.
* :ghpull:`556`: Release v1.4.0
* :ghpull:`550`: Added support for loading NAME files
* :ghpull:`552`: Switched from sample data to test data.
* :ghpull:`546`: PP save default gridded polar axis.
* :ghpull:`548`: Export to GeoTIFF refactor.
* :ghpull:`524`: Added support for checking coding standards of docs
* :ghpull:`544`: 1.4.0rc1 version number
* :ghpull:`540`: 1.4.0 changes and what's new
* :ghpull:`538`: MAIN: Removed all whitespace and file-end lines
* :ghpull:`541`: Added git clone depth of zero to travis CI config.
* :ghpull:`539`: Specified depth of zero for travis ci clone
* :ghpull:`537`: update version to v1.5.0-dev
* :ghpull:`518`: polar stereo grib
* :ghpull:`521`: Added netcdf load support for transverse Mercator grid mapping and climatology
* :ghpull:`531`: PEP8 fixes for aggregation tests
* :ghpull:`494`: updating iris release to use latest cartopy (master)
* :ghpull:`297`: Adds support for ieee 32bit fieldsfiles to iris.load
* :ghpull:`527`: BF - preserve masked arrays during aggregation
* :ghpull:`514`: New pp rule to calculate forecast period.
* :ghpull:`463`: Modified plot.py to pass coords arg through to _map_common()
* :ghpull:`482`: Revised grib load+save
* :ghpull:`523`: Moved to SHA for iris-test-data until we tag the next release.
* :ghpull:`520`: BF - handle missing values from grib messages
* :ghpull:`453`: Esmf conserve
* :ghpull:`515`: Fixed PEP8 issues in netcdf.py
* :ghpull:`511`: Resolved pickling issues with deferred loading.
* :ghpull:`513`: Missing import
* :ghpull:`510`: Documentation change on coordinates parameter for slices()
* :ghpull:`507`: Load of nimrod files with multiple fields and period of interest
* :ghpull:`499`: Propogate all coord names in construct midpoint
* :ghpull:`508`: Modified linear() to handle non-scalar length one coords.
* :ghpull:`504`: Aggregate by on str dim err handling
* :ghpull:`479`: Make tests optional that depend on pydot and pandas (optional installs).
* :ghpull:`498`: PP save with no time forecast.
* :ghpull:`495`: Removal of strict constraint in izip().
* :ghpull:`490`: Refactor seasons categorisation functions.
* :ghpull:`491`: PEP8 corrections to iterate.py.
* :ghpull:`464`: Area weighted regridding
* :ghpull:`480`: Unambiguous season year naming
* :ghpull:`428`: Add an optimisation for single valued coordinate constraints.
* :ghpull:`488`: fixed netcdf save cubelist bug
* :ghpull:`487`: Fix to copyright licence check logic
* :ghpull:`475`: Cube merge string coords.
* :ghpull:`486`: Fixed copyright date of an experimental file.
* :ghpull:`483`: Fixed copyright date on one of the tests.
* :ghpull:`477`: Added a test for license checking of code files. Closes #454.
* :ghpull:`471`: gribsave int32 scalar
* :ghpull:`476`: Added a PEP8 test to ensure coding standards.
* :ghpull:`424`: Added geotiff to what's new v1.4
* :ghpull:`422`: Working OpenDAP functionality.
* :ghpull:`451`: Added depth rules for bounds.
* :ghpull:`472`: Modified logic in Unit.convert to handle numpy scalars.
* :ghpull:`199`: nimrod level type 12
* :ghpull:`473`: Pandas 0.11
* :ghpull:`439`: pandas 1d
* :ghpull:`427`: Pep8 init and constraints
* :ghpull:`465`: change unit print-out syntax
* :ghpull:`459`: Added tolerances to Coord.is_contiguous()
* :ghpull:`449`: Added cfchecker to travis-ci
* :ghpull:`435`: Safely build PyKE rule base.
* :ghpull:`442`: Add bilinear interpolation between rectilinear grids.
* :ghpull:`440`: Added iris.tests.assertArrayAllClose 
* :ghpull:`441`: Fixed ref to old scitools.org page.
* :ghpull:`436`: Travis-CI non git clone of Cartopy.
* :ghpull:`437`: Helper function for regridding that returns x and y coords.
* :ghpull:`426`: V1x3 release
* :ghpull:`383`: GeoTiff export with test
* :ghpull:`420`: Fixed sphinx error when building netcdf save docs
* :ghpull:`415`: Add dtype and flags to CML
* :ghpull:`418`: Update to iris-test-data commit in .travis.yml
* :ghpull:`419`: Removed data_repository from site.cfg.template
* :ghpull:`406`: Simplified resource configuration
* :ghpull:`416`: Remove redundant code.
* :ghpull:`414`: Remove redundant method definition
* :ghpull:`391`: Rework Coord nearest_neighbour calc to support circular cases.
* :ghpull:`411`: Explicitly set the SHAs of dependencies in the travis.yml file.
* :ghpull:`409`: Update changelog and what's new pages for 1.3
* :ghpull:`367`: Saving multiple cubes to a netcdf
* :ghpull:`410`: Moved from master to a specific commit for iris test data.
* :ghpull:`405`: Return a CubeList when slicing a CubeList
* :ghpull:`397`: corrected syntax for named tuple def
* :ghpull:`393`: Copy button in docs.
* :ghpull:`399`: Update to README.md to include Travis-CI build status.
* :ghpull:`390`: fixed converting reference times
* :ghpull:`336`: Cube concatenate.
* :ghpull:`359`: add exponential to math
* :ghpull:`392`: travis (full)
* :ghpull:`389`: Iris citation added into user guide
* :ghpull:`382`: Removed most redundant imports
* :ghpull:`381`: pep8 coords.py
* :ghpull:`380`: version 1.2.1-dev
* :ghpull:`379`: V1.2.0 release
* :ghpull:`377`: V1.2.0 release
* :ghpull:`375`: V1.2.0 release
* :ghpull:`368`: NetCDF CF profile support.
* :ghpull:`371`: Attempt to fix pep8 issues in cube.py
* :ghpull:`374`: pep8 and various re-formatting
* :ghpull:`373`: Bug fix of pp saving, closses #277
* :ghpull:`370`: Added missing blank lines after sphinx directives.
* :ghpull:`361`: Modified cube.summary() to handle unicode cube attributes.
* :ghpull:`362`: Refactored is_x boolean methods in unit module.
* :ghpull:`261`: Changes from PR.212 -- those already ok-ed.
* :ghpull:`350`: Basic support for varying longitude ranges.
* :ghpull:`358`: Prevented the ability to add AuxCoord instances to the dim_coords.
* :ghpull:`357`: Use position independent code flag for compilation of gribapi
* :ghpull:`354`: Default tests to the 'agg' matplotlib backend.
* :ghpull:`313`: Rm pyproj dep
* :ghpull:`349`: Corrected len(coord) to len(coord.points) and added test
* :ghpull:`172`: pcol msg fix
* :ghpull:`347`: Revert Iris version from 1.2.0-rc1 back to 1.2.0-dev.
* :ghpull:`238`: Abf loading
* :ghpull:`343`: Changes for rc1.
* :ghpull:`342`: Increment version number
* :ghpull:`203`: nimrod orography loading
* :ghpull:`317`: Added support for CF variable names to cubes and coords
* :ghpull:`268`: Changes to cartography.py, copied from PR.212
* :ghpull:`180`: GRIB1 duplicate rule
* :ghpull:`265`: Changes to loading_iris_cubes.rst, copied from PR.212
* :ghpull:`173`: Empty file handing
* :ghpull:`323`: datetime human-read + cube unit print
* :ghpull:`338`: Standardising numpy namespaces
* :ghpull:`334`: Added convert_units() to cubes and coords.
* :ghpull:`327`: Prevented the ability to add AuxCoord instances to the dim_coords.
* :ghpull:`331`: Defer format-specific imports
* :ghpull:`333`: Remove obsolete unused code (very small)
* :ghpull:`174`: Grib hindcast workaround
* :ghpull:`326`: Unused "import unittest" removed from extests
* :ghpull:`316`: Updated the extests to easily support image tolerances.
* :ghpull:`311`: Consistent Cell-Cell ordering, and seperate Cell-scalar semantics.
* :ghpull:`273`: control cube iteration
* :ghpull:`310`: Fixed issue introduced by #300.
* :ghpull:`315`: Document sitecfg patch
* :ghpull:`309`: Added an example of rolling_window with weights.
* :ghpull:`304`: Travis nodata
* :ghpull:`306`: full data cml
* :ghpull:`300`: Supported merging of accumulations.
* :ghpull:`296`: Added a simple rule, which needed a lot more work to test it!.
* :ghpull:`285`: NF - allow weights in rolling window aggregations
* :ghpull:`288`: Nimrod xy names
* :ghpull:`293`: travis
* :ghpull:`302`: Missed points in previous merges
* :ghpull:`207`: Plot 2dxy
* :ghpull:`290`: Docs still mention "Basemap" (new rm basemap)
* :ghpull:`294`: Fixed lone typo in PP rules. Closes #258.
* :ghpull:`286`: NF - normalized area weights
* :ghpull:`221`: g1 rule update
* :ghpull:`282`: Deal with netCDF non-monotonic and masked coordinates.
* :ghpull:`283`: Graceful PP loading of invalid units.
* :ghpull:`247`: NF - add new weighting methods
* :ghpull:`279`: Faster monotonic check
* :ghpull:`276`: Analysis root mean square calculation. Closes issue #274.
* :ghpull:`239`: externalise tests -5
* :ghpull:`272`: Ocean depth PP rule and minor merge optimisation.
* :ghpull:`270`: Added missing @iris.tests.skip_data decorators.
* :ghpull:`184`: Grib1unit10 additional
* :ghpull:`235`: Externalise tests take-4
* :ghpull:`263`: Changes to cube_statistics.rst, copied from PR.212
* :ghpull:`266`: Changes to whats_new.rst, copied from PR.212
* :ghpull:`269`: A series of small but important fixes to the docs.
* :ghpull:`267`: remove '1.0'
* :ghpull:`264`: Update documentation.
* :ghpull:`262`: Updated minimum dependency of cartopy to v0.5.
* :ghpull:`204`: Implicit out of place guess_bounds when using pcolor and pcolormesh
* :ghpull:`254`: save netcdf traj
* :ghpull:`209`: Hybrid height trajectory
* :ghpull:`200`: Grib steps bug
* :ghpull:`250`: NF - new coord categorisation functions
* :ghpull:`255`: BF - fix typo in udunits2 library locator
* :ghpull:`246`: unit.py on osx
* :ghpull:`253`: Coordinate CML id sorted for attribute dictionary.
* :ghpull:`231`: Externalise tests take-2 : "test_merge" + "test_hybrid"
* :ghpull:`243`: Fieldsfile ancillary loading capability with PP LBTIM=2 support.
* :ghpull:`225`: Constraint extract slice optimisation.
* :ghpull:`237`: PP load optimisation.
* :ghpull:`249`: Fixed failing doctest introduced by #248
* :ghpull:`248`: Added lbtim.ib == 3 forecast_reference_time rule
* :ghpull:`175`: Tidy XML tests
* :ghpull:`183`: Added forecast reference time to a field with lbtim of 11 (one of the most common lbtim-s)
* :ghpull:`194`: BF - fix broadcasting of area weights
* :ghpull:`245`: Fixed lowercase github link.
* :ghpull:`216`: Deferred loading of AuxCoord points/bounds.
* :ghpull:`217`: Changed data_dim to an argument not keyword
* :ghpull:`229`: grib save : minor update
* :ghpull:`232`: Externalise tests take-3 : test_plot
* :ghpull:`218`: Faster masked array creation in DataManager.
* :ghpull:`213`: Cache netCDF attributes
* :ghpull:`210`: Fieldsfile and PP support for unpacked and CRAY 32-bit packed data.
* :ghpull:`208`: Externalise tests
* :ghpull:`205`: Track Cartopy version
* :ghpull:`121`: new cell comparison
* :ghpull:`195`: Add CONTRIBUTING.md
* :ghpull:`190`: Remove redundant and erroneous standard_name to LBFC/STASH PP save rules.
* :ghpull:`171`: grib1 unit 10 handling
* :ghpull:`157`: logo
* :ghpull:`160`: Update Iris version for bug-fix development.
* :ghpull:`159`: Release version 1.0.0
* :ghpull:`158`: Autodoc fix
* :ghpull:`156`: Update to visual test result following orthographic fix in cartopy.
* :ghpull:`155`: linear: numpy masked append workaround (replacement pr)
* :ghpull:`153`: Gitwash build and Iris URL update.
* :ghpull:`141`: Avoid unnecessary analysis imports.
* :ghpull:`150`: Update test_plot.test_missing_cs expected png result for non-brewer
* :ghpull:`144`: Undo mistaken version change on master.
* :ghpull:`147`: Revert version back to 1.0.0-dev after rc1 tag
* :ghpull:`146`: Release version 1.0.0rc1
* :ghpull:`145`: Release 1.0.0 documentation changes.
* :ghpull:`143`: Release version 1.0.0rc1
* :ghpull:`142`: Release 1.0.0 documentation changes.
* :ghpull:`136`: Map setup purge
* :ghpull:`138`: bug fix: plot with no cs
* :ghpull:`139`: A quick word about future plans
* :ghpull:`137`: Turn off auto palette selection of brewer
* :ghpull:`134`: grib1 time bound
* :ghpull:`111`: Added cube project function to iris.analysis.cartography
* :ghpull:`120`: no default cf cs
* :ghpull:`97`: Axis unit labels
* :ghpull:`78`: New load/merge API
* :ghpull:`128`: Minor correction to what's new and CHANGES, plus apostrophe.
* :ghpull:`124`: Use of tuples for dimensions in coord to dim mapping
* :ghpull:`116`: Nimrod loader patch.
* :ghpull:`126`: cartopy replaces basemap
* :ghpull:`125`: What's new
* :ghpull:`122`: collapse nd bounds
* :ghpull:`117`: Consistent order in idiff
* :ghpull:`114`: Extend attribute comparison to deal with NumPy arrays
* :ghpull:`113`: Addition of cartopy_crs method to CoordSystem.
* :ghpull:`109`: Mpl 1.2 rc2
* :ghpull:`108`: Added real image testing.
* :ghpull:`96`: Re-introduce Cynthia Brewer palettes.
* :ghpull:`95`: First cut of hybrid-pressure for GRIB.
* :ghpull:`92`: Remove test_trui, and cut the fat from test_uri_callback.py.
* :ghpull:`94`: Fixes to unguarded access to circular attribute.
* :ghpull:`64`: Test cleanup
* :ghpull:`93`: Allow GRIB loader to handle cross-references
* :ghpull:`90`: Add copyright/licence header check
* :ghpull:`91`: Removal of test data from test_iterate.py
* :ghpull:`88`: Speed-up & tidy test_cube_to_pp
* :ghpull:`89`: Change to cube summary for cubes with bounded scalar coords
* :ghpull:`54`: NIMROD loading
* :ghpull:`81`: Update change log for release v0.9.1
* :ghpull:`80`: CF file handle bug fix.
* :ghpull:`72`: Fix to bug causing files to remain open after loading netCDF files.
* :ghpull:`65`: CF CoordSystems (replacement pull)
* :ghpull:`68`: Exposed merge keyword argument to public load api.
* :ghpull:`70`: Speed up the CDM tests...
* :ghpull:`63`: Make netCDF save outermost dimension as unlimited
* :ghpull:`66`: Clean up of setup.py, install docs and removal of symlinks
* :ghpull:`33`: installation instructions
* :ghpull:`46`: Hybrid pressure
* :ghpull:`56`: plot doc error
* :ghpull:`60`: iris.util.guess_coord_axis() behaviour
* :ghpull:`55`: math doc typo
* :ghpull:`24`: Deprecation of coord trig methods
* :ghpull:`39`: Convert source from coord to attribute. Fixes #10
* :ghpull:`42`: Build of gh-pages for iris release v0.9.
* :ghpull:`34`: Sample data usage
* :ghpull:`38`: Removed details of earlier api change from docs.
* :ghpull:`36`: 0.9 change log
* :ghpull:`28`: Pyke rules unicode warning tweak. Fixes #27.
* :ghpull:`31`: Removal of TEST_COMPAT and OLD_XML code
* :ghpull:`32`: For #30, update gitwash folder
* :ghpull:`29`: patch for IE9 support
* :ghpull:`25`: Cube merge dimension hint and ordering.
* :ghpull:`16`: Fixed unit usage and updated a warning to be more explicit.
* :ghpull:`21`: Use new xml
* :ghpull:`5`: Cube summary misalignment and coordinate name clipping. Fixes #4.
* :ghpull:`17`: Graceful netCDF units loading.

Issues (614):

* :ghissue:`750`: Fix GeoTIFF export
* :ghissue:`751`: BUG: Fixed typo in OSGB definition
* :ghissue:`752`: BUG: Fixed pyke rules for transverse mercator.
* :ghissue:`754`: Transverse Mercator from netCDF
* :ghissue:`745`: Added flexibilty when saving integer types to netCDF3
* :ghissue:`744`: Gracefully accept either NetCDF longitude std name.
* :ghissue:`670`: BUG: Stop checking docs build in coding standards test
* :ghissue:`748`: Respect major/minor axes when converting GeogCS to Cartopy.
* :ghissue:`187`: Improve ionosphere gallery plot
* :ghissue:`749`: Corrected 'temperature' var name to 'air_pressure', plus a couple of PEP8 changes
* :ghissue:`700`: conservative regrid : unexpectedly masked data
* :ghissue:`731`: Ensure consistent memory layout for checksums.
* :ghissue:`740`: Added the magic lines to support UM ancillary files trunkated with the UM utility ieee.
* :ghissue:`743`: Handle directory creation race condition.
* :ghissue:`703`: Graceful identification of longitude.
* :ghissue:`742`: Add netCDF export support for transverse Mercator.
* :ghissue:`739`: Add netCDF export support for transverse Mercator.
* :ghissue:`682`: quasi-regular grib1
* :ghissue:`364`: Documentation of iris.analysis.interpolate.linear()
* :ghissue:`735`: Linear doc example
* :ghissue:`734`: From reg fix
* :ghissue:`170`: Erratic/broken docs
* :ghissue:`603`: Simplify ionosphere example
* :ghissue:`732`: Simplified ionosphere example.
* :ghissue:`450`: dim_coords docstring
* :ghissue:`733`: Modified cube dim_coords and aux_coords docstrings
* :ghissue:`481`: Missing sample data for Iris docs
* :ghissue:`728`: Cube extraction doc examples
* :ghissue:`400`: git check sheet URL broken
* :ghissue:`724`: doc updates for iris.save
* :ghissue:`474`: Missing dependencies (GDAL and Pandas) in INSTALL
* :ghissue:`729`: Added gdal and pandas to INSTALL
* :ghissue:`730`: Fix coding standards exclusions after 1.4.x merge.
* :ghissue:`727`: V1.4.x merge
* :ghissue:`505`: Make license headers tests more friendly
* :ghissue:`726`: Makes TestLicenseHeaders error message better. Closes #505.
* :ghissue:`49`: Mac-compatible shared library refs
* :ghissue:`185`: Problems reading fieldsfile with LBPACK=0
* :ghissue:`188`: remove two usused funcs
* :ghissue:`214`: Scatter bounds
* :ghissue:`244`: PP loading optimisation
* :ghissue:`271`: Add a full release history in the documentation
* :ghissue:`339`: Remove '>>>' from documentation to allow copy/paste
* :ghissue:`725`: Added the implied heights for several common PP STASH codes.
* :ghissue:`705`: Add new stashcodes for dewpoint and RH.
* :ghissue:`714`: ENH: Added support for checking netcdf file saving
* :ghissue:`711`: Modified attribute behaviour for netCDF save.
* :ghissue:`522`: y-axis inversion for vertical coords
* :ghissue:`694`: reordered phenomenon dictionaries
* :ghissue:`697`: Fix coding of maximum and minimum time-stats in Grib2 save.
* :ghissue:`719`: Added more complete tmerc to cartopy translation
* :ghissue:`707`: Nearest neighbour indexing no longer loads the data.
* :ghissue:`721`: BF - allow model level GRIB files to be loaded.
* :ghissue:`720`: Can't load model level data in GRIB
* :ghissue:`718`: Fix for pp load rule with hybrid model level number
* :ghissue:`689`: Added support for loading regular gaussian grids from GRIB files.
* :ghissue:`702`: Make data checksum ignore masked values.
* :ghissue:`690`: Saved GRIB2 files lose coordinate system details
* :ghissue:`691`: BF - Correct Earth radius when saving GRIB2 files, fixes #690.
* :ghissue:`555`: Calculates correlation between cubes
* :ghissue:`619`: Varff
* :ghissue:`713`: Variable resolution fieldsfiles
* :ghissue:`710`: Update trajectory.py
* :ghissue:`708`: Update interpolate.py
* :ghissue:`692`: Alternative approach required for graphic testing
* :ghissue:`709`: allow interpolation of an array of values
* :ghissue:`669`: Relax NetCDF longitude circularity criteria.
* :ghissue:`684`: Limit unit calendar argument
* :ghissue:`643`: Added support for bool array indexing on a cube.
* :ghissue:`663`: ENH: Added support to setup.py for quick testing
* :ghissue:`496`: working wrapper for animating in iris
* :ghissue:`701`: PEP 8 iris.fileformats.grib.grib_save_rules
* :ghissue:`553`: Add ability to pass data dimensions for slicing cube
* :ghissue:`671`: BUG: Fix broken test result TestAtlanticProfiles
* :ghissue:`698`: lamb/polar coord units
* :ghissue:`680`: Local and global attributes
* :ghissue:`630`: abf for 32-bit arch
* :ghissue:`693`: DimCoord CML circular flag.
* :ghissue:`695`: Fix netCDF4 python package at version 1.0.2 for travis-ci
* :ghissue:`688`: verbose install of netcdf4 python package for testing
* :ghissue:`666`: standard --> gregorian
* :ghissue:`667`: Pep8 compliance of unit.py and whitespace removal
* :ghissue:`681`: Correct dim coord behaviour for aggregated by.
* :ghissue:`679`: Correct dim coord behaviour for aggregated by.
* :ghissue:`676`: Updated docs to use latest what's new.
* :ghissue:`610`: added 42-byte wmo header, requiring code tidy
* :ghissue:`625`: Percentile agg
* :ghissue:`672`: Cope with missing bounds
* :ghissue:`589`: Speedup PP load by caching rules action results on relevant parts of the field.
* :ghissue:`583`: Speedup PP rules cube construction.
* :ghissue:`659`: Fast Cell construction during merge
* :ghissue:`668`: Added Grib2 translations for 'soil_temperature'.
* :ghissue:`637`: Fast add coords
* :ghissue:`642`: Switch to using DimCoord.from_regular
* :ghissue:`653`: Add new ocean profile plot example.
* :ghissue:`662`: PEP 8 iris.symbols
* :ghissue:`661`: PEP 8 iris.config
* :ghissue:`657`: Use id() based lookup to find aux_coords.
* :ghissue:`658`: Fix indent typo in BitwiseInt
* :ghissue:`631`: TestPPSave for 32-bit arch
* :ghissue:`632`: test_ff for 32 -bit arch
* :ghissue:`641`: ne logic
* :ghissue:`639`: __ne__() logic
* :ghissue:`655`: MAINT: Refactoring of unit equality
* :ghissue:`635`: Global unit cache
* :ghissue:`654`: PEP 8 follow-up: fixup comments in iris.coord_categorisation
* :ghissue:`645`: PEP 8 iris.coord_categorisation
* :ghissue:`651`: ENH: Optimised unit equality special method
* :ghissue:`561`: Lambert conformal grib2 loading
* :ghissue:`649`: Support qplt.show in extest.
* :ghissue:`646`: PEP 8 iris.cube
* :ghissue:`644`: PEP 8 iris.aux_factory
* :ghissue:`650`: BUG: Fix pandas test for version update
* :ghissue:`636`: copyright dates 'out of date'
* :ghissue:`633`: Cell is a named tuple
* :ghissue:`648`: Avoid version-specific formatting issues by checking data directly.
* :ghissue:`647`: Remove workaround from filtering example.
* :ghissue:`629`: PEP 8 iris.quickplot
* :ghissue:`640`: BUG: Updated copyright dates on docs
* :ghissue:`626`: Convert iris.analysis.SUM to a weighted aggregator.
* :ghissue:`595`: Weighted sum operator
* :ghissue:`547`: grib hind doc
* :ghissue:`638`: BUG: Updated copyright dates for docs
* :ghissue:`617`: NF - add 1D scatter plot functions.
* :ghissue:`628`: PEP 8 iris.exceptions
* :ghissue:`627`: PEP 8 iris.palette and iris.pandas
* :ghissue:`634`: Update PPA to get a working libgeos-dev
* :ghissue:`622`: PEP 8 iris._merge
* :ghissue:`620`: PEP 8 iris._cube_coord_common
* :ghissue:`621`: PEP 8 iris.fileformats.ff
* :ghissue:`611`: PP/GRIB load rules as a function
* :ghissue:`618`: Convert post-item documentation to pre-item #: form
* :ghissue:`591`: FormatAgent is irrelevant
* :ghissue:`615`: Incorrect time axis units for `iris.quickplot.plot`
* :ghissue:`616`: Remove quickplot unit label for time reference coordinates. Closes #615.
* :ghissue:`462`: Added verbose functionality to is_compatible
* :ghissue:`580`: unit doc
* :ghissue:`614`: BUG: creating coordinate from another crs copy
* :ghissue:`607`: quickplot should provide a show method
* :ghissue:`613`: Add show to quickplot. Closes #607.
* :ghissue:`593`: Extend 1D plotting capabilities. Resolves #581.
* :ghissue:`581`: 1D plots with coord on vertical axis
* :ghissue:`429`: bluemarble demo broken
* :ghissue:`587`: Optimise constructing ordered metadata (merge path)
* :ghissue:`578`: Corrected filtered values in SOI example.
* :ghissue:`526`: Missing in_place keywords in analysis.math
* :ghissue:`577`: 2d plotting should handle date time axis
* :ghissue:`602`: Added missing in_place keywords in analysis.math
* :ghissue:`601`: Added Polar stereographic plot
* :ghissue:`599`: Adds Polar Stereo example code
* :ghissue:`600`: Error reading "colpex.pp"
* :ghissue:`592`: Revert version-specific CF standard-names usage in .travis.yml.
* :ghissue:`590`: memory usage when plotting through iris
* :ghissue:`597`: Simple PEP8 fix.
* :ghissue:`594`: Consistant title
* :ghissue:`559`: Merge 1.4.x changes into master branch
* :ghissue:`579`: name trajectory loading
* :ghissue:`588`: removed workaround to multiple coordinates been present when plotting.
* :ghissue:`586`: fix travis ref to new sample data
* :ghissue:`582`: Allow 1D plots to have the coordinate on the y-axis.
* :ghissue:`585`: Pyke Issue
* :ghissue:`570`: Updated example to work with newer sample data.
* :ghissue:`574`: Save (not export) to geotiff
* :ghissue:`576`: Pandas as_series should handle time coordinates
* :ghissue:`566`: added packaged grib support
* :ghissue:`332`: Read GRIB files with WMO bulletin headers
* :ghissue:`575`: PEP8 fixes for cartography module
* :ghissue:`571`: NF - Remove 1d restriction from cosine latitude weights
* :ghissue:`568`: Corrected rendering of code examples.
* :ghissue:`560`: Spherical Harmonic GRIB
* :ghissue:`543`: Stash std name
* :ghissue:`542`: Mr std name
* :ghissue:`564`: Fix to prevent unnecessary rebuild of pyke rules.
* :ghissue:`226`: Support for OS X Mountain Lion 
* :ghissue:`558`: Doc tweaks for custom season categorisation
* :ghissue:`528`: occasional failure of test_phenom_unknown 
* :ghissue:`557`: Reset warnings in test_phenom_unknown.
* :ghissue:`556`: Release v1.4.0
* :ghissue:`550`: Added support for loading NAME files
* :ghissue:`554`: local test failure: test_no_coord_system (iris.tests.test_analysis.TestProject)
* :ghissue:`552`: Switched from sample data to test data.
* :ghissue:`546`: PP save default gridded polar axis.
* :ghissue:`548`: Export to GeoTIFF refactor.
* :ghissue:`549`: gribapi 1.10.4
* :ghissue:`524`: Added support for checking coding standards of docs
* :ghissue:`535`: Export to GeoTIFF refactor.
* :ghissue:`545`: Broken GRIB load
* :ghissue:`544`: 1.4.0rc1 version number
* :ghissue:`540`: 1.4.0 changes and what's new
* :ghissue:`538`: MAIN: Removed all whitespace and file-end lines
* :ghissue:`541`: Added git clone depth of zero to travis CI config.
* :ghissue:`539`: Specified depth of zero for travis ci clone
* :ghissue:`534`: fix new licence failures
* :ghissue:`537`: update version to v1.5.0-dev
* :ghissue:`532`: whitespace removal
* :ghissue:`536`: license dates fix
* :ghissue:`533`: dummy change
* :ghissue:`518`: polar stereo grib
* :ghissue:`521`: Added netcdf load support for transverse Mercator grid mapping and climatology
* :ghissue:`531`: PEP8 fixes for aggregation tests
* :ghissue:`494`: updating iris release to use latest cartopy (master)
* :ghissue:`297`: Adds support for ieee 32bit fieldsfiles to iris.load
* :ghissue:`447`: aggregating dimensions of entirely masked data results erroneous values
* :ghissue:`319`: Cube.aggregated_by() removes mask
* :ghissue:`527`: BF - preserve masked arrays during aggregation
* :ghissue:`516`: GRIB missing data handling
* :ghissue:`514`: New pp rule to calculate forecast period.
* :ghissue:`468`: default expected coordinates used for plotting should be the dimension coordinates
* :ghissue:`448`: iris plot bug when certain aux and dim coords present
* :ghissue:`463`: Modified plot.py to pass coords arg through to _map_common()
* :ghissue:`482`: Revised grib load+save
* :ghissue:`523`: Moved to SHA for iris-test-data until we tag the next release.
* :ghissue:`520`: BF - handle missing values from grib messages
* :ghissue:`517`: Polar Stereographic CS
* :ghissue:`453`: Esmf conserve
* :ghissue:`485`: Plot bug when plotting on non-angular coordinate systems
* :ghissue:`515`: Fixed PEP8 issues in netcdf.py
* :ghissue:`503`: New pp save rule to handle missing forecast period
* :ghissue:`511`: Resolved pickling issues with deferred loading.
* :ghissue:`513`: Missing import
* :ghissue:`501`: Clarify cube.slices help information on coords_to_slice
* :ghissue:`510`: Documentation change on coordinates parameter for slices()
* :ghissue:`469`: Gracefully merge cubes with different history.
* :ghissue:`433`: UDUNITS2 correct bad timestamp clock time.
* :ghissue:`502`: nimrod file load only loads one field
* :ghissue:`507`: Load of nimrod files with multiple fields and period of interest
* :ghissue:`499`: Propogate all coord names in construct midpoint
* :ghissue:`508`: Modified linear() to handle non-scalar length one coords.
* :ghissue:`504`: Aggregate by on str dim err handling
* :ghissue:`369`: removal of cf conventions, title and history global attr, closes #365
* :ghissue:`479`: Make tests optional that depend on pydot and pandas (optional installs).
* :ghissue:`275`: Ubuntu 12.04 server build (similar to a travis VM) on top of cartopy build
* :ghissue:`461`: Aggregate by on str dim err handling
* :ghissue:`500`: izip docstring incorrect
* :ghissue:`452`: g1_level1 handle surface level type
* :ghissue:`498`: PP save with no time forecast.
* :ghissue:`495`: Removal of strict constraint in izip().
* :ghissue:`490`: Refactor seasons categorisation functions.
* :ghissue:`497`: propogate all coord names in construct midpoint
* :ghissue:`492`: Warn coord present
* :ghissue:`491`: PEP8 corrections to iterate.py.
* :ghissue:`464`: Area weighted regridding
* :ghissue:`443`: added gallery example calculating distance using pyproj
* :ghissue:`480`: Unambiguous season year naming
* :ghissue:`428`: Add an optimisation for single valued coordinate constraints.
* :ghissue:`488`: fixed netcdf save cubelist bug
* :ghissue:`460`: Unambiguous season year naming
* :ghissue:`487`: Fix to copyright licence check logic
* :ghissue:`475`: Cube merge string coords.
* :ghissue:`486`: Fixed copyright date of an experimental file.
* :ghissue:`483`: Fixed copyright date on one of the tests.
* :ghissue:`477`: Added a test for license checking of code files. Closes #454.
* :ghissue:`454`: Add licence header check to tests
* :ghissue:`471`: gribsave int32 scalar
* :ghissue:`281`: XYT cube movie
* :ghissue:`470`: Unit.convert vs np.int32
* :ghissue:`325`: Is cube.metadata valid?
* :ghissue:`256`: Errors when testing with no data (-n) flag
* :ghissue:`69`: Use of CF variable names
* :ghissue:`112`: reconsider map_setup
* :ghissue:`130`: Convert "Note:" to rst
* :ghissue:`179`: poor pcolormesh exception
* :ghissue:`189`: NetCDF File Export with multiple Cubes
* :ghissue:`219`: guess_bounds in coord creation
* :ghissue:`318`: travis : full tests
* :ghissue:`366`: What's new page for upstream/master
* :ghissue:`407`: Allowing NetCDF4 to open OPeNDAP URLS
* :ghissue:`444`: The Travis-CI tests are skipping the cfchecker tests
* :ghissue:`431`: Cannot write GRIB2 file with integer pressure level.
* :ghissue:`476`: Added a PEP8 test to ensure coding standards.
* :ghissue:`424`: Added geotiff to what's new v1.4
* :ghissue:`422`: Working OpenDAP functionality.
* :ghissue:`451`: Added depth rules for bounds.
* :ghissue:`162`: Coord.__deepcopy__()
* :ghissue:`472`: Modified logic in Unit.convert to handle numpy scalars.
* :ghissue:`455`: Esmf trytravis
* :ghissue:`199`: nimrod level type 12
* :ghissue:`473`: Pandas 0.11
* :ghissue:`439`: pandas 1d
* :ghissue:`427`: Pep8 init and constraints
* :ghissue:`446`: grid map variable type in NetCDF load requires specific type
* :ghissue:`401`: unit print syntax
* :ghissue:`465`: change unit print-out syntax
* :ghissue:`459`: Added tolerances to Coord.is_contiguous()
* :ghissue:`71`: iris pp loading
* :ghissue:`457`: name=
* :ghissue:`449`: Added cfchecker to travis-ci
* :ghissue:`435`: Safely build PyKE rule base.
* :ghissue:`442`: Add bilinear interpolation between rectilinear grids.
* :ghissue:`440`: Added iris.tests.assertArrayAllClose 
* :ghissue:`441`: Fixed ref to old scitools.org page.
* :ghissue:`436`: Travis-CI non git clone of Cartopy.
* :ghissue:`437`: Helper function for regridding that returns x and y coords.
* :ghissue:`434`: Force mpl v1.2.0 for Travis-CI.
* :ghissue:`426`: V1x3 release
* :ghissue:`193`: CubeList slices
* :ghissue:`383`: GeoTiff export with test
* :ghissue:`420`: Fixed sphinx error when building netcdf save docs
* :ghissue:`415`: Add dtype and flags to CML
* :ghissue:`418`: Update to iris-test-data commit in .travis.yml
* :ghissue:`417`: Make Cube.data deal in views.
* :ghissue:`419`: Removed data_repository from site.cfg.template
* :ghissue:`406`: Simplified resource configuration
* :ghissue:`416`: Remove redundant code.
* :ghissue:`414`: Remove redundant method definition
* :ghissue:`391`: Rework Coord nearest_neighbour calc to support circular cases.
* :ghissue:`411`: Explicitly set the SHAs of dependencies in the travis.yml file.
* :ghissue:`413`: Experimental convenience hooks
* :ghissue:`220`: Missing exponential function in anal. math.
* :ghissue:`305`: pyproj dependency removal
* :ghissue:`408`: Renamed test data directory
* :ghissue:`412`: Use grib-api from launchpad in the travis.yml configuration.
* :ghissue:`409`: Update changelog and what's new pages for 1.3
* :ghissue:`367`: Saving multiple cubes to a netcdf
* :ghissue:`410`: Moved from master to a specific commit for iris test data.
* :ghissue:`404`: problem with UDUNITS2 after install
* :ghissue:`398`: pp.save always writes zeros for missing values
* :ghissue:`405`: Return a CubeList when slicing a CubeList
* :ghissue:`378`: Switch to biggus for deferred loading (Experimental!)
* :ghissue:`402`: cube history attribute, preserve information
* :ghissue:`224`: Export to geotiff
* :ghissue:`385`: Iris citation
* :ghissue:`397`: corrected syntax for named tuple def
* :ghissue:`393`: Copy button in docs.
* :ghissue:`376`: Metaclass to disable cf profile for testing.
* :ghissue:`399`: Update to README.md to include Travis-CI build status.
* :ghissue:`390`: fixed converting reference times
* :ghissue:`336`: Cube concatenate.
* :ghissue:`359`: add exponential to math
* :ghissue:`392`: travis (full)
* :ghissue:`389`: Iris citation added into user guide
* :ghissue:`252`: Coord.unit_converted limitation
* :ghissue:`387`: Better history from multiple addition
* :ghissue:`312`: wrapped extract
* :ghissue:`384`: small data
* :ghissue:`382`: Removed most redundant imports
* :ghissue:`381`: pep8 coords.py
* :ghissue:`380`: version 1.2.1-dev
* :ghissue:`379`: V1.2.0 release
* :ghissue:`377`: V1.2.0 release
* :ghissue:`375`: V1.2.0 release
* :ghissue:`368`: NetCDF CF profile support.
* :ghissue:`372`: First attempt at GeoTiff export for early review
* :ghissue:`371`: Attempt to fix pep8 issues in cube.py
* :ghissue:`374`: pep8 and various re-formatting
* :ghissue:`277`: Saving to PP and reading back scrambles time coordinates.
* :ghissue:`373`: Bug fix of pp saving, closses #277
* :ghissue:`370`: Added missing blank lines after sphinx directives.
* :ghissue:`361`: Modified cube.summary() to handle unicode cube attributes.
* :ghissue:`362`: Refactored is_x boolean methods in unit module.
* :ghissue:`261`: Changes from PR.212 -- those already ok-ed.
* :ghissue:`360`: turning single cube into a cubelist
* :ghissue:`355`: Fixed cube.summary() to handle unicode cube attributes.
* :ghissue:`350`: Basic support for varying longitude ranges.
* :ghissue:`358`: Prevented the ability to add AuxCoord instances to the dim_coords.
* :ghissue:`337`: Prevented the ability to add AuxCoord instances to the dim_coords.
* :ghissue:`357`: Use position independent code flag for compilation of gribapi
* :ghissue:`354`: Default tests to the 'agg' matplotlib backend.
* :ghissue:`313`: Rm pyproj dep
* :ghissue:`110`: refactor: one test per test
* :ghissue:`234`: Erroneous use of len(coord)
* :ghissue:`349`: Corrected len(coord) to len(coord.points) and added test
* :ghissue:`41`: Improve the error message when pcolormeshing a non-bounded cube
* :ghissue:`172`: pcol msg fix
* :ghissue:`347`: Revert Iris version from 1.2.0-rc1 back to 1.2.0-dev.
* :ghissue:`238`: Abf loading
* :ghissue:`343`: Changes for rc1.
* :ghissue:`342`: Increment version number
* :ghissue:`203`: nimrod orography loading
* :ghissue:`206`: NIMROD NG coord names
* :ghissue:`317`: Added support for CF variable names to cubes and coords
* :ghissue:`101`: Avoid obscure crash when loading from empty file
* :ghissue:`181`: colorbar ticks
* :ghissue:`268`: Changes to cartography.py, copied from PR.212
* :ghissue:`180`: GRIB1 duplicate rule
* :ghissue:`265`: Changes to loading_iris_cubes.rst, copied from PR.212
* :ghissue:`173`: Empty file handing
* :ghissue:`323`: datetime human-read + cube unit print
* :ghissue:`338`: Standardising numpy namespaces
* :ghissue:`322`: Update to Iris software stack versions.
* :ghissue:`334`: Added convert_units() to cubes and coords.
* :ghissue:`327`: Prevented the ability to add AuxCoord instances to the dim_coords.
* :ghissue:`335`: Documenting the review process of a pull request
* :ghissue:`331`: Defer format-specific imports
* :ghissue:`333`: Remove obsolete unused code (very small)
* :ghissue:`174`: Grib hindcast workaround
* :ghissue:`295`: Document site.cfg usage
* :ghissue:`324`: extests: remove unused import unittest
* :ghissue:`326`: Unused "import unittest" removed from extests
* :ghissue:`320`: Changing units
* :ghissue:`316`: Updated the extests to easily support image tolerances.
* :ghissue:`284`: Changing units
* :ghissue:`311`: Consistent Cell-Cell ordering, and seperate Cell-scalar semantics.
* :ghissue:`86`: Remove the iterability of a cube
* :ghissue:`273`: control cube iteration
* :ghissue:`310`: Fixed issue introduced by #300.
* :ghissue:`315`: Document sitecfg patch
* :ghissue:`307`: Document site.cfg
* :ghissue:`309`: Added an example of rolling_window with weights.
* :ghissue:`304`: Travis nodata
* :ghissue:`258`: "lone_name"
* :ghissue:`289`: iris._constraints._ColumnIndexManager description
* :ghissue:`306`: full data cml
* :ghissue:`300`: Supported merging of accumulations.
* :ghissue:`296`: Added a simple rule, which needed a lot more work to test it!.
* :ghissue:`285`: NF - allow weights in rolling window aggregations
* :ghissue:`288`: Nimrod xy names
* :ghissue:`293`: travis
* :ghissue:`177`: Docs still mention "Basemap"
* :ghissue:`302`: Missed points in previous merges
* :ghissue:`301`: Travis nodata
* :ghissue:`207`: Plot 2dxy
* :ghissue:`290`: Docs still mention "Basemap" (new rm basemap)
* :ghissue:`83`: Rolling cubes
* :ghissue:`294`: Fixed lone typo in PP rules. Closes #258.
* :ghissue:`242`: Rm basemap doc
* :ghissue:`286`: NF - normalized area weights
* :ghissue:`221`: g1 rule update
* :ghissue:`280`: address grib api import
* :ghissue:`282`: Deal with netCDF non-monotonic and masked coordinates.
* :ghissue:`283`: Graceful PP loading of invalid units.
* :ghissue:`247`: NF - add new weighting methods
* :ghissue:`279`: Faster monotonic check
* :ghissue:`278`: Faster monotonic check
* :ghissue:`274`: Add support for root mean square (RMS)
* :ghissue:`276`: Analysis root mean square calculation. Closes issue #274.
* :ghissue:`239`: externalise tests -5
* :ghissue:`272`: Ocean depth PP rule and minor merge optimisation.
* :ghissue:`270`: Added missing @iris.tests.skip_data decorators.
* :ghissue:`184`: Grib1unit10 additional
* :ghissue:`235`: Externalise tests take-4
* :ghissue:`263`: Changes to cube_statistics.rst, copied from PR.212
* :ghissue:`266`: Changes to whats_new.rst, copied from PR.212
* :ghissue:`269`: A series of small but important fixes to the docs.
* :ghissue:`191`: trajectory.interpolate aux factory bug
* :ghissue:`267`: remove '1.0'
* :ghissue:`152`: Using units.py on OSX
* :ghissue:`230`: iris.save fails for trajectories (NetCDF)
* :ghissue:`264`: Update documentation.
* :ghissue:`262`: Updated minimum dependency of cartopy to v0.5.
* :ghissue:`260`: Updated minimum dependency of cartopy to v0.5.
* :ghissue:`212`: Old doc changes
* :ghissue:`204`: Implicit out of place guess_bounds when using pcolor and pcolormesh
* :ghissue:`254`: save netcdf traj
* :ghissue:`209`: Hybrid height trajectory
* :ghissue:`200`: Grib steps bug
* :ghissue:`250`: NF - new coord categorisation functions
* :ghissue:`255`: BF - fix typo in udunits2 library locator
* :ghissue:`246`: unit.py on osx
* :ghissue:`253`: Coordinate CML id sorted for attribute dictionary.
* :ghissue:`231`: Externalise tests take-2 : "test_merge" + "test_hybrid"
* :ghissue:`154`: lbtim.ib == 0 missing from pp_rules
* :ghissue:`228`: Fieldsfiles magic numbers
* :ghissue:`243`: Fieldsfile ancillary loading capability with PP LBTIM=2 support.
* :ghissue:`225`: Constraint extract slice optimisation.
* :ghissue:`237`: PP load optimisation.
* :ghissue:`249`: Fixed failing doctest introduced by #248
* :ghissue:`248`: Added lbtim.ib == 3 forecast_reference_time rule
* :ghissue:`175`: Tidy XML tests
* :ghissue:`183`: Added forecast reference time to a field with lbtim of 11 (one of the most common lbtim-s)
* :ghissue:`194`: BF - fix broadcasting of area weights
* :ghissue:`198`: NetCDF loading time breakdown
* :ghissue:`245`: Fixed lowercase github link.
* :ghissue:`216`: Deferred loading of AuxCoord points/bounds.
* :ghissue:`178`: add_dim_coord Documentation is misleading
* :ghissue:`217`: Changed data_dim to an argument not keyword
* :ghissue:`241`: Rm basemap doc #177
* :ghissue:`240`: Rm basemap doc
* :ghissue:`229`: grib save : minor update
* :ghissue:`236`: Improved userguide page on cube statistics.
* :ghissue:`232`: Externalise tests take-3 : test_plot
* :ghissue:`218`: Faster masked array creation in DataManager.
* :ghissue:`222`: Export to geotiff
* :ghissue:`215`: pretty cubelist extraction
* :ghissue:`163`: various documentation tweaks
* :ghissue:`213`: Cache netCDF attributes
* :ghissue:`210`: Fieldsfile and PP support for unpacked and CRAY 32-bit packed data.
* :ghissue:`208`: Externalise tests
* :ghissue:`98`: Moved examples to lib/iris/examples which is now importable.
* :ghissue:`205`: Track Cartopy version
* :ghissue:`104`: Cell comparison inconsistency
* :ghissue:`121`: new cell comparison
* :ghissue:`195`: Add CONTRIBUTING.md
* :ghissue:`100`: Add CONTRIBUTING[.md]
* :ghissue:`190`: Remove redundant and erroneous standard_name to LBFC/STASH PP save rules.
* :ghissue:`171`: grib1 unit 10 handling
* :ghissue:`118`: IO rule representation
* :ghissue:`161`: Used cartopy's new wrapping functionality when using pcolormesh.
* :ghissue:`169`: Subplot multiprocessing
* :ghissue:`151`: Grib hindcast workaround
* :ghissue:`165`: empty file handling
* :ghissue:`166`: pcol msg fix
* :ghissue:`168`: grib1 unit 10 handling
* :ghissue:`164`: Merge v1.0.x back into master
* :ghissue:`131`: Remove obsolete "unrotate" testing code
* :ghissue:`140`: logo
* :ghissue:`157`: logo
* :ghissue:`160`: Update Iris version for bug-fix development.
* :ghissue:`159`: Release version 1.0.0
* :ghissue:`158`: Autodoc fix
* :ghissue:`156`: Update to visual test result following orthographic fix in cartopy.
* :ghissue:`106`: iris.analysis.interpolate.linear breaks masked array
* :ghissue:`155`: linear: numpy masked append workaround (replacement pr)
* :ghissue:`123`: linear: numpy masked append workaround
* :ghissue:`153`: Gitwash build and Iris URL update.
* :ghissue:`141`: Avoid unnecessary analysis imports.
* :ghissue:`148`: NetCDF units longitude/latitude
* :ghissue:`150`: Update test_plot.test_missing_cs expected png result for non-brewer
* :ghissue:`149`: Grib hindcast workaround
* :ghissue:`144`: Undo mistaken version change on master.
* :ghissue:`147`: Revert version back to 1.0.0-dev after rc1 tag
* :ghissue:`146`: Release version 1.0.0rc1
* :ghissue:`145`: Release 1.0.0 documentation changes.
* :ghissue:`143`: Release version 1.0.0rc1
* :ghissue:`133`: Updated userguide's definition of DimCoord
* :ghissue:`142`: Release 1.0.0 documentation changes.
* :ghissue:`136`: Map setup purge
* :ghissue:`138`: bug fix: plot with no cs
* :ghissue:`139`: A quick word about future plans
* :ghissue:`137`: Turn off auto palette selection of brewer
* :ghissue:`73`: Remove CF default ellipsoid
* :ghissue:`134`: grib1 time bound
* :ghissue:`111`: Added cube project function to iris.analysis.cartography
* :ghissue:`120`: no default cf cs
* :ghissue:`97`: Axis unit labels
* :ghissue:`85`: collapsing a multidimensional coordinate with bounds
* :ghissue:`14`: Plot cube over OS map
* :ghissue:`15`: Use cartopy for maps
* :ghissue:`84`: Nimrod plotting
* :ghissue:`78`: New load/merge API
* :ghissue:`128`: Minor correction to what's new and CHANGES, plus apostrophe.
* :ghissue:`103`: Contour levels on a wrapped map
* :ghissue:`124`: Use of tuples for dimensions in coord to dim mapping
* :ghissue:`116`: Nimrod loader patch.
* :ghissue:`126`: cartopy replaces basemap
* :ghissue:`119`: Pycarto++
* :ghissue:`125`: What's new
* :ghissue:`122`: collapse nd bounds
* :ghissue:`115`: Pycarto++
* :ghissue:`67`: cartopy replaces basemap
* :ghissue:`117`: Consistent order in idiff
* :ghissue:`114`: Extend attribute comparison to deal with NumPy arrays
* :ghissue:`113`: Addition of cartopy_crs method to CoordSystem.
* :ghissue:`109`: Mpl 1.2 rc2
* :ghissue:`108`: Added real image testing.
* :ghissue:`48`: Cynthia Brewer colour schemes
* :ghissue:`96`: Re-introduce Cynthia Brewer palettes.
* :ghissue:`1`: Hybrid pressure
* :ghissue:`74`: grib hybrid pressure
* :ghissue:`95`: First cut of hybrid-pressure for GRIB.
* :ghissue:`92`: Remove test_trui, and cut the fat from test_uri_callback.py.
* :ghissue:`94`: Fixes to unguarded access to circular attribute.
* :ghissue:`64`: Test cleanup
* :ghissue:`93`: Allow GRIB loader to handle cross-references
* :ghissue:`90`: Add copyright/licence header check
* :ghissue:`91`: Removal of test data from test_iterate.py
* :ghissue:`76`: doctest failure
* :ghissue:`88`: Speed-up & tidy test_cube_to_pp
* :ghissue:`89`: Change to cube summary for cubes with bounded scalar coords
* :ghissue:`62`: Record dimension when writing NETCDF files with Iris
* :ghissue:`82`: setup.py fails on Windows due to incorrect handling of sub-directories for packages
* :ghissue:`13`: Load NIMROD data
* :ghissue:`54`: NIMROD loading
* :ghissue:`81`: Update change log for release v0.9.1
* :ghissue:`80`: CF file handle bug fix.
* :ghissue:`72`: Fix to bug causing files to remain open after loading netCDF files.
* :ghissue:`45`: Convert coord-systems to CF
* :ghissue:`65`: CF CoordSystems (replacement pull)
* :ghissue:`68`: Exposed merge keyword argument to public load api.
* :ghissue:`70`: Speed up the CDM tests...
* :ghissue:`20`: Update installation data file list
* :ghissue:`12`: Remove symlinks
* :ghissue:`63`: Make netCDF save outermost dimension as unlimited
* :ghissue:`66`: Clean up of setup.py, install docs and removal of symlinks
* :ghissue:`33`: installation instructions
* :ghissue:`52`: CF coord systems
* :ghissue:`43`: Error in plot.py docstrings
* :ghissue:`46`: Hybrid pressure
* :ghissue:`56`: plot doc error
* :ghissue:`11`: Review iris.util.guess_coord_axis() behaviour
* :ghissue:`60`: iris.util.guess_coord_axis() behaviour
* :ghissue:`44`: cube_maths documentation typo
* :ghissue:`55`: math doc typo
* :ghissue:`9`: Remove Coord trig methods
* :ghissue:`24`: Deprecation of coord trig methods
* :ghissue:`23`: Installation of additional files
* :ghissue:`10`: Convert source/history to attributes
* :ghissue:`39`: Convert source from coord to attribute. Fixes #10
* :ghissue:`42`: Build of gh-pages for iris release v0.9.
* :ghissue:`40`: Cube.aggregated_by() broken
* :ghissue:`34`: Sample data usage
* :ghissue:`38`: Removed details of earlier api change from docs.
* :ghissue:`36`: 0.9 change log
* :ghissue:`26`: Remove TEST_COMPAT code
* :ghissue:`27`: Unsighlty unicode 'u' in warnings
* :ghissue:`28`: Pyke rules unicode warning tweak. Fixes #27.
* :ghissue:`31`: Removal of TEST_COMPAT and OLD_XML code
* :ghissue:`35`: Update the changelog for a 0.9 release.
* :ghissue:`19`: Sampledata
* :ghissue:`32`: For #30, update gitwash folder
* :ghissue:`30`: Fix gitwash docs
* :ghissue:`29`: patch for IE9 support
* :ghissue:`7`: Cube merge dimension hint
* :ghissue:`8`: Cube merge dimension order
* :ghissue:`25`: Cube merge dimension hint and ordering.
* :ghissue:`22`: Automatic dimension choosing hints for merge.
* :ghissue:`6`: Transition to new XML format
* :ghissue:`16`: Fixed unit usage and updated a warning to be more explicit.
* :ghissue:`21`: Use new xml
* :ghissue:`5`: Cube summary misalignment and coordinate name clipping. Fixes #4.
* :ghissue:`17`: Graceful netCDF units loading.
* :ghissue:`4`: Cube summary misalignment.
* :ghissue:`2`: Example data
* :ghissue:`3`: Publish documentation
