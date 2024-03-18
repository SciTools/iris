# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.

"""Extensions to Iris' CF variable representation to represent CF UGrid variables.

Eventual destination: :mod:`iris.fileformats.cf`.

"""

import warnings

from ...fileformats import cf
from ...warnings import IrisCfLabelVarWarning, IrisCfMissingVarWarning
from .mesh import Connectivity


class CFUGridConnectivityVariable(cf.CFVariable):
    """A CF_UGRID connectivity variable.

    A CF_UGRID connectivity variable points to an index variable identifying
    for every element (edge/face/volume) the indices of its corner nodes. The
    connectivity array will thus be a matrix of size n-elements x n-corners.
    For the indexing one may use either 0- or 1-based indexing; the convention
    used should be specified using a ``start_index`` attribute to the index
    variable.

    For face elements: the corner nodes should be specified in anticlockwise
    direction as viewed from above. For volume elements: use the
    additional attribute ``volume_shape_type`` which points to a flag variable
    that specifies for every volume its shape.

    Identified by a CF-netCDF variable attribute equal to any one of the values
    in :attr:`~iris.experimental.ugrid.mesh.Connectivity.UGRID_CF_ROLES`.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    cf_identity = NotImplemented
    cf_identities = Connectivity.UGRID_CF_ROLES

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF-UGRID connectivity variables.
        for nc_var_name, nc_var in target.items():
            # Check for connectivity variable references, iterating through
            # the valid cf roles.
            for identity in cls.cf_identities:
                nc_var_att = getattr(nc_var, identity, None)

                if nc_var_att is not None:
                    # UGRID only allows for one of each connectivity cf role.
                    name = nc_var_att.strip()
                    if name not in ignore:
                        if name not in variables:
                            message = (
                                f"Missing CF-UGRID connectivity variable "
                                f"{name}, referenced by netCDF variable "
                                f"{nc_var_name}"
                            )
                            if warn:
                                warnings.warn(message, category=IrisCfMissingVarWarning)
                        else:
                            # Restrict to non-string type i.e. not a
                            # CFLabelVariable.
                            if not cf._is_str_dtype(variables[name]):
                                result[name] = CFUGridConnectivityVariable(
                                    name, variables[name]
                                )
                            else:
                                message = (
                                    f"Ignoring variable {name}, identified "
                                    f"as a CF-UGRID connectivity - is a "
                                    f"CF-netCDF label variable."
                                )
                                if warn:
                                    warnings.warn(
                                        message, category=IrisCfLabelVarWarning
                                    )

        return result


class CFUGridAuxiliaryCoordinateVariable(cf.CFVariable):
    """A CF-UGRID auxiliary coordinate variable.

    A CF-UGRID auxiliary coordinate variable is a CF-netCDF auxiliary
    coordinate variable representing the element (node/edge/face/volume)
    locations (latitude, longitude or other spatial coordinates, and optional
    elevation or other coordinates). These auxiliary coordinate variables will
    have length n-elements.

    For elements other than nodes, these auxiliary coordinate variables may
    have in turn a ``bounds`` attribute that specifies the bounding coordinates
    of the element (thereby duplicating the data in the ``node_coordinates``
    variables).

    Identified by the CF-netCDF variable attribute
    ``node_``/``edge_``/``face_``/``volume_coordinates``.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    cf_identity = NotImplemented
    cf_identities = [
        "node_coordinates",
        "edge_coordinates",
        "face_coordinates",
        "volume_coordinates",
    ]

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify any CF-UGRID-relevant auxiliary coordinate variables.
        for nc_var_name, nc_var in target.items():
            # Check for UGRID auxiliary coordinate variable references.
            for identity in cls.cf_identities:
                nc_var_att = getattr(nc_var, identity, None)

                if nc_var_att is not None:
                    for name in nc_var_att.split():
                        if name not in ignore:
                            if name not in variables:
                                message = (
                                    f"Missing CF-netCDF auxiliary coordinate "
                                    f"variable {name}, referenced by netCDF "
                                    f"variable {nc_var_name}"
                                )
                                if warn:
                                    warnings.warn(
                                        message,
                                        category=IrisCfMissingVarWarning,
                                    )
                            else:
                                # Restrict to non-string type i.e. not a
                                # CFLabelVariable.
                                if not cf._is_str_dtype(variables[name]):
                                    result[name] = CFUGridAuxiliaryCoordinateVariable(
                                        name, variables[name]
                                    )
                                else:
                                    message = (
                                        f"Ignoring variable {name}, "
                                        f"identified as a CF-netCDF "
                                        f"auxiliary coordinate - is a "
                                        f"CF-netCDF label variable."
                                    )
                                    if warn:
                                        warnings.warn(
                                            message,
                                            category=IrisCfLabelVarWarning,
                                        )

        return result


class CFUGridMeshVariable(cf.CFVariable):
    """A CF-UGRID mesh variable is a dummy variable for storing topology information as attributes.

    A CF-UGRID mesh variable is a dummy variable for storing topology
    information as attributes. The mesh variable has the ``cf_role``
    'mesh_topology'.

    The UGRID conventions describe define the mesh topology as the
    interconnection of various geometrical elements of the mesh. The pure
    interconnectivity is independent of georeferencing the individual
    geometrical elements, but for the practical applications for which the
    UGRID CF extension is defined, coordinate data will always be added.

    Identified by the CF-netCDF variable attribute 'mesh'.

    .. seealso::

        The UGRID Conventions, https://ugrid-conventions.github.io/ugrid-conventions/

    """

    cf_identity = "mesh"

    @classmethod
    def identify(cls, variables, ignore=None, target=None, warn=True):
        result = {}
        ignore, target = cls._identify_common(variables, ignore, target)

        # Identify all CF-UGRID mesh variables.
        all_vars = target == variables
        for nc_var_name, nc_var in target.items():
            if all_vars:
                # SPECIAL BEHAVIOUR FOR MESH VARIABLES.
                # We are looking for all mesh variables. Check if THIS variable
                #  is a mesh using its own attributes.
                if getattr(nc_var, "cf_role", "") == "mesh_topology":
                    result[nc_var_name] = CFUGridMeshVariable(nc_var_name, nc_var)

            # Check for mesh variable references.
            nc_var_att = getattr(nc_var, cls.cf_identity, None)

            if nc_var_att is not None:
                # UGRID only allows for 1 mesh per variable.
                name = nc_var_att.strip()
                if name not in ignore:
                    if name not in variables:
                        message = (
                            f"Missing CF-UGRID mesh variable {name}, "
                            f"referenced by netCDF variable {nc_var_name}"
                        )
                        if warn:
                            warnings.warn(message, category=IrisCfMissingVarWarning)
                    else:
                        # Restrict to non-string type i.e. not a
                        # CFLabelVariable.
                        if not cf._is_str_dtype(variables[name]):
                            result[name] = CFUGridMeshVariable(name, variables[name])
                        else:
                            message = (
                                f"Ignoring variable {name}, identified as a "
                                f"CF-UGRID mesh - is a CF-netCDF label "
                                f"variable."
                            )
                            if warn:
                                warnings.warn(message, category=IrisCfLabelVarWarning)

        return result


class CFUGridGroup(cf.CFGroup):
    """Represents a collection of CF Metadata Conventions variables and netCDF global attributes.

    Represents a collection of 'NetCDF Climate and Forecast (CF) Metadata
    Conventions' variables and netCDF global attributes.

    Specialisation of :class:`~iris.fileformats.cf.CFGroup` that includes extra
    collections for CF-UGRID-specific variable types.

    """

    @property
    def connectivities(self):
        """Collection of CF-UGRID connectivity variables."""
        return self._cf_getter(CFUGridConnectivityVariable)

    @property
    def ugrid_coords(self):
        """Collection of CF-UGRID-relevant auxiliary coordinate variables."""
        return self._cf_getter(CFUGridAuxiliaryCoordinateVariable)

    @property
    def meshes(self):
        """Collection of CF-UGRID mesh variables."""
        return self._cf_getter(CFUGridMeshVariable)

    @property
    def non_data_variable_names(self):
        """:class:`set` of names of the CF-netCDF/CF-UGRID variables that are not the data pay-load."""
        extra_variables = (self.connectivities, self.ugrid_coords, self.meshes)
        extra_result = set()
        for variable in extra_variables:
            extra_result |= set(variable)
        return super().non_data_variable_names | extra_result


class CFUGridReader(cf.CFReader):
    """Allows the contents of a netCDF file to be.

    This class allows the contents of a netCDF file to be interpreted according
    to the 'NetCDF Climate and Forecast (CF) Metadata Conventions'.

    Specialisation of :class:`~iris.fileformats.cf.CFReader` that can also
    handle CF-UGRID-specific variable types.

    """

    _variable_types = cf.CFReader._variable_types + (
        CFUGridConnectivityVariable,
        CFUGridAuxiliaryCoordinateVariable,
        CFUGridMeshVariable,
    )

    CFGroup = CFUGridGroup
