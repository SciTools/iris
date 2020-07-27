# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from collections import namedtuple
from collections.abc import Iterable
import logging

import numpy as np

from iris.common import LENIENT


__all__ = ["Resolve"]


# Configure the logger.
logger = logging.getLogger(__name__)


_AuxCoverage = namedtuple(
    "AuxCoverage",
    [
        "cube",
        "common_items_aux",
        "common_items_scalar",
        "local_items_aux",
        "local_items_scalar",
        "dims_common",
        "dims_local",
        "dims_free",
    ],
)

_DimCoverage = namedtuple(
    "DimCoverage",
    ["cube", "metadata", "coords", "dims_common", "dims_local", "dims_free"],
)

_Item = namedtuple("Item", ["metadata", "coord", "dims"])

_CategoryItems = namedtuple(
    "CategoryItems", ["items_dim", "items_aux", "items_scalar"],
)

_PreparedFactory = namedtuple("PreparedFactory", ["container", "dependencies"])

_PreparedItem = namedtuple(
    "PreparedItem", ["metadata", "points", "bounds", "dims", "container"],
)

_PreparedMetadata = namedtuple("PreparedMetadata", ["combined", "src", "tgt"])


class Resolve:
    def __init__(self, lhs=None, rhs=None):
        if lhs is not None or rhs is not None:
            self(lhs, rhs)

    def __call__(self, lhs, rhs):
        self._init(lhs, rhs)

        self._metadata_resolve()
        self._metadata_coverage()

        if self._debug:
            self.show_dim(self.lhs_cube_dim_coverage)
            self.show_dim(self.rhs_cube_dim_coverage)
            self.show_aux(self.lhs_cube_aux_coverage)
            self.show_aux(self.rhs_cube_aux_coverage)
            self.show_items(self.lhs_cube_category_local, title="LHS local")
            self.show_items(self.rhs_cube_category_local, title="RHS local")
            self.show_items(self.category_common, title="common")
            print(f"map_rhs_to_lhs: {self.map_rhs_to_lhs}")

        self._metadata_mapping()
        self._metadata_prepare()

        if self._debug:
            self.show_prepared()

    @staticmethod
    def _aux_coverage(
        cube,
        cube_items_aux,
        cube_items_scalar,
        common_aux_metadata,
        common_scalar_metadata,
    ):
        common_items_aux = []
        common_items_scalar = []
        local_items_aux = []
        local_items_scalar = []
        dims_common = []
        dims_local = []
        dims_free = set(range(cube.ndim))

        for item in cube_items_aux:
            [dims_free.discard(dim) for dim in item.dims]

            if item.metadata in common_aux_metadata:
                common_items_aux.append(item)
                dims_common.extend(item.dims)
            else:
                local_items_aux.append(item)
                dims_local.extend(item.dims)

        for item in cube_items_scalar:
            if item.metadata in common_scalar_metadata:
                common_items_scalar.append(item)
            else:
                local_items_scalar.append(item)

        return _AuxCoverage(
            cube=cube,
            common_items_aux=common_items_aux,
            common_items_scalar=common_items_scalar,
            local_items_aux=local_items_aux,
            local_items_scalar=local_items_scalar,
            dims_common=sorted(set(dims_common)),
            dims_local=sorted(set(dims_local)),
            dims_free=sorted(dims_free),
        )

    def _aux_mapping(self, src_coverage, tgt_coverage):
        for tgt_item in tgt_coverage.common_items_aux:
            # Search for a src aux metadata match.
            tgt_metadata = tgt_item.metadata
            src_items = tuple(
                filter(
                    lambda src_item: src_item.metadata == tgt_metadata,
                    src_coverage.common_items_aux,
                )
            )
            if src_items:
                # Multiple matching src metadata must cover the same src
                # dimensions.
                src_dims = src_items[0].dims
                if all(map(lambda item: item.dims == src_dims, src_items)):
                    # Ensure src and tgt have equal rank.
                    tgt_dims = tgt_item.dims
                    if len(src_dims) == len(tgt_dims):
                        for src_dim, tgt_dim in zip(src_dims, tgt_dims):
                            self.mapping[src_dim] = tgt_dim
                            logger.debug(f"{src_dim}->{tgt_dim}")
            else:
                # This situation can only occur due to a systemic internal
                # failure to correctly identify common aux coordinate metadata
                # coverage between the cubes.
                emsg = (
                    "Failed to map common aux coordinate metadata from "
                    "source cube {!r} to target cube {!r}, using {!r} on "
                    "target cube dimension{} {}."
                )
                raise ValueError(
                    emsg.format(
                        src_coverage.cube.name(),
                        tgt_coverage.cube.name(),
                        tgt_metadata,
                        "s" if len(tgt_item.dims) > 1 else "",
                        tgt_item.dims,
                    )
                )

    @staticmethod
    def _categorise_items(cube):
        category = _CategoryItems(items_dim=[], items_aux=[], items_scalar=[])

        # Categorise the dim coordinates of the cube.
        for coord in cube.dim_coords:
            item = _Item(
                metadata=coord.metadata,
                coord=coord,
                dims=cube.coord_dims(coord),
            )
            category.items_dim.append(item)

        # Categorise the aux and scalar coordinates of the cube.
        for coord in cube.aux_coords:
            dims = cube.coord_dims(coord)
            item = _Item(metadata=coord.metadata, coord=coord, dims=dims)
            if dims:
                category.items_aux.append(item)
            else:
                category.items_scalar.append(item)

        return category

    @staticmethod
    def _create_prepared_item(coord, dims, src=None, tgt=None):
        if src is not None and tgt is not None:
            combined = src.combine(tgt)
        else:
            combined = src or tgt
        if not isinstance(dims, Iterable):
            dims = (dims,)
        prepared_metadata = _PreparedMetadata(
            combined=combined, src=src, tgt=tgt
        )
        bounds = coord.bounds
        result = _PreparedItem(
            metadata=prepared_metadata,
            points=coord.points.copy(),
            bounds=bounds if bounds is None else bounds.copy(),
            dims=dims,
            container=type(coord),
        )
        return result

    @property
    def _debug(self):
        return logger.getEffectiveLevel() <= logging.DEBUG

    @staticmethod
    def _dim_coverage(cube, cube_items_dim, common_dim_metadata):
        ndim = cube.ndim
        metadata = [None] * ndim
        coords = [None] * ndim
        dims_common = []
        dims_local = []
        dims_free = set(range(ndim))

        for item in cube_items_dim:
            (dim,) = item.dims
            dims_free.discard(dim)
            metadata[dim] = item.metadata
            coords[dim] = item.coord
            if item.metadata in common_dim_metadata:
                dims_common.append(dim)
            else:
                dims_local.append(dim)

        return _DimCoverage(
            cube=cube,
            metadata=metadata,
            coords=coords,
            dims_common=sorted(dims_common),
            dims_local=sorted(dims_local),
            dims_free=sorted(dims_free),
        )

    def _dim_mapping(self, src_coverage, tgt_coverage):
        for tgt_dim in tgt_coverage.dims_common:
            # Search for a src dim metadata match.
            tgt_metadata = tgt_coverage.metadata[tgt_dim]
            try:
                src_dim = src_coverage.metadata.index(tgt_metadata)
                self.mapping[src_dim] = tgt_dim
                logger.debug(f"{src_dim}->{tgt_dim}")
            except ValueError:
                # This exception can only occur due to a systemic internal
                # failure to correctly identify common dim coordinate metadata
                # coverage between the cubes.
                emsg = (
                    "Failed to map common dim coordinate metadata from "
                    "source cube {!r} to target cube {!r}, using {!r} on "
                    "target cube dimension {}."
                )
                raise ValueError(
                    emsg.format(
                        src_coverage.cube.name(),
                        tgt_coverage.cube.name(),
                        tgt_metadata,
                        tuple([tgt_dim]),
                    )
                )

    def _init(self, lhs, rhs):
        from iris.cube import Cube

        emsg = "{cls} requires {arg!r} argument to be a 'Cube', got {actual}."
        clsname = self.__class__.__name__

        if not isinstance(lhs, Cube):
            raise TypeError(
                emsg.format(cls=clsname, arg="lhs", actual=type(lhs))
            )

        if not isinstance(rhs, Cube):
            raise TypeError(
                emsg.format(cls=clsname, arg="rhs", actual=type(rhs))
            )

        # The LHS cube to be resolved into the resultant cube.
        self.lhs_cube = lhs
        # The RHS cube to be resolved into the resultant cube.
        self.rhs_cube = rhs

        # Categorised dim, aux and scalar coordinate items for LHS cube.
        self.lhs_cube_category = None
        # Categorised dim, aux and scalar coordinate items for RHS cube.
        self.rhs_cube_category = None

        # Categorised dim, aux and scalar coordinate items local to LHS cube only.
        self.lhs_cube_category_local = None
        # Categorised dim, aux and scalar coordinate items local to RHS cube only.
        self.rhs_cube_category_local = None
        # Categorised dim, aux and scalar coordinate items common to both
        # LHS cube and RHS cube.
        self.category_common = None

        # Analysis of dim coordinates spanning LHS cube.
        self.lhs_cube_dim_coverage = None
        # Analysis of aux and scalar coordinates spanning LHS cube.
        self.lhs_cube_aux_coverage = None
        # Analysis of dim coordinates spanning RHS cube.
        self.rhs_cube_dim_coverage = None
        # Analysis of aux and scalar coordinates spanning RHS cube.
        self.rhs_cube_aux_coverage = None

        # Map common metadata from RHS cube to LHS cube if LHS-rank >= RHS-rank,
        # otherwise map common metadata from LHS cube to RHS cube.
        if self.lhs_cube.ndim >= self.rhs_cube.ndim:
            self.map_rhs_to_lhs = True
        else:
            self.map_rhs_to_lhs = False

        # Mapping of the dimensions between common metadata for the cubes,
        # where the direction of the mapping is governed by map_rhs_to_lhs.
        self.mapping = None

        # Cache containing a list of dim, aux and scalar coordinates prepared
        # and ready for creating and attaching to the resultant cube.
        self.prepared_category = None

        # Cache containing a list of aux factories prepared and ready for
        # creating and attaching to the resultant cube.
        self.prepared_factories = None

    def _metadata_coverage(self):
        # Determine the common dim coordinate metadata coverage.
        common_dim_metadata = [
            item.metadata for item in self.category_common.items_dim
        ]

        self.lhs_cube_dim_coverage = self._dim_coverage(
            self.lhs_cube,
            self.lhs_cube_category.items_dim,
            common_dim_metadata,
        )
        self.rhs_cube_dim_coverage = self._dim_coverage(
            self.rhs_cube,
            self.rhs_cube_category.items_dim,
            common_dim_metadata,
        )

        # Determine the common aux and scalar coordinate metadata coverage.
        common_aux_metadata = [
            item.metadata for item in self.category_common.items_aux
        ]
        common_scalar_metadata = [
            item.metadata for item in self.category_common.items_scalar
        ]

        self.lhs_cube_aux_coverage = self._aux_coverage(
            self.lhs_cube,
            self.lhs_cube_category.items_aux,
            self.lhs_cube_category.items_scalar,
            common_aux_metadata,
            common_scalar_metadata,
        )
        self.rhs_cube_aux_coverage = self._aux_coverage(
            self.rhs_cube,
            self.rhs_cube_category.items_aux,
            self.rhs_cube_category.items_scalar,
            common_aux_metadata,
            common_scalar_metadata,
        )

    def _metadata_mapping(self):
        # Initialise the state.
        self.mapping = {}

        # Map RHS cube to LHS cube, or smaller to larger cube rank.
        if self.map_rhs_to_lhs:
            src_dim_coverage = self.rhs_cube_dim_coverage
            src_aux_coverage = self.rhs_cube_aux_coverage
            tgt_dim_coverage = self.lhs_cube_dim_coverage
            tgt_aux_coverage = self.lhs_cube_aux_coverage
        else:
            src_dim_coverage = self.lhs_cube_dim_coverage
            src_aux_coverage = self.lhs_cube_aux_coverage
            tgt_dim_coverage = self.rhs_cube_dim_coverage
            tgt_aux_coverage = self.rhs_cube_aux_coverage

        self._dim_mapping(src_dim_coverage, tgt_dim_coverage)
        logger.debug(f"mapping={self.mapping}")

        if not self.mapped:
            self._aux_mapping(src_aux_coverage, tgt_aux_coverage)
            logger.debug(f"mapping={self.mapping}")

        self._verify_mapping()

    def _metadata_prepare(self):
        # Initialise the state.
        self.prepared_category = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.prepared_factories = []

        # Map RHS cube to LHS cube, or smaller to larger cube rank.
        if self.map_rhs_to_lhs:
            src_cube = self.rhs_cube
            src_category_local = self.rhs_cube_category_local
            src_dim_coverage = self.rhs_cube_dim_coverage
            src_aux_coverage = self.rhs_cube_aux_coverage
            tgt_cube = self.lhs_cube
            tgt_category_local = self.lhs_cube_category_local
            tgt_dim_coverage = self.lhs_cube_dim_coverage
            tgt_aux_coverage = self.lhs_cube_aux_coverage
        else:
            src_cube = self.lhs_cube
            src_category_local = self.lhs_cube_category_local
            src_dim_coverage = self.lhs_cube_dim_coverage
            src_aux_coverage = self.lhs_cube_aux_coverage
            tgt_cube = self.rhs_cube
            tgt_category_local = self.rhs_cube_category_local
            tgt_dim_coverage = self.rhs_cube_dim_coverage
            tgt_aux_coverage = self.rhs_cube_aux_coverage

        # Determine the resultant cube dim coordinate/s.
        self._prepare_common_dim_payload(src_dim_coverage, tgt_dim_coverage)

        # Determine the resultant cube aux coordinate/s.
        self._prepare_common_aux_payload(
            src_aux_coverage.common_items_aux,  # input
            tgt_aux_coverage.common_items_aux,  # input
            self.prepared_category.items_aux,  # output
        )

        # Determine the resultant cube scalar coordinate/s.
        self._prepare_common_aux_payload(
            src_aux_coverage.common_items_scalar,  # input
            tgt_aux_coverage.common_items_scalar,  # input
            self.prepared_category.items_scalar,  # output
        )

        self._prepare_local_payload(
            src_dim_coverage,
            src_aux_coverage,
            tgt_dim_coverage,
            tgt_aux_coverage,
        )

        self._prepare_factory_payload(
            tgt_cube, tgt_category_local, from_src=False
        )
        self._prepare_factory_payload(src_cube, src_category_local)

    def _metadata_resolve(self):
        """
        Categorise the coordinate metadata of the cubes into three distinct
        groups; metadata from coordinates only available in 'cube1', metadata
        from coordinates only available in 'cube2', and metadata from
        coordinates common to both cubes.

        This is only applicable to coordinates that are members of the
        'aux_coords' or dim_coords' of the participating cubes.

        .. note::
            Coordinate metadata specific to each cube, but with a shared common
            name will be removed, as the difference in metadata is in conflict
            and cannot be resolved.

        """

        # Initialise the local and common category state.
        self.lhs_cube_category_local = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.rhs_cube_category_local = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )
        self.category_common = _CategoryItems(
            items_dim=[], items_aux=[], items_scalar=[]
        )

        # Determine the cube dim, aux and scalar coordinate items.
        self.lhs_cube_category = self._categorise_items(self.lhs_cube)
        self.rhs_cube_category = self._categorise_items(self.rhs_cube)

        # Map RHS cube to LHS cube, or smaller to larger ranked cube.
        if self.map_rhs_to_lhs:
            args = (
                self.lhs_cube_category,  # input
                self.rhs_cube_category,  # input
                self.lhs_cube_category_local,  # output
                self.rhs_cube_category_local,  # output
                self.category_common,  # output
            )
        else:
            args = (
                self.rhs_cube_category,  # input
                self.lhs_cube_category,  # input
                self.rhs_cube_category_local,  # output
                self.lhs_cube_category_local,  # output
                self.category_common,  # output
            )

        # Resolve local and common category items.
        self._resolve_category_items(*args)

        # TODO: remove this!
        # # Purge different metadata with common names.
        # cube1_names = set(
        #     [item.metadata.name() for item in self.cube1_local_items]
        # )
        # cube2_names = set(
        #     [item.metadata.name() for item in self.cube2_local_items]
        # )
        # common_names = cube1_names & cube2_names
        #
        # if common_names:
        #     self.cube1_local_items = [
        #         item
        #         for item in self.cube1_local_items
        #         if item.metadata.name() not in common_names
        #     ]
        #     self.cube2_local_items = [
        #         item
        #         for item in self.cube2_local_items
        #         if item.metadata.name() not in common_names
        #     ]

    def _prepare_common_aux_payload(
        self, src_common_items, tgt_common_items, prepared_items
    ):
        from iris.coords import AuxCoord

        for src_item in src_common_items:
            src_metadata = src_item.metadata
            tgt_items = tuple(
                filter(
                    lambda tgt_item: tgt_item.metadata == src_metadata,
                    tgt_common_items,
                )
            )
            if not tgt_items:
                dmsg = f"ignoring src {src_metadata}, does not match any common tgt metadata"
                logger.debug(dmsg)
            elif len(tgt_items) > 1:
                dmsg = (
                    f"ignoring src {src_metadata}, matches multiple "
                    f"[{len(tgt_items)}] common tgt metadata"
                )
                logger.debug(dmsg)
            else:
                (tgt_item,) = tgt_items
                src_coord = src_item.coord
                tgt_coord = tgt_item.coord
                points, bounds = self._prepare_points_and_bounds(
                    src_coord, tgt_coord, src_item.dims, tgt_item.dims
                )
                if points is not None:
                    src_type = type(src_coord)
                    tgt_type = type(tgt_coord)
                    # Downcast to aux if there are mixed container types.
                    container = src_type if src_type is tgt_type else AuxCoord
                    prepared_metadata = _PreparedMetadata(
                        combined=src_metadata.combine(tgt_item.metadata),
                        src=src_metadata,
                        tgt=tgt_item.metadata,
                    )
                    prepared_item = _PreparedItem(
                        metadata=prepared_metadata,
                        points=points.copy(),
                        bounds=bounds if bounds is None else bounds.copy(),
                        dims=tgt_item.dims,
                        container=container,
                    )
                    prepared_items.append(prepared_item)

    def _prepare_common_dim_payload(self, src_coverage, tgt_coverage):
        from iris.coords import DimCoord

        for src_dim in src_coverage.dims_common:
            src_metadata = src_coverage.metadata[src_dim]
            src_coord = src_coverage.coords[src_dim]

            tgt_dim = self.mapping[src_dim]
            tgt_metadata = tgt_coverage.metadata[tgt_dim]
            tgt_coord = tgt_coverage.coords[tgt_dim]

            points, bounds = self._prepare_points_and_bounds(
                src_coord, tgt_coord, src_dim, tgt_dim
            )

            if points is not None:
                prepared_metadata = _PreparedMetadata(
                    combined=src_metadata.combine(tgt_metadata),
                    src=src_metadata,
                    tgt=tgt_metadata,
                )
                prepared_item = _PreparedItem(
                    metadata=prepared_metadata,
                    points=points.copy(),
                    bounds=bounds if bounds is None else bounds.copy(),
                    dims=(tgt_dim,),
                    container=DimCoord,
                )
                self.prepared_category.items_dim.append(prepared_item)

    def _prepare_factory_payload(self, cube, category_local, from_src=True):
        def _get_prepared_item(metadata, from_src=True, from_local=False):
            result = None
            if from_local:
                category = category_local
                match = lambda item: item.metadata == metadata
            else:
                category = self.prepared_category
                if from_src:
                    match = lambda item: item.metadata.src == metadata
                else:
                    match = lambda item: item.metadata.tgt == metadata
            for member in category._fields:
                category_items = getattr(category, member)
                matched_items = tuple(filter(match, category_items))
                if matched_items:
                    if len(matched_items) > 1:
                        dmsg = (
                            f"ignoring factory dependency {metadata}, multiple {'src' if from_src else 'tgt'} "
                            f"{'local' if from_local else 'prepared'} metadata matches"
                        )
                        logger.debug(dmsg)
                    else:
                        (item,) = matched_items
                        if from_local:
                            src = tgt = None
                            if from_src:
                                src = item.metadata
                                dims = tuple(
                                    [self.mapping[dim] for dim in item.dims]
                                )
                            else:
                                tgt = item.metadata
                                dims = item.dims
                            result = self._create_prepared_item(
                                item.coord, dims, src=src, tgt=tgt
                            )
                            getattr(self.prepared_category, member).append(
                                result
                            )
                        else:
                            result = item
                    break
            return result

        for factory in cube.aux_factories:
            container = type(factory)
            dependencies = {}
            prepared_item = None

            if tuple(
                filter(
                    lambda item: item.container is container,
                    self.prepared_factories,
                )
            ):
                # debug: skipping, factory already exists
                dmsg = (
                    f"ignoring {'src' if from_src else 'tgt'} {container}, "
                    f"a similar factory has already been prepared"
                )
                logger.debug(dmsg)
                continue

            for (
                dependency_name,
                dependency_coord,
            ) in factory.dependencies.items():
                metadata = dependency_coord.metadata
                prepared_item = _get_prepared_item(metadata, from_src=from_src)
                if prepared_item is None:
                    prepared_item = _get_prepared_item(
                        metadata, from_src=from_src, from_local=True
                    )
                    if prepared_item is None:
                        dmsg = f"cannot find matching {metadata} for {container} dependency {dependency_name}"
                        logger.debug(dmsg)
                        break
                dependencies[dependency_name] = prepared_item.metadata

            if prepared_item is not None:
                prepared_factory = _PreparedFactory(
                    container=container, dependencies=dependencies
                )
                self.prepared_factories.append(prepared_factory)
            else:
                dmsg = f"ignoring {'src' if from_src else 'tgt'} {container}, cannot find all dependencies"
                logger.debug(dmsg)

    def _prepare_local_payload_aux(self, src_aux_coverage, tgt_aux_coverage):
        # Determine whether there are extra tgt dimensions that may
        # require local tgt aux coordinates.
        delta = tgt_aux_coverage.cube.ndim - src_aux_coverage.cube.ndim
        extra_tgt_dims = set([dim for dim in range(delta)])

        if LENIENT["maths"]:
            mapped_src_dims = set(self.mapping.keys())
            mapped_tgt_dims = set(self.mapping.values())

            # Add local src aux coordinates.
            for item in src_aux_coverage.local_items_aux:
                if all([dim in mapped_src_dims for dim in item.dims]):
                    tgt_dims = tuple([self.mapping[dim] for dim in item.dims])
                    prepared_item = self._create_prepared_item(
                        item.coord, tgt_dims, src=item.metadata
                    )
                    self.prepared_category.items_aux.append(prepared_item)
                else:
                    dmsg = (
                        f"ignoring local src aux coordinate {item.metadata}, "
                        f"as not all src dimensions {item.dims} are mapped."
                    )
                    logger.debug(dmsg)
        else:
            # For strict maths, only local tgt aux coordinates covering
            # the extra dimensions of the tgt cube may be added.
            mapped_tgt_dims = set()

        # Add local tgt aux coordinates.
        for item in tgt_aux_coverage.local_items_aux:
            tgt_dims = item.dims
            if all([dim in mapped_tgt_dims for dim in tgt_dims]) or any(
                [dim in extra_tgt_dims for dim in tgt_dims]
            ):
                prepared_item = self._create_prepared_item(
                    item.coord, tgt_dims, tgt=item.metadata
                )
                self.prepared_category.items_aux.append(prepared_item)
            else:
                dmsg = (
                    f"ignoring local tgt aux coordinate {item.metadata} "
                    f"as not all tgt dimensions {tgt_dims} are mapped."
                )
                logger.debug(dmsg)

    def _prepare_local_payload_dim(self, src_dim_coverage, tgt_dim_coverage):
        # Determine whether there are extra tgt dimensions that require
        # local tgt dim coordinates.
        delta = tgt_dim_coverage.cube.ndim - src_dim_coverage.cube.ndim
        extra_tgt_dims = set(range(delta))

        if LENIENT["maths"]:
            tgt_dims_mapped = set()

            # Add local src dim coordinates.
            for src_dim in src_dim_coverage.dims_local:
                tgt_dim = self.mapping.get(src_dim)
                # Only add the local src dim coordinate iff there is no
                # associated local tgt dim coordinate.
                if (
                    tgt_dim is not None
                    and tgt_dim not in tgt_dim_coverage.dims_local
                ):
                    tgt_dims_mapped.add(tgt_dim)
                    metadata = src_dim_coverage.metadata[src_dim]
                    coord = src_dim_coverage.coords[src_dim]
                    prepared_item = self._create_prepared_item(
                        coord, tgt_dim, src=metadata
                    )
                    self.prepared_category.items_dim.append(prepared_item)
                else:
                    if self._debug:
                        src_metadata = src_dim_coverage.metadata[src_dim]
                        dmsg = f"ignoring local src dim coordinate {src_metadata}, "
                        if tgt_dim is None:
                            dmsg += (
                                f"as src dimension ({src_dim},) is not mapped."
                            )
                        else:
                            tgt_metadata = tgt_dim_coverage.metadata[tgt_dim]
                            dmsg += (
                                f"conflicts with tgt dim coordinate {tgt_metadata}, "
                                f"mapping ({src_dim},)->({tgt_dim},)."
                            )
                        logger.debug(dmsg)

            # Determine whether there are any tgt dims free to be mapped
            # by an available local tgt dim coordinate.
            tgt_dims_local_unmapped = (
                set(tgt_dim_coverage.dims_local) - tgt_dims_mapped
            )
        else:
            # For strict maths, only local tgt dim coordinates covering
            # the extra dimensions of the tgt cube may be added.
            tgt_dims_local_unmapped = extra_tgt_dims

        mapped_tgt_dims = self.mapping.values()

        # Add local tgt dim coordinates.
        for tgt_dim in tgt_dims_local_unmapped:
            if tgt_dim in mapped_tgt_dims or tgt_dim in extra_tgt_dims:
                metadata = tgt_dim_coverage.metadata[tgt_dim]
                if metadata is not None:
                    coord = tgt_dim_coverage.coords[tgt_dim]
                    prepared_item = self._create_prepared_item(
                        coord, tgt_dim, tgt=metadata
                    )
                    self.prepared_category.items_dim.append(prepared_item)

    def _prepare_local_payload_scalar(
        self, src_aux_coverage, tgt_aux_coverage
    ):
        # Add all local tgt scalar coordinates iff the src cube is a
        # scalar cube with no local src scalar coordinates.
        # Only for strict maths.
        src_scalar_cube = (
            not LENIENT["maths"]
            and src_aux_coverage.cube.ndim == 0
            and len(src_aux_coverage.local_items_scalar) == 0
        )

        if src_scalar_cube or LENIENT["maths"]:
            # Add any local src scalar coordinates, if available.
            for item in src_aux_coverage.local_items_scalar:
                prepared_item = self._create_prepared_item(
                    item.coord, item.dims, src=item.metadata
                )
                self.prepared_category.items_scalar.append(prepared_item)

            # Add any local tgt scalar coordinates, if available.
            for item in tgt_aux_coverage.local_items_scalar:
                prepared_item = self._create_prepared_item(
                    item.coord, item.dims, tgt=item.metadata
                )
                self.prepared_category.items_scalar.append(prepared_item)

    def _prepare_local_payload(
        self,
        src_dim_coverage,
        src_aux_coverage,
        tgt_dim_coverage,
        tgt_aux_coverage,
    ):
        # Add local src/tgt dim coordinates.
        self._prepare_local_payload_dim(src_dim_coverage, tgt_dim_coverage)

        # Add local src/tgt aux coordinates.
        self._prepare_local_payload_aux(src_aux_coverage, tgt_aux_coverage)

        # Add local src/tgt scalar coordinates.
        self._prepare_local_payload_scalar(src_aux_coverage, tgt_aux_coverage)

    def _prepare_points_and_bounds(
        self, src_coord, tgt_coord, src_dims, tgt_dims
    ):
        from iris.util import array_equal

        points, bounds = None, None
        eq_points = array_equal(
            src_coord.points, tgt_coord.points, withnans=True
        )
        if eq_points:
            points = src_coord.points
            src_has_bounds = src_coord.has_bounds()
            tgt_has_bounds = tgt_coord.has_bounds()
            if src_has_bounds and tgt_has_bounds:
                src_bounds = src_coord.bounds
                eq_bounds = array_equal(
                    src_bounds, tgt_coord.bounds, withnans=True
                )
                if eq_bounds:
                    bounds = src_bounds
                else:
                    if LENIENT["maths"]:
                        # For lenient, ignore coordinate with mis-matched bounds.
                        if not isinstance(src_dims, Iterable):
                            src_dims = (src_dims,)
                        if not isinstance(tgt_dims, Iterable):
                            tgt_dims = (tgt_dims,)
                        dmsg = (
                            f"ignoring src {src_coord.metadata}, "
                            f"unequal bounds with tgt {src_dims}->{tgt_dims}"
                        )
                        logger.debug(dmsg)
                    else:
                        # For strict, the coordinate bounds must match.
                        emsg = (
                            f"Coordinate {src_coord.name()!r} has different bounds for the "
                            f"LHS cube {self.lhs_cube.name()!r} and "
                            f"RHS cube {self.rhs_cube.name()!r}."
                        )
                        raise ValueError(emsg)
            else:
                # For lenient, use either of the coordinate bounds, if they exist.
                if LENIENT["maths"]:
                    if src_has_bounds:
                        dmsg = f"using src {src_coord.metadata} bounds, tgt has no bounds"
                        logger.debug(dmsg)
                        bounds = src_coord.bounds
                    else:
                        dmsg = f"using tgt {tgt_coord.metadata} bounds, src has no bounds"
                        logger.debug(dmsg)
                        bounds = tgt_coord.bounds
                else:
                    # For strict, both coordinates must have bounds, or both
                    # coordinates must not have bounds.
                    if src_has_bounds:
                        emsg = (
                            f"Coordinate {src_coord.name()!r} has bounds for the "
                            f"{self._src_cube_position} cube {self._src_cube.name()!r}, "
                            f"but not the {self._tgt_cube_position} cube {self._tgt_cube.name()!r}."
                        )
                        raise ValueError(emsg)
                    if tgt_has_bounds:
                        emsg = (
                            f"Coordinate {tgt_coord.name()!r} has bounds for the "
                            f"{self._tgt_cube_position} cube {self._tgt_cube.name()!r}, "
                            f"but not the {self._src_cube_position} cube {self._src_cube.name()!r}."
                        )
                        raise ValueError(emsg)
        else:
            if LENIENT["maths"]:
                # For lenient, ignore coordinate with mis-matched points.
                if not isinstance(src_dims, Iterable):
                    src_dims = (src_dims,)
                if not isinstance(tgt_dims, Iterable):
                    tgt_dims = (tgt_dims,)
                dmsg = f"ignoring src {src_coord.metadata}, unequal points with tgt {src_dims}->{tgt_dims}"
                logger.debug(dmsg)
            else:
                # For strict, the coordinate points must match.
                emsg = (
                    f"Coordinate {src_coord.name()!r} has different points for the "
                    f"LHS cube {self.lhs_cube.name()!r} and "
                    f"RHS cube {self.rhs_cube.name()!r}."
                )
                raise ValueError(emsg)
        return points, bounds

    @staticmethod
    def _resolve_category_items(
        src_category,
        tgt_category,
        src_category_local,
        tgt_category_local,
        category_common,
    ):
        def _categorise(
            src_items,
            tgt_items,
            src_local_items,
            tgt_local_items,
            common_items,
        ):
            tgt_items_metadata = [item.metadata for item in tgt_items]
            # Track common metadata here as a temporary convenience.
            common_metadata = []

            # Determine items local to the src, and shared items
            # common to both src and tgt.
            for item in src_items:
                metadata = item.metadata
                if metadata in tgt_items_metadata:
                    # The metadata is common between src and tgt.
                    if metadata not in common_metadata:
                        common_items.append(item)
                        common_metadata.append(metadata)
                else:
                    # The metadata is local to the src.
                    src_local_items.append(item)

            # Determine items local to the tgt.
            for item in tgt_items:
                if item.metadata not in common_metadata:
                    tgt_local_items.append(item)

        # Resolve local and common dim category items.
        _categorise(
            src_category.items_dim,  # input
            tgt_category.items_dim,  # input
            src_category_local.items_dim,  # output
            tgt_category_local.items_dim,  # output
            category_common.items_dim,  # output
        )

        # Resolve local and common aux category items.
        _categorise(
            src_category.items_aux,  # input
            tgt_category.items_aux,  # input
            src_category_local.items_aux,  # output
            tgt_category_local.items_aux,  # output
            category_common.items_aux,  # output
        )

        # Resolve local and common scalar category items.
        _categorise(
            src_category.items_scalar,  # input
            tgt_category.items_scalar,  # input
            src_category_local.items_scalar,  # output
            tgt_category_local.items_scalar,  # output
            category_common.items_scalar,  # output
        )

        # Sort the result categories by metadata name for consistency.
        results = (src_category_local, tgt_category_local, category_common)
        key_func = lambda item: item.metadata.name()
        for result in results:
            result.items_dim.sort(key=key_func)
            result.items_aux.sort(key=key_func)
            result.items_scalar.sort(key=key_func)

    @property
    def _src_cube(self):
        if self.map_rhs_to_lhs:
            result = self.rhs_cube
        else:
            result = self.lhs_cube
        return result

    @property
    def _src_cube_position(self):
        if self.map_rhs_to_lhs:
            result = "RHS"
        else:
            result = "LHS"
        return result

    @property
    def _tgt_cube(self):
        if self.map_rhs_to_lhs:
            result = self.lhs_cube
        else:
            result = self.rhs_cube
        return result

    @property
    def _tgt_cube_position(self):
        if self.map_rhs_to_lhs:
            result = "LHS"
        else:
            result = "RHS"
        return result

    def _tgt_cube_clear(self):
        cube = self._tgt_cube

        # clear the aux factories.
        for factory in cube.aux_factories:
            cube.remove_aux_factory(factory)

        # clear the cube coordinates.
        for coord in cube.coords():
            cube.remove_coord(coord)

        # clear the cube cell measures.
        for cm in cube.cell_measures():
            cube.remove_cell_measure(cm)

        # clear the ancillary variables.
        for av in cube.ancillary_variables():
            cube.remove_ancillary_variable(av)

        return cube

    def _verify_mapping(self):
        from iris.exceptions import NotYetImplementedError

        def _shape(cube, dims):
            if not isinstance(dims, Iterable):
                dims = (dims,)
            cube_shape = cube.shape
            shape = [None] * cube.ndim
            for dim in dims:
                shape[dim] = cube_shape[dim]
            return tuple(filter(lambda extent: extent is not None, shape))

        src_cube = self._src_cube
        tgt_cube = self._tgt_cube
        src_dims = sorted(self.mapping.keys())
        tgt_dims = [self.mapping[src_dim] for src_dim in src_dims]

        # common exception message.
        emsg = "Cannot resolve the cubes, as the {!r} cube {!r} requires to be transposed/reshaped."
        emsg = emsg.format(self._src_cube_position, src_cube.name())

        # TODO: future support for generic transpose and/or reshape required.
        if not np.all(np.diff(tgt_dims) > 0):
            raise NotYetImplementedError(emsg)

        # special case: single common dimension.
        if len(src_dims) == 1:
            delta = tgt_cube.ndim - src_cube.ndim
            (src_dim,), (tgt_dim,) = src_dims, tgt_dims
            if (src_dim + delta) != tgt_dim:
                raise NotYetImplementedError(emsg)

        src_shape = _shape(src_cube, src_dims)
        tgt_shape = _shape(tgt_cube, tgt_dims)

        if src_shape != tgt_shape:
            items = []
            for src_dim, src_size, tgt_dim, tgt_size in zip(
                src_dims, src_shape, tgt_dims, tgt_shape
            ):
                if src_size != tgt_size:
                    msg = f"dim[{src_dim}]!=dim[{tgt_dim}]"
                    items.append(msg)
            emap = ", ".join(items)
            emsg = (
                "Incompatible dimension shapes for cube {!r}->{} and "
                "cube {!r}->{}, got {} respectively."
            )
            raise ValueError(
                emsg.format(
                    src_cube.name(),
                    src_cube.shape,
                    tgt_cube.name(),
                    tgt_cube.shape,
                    emap,
                )
            )

    def cube(self, data, in_place=False):
        result = None
        shape = self.shape

        if shape is None:
            dmsg = f"cannot resolve resultant cube, as no candidate cubes have been provided"
            logger.debug(dmsg)
        else:
            from iris.cube import Cube

            # Ensure that the shape of the provided data is the expected
            # shape of the resultant resolved cube.
            if data.shape != shape:
                emsg = f"Cannot resolve resultant cube, expect data with shape {shape}, got {data.shape}."
                raise ValueError(emsg)

            if in_place:
                # Prepare in-place target cube for population with prepared content.
                result = self._tgt_cube_clear()
            else:
                # Create the resultant resolved cube.
                result = Cube(data)

            # Add the combined cube metadata from both the candidate cubes.
            result.metadata = self.lhs_cube.metadata.combine(
                self.rhs_cube.metadata
            )

            # Add the prepared dim coordinates.
            for item in self.prepared_category.items_dim:
                coord = item.container(item.points, bounds=item.bounds)
                coord.metadata = item.metadata.combined
                result.add_dim_coord(coord, item.dims)

            # Add the prepared aux and scalar coordinates.
            prepared_aux_coords = (
                self.prepared_category.items_aux
                + self.prepared_category.items_scalar
            )
            for item in prepared_aux_coords:
                coord = item.container(item.points, bounds=item.bounds)
                coord.metadata = item.metadata.combined
                result.add_aux_coord(coord, item.dims)

            # Add the prepared aux factories.
            for prepared_factory in self.prepared_factories:
                dependencies = dict()
                for (
                    dependency_name,
                    prepared_metadata,
                ) in prepared_factory.dependencies.items():
                    coord = result.coord(prepared_metadata.combined)
                    dependencies[dependency_name] = coord
                factory = prepared_factory.container(**dependencies)
                result.add_aux_factory(factory)

        return result

    @property
    def mapped(self):
        # Map RHS cube to LHS cube, or smaller to larger cube rank.
        if self.map_rhs_to_lhs:
            dim_coverage, aux_coverage = (
                self.rhs_cube_dim_coverage,
                self.rhs_cube_aux_coverage,
            )
        else:
            dim_coverage, aux_coverage = (
                self.lhs_cube_dim_coverage,
                self.lhs_cube_aux_coverage,
            )

        dims_common = set(dim_coverage.dims_common) | set(
            aux_coverage.dims_common
        )

        return (dims_common & set(self.mapping)) == dims_common

    @property
    def shape(self):
        result = None
        map_rhs_to_lhs = getattr(self, "map_rhs_to_lhs", None)
        if map_rhs_to_lhs is not None:
            cube = self.lhs_cube if map_rhs_to_lhs else self.rhs_cube
            result = cube.shape
        return result

    ###########################################################################

    # TODO:: remove this!
    @staticmethod
    def show_dim(coverage):
        from pprint import pprint

        print("dim coverage:")
        pprint(coverage.metadata)
        print(
            f"name: {coverage.cube.name()}\ncommon: {coverage.dims_common}, "
            f"local: {coverage.dims_local}, free: {coverage.dims_free}\n"
            f"ndim: {coverage.cube.ndim}\n"
        )

    # TODO:: remove this!
    @staticmethod
    def show_aux(coverage):
        from pprint import pprint

        print("aux coverage:")
        items = [
            (item.metadata, item.dims) for item in coverage.common_items_aux
        ]
        pprint(items)
        items = [
            (item.metadata, item.dims) for item in coverage.common_items_scalar
        ]
        pprint(items)
        items = [
            (item.metadata, item.dims) for item in coverage.local_items_aux
        ]
        pprint(items)
        items = [
            (item.metadata, item.dims) for item in coverage.local_items_scalar
        ]
        pprint(items)
        print(
            f"name: {coverage.cube.name()}\ncommon: {coverage.dims_common}, "
            f"local: {coverage.dims_local}, free: {coverage.dims_free}\n"
            f"ndim: {coverage.cube.ndim}\n"
        )

    # TODO:: remove this!
    @staticmethod
    def show_items(items, title=None):
        from pprint import pprint

        title = f"{title} " if title else ""
        print(f"{title}dim:")
        out = [
            (item.metadata, item.dims, item.coord.has_bounds())
            for item in items.items_dim
        ]
        pprint(out)

        print(f"{title}aux:")
        out = [
            (item.metadata, item.dims, item.coord.has_bounds())
            for item in items.items_aux
        ]
        pprint(out)

        print(f"{title}scalar:")
        out = [
            (item.metadata, item.dims, item.coord.has_bounds())
            for item in items.items_scalar
        ]
        pprint(out)
        print()

    # TODO:: remove this!
    def show_prepared(self):
        from pprint import pprint

        title = "prepared "
        print(f"{title}dim:")
        out = [
            (
                item.metadata.combined,
                item.dims,
                item.bounds is not None,
                item.container,
            )
            for item in self.prepared_category.items_dim
        ]
        pprint(out)

        print(f"{title}aux:")
        out = [
            (
                item.metadata.combined,
                item.dims,
                item.bounds is not None,
                item.container,
            )
            for item in self.prepared_category.items_aux
        ]
        pprint(out)

        print(f"{title}scalar:")
        out = [
            (
                item.metadata.combined,
                item.dims,
                item.bounds is not None,
                item.container,
            )
            for item in self.prepared_category.items_scalar
        ]
        pprint(out)

        print(f"{title}factories:")
        out = [
            (item.container, item.dependencies,)
            for item in self.prepared_factories
        ]
        pprint(out)
        print()

    # TODO:: remove this!
    @classmethod
    def test(cls, scenario):
        from pathlib import Path
        import pickle

        dname = Path(
            f"/project/avd/bill/cube-arithmetic/data/ehogan/scenario{scenario}"
        )

        with open(dname / f"cube{scenario}a.pkl", "rb") as fi:
            lhs = pickle.load(fi)

        with open(dname / f"cube{scenario}b.pkl", "rb") as fi:
            rhs = pickle.load(fi)

        resolve = cls(lhs, rhs)
        return resolve
