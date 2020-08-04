# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.

from collections import namedtuple
from collections.abc import Iterable
import logging

from dask.array.core import broadcast_shapes

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

        # TODO: remote this!
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

        # TODO: remove this!
        if self._debug:
            self.show_prepared()

    def _as_compatible_cubes(self):
        from iris.cube import Cube

        src_cube = self._src_cube
        tgt_cube = self._tgt_cube

        # Use the mapping to calculate the new src cube shape.
        new_src_shape = [1] * tgt_cube.ndim
        for src_dim, tgt_dim in self.mapping.items():
            new_src_shape[tgt_dim] = src_cube.shape[src_dim]
        new_src_shape = tuple(new_src_shape)

        try:
            # Determine whether the tgt cube and proposed new src
            # cube (shape) will successfully broadcast together.
            self._broadcast_shape = broadcast_shapes(
                tgt_cube.shape, new_src_shape
            )
        except ValueError:
            emsg = (
                "Cannot resolve cubes, as a suitable transpose of the "
                f"{self._src_cube_position} cube {src_cube.name()!r} "
                f"will not broadcast with the {self._tgt_cube_position} cube "
                f"{tgt_cube.name()!r}."
            )
            raise ValueError(emsg)

        new_src_data = src_cube.core_data().copy()

        # Use the mapping to determine the transpose sequence of
        # src dimensions in increasing tgt dimension order.
        order = [
            src_dim
            for src_dim, tgt_dim in sorted(
                self.mapping.items(), key=lambda pair: pair[1]
            )
        ]

        # Determine whether a transpose of the src cube is necessary.
        if order != sorted(order):
            new_src_data = new_src_data.transpose(order)

        # Determine whether a reshape is necessary.
        if new_src_shape != new_src_data.shape:
            new_src_data = new_src_data.reshape(new_src_shape)

        # Create the new src cube.
        new_src_cube = Cube(new_src_data)
        new_src_cube.metadata = src_cube.metadata

        def add_coord(coord, dim_coord=False):
            src_dims = src_cube.coord_dims(coord)
            tgt_dims = [self.mapping[src_dim] for src_dim in src_dims]
            if dim_coord:
                new_src_cube.add_dim_coord(coord, tgt_dims)
            else:
                new_src_cube.add_aux_coord(coord, tgt_dims)

        # Add the dim coordinates to the new src cube.
        for coord in src_cube.dim_coords:
            add_coord(coord, dim_coord=True)

        # Add the aux and scalar coordinates to the new src cube.
        for coord in src_cube.aux_coords:
            add_coord(coord)

        # Add the aux factories to the new src cube.
        for factory in src_cube.aux_factories:
            new_src_cube.add_aux_factory(factory)

        # Set the resolved cubes.
        self._src_cube_resolved = new_src_cube
        self._tgt_cube_resolved = tgt_cube

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

    def _free_mapping(
        self,
        src_dim_coverage,
        tgt_dim_coverage,
        src_aux_coverage,
        tgt_aux_coverage,
    ):
        src_cube = src_dim_coverage.cube
        tgt_cube = tgt_dim_coverage.cube
        src_ndim = src_cube.ndim
        tgt_ndim = tgt_cube.ndim

        # mapping src to tgt, involving free dimensions on either the src/tgt.
        free_mapping = {}

        # Determine the src/tgt dimensions that are not mapped,
        # and not covered by any metadata.
        src_free = set(src_dim_coverage.dims_free) & set(
            src_aux_coverage.dims_free
        )
        tgt_free = set(tgt_dim_coverage.dims_free) & set(
            tgt_aux_coverage.dims_free
        )

        if src_free or tgt_free:
            # Determine the src/tgt dimensions that are not mapped.
            src_unmapped = set(range(src_ndim)) - set(self.mapping)
            tgt_unmapped = set(range(tgt_ndim)) - set(self.mapping.values())

            # Determine the src/tgt dimensions that are not mapped,
            # but are covered by a src/tgt local coordinate.
            src_unmapped_local = src_unmapped - src_free
            tgt_unmapped_local = tgt_unmapped - tgt_free

            src_shape = src_cube.shape
            tgt_shape = tgt_cube.shape
            src_max, tgt_max = max(src_shape), max(tgt_shape)

            def assign_mapping(extent, unmapped_local_items, free_items=None):
                result = None
                if free_items is None:
                    free_items = []
                if extent == 1:
                    if unmapped_local_items:
                        result, _ = unmapped_local_items.pop(0)
                    elif free_items:
                        result, _ = free_items.pop(0)
                else:

                    def _filter(items):
                        return list(
                            filter(lambda item: item[1] == extent, items)
                        )

                    def _pop(item, items):
                        result, _ = item
                        index = items.index(item)
                        items.pop(index)
                        return result

                    items = _filter(unmapped_local_items)
                    if items:
                        result = _pop(items[0], unmapped_local_items)
                    else:
                        items = _filter(free_items)
                        if items:
                            result = _pop(items[0], free_items)
                return result

            if src_free:
                # Attempt to map src free dimensions to tgt unmapped local or free dimensions.
                tgt_unmapped_local_items = [
                    (dim, tgt_shape[dim]) for dim in tgt_unmapped_local
                ]
                tgt_free_items = [(dim, tgt_shape[dim]) for dim in tgt_free]

                for src_dim in sorted(
                    src_free, key=lambda dim: (src_max - src_shape[dim], dim)
                ):
                    tgt_dim = assign_mapping(
                        src_shape[src_dim],
                        tgt_unmapped_local_items,
                        tgt_free_items,
                    )
                    if tgt_dim is None:
                        # Failed to map the src free dimension
                        # to a suitable tgt local/free dimension.
                        dmsg = (
                            f"failed to map src free dimension ({src_dim},) from "
                            f"{self._src_cube_position} cube {src_cube.name()!r} to "
                            f"{self._tgt_cube_position} cube {tgt_cube.name()!r}."
                        )
                        logger.debug(dmsg)
                        break
                    free_mapping[src_dim] = tgt_dim
            else:
                # Attempt to map tgt free dimensions to src unmapped local dimensions.
                src_unmapped_local_items = [
                    (dim, src_shape[dim]) for dim in src_unmapped_local
                ]

                for tgt_dim in sorted(
                    tgt_free, key=lambda dim: (tgt_max - tgt_shape[dim], dim)
                ):
                    src_dim = assign_mapping(
                        tgt_shape[tgt_dim], src_unmapped_local_items
                    )
                    if src_dim is not None:
                        free_mapping[src_dim] = tgt_dim
                        if not src_unmapped_local_items:
                            # There are no more src unmapped local dimensions.
                            break

        # Determine whether there are still unmapped src dimensions.
        src_unmapped = (
            set(range(src_cube.ndim)) - set(self.mapping) - set(free_mapping)
        )

        if src_unmapped:
            plural = "s" if len(src_unmapped) > 1 else ""
            emsg = (
                "Insufficient matching coordinate metadata to resolve cubes, "
                f"cannot map dimension{plural} {tuple(sorted(src_unmapped))} "
                f"of the {self._src_cube_position} cube {src_cube.name()!r} "
                f"to the {self._tgt_cube_position} cube {tgt_cube.name()!r}."
            )
            raise ValueError(emsg)

        # Update the mapping.
        self.mapping.update(free_mapping)

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

        # The transposed/reshaped (if required) LHS cube, which
        # can be broadcast with RHS cube.
        self.lhs_cube_resolved = None
        # The transposed/reshaped (if required) RHS cube, which
        # can be broadcast with LHS cube.
        self.rhs_cube_resolved = None

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

        dmsg = f"map_rhs_to_lhs={self.map_rhs_to_lhs}"
        logger.debug(dmsg)

        # Mapping of the dimensions between common metadata for the cubes,
        # where the direction of the mapping is governed by map_rhs_to_lhs.
        self.mapping = None

        # Cache containing a list of dim, aux and scalar coordinates prepared
        # and ready for creating and attaching to the resultant cube.
        self.prepared_category = None

        # Cache containing a list of aux factories prepared and ready for
        # creating and attaching to the resultant cube.
        self.prepared_factories = None

        # The shape of the resultant resolved cube.
        self._broadcast_shape = None

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
            src_cube = self.rhs_cube
            src_dim_coverage = self.rhs_cube_dim_coverage
            src_aux_coverage = self.rhs_cube_aux_coverage
            tgt_cube = self.lhs_cube
            tgt_dim_coverage = self.lhs_cube_dim_coverage
            tgt_aux_coverage = self.lhs_cube_aux_coverage
        else:
            src_cube = self.lhs_cube
            src_dim_coverage = self.lhs_cube_dim_coverage
            src_aux_coverage = self.lhs_cube_aux_coverage
            tgt_cube = self.rhs_cube
            tgt_dim_coverage = self.rhs_cube_dim_coverage
            tgt_aux_coverage = self.rhs_cube_aux_coverage

        # Use the dim coordinates to fully map the
        # src cube dimensions to the tgt cube dimensions.
        self._dim_mapping(src_dim_coverage, tgt_dim_coverage)
        logger.debug(f"mapping={self.mapping}")

        # If necessary, use the aux coordinates to fully map the
        # src cube dimensions to the tgt cube dimensions.
        if not self.mapped:
            self._aux_mapping(src_aux_coverage, tgt_aux_coverage)
            logger.debug(f"mapping={self.mapping}")

        if not self.mapped:
            # Attempt to complete the mapping using src/tgt free dimensions.
            self._free_mapping(
                src_dim_coverage,
                tgt_dim_coverage,
                src_aux_coverage,
                tgt_aux_coverage,
            )

        # Attempt to transpose/reshape the cubes into compatible broadcast shapes.
        self._as_compatible_cubes()

        # Given the resultant broadcast shape, determine whether the
        # mapping requires to be reversed.
        if (
            src_cube.ndim == tgt_cube.ndim
            and self._tgt_cube_resolved.shape != self.shape
            and self._src_cube_resolved.shape == self.shape
        ):
            flip_mapping = {
                tgt_dim: src_dim for src_dim, tgt_dim in self.mapping.items()
            }
            self.map_rhs_to_lhs = not self.map_rhs_to_lhs
            dmsg = (
                f"reversing the mapping from {self.mapping} to {flip_mapping}, "
                f"now map_rhs_to_lhs={self.map_rhs_to_lhs}"
            )
            logger.debug(dmsg)
            self.mapping = flip_mapping

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
        # Determine whether there are tgt dimensions not mapped to by an
        # associated src dimension, and thus may be covered by any local
        # tgt aux coordinates.
        extra_tgt_dims = set(range(tgt_aux_coverage.cube.ndim)) - set(
            self.mapping.values()
        )

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
                    f"ignoring local tgt aux coordinate {item.metadata}, "
                    f"as not all tgt dimensions {tgt_dims} are mapped."
                )
                logger.debug(dmsg)

    def _prepare_local_payload_dim(self, src_dim_coverage, tgt_dim_coverage):
        mapped_tgt_dims = self.mapping.values()

        # Determine whether there are tgt dimensions not mapped to by an
        # associated src dimension, and thus may be covered by any local
        # tgt dim coordinates.
        extra_tgt_dims = set(range(tgt_dim_coverage.cube.ndim)) - set(
            mapped_tgt_dims
        )

        if LENIENT["maths"]:
            tgt_dims_conflict = set()

            # Add local src dim coordinates.
            for src_dim in src_dim_coverage.dims_local:
                tgt_dim = self.mapping[src_dim]
                # Only add the local src dim coordinate iff there is no
                # associated local tgt dim coordinate.
                if tgt_dim not in tgt_dim_coverage.dims_local:
                    metadata = src_dim_coverage.metadata[src_dim]
                    coord = src_dim_coverage.coords[src_dim]
                    prepared_item = self._create_prepared_item(
                        coord, tgt_dim, src=metadata
                    )
                    self.prepared_category.items_dim.append(prepared_item)
                else:
                    tgt_dims_conflict.add(tgt_dim)
                    if self._debug:
                        src_metadata = src_dim_coverage.metadata[src_dim]
                        dmsg = f"ignoring local src dim coordinate {src_metadata}, "
                        tgt_metadata = tgt_dim_coverage.metadata[tgt_dim]
                        dmsg += (
                            f"as conflicts with tgt dim coordinate {tgt_metadata}, "
                            f"mapping ({src_dim},)->({tgt_dim},)."
                        )
                        logger.debug(dmsg)

            # Determine whether there are any tgt dims free to be mapped
            # by an available local tgt dim coordinate.
            tgt_dims_unmapped = (
                set(tgt_dim_coverage.dims_local) - tgt_dims_conflict
            )
        else:
            # For strict maths, only local tgt dim coordinates covering
            # the extra dimensions of the tgt cube may be added.
            tgt_dims_unmapped = extra_tgt_dims

        # Add local tgt dim coordinates.
        for tgt_dim in tgt_dims_unmapped:
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

        if not isinstance(src_dims, Iterable):
            src_dims = (src_dims,)

        if not isinstance(tgt_dims, Iterable):
            tgt_dims = (tgt_dims,)

        if src_coord.points.shape != tgt_coord.points.shape:
            # Check whether the src coordinate is broadcasting.
            dims = tuple([self.mapping[dim] for dim in src_dims])
            src_shape_broadcast = tuple([self.shape[dim] for dim in dims])
            src_cube_shape = self._src_cube.shape
            src_shape = tuple([src_cube_shape[dim] for dim in src_dims])
            src_broadcasting = src_shape != src_shape_broadcast

            # Check whether the tgt coordinate is broadcasting.
            tgt_shape_broadcast = tuple([self.shape[dim] for dim in tgt_dims])
            tgt_cube_shape = self._tgt_cube.shape
            tgt_shape = tuple([tgt_cube_shape[dim] for dim in tgt_dims])
            tgt_broadcasting = tgt_shape != tgt_shape_broadcast

            if src_broadcasting and tgt_broadcasting:
                emsg = (
                    f"Cannot broadcast the coordinate {src_coord.name()!r} on "
                    f"{self._src_cube_position} cube {self._src_cube.name()!r} and "
                    f"coordinate {tgt_coord.name()!r} on "
                    f"{self._tgt_cube_position} cube {self._tgt_cube.name()!r} to "
                    f"broadcast shape {tgt_shape_broadcast}."
                )
                raise ValueError(emsg)
            elif src_broadcasting:
                # Use the tgt coordinate points/bounds.
                points = tgt_coord.points
                bounds = tgt_coord.bounds
            elif tgt_broadcasting:
                # Use the src coordinate points/bounds.
                points = src_coord.points
                bounds = src_coord.bounds

        if points is None and bounds is None:
            # Note that, this also ensures shape equality.
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
    def _src_cube_resolved(self):
        if self.map_rhs_to_lhs:
            result = self.rhs_cube_resolved
        else:
            result = self.lhs_cube_resolved
        return result

    @_src_cube_resolved.setter
    def _src_cube_resolved(self, cube):
        if self.map_rhs_to_lhs:
            self.rhs_cube_resolved = cube
        else:
            self.lhs_cube_resolved = cube

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

    @property
    def _tgt_cube_resolved(self):
        if self.map_rhs_to_lhs:
            result = self.lhs_cube_resolved
        else:
            result = self.rhs_cube_resolved
        return result

    @_tgt_cube_resolved.setter
    def _tgt_cube_resolved(self, cube):
        if self.map_rhs_to_lhs:
            self.lhs_cube_resolved = cube
        else:
            self.rhs_cube_resolved = cube

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
                emsg = (
                    "Cannot resolve resultant cube, expected data with shape "
                    f"{shape}, got {data.shape}."
                )
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
                try:
                    result.add_aux_coord(coord, item.dims)
                except ValueError as err:
                    scalar = dims = ""
                    if item.dims:
                        plural = "s" if len(item.dims) > 1 else ""
                        dims = f" with tgt dim{plural} {item.dims}"
                    else:
                        scalar = "scalar "
                    dmsg = (
                        f"ignoring prepared {scalar}coordinate "
                        f"{coord.metadata}{dims}, got {err!r}"
                    )
                    logger.debug(dmsg)

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
        return self._src_cube.ndim == len(self.mapping)

    @property
    def shape(self):
        """The shape of the resultant resolved cube."""
        return self._broadcast_shape

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
