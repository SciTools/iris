# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.experimental.mesh_coord_indexing` module."""

import threading

import pytest

from iris.experimental import mesh_coord_indexing


@pytest.fixture(autouse=True)
def restore_setting_value():
    """Keep global runtime setting isolated between tests."""
    original_value = mesh_coord_indexing.SETTING.value
    try:
        yield
    finally:
        mesh_coord_indexing.SETTING.value = original_value


class TestOptions:
    def test_members(self):
        assert mesh_coord_indexing.Options.AUX_COORD.name == "AUX_COORD"
        assert mesh_coord_indexing.Options.NEW_MESH.name == "NEW_MESH"
        assert mesh_coord_indexing.Options.MESH_INDEX_SET.name == "MESH_INDEX_SET"


class TestSettingValue:
    def test_default(self):
        assert (
            mesh_coord_indexing.SETTING.value == mesh_coord_indexing.Options.AUX_COORD
        )

    def test_set_enum_member(self):
        mesh_coord_indexing.SETTING.value = mesh_coord_indexing.Options.NEW_MESH

        assert mesh_coord_indexing.SETTING.value == mesh_coord_indexing.Options.NEW_MESH

    def test_set_member_value(self):
        mesh_coord_indexing.SETTING.value = (
            mesh_coord_indexing.Options.MESH_INDEX_SET.value
        )

        assert (
            mesh_coord_indexing.SETTING.value
            == mesh_coord_indexing.Options.MESH_INDEX_SET
        )

    def test_invalid_value(self):
        with pytest.raises(ValueError, match="is not a valid Options"):
            mesh_coord_indexing.SETTING.value = "not-an-option"


class TestSettingContext:
    def test_temporary_override(self):
        assert (
            mesh_coord_indexing.SETTING.value == mesh_coord_indexing.Options.AUX_COORD
        )

        with mesh_coord_indexing.SETTING.context(mesh_coord_indexing.Options.NEW_MESH):
            assert (
                mesh_coord_indexing.SETTING.value
                == mesh_coord_indexing.Options.NEW_MESH
            )

        assert (
            mesh_coord_indexing.SETTING.value == mesh_coord_indexing.Options.AUX_COORD
        )

    def test_restores_after_exception(self):
        class LocalTestException(Exception):
            pass

        with pytest.raises(LocalTestException):
            with mesh_coord_indexing.SETTING.context(
                mesh_coord_indexing.Options.MESH_INDEX_SET
            ):
                raise LocalTestException

        assert (
            mesh_coord_indexing.SETTING.value == mesh_coord_indexing.Options.AUX_COORD
        )

    def test_invalid_context_value(self):
        with pytest.raises(ValueError, match="is not a valid Options"):
            with mesh_coord_indexing.SETTING.context("not-an-option"):
                pass


class TestThreadLocalSetting:
    def test_independent_values_per_thread(self):
        thread_state = {}

        mesh_coord_indexing.SETTING.value = mesh_coord_indexing.Options.NEW_MESH

        def worker():
            thread_state["before"] = mesh_coord_indexing.SETTING.value
            mesh_coord_indexing.SETTING.value = (
                mesh_coord_indexing.Options.MESH_INDEX_SET
            )
            thread_state["during"] = mesh_coord_indexing.SETTING.value

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

        assert thread_state["before"] == mesh_coord_indexing.Options.AUX_COORD
        assert thread_state["during"] == mesh_coord_indexing.Options.MESH_INDEX_SET
        assert mesh_coord_indexing.SETTING.value == mesh_coord_indexing.Options.NEW_MESH
