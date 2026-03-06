# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.fileformats._nc_load_rules.engine` module."""

import pytest

from iris.fileformats._nc_load_rules.engine import Engine, FactEntity


class Test_Engine:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.empty_engine = Engine()
        engine = Engine()
        engine.add_fact("this", ("that", "other"))
        self.nonempty_engine = engine

    def test__init(self):
        # Check that init creates an empty Engine.
        engine = Engine()
        assert isinstance(engine, Engine)
        assert isinstance(engine.facts, FactEntity)
        assert list(engine.facts.entity_lists.keys()) == []

    def test_reset(self):
        # Check that calling reset() causes a non-empty engine to be emptied.
        engine = self.nonempty_engine
        fact_names = list(engine.facts.entity_lists.keys())
        assert len(fact_names) != 0
        engine.reset()
        fact_names = list(engine.facts.entity_lists.keys())
        assert len(fact_names) == 0

    def test_activate(self, mocker):
        # Check that calling engine.activate() --> actions.run_actions(engine)
        engine = self.empty_engine
        target = "iris.fileformats._nc_load_rules.engine.run_actions"
        run_call = mocker.patch(target)
        engine.activate()
        assert run_call.call_args_list == [mocker.call(engine)]

    def test_add_case_specific_fact__newname(self):
        # Adding a new fact to a new fact-name records as expected.
        engine = self.nonempty_engine
        engine.add_case_specific_fact("new_fact", ("a1", "a2"))
        assert engine.fact_list("new_fact") == [("a1", "a2")]

    def test_add_case_specific_fact__existingname(self):
        # Adding a new fact to an existing fact-name records as expected.
        engine = self.nonempty_engine
        name = "this"
        assert engine.fact_list(name) == [("that", "other")]
        engine.add_case_specific_fact(name, ("yetanother",))
        assert engine.fact_list(name) == [("that", "other"), ("yetanother",)]

    def test_add_case_specific_fact__emptyargs(self):
        # Check that empty args work ok, and will create a new fact.
        engine = self.empty_engine
        engine.add_case_specific_fact("new_fact", ())
        assert "new_fact" in engine.facts.entity_lists
        assert engine.fact_list("new_fact") == [()]

    def test_add_fact(self, mocker):
        # Check that 'add_fact' is equivalent to (short for) a call to
        # 'add_case_specific_fact'.
        engine = self.empty_engine
        target = "iris.fileformats._nc_load_rules.engine.Engine.add_case_specific_fact"
        acsf_call = mocker.patch(target)
        engine.add_fact("extra", ())
        assert acsf_call.call_count == 1
        assert acsf_call.call_args_list == [
            mocker.call(fact_name="extra", fact_arglist=())
        ]

    def test_get_kb(self):
        # Check that this stub just returns the facts database.
        engine = self.nonempty_engine
        kb = engine.get_kb()
        assert isinstance(kb, FactEntity)
        assert kb is engine.facts

    def test_fact_list__existing(self):
        assert self.nonempty_engine.fact_list("this") == [("that", "other")]

    def test_fact_list__nonexisting(self):
        assert self.empty_engine.fact_list("odd-unknown") == []
