# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the :mod:`iris.fileformats._nc_load_rules.engine` module."""

from unittest import mock

from iris.fileformats._nc_load_rules.engine import Engine, FactEntity
import iris.tests as tests


class Test_Engine(tests.IrisTest):
    def setUp(self):
        self.empty_engine = Engine()
        engine = Engine()
        engine.add_fact("this", ("that", "other"))
        self.nonempty_engine = engine

    def test__init(self):
        # Check that init creates an empty Engine.
        engine = Engine()
        self.assertIsInstance(engine, Engine)
        self.assertIsInstance(engine.facts, FactEntity)
        self.assertEqual(list(engine.facts.entity_lists.keys()), [])

    def test_reset(self):
        # Check that calling reset() causes a non-empty engine to be emptied.
        engine = self.nonempty_engine
        fact_names = list(engine.facts.entity_lists.keys())
        self.assertNotEqual(len(fact_names), 0)
        engine.reset()
        fact_names = list(engine.facts.entity_lists.keys())
        self.assertEqual(len(fact_names), 0)

    def test_activate(self):
        # Check that calling engine.activate() --> actions.run_actions(engine)
        engine = self.empty_engine
        target = "iris.fileformats._nc_load_rules.engine.run_actions"
        run_call = self.patch(target)
        engine.activate()
        self.assertEqual(run_call.call_args_list, [mock.call(engine)])

    def test_add_case_specific_fact__newname(self):
        # Adding a new fact to a new fact-name records as expected.
        engine = self.nonempty_engine
        engine.add_case_specific_fact("new_fact", ("a1", "a2"))
        self.assertEqual(engine.fact_list("new_fact"), [("a1", "a2")])

    def test_add_case_specific_fact__existingname(self):
        # Adding a new fact to an existing fact-name records as expected.
        engine = self.nonempty_engine
        name = "this"
        self.assertEqual(engine.fact_list(name), [("that", "other")])
        engine.add_case_specific_fact(name, ("yetanother",))
        self.assertEqual(engine.fact_list(name), [("that", "other"), ("yetanother",)])

    def test_add_case_specific_fact__emptyargs(self):
        # Check that empty args work ok, and will create a new fact.
        engine = self.empty_engine
        engine.add_case_specific_fact("new_fact", ())
        self.assertIn("new_fact", engine.facts.entity_lists)
        self.assertEqual(engine.fact_list("new_fact"), [()])

    def test_add_fact(self):
        # Check that 'add_fact' is equivalent to (short for) a call to
        # 'add_case_specific_fact'.
        engine = self.empty_engine
        target = "iris.fileformats._nc_load_rules.engine.Engine.add_case_specific_fact"
        acsf_call = self.patch(target)
        engine.add_fact("extra", ())
        self.assertEqual(acsf_call.call_count, 1)
        self.assertEqual(
            acsf_call.call_args_list,
            [mock.call(fact_name="extra", fact_arglist=())],
        )

    def test_get_kb(self):
        # Check that this stub just returns the facts database.
        engine = self.nonempty_engine
        kb = engine.get_kb()
        self.assertIsInstance(kb, FactEntity)
        self.assertIs(kb, engine.facts)

    def test_fact_list__existing(self):
        self.assertEqual(self.nonempty_engine.fact_list("this"), [("that", "other")])

    def test_fact_list__nonexisting(self):
        self.assertEqual(self.empty_engine.fact_list("odd-unknown"), [])


if __name__ == "__main__":
    tests.main()
