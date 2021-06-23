# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""
A simple mimic of the Pyke 'knowledge_engine', for interfacing to the routines
in 'iris.fileformats.netcdf' with minimal changes to that code.

This allows us to replace the Pyke rules operation with the simpler pure-Python
translation operations in :mod:`iris.fileformats._nc_load_rules.actions`.

The core of this is the 'Engine' class, which mimics the Pyke engine operations,
as used by our code to translate each data cube.

engine.get_kb() also returns a FactEntity object, which mimics *just enough*
API of a Pyke.knowlege_base, so that we can list its case-specific facts, as
used in :meth:`iris.fileformats.netcdf._actions_activation_stats`.

"""
from .actions import run_actions


class FactEntity:
    """
    An object with an 'entity_lists' property  which is a dict of 'FactList's.

    A Factlist, in turn, is an object with property 'case_specific_facts',
    which is a list of tuples of strings
    (each of which is a 'fact' of the named class).

    To support the debug code :
        kb_facts = engine.get_kb(_PYKE_FACT_BASE)
        for key in kb_facts.entity_lists.keys():
            for arg in kb_facts.entity_lists[key].case_specific_facts:
                print("\t%s%s" % (key, arg))

    """

    def __init__(self):
        self.entity_lists = {}

    class _FactList:
        # Just "an object with a 'case_specific_facts' property" (which is a list).
        def __init__(self):
            self.case_specific_facts = []

    def add_fact(self, fact_name, args):
        # Add a fact "fact_name(*args)".
        if fact_name not in self.entity_lists:
            self.entity_lists[fact_name] = self._FactList()
        fact_list = self.entity_lists[fact_name]
        fact_list.case_specific_facts.append(tuple(args))

    def sect_facts(self, fact_name):
        # Lookup all facts "fact_name(*args)" for a given fact_name.
        if fact_name in self.entity_lists:
            facts = self.entity_lists.get(fact_name).case_specific_facts
        else:
            facts = []
        return facts


class Engine:
    """
    A minimal mimic of a Pyke.engine.

    Provides just enough API so that the existing code in
    :mod:`iris.fileformats.netcdf` can interface with our new rules functions.

    A list of possible fact-arglists is stored, for each of a set of fact-names
    (which are strings).
    Each fact-argslist is represented by a tuple of values
    -- at present, in practice, those are all strings too.

    """

    def __init__(self):
        """Init new engine."""
        self.reset()

    def reset(self):
        """Reset the engine = remove all facts."""
        self.facts = FactEntity()

    def activate(self):
        """
        Run all the translation rules to produce a single output cube.

        This implicitly references the output variable for this operation,
        set by engine.cf_var (a CFDataVariable).

        The rules operation itself is coded elsewhere,
        in :mod:`iris.fileformats.netcdf._nc_load_rules.actions`.

        """
        run_actions(self)

    def get_kb(self):
        """
        Get a FactEntity, which mimic (bits of) a knowledge-base.

        Just allowing
        :meth:`iris.fileformats.netcdf._action_activation_stats` to list the
        facts.

        """
        return self.facts

    def print_stats(self):
        """
        No-op, called by
        :meth:`iris.fileformats.netcdf._action_activation_stats`.

        """
        pass

    def add_case_specific_fact(self, fact_name, fact_arglist):
        """
        Record a fact about the current output operation.

        Roughly,
          facts = self.facts.entity_lists[fact_name].case_specific_facts
          facts.append(fact_arglist)

        """
        self.facts.add_fact(fact_name, fact_arglist)

    def fact_list(self, fact_name):
        """
        Return the facts (arg-lists) for one fact name.

        A shorthand form used only by the new 'actions' routines.

        AKA 'case-specific-facts', in the original.
        Roughly = "self.facts.entity_lists[fact_name].case_specific_facts".

        """
        return self.facts.sect_facts(fact_name)

    def add_fact(self, fact_name, fact_arglist):
        """
        Add a new fact.

        A shorthand form used only by the new 'actions' routines.

        """
        self.add_case_specific_fact(
            fact_name=fact_name, fact_arglist=fact_arglist
        )
