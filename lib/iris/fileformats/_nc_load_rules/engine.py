"""
A simple mimic of the Pyke 'knowledge_engine', for interfacing to the routines
in 'iris.fileformats.netcdf' with minimal changes to that code.

This allows us to replace the Pyke rules operation with the simpler pure-Python
translation operations in :mod:`iris.fileformats._nc_load_rules.actions`.

The core of this is the 'Engine' class, which mimics the Pyke engine operations,
as used by our code to translate each data cube.

engine.get_kb() also returns a FactEntity object, which mimics *just enough*
API of a Pyke.knowlege_base, so that we can list its case-specific facts, as
used in :meth:`iris.fileformats.netcdf.pyke_stats`.

"""
from .actions import run_actions


class FactList:
    def __init__(self):
        self.case_specific_facts = []


class FactEntity:
    # To support:
    """
    kb_facts = engine.get_kb(_PYKE_FACT_BASE)

    for key in kb_facts.entity_lists.keys():
        for arg in kb_facts.entity_lists[key].case_specific_facts:
            print("\t%s%s" % (key, arg))

    """

    def __init__(self):
        self.entity_lists = {}

    def add_fact(self, fact_name, args):
        if fact_name not in self.entity_lists:
            self.entity_lists[fact_name] = FactList()
        fact_list = self.entity_lists[fact_name]
        fact_list.case_specific_facts.append(tuple(args))

    def sect_facts(self, entity_name):
        if entity_name in self.entity_lists:
            facts = self.entity_lists.get(entity_name).case_specific_facts
        else:
            facts = []
        return facts


class Engine:
    """
    A minimal mimic of a Pyke.engine.

    Provides just enough API so that the existing code in
    :mod:`iris.fileformats.netcdf` can interface with our new rules functions.

    """

    def __init__(self):
        """Init new engine."""
        self.reset()

    def reset(self):
        """Reset the engine = remove all facts."""
        self.facts = FactEntity()

    def activate(self, rules_base_str=None):
        """
        Run all the translation rules to produce a single output cube.

        This implicitly references the output variable for this operation,
        set by engine.cf_var (the variable name).

        The rules operation itself is coded elsewhere,
        in :mod:`iris.fileformats.netcdf._nc_load_rules.rules`.

        """
        run_actions(self)

    def print_stats(self):
        """No-op, called by :meth:`iris.fileformats.netcdf.pyke_stats`."""
        pass

    def add_case_specific_fact(self, kb_name, fact_name, fact_arglist):
        """
        Record a fact about the current output operation.

        Roughly, self.facts.entity_lists[fact_name].append(fact_arglist).

        """
        self.facts.add_fact(fact_name, fact_arglist)

    def get_kb(self, fact_base_str=None):
        """
        Get a FactEntity, which mimic (bits of) a knowledge-base.

        Just allowing
        :meth:`iris.fileformats.netcdf.pyke_stats` to list the facts.

        """
        return self.facts

    def fact_list(self, fact_name):
        """
        Return the facts (arg-lists) for one fact name.

        A shorthand form used only by the new rules routines.

        AKA 'case-specific-facts', in the original.
        Roughly "return self.facts.entity_lists[fact_name]".

        """
        return self.facts.sect_facts(fact_name)

    def add_fact(self, fact_name, fact_arglist):
        """
        Add a new fact.

        A shorthand form used only by the new rules routines.

        """
        self.add_case_specific_fact(
            kb_name="<n/a>", fact_name=fact_name, fact_arglist=fact_arglist
        )
