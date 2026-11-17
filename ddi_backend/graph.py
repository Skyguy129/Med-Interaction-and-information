# Module docstring: says the graph has medications as nodes and interactions as edges, and when to use it
"""The knowledge graph: medications are nodes, interactions are edges.

Built from the interactions table with plain dictionaries. Pair lookups for a
user's answer go through ``ingest.pair_records`` (a direct query); the graph is
for whole-graph questions (avoid lists) and the safer-alternative cross-check.
"""

from collections import defaultdict  # dict that creates a default value for missing keys, used for the indexes

from ddi_backend import ingest  # database access helpers (load_interactions, drug_names)
from ddi_backend.common import severity_rank  # imported from the shared helpers; not used directly in this file


class Graph:  # in-memory graph of drugs (nodes) and interaction records (edges)
    def __init__(self, records, drug_names=()):  # build all lookup tables once from the records
        self.records = list(records)  # keep a plain list copy of every interaction record
        self._pairs = defaultdict(list)       # frozenset({id_a, id_b}) -> [record]
        self._neighbors = defaultdict(set)    # drug_id -> {drug_id}
        self._names = {}                      # drug_id -> name
        self._ids = {}                        # casefolded name -> drug_id
        for record in self.records:  # walk every interaction record (one edge each)
            a, b = record["drug_a_id"], record["drug_b_id"]  # the two drug IDs at the ends of this edge
            self._pairs[frozenset((a, b))].append(record)  # file the record under the unordered pair, so A+B equals B+A
            self._neighbors[a].add(b)  # b is a neighbour of a
            self._neighbors[b].add(a)  # and a is a neighbour of b (edges have no direction)
            self._names[a], self._names[b] = record["drug_a"], record["drug_b"]  # remember each ID's display name
        for drug_id, name in self._names.items():  # build the reverse name -> ID table
            self._ids[name.casefold()] = drug_id  # case-insensitive key so lookups ignore capitalisation
        for name in drug_names:  # drugs that exist in the data but have no interaction row
            self._ids.setdefault(name.casefold(), None)  # known name with no ID here; keeps any ID already set

    @classmethod  # alternate constructor that takes a database connection
    def from_connection(cls, conn):
        return cls(ingest.load_interactions(conn), ingest.drug_names(conn))  # load all rows and all drug names, then build

    def drug_id(self, name):  # name -> drug ID
        """The drug ID for an exact (case-insensitive) name, or None."""
        return self._ids.get((name or "").casefold())  # treat None as empty text; unknown names give None

    def name(self, drug_id):  # drug ID -> name
        return self._names.get(drug_id)  # None when the ID is not in the graph

    def pair(self, id_a, id_b):  # all records between two drugs
        return sorted(self._pairs.get(frozenset((id_a, id_b)), []), key=lambda r: r["id"])  # order-free lookup, stable order by record ID

    def neighbors(self, drug_id):  # every drug that shares an interaction with this one
        return sorted(self._neighbors.get(drug_id, ()), key=lambda i: self._names[i].casefold())  # alphabetical by name

    def edge_count(self):  # number of distinct drug pairs (repeated pairs count once)
        return len(self._pairs)  # one entry per unordered pair


# ---------------------------------------------------------------------------
# FILE SUMMARY: Holds the in-memory knowledge graph. Drugs are nodes and each documented interaction is an edge, with
# indexes for pair lookups, neighbour lists and case-insensitive name-to-ID lookups. The checker and the rules use it
# for avoid lists and for cross-checking suggested safer alternatives.
# ASSIGNMENT TOOL: Knowledge graph with rule-based reasoning (the core engine). This file is the graph half; rules.py
# is the reasoning half.
# ---------------------------------------------------------------------------
