"""The knowledge graph: medications are nodes, interactions are edges.

Built from the interactions table with plain dictionaries. Pair lookups for a
user's answer go through ``ingest.pair_records`` (a direct query); the graph is
for whole-graph questions (avoid lists) and the safer-alternative cross-check.
"""

from collections import defaultdict

from ddi_backend import ingest
from ddi_backend.common import severity_rank


class Graph:
    def __init__(self, records, drug_names=()):
        self.records = list(records)
        self._pairs = defaultdict(list)       # frozenset({id_a, id_b}) -> [record]
        self._neighbors = defaultdict(set)    # drug_id -> {drug_id}
        self._names = {}                      # drug_id -> name
        self._ids = {}                        # casefolded name -> drug_id
        for record in self.records:
            a, b = record["drug_a_id"], record["drug_b_id"]
            self._pairs[frozenset((a, b))].append(record)
            self._neighbors[a].add(b)
            self._neighbors[b].add(a)
            self._names[a], self._names[b] = record["drug_a"], record["drug_b"]
        for drug_id, name in self._names.items():
            self._ids[name.casefold()] = drug_id
        for name in drug_names:
            self._ids.setdefault(name.casefold(), None)

    @classmethod
    def from_connection(cls, conn):
        return cls(ingest.load_interactions(conn), ingest.drug_names(conn))

    def drug_id(self, name):
        """The drug ID for an exact (case-insensitive) name, or None."""
        return self._ids.get((name or "").casefold())

    def name(self, drug_id):
        return self._names.get(drug_id)

    def pair(self, id_a, id_b):
        return sorted(self._pairs.get(frozenset((id_a, id_b)), []), key=lambda r: r["id"])

    def neighbors(self, drug_id):
        return sorted(self._neighbors.get(drug_id, ()), key=lambda i: self._names[i].casefold())

    def edge_count(self):
        return len(self._pairs)
