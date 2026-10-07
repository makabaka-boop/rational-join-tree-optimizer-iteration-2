#!/usr/bin/env python3
"""joinplan — exhaustive optimal join-order planner (pure stdlib, no database).

Reads one JSON problem from stdin (or from the file named in argv[1]) and
writes one JSON plan to stdout.

Input
-----
{
  "tables": [
    {"name": "A", "rows": 100}, ...          # 2..9 tables, unique printable
                                             # ASCII names without '(' / ')',
                                             # rows a positive integer
  ],
  "predicates": [                            # optional, default []
    {"left": "A", "right": "B", "selectivity": "1/10"}, ...
  ],                                         # several predicates per table
                                             # pair are allowed
  "selectivities2": ["1/5", ...]             # optional second-scenario
                                             # selectivities, exactly one per
                                             # predicate, aligned by index;
                                             # limits the input to 2..6 tables
  "materialized": {                          # optional cached join result
    "tables": ["A", "B"],                    # a proper subset, 2..n-1 tables,
                                             # whose induced predicate graph
                                             # is connected; limits the input
                                             # to 3..6 tables
    "rows": 2000,                            # observed row count (>= 0 int)
    "read_cost": "5",                        # cost of scanning the cache
                                             # (non-negative rational p/q)
    "rows2": 3000, "read_cost2": "7"         # required in dual-scenario mode
  }
}

A selectivity is a rational in [0, 1] given as "p/q" (or the integer 0 / 1).
Non-reduced fractions such as "2/4" are accepted and reduced internally.

Model
-----
Any binary join tree is allowed, but every merge must have at least one
predicate between its two sides.  The estimated row count of a subtree is the
product of its base-table row counts times the selectivities of *all*
predicates inside the subtree; the total cost of a tree is the sum of the
estimated row counts of its non-leaf nodes.

Single scenario (no "selectivities2"): among all minimum-cost trees the one
with the lexicographically smallest fully-parenthesized tree string wins,
where every internal node keeps its left/right children ordered by
tree-string byte order.

Dual scenario ("selectivities2" present): every tree is costed under both
data distributions with exact rationals — join legality is still decided by
the original predicate graph alone.  The chosen tree minimizes,
lexicographically:

  1. max(cost1, cost2)   — neither scenario may become too expensive,
  2. cost1 + cost2,
  3. the canonical tree string (as in the single-scenario rule).

To stay exact the subset DP keeps, per table set, every non-dominated
(cost1, cost2) pair and its witness strings, instead of only the single
cheapest subtree per scenario: a dominated pair can never be part of the
global optimum, but a non-dominated one may.

Optional cached join result ("materialized")
--------------------------------------------

A reporting engineer may already hold the materialized join of a proper,
internally connected subset of the tables (at least two tables, never all
of them).  When the field is present the input is limited to 3..6 tables
and the planner solves the component containing the subset twice:

* recompute: the ordinary problem above;
* cached: the subset is one indivisible leaf node with the *observed* row
  count instead of base-rows-times-internal-selectivities, and scanning it
  costs "read_cost" (so the read cost is paid once, inside the total, and
  the subset's internal predicates are never re-multiplied).  Predicates
  with both endpoints inside the subset are already applied and never
  appear in the cached tree; predicates crossing the subset boundary still
  take effect the first time their two sides are merged, together with the
  rest of that component exactly as usual.

The two plans are compared with the same objective as without the field:
minimum cost, or in dual mode (where "rows2"/"read_cost2" are mandatory)
(max(cost1, cost2), cost1 + cost2).  An exact tie prefers recompute.

Output
------
Connected predicate graph:
  {"status": "ok", "cost": "p/q", "tree_string": "((AB)C)", "tree": {...}}
Disconnected graph:
  {"status": "disconnected", "components": [{"tables": [...], "cost": "p/q",
     "tree_string": "...", "tree": {...}}, ...]}
Invalid input (exit code 2; nothing but the error object is emitted):
  {"status": "error", "error": "..."}

In dual-scenario mode every "cost" gains a "cost2" sibling, every join node
gains "rows2" next to "rows", and every predicate listed in a join node gains
"selectivity2".  Without "selectivities2" the output is exactly the document
described above, with no extra keys.

When "materialized" is given, the document gains a trailing "materialized"
block describing the chosen strategy ("chosen": "recompute" or "cached"),
the cached leaf node ("materialized_leaf"), the predicates covered by the
cache ("covered_predicates", input order), and the per-scenario costs of
both strategies ("recompute_cost"/"cached_cost", with "cost2" siblings in
dual mode).  A chosen cached tree embeds the same materialized leaf as
{"type": "materialized", "tables": [...], "rows": <int>, "read_cost": ...}
(plus "rows2"/"read_cost2" in dual mode) in place of the subset's subtree;
a chosen recompute plan's tree is the ordinary tree.  Without
"materialized" the output has none of these keys.

Tree nodes: ordinary leaves are {"type": "table", "name": ..., "rows": <int>};
in a cached plan the materialized subset replaces its whole subtree with one
{"type": "materialized", "tables": [...sorted...], "rows": <int>,
"read_cost": "p/q"} leaf (with "rows2"/"read_cost2" siblings in dual mode);
joins are {"type": "join", "rows": "p/q", "tables": [...sorted...],
"predicates": [<predicates taking effect for the first time at this merge, in
input order>], "children": [left, right]} with children ordered so that the
left child's tree string is byte-wise <= the right child's.
"""

import json
import re
import sys
from fractions import Fraction

MIN_TABLES = 2
MAX_TABLES = 9
MAX_TABLES_DUAL = 6  # table limit when "selectivities2" is given
MIN_TABLES_MATERIALIZED = 3  # table floor when "materialized" is given
MAX_TABLES_MATERIALIZED = 6  # table ceiling when "materialized" is given

_SELECTIVITY_RE = re.compile(r"([0-9]+)(?:/([0-9]+))?")


class InputError(Exception):
    """Raised when the problem statement is invalid."""


def format_fraction(value):
    """Render a Fraction as a reduced 'p/q' string."""
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_name(name):
    if not isinstance(name, str) or not name:
        raise InputError("table names must be non-empty strings")
    for ch in name:
        code = ord(ch)
        if code < 0x20 or code > 0x7E:
            raise InputError(f"table name {name!r} must be printable ASCII")
        if ch in "()":
            raise InputError(f"table name {name!r} must not contain '(' or ')'")


def _parse_selectivity(value):
    if _is_int(value):
        if value in (0, 1):
            return Fraction(value)
        raise InputError(f"selectivity must be within [0, 1], got {value}")
    if isinstance(value, str):
        match = _SELECTIVITY_RE.fullmatch(value.strip())
        if not match:
            raise InputError(
                f"selectivity {value!r} must look like 'p/q' with 0 <= p <= q"
            )
        num = int(match.group(1))
        den = int(match.group(2)) if match.group(2) is not None else 1
        if den == 0:
            raise InputError(f"selectivity {value!r} has a zero denominator")
        if num > den:
            raise InputError(f"selectivity {value!r} is greater than 1")
        return Fraction(num, den)
    raise InputError(
        f"selectivity must be a string 'p/q' or the integer 0/1, got {value!r}"
    )


def _parse_nonnegative_fraction(value, field):
    """Parse a non-negative rational ('p/q' or non-negative int)."""
    if _is_int(value):
        if value >= 0:
            return Fraction(value)
        raise InputError(f"{field} must be non-negative, got {value}")
    if isinstance(value, str):
        match = _SELECTIVITY_RE.fullmatch(value.strip())
        if not match:
            raise InputError(f"{field} {value!r} must look like 'p/q' with p >= 0")
        num = int(match.group(1))
        den = int(match.group(2)) if match.group(2) is not None else 1
        if den == 0:
            raise InputError(f"{field} {value!r} has a zero denominator")
        return Fraction(num, den)
    raise InputError(
        f"{field} must be a string 'p/q' or a non-negative integer, got {value!r}"
    )


def _nonnegative_int(value, field):
    if not _is_int(value) or value < 0:
        raise InputError(f"{field} must be a non-negative integer, got {value!r}")
    return value


def parse_problem(data):
    """Validate the raw JSON document.

    Returns (names, rows, predicates, selectivities2, materialized) where
    names is a list of table names in input order, rows maps name ->
    positive int, predicates is a list of (left, right, Fraction) in input
    order, selectivities2 is either None or a list of Fraction aligned with
    predicates by index, and materialized is either None or a dict with
    "tables" (sorted list), "rows" (non-negative int), "read_cost"
    (Fraction) and, in dual mode, "rows2" / "read_cost2".

    Everything is validated here, before any planning happens, so an invalid
    field can never lead to a partially computed plan.
    """
    if not isinstance(data, dict):
        raise InputError("the top level must be a JSON object")
    unknown = set(data) - {
        "tables", "predicates", "selectivities2", "materialized"
    }
    if unknown:
        raise InputError(f"unknown top-level keys: {sorted(unknown)}")
    if "tables" not in data:
        raise InputError("missing 'tables'")

    dual = "selectivities2" in data
    has_materialized = "materialized" in data
    if has_materialized:
        min_tables, max_tables = MIN_TABLES_MATERIALIZED, MAX_TABLES_MATERIALIZED
    else:
        min_tables, max_tables = MIN_TABLES, (
            MAX_TABLES_DUAL if dual else MAX_TABLES
        )

    tables = data["tables"]
    if not isinstance(tables, list):
        raise InputError("'tables' must be a list")
    if not MIN_TABLES <= len(tables) <= max_tables:
        raise InputError(
            f"need between {min_tables} and {max_tables} tables, "
            f"got {len(tables)}"
        )

    names = []
    rows = {}
    for entry in tables:
        if not isinstance(entry, dict) or set(entry) != {"name", "rows"}:
            raise InputError(
                "each table must be an object with exactly 'name' and 'rows'"
            )
        name = entry["name"]
        _check_name(name)
        if name in rows:
            raise InputError(f"duplicate table name {name!r}")
        count = entry["rows"]
        if not _is_int(count) or count <= 0:
            raise InputError(f"rows of table {name!r} must be a positive integer")
        rows[name] = count
        names.append(name)

    raw_predicates = data.get("predicates", [])
    if not isinstance(raw_predicates, list):
        raise InputError("'predicates' must be a list")
    predicates = []
    for entry in raw_predicates:
        if not isinstance(entry, dict) or set(entry) != {"left", "right", "selectivity"}:
            raise InputError(
                "each predicate must be an object with exactly "
                "'left', 'right' and 'selectivity'"
            )
        left, right = entry["left"], entry["right"]
        for endpoint in (left, right):
            if endpoint not in rows:
                raise InputError(f"predicate references unknown table {endpoint!r}")
        if left == right:
            raise InputError(f"predicate on {left!r} must join two distinct tables")
        predicates.append((left, right, _parse_selectivity(entry["selectivity"])))

    selectivities2 = None
    if dual:
        raw_second = data["selectivities2"]
        if not isinstance(raw_second, list):
            raise InputError("'selectivities2' must be a list")
        if len(raw_second) != len(predicates):
            raise InputError(
                "'selectivities2' must have exactly one entry per predicate "
                f"({len(predicates)}), got {len(raw_second)}"
            )
        selectivities2 = [_parse_selectivity(value) for value in raw_second]

    materialized = None
    if has_materialized:
        raw_mat = data["materialized"]
        required = {"tables", "rows", "read_cost"}
        if dual:
            required |= {"rows2", "read_cost2"}
        if not isinstance(raw_mat, dict):
            raise InputError("'materialized' must be an object")
        unknown_mat = set(raw_mat) - required
        if unknown_mat:
            raise InputError(
                f"unknown materialized keys: {sorted(unknown_mat)}"
            )
        missing = required - set(raw_mat)
        if missing:
            raise InputError(
                f"missing materialized fields: {sorted(missing)}"
            )
        raw_group = raw_mat["tables"]
        if not isinstance(raw_group, list) or not raw_group:
            raise InputError("materialized 'tables' must be a non-empty list")
        group = []
        for name in raw_group:
            if not isinstance(name, str):
                raise InputError(
                    f"materialized table names must be strings, got {name!r}"
                )
            if name not in rows:
                raise InputError(
                    f"materialized table {name!r} is not declared in 'tables'"
                )
            if name in group:
                raise InputError(
                    f"materialized table {name!r} listed more than once"
                )
            group.append(name)
        if len(group) < 2:
            raise InputError(
                "the materialized subset must contain at least two tables"
            )
        if len(group) >= len(names):
            raise InputError(
                "the materialized subset must be a proper subset of the tables"
            )
        members = set(group)
        induced = {
            (left, right)
            for left, right, _ in predicates
            if left in members and right in members
        }
        # connectivity of the subgraph induced by the subset: it must admit a
        # legal join tree under the original predicate graph
        reached = {group[0]}
        while True:
            added = {
                right if left in reached else left
                for left, right in induced
                if (left in reached) != (right in reached)
            }
            if not added:
                break
            reached |= added
        if reached != members:
            raise InputError(
                "the materialized subset cannot be joined: its induced "
                "predicate graph is disconnected"
            )
        materialized = {
            "tables": sorted(group),
            "rows": _nonnegative_int(raw_mat["rows"], "materialized 'rows'"),
            "read_cost": _parse_nonnegative_fraction(
                raw_mat["read_cost"], "materialized 'read_cost'"
            ),
        }
        if dual:
            materialized["rows2"] = _nonnegative_int(
                raw_mat["rows2"], "materialized 'rows2'"
            )
            materialized["read_cost2"] = _parse_nonnegative_fraction(
                raw_mat["read_cost2"], "materialized 'read_cost2'"
            )

    return names, rows, predicates, selectivities2, materialized


# ---------------------------------------------------------------------------
# planner core
# ---------------------------------------------------------------------------

def _fuse(left_string, right_string):
    """Canonical fully-parenthesized string of a join: children byte-sorted."""
    if left_string <= right_string:
        return "(" + left_string + right_string + ")"
    return "(" + right_string + left_string + ")"


def materialized_token(group, names):
    """Canonical tree-string token standing for the cached subset.

    It renders as the sorted subset wrapped in brackets, widened with extra
    brackets until it collides with no table name (the full set of names
    bounds the search), so it is a fresh leaf symbol whose place in byte
    order is well defined.
    """
    stem = "[" + "".join(group) + "]"
    token = stem
    while token in names:
        token = "[" + token + "]"
    return token


def _prefix_free(names):
    """True when no table name is a proper prefix of another name."""
    ordered = sorted(names)
    return all(
        not ordered[i + 1].startswith(ordered[i]) for i in range(len(ordered) - 1)
    )


class _ComponentPlanner:
    """Exact subset DP over the tables of one connected component.

    Single scenario (selectivities2 is None):

    cost[mask]   — minimum total cost over all legal join trees for `mask`
                   (None when the tables cannot be joined at all).
    trees[mask]  — dict mapping canonical tree string -> witness
                   (submask, left_string, right_string); a single
                   {name: None} entry for leaves.

    Dual scenario (selectivities2 given):

    pairs[mask]  — dict mapping each non-dominated (cost1, cost2) pair to a
                   dict of canonical tree string -> witness
                   (submask, left_pair, left_string, right_pair, right_string).

    When table names are prefix-free the canonical tree strings form a
    prefix-free code, so keeping only the single cheapest-and-smallest string
    per subset (per cost pair in dual mode) is exact.  With prefix-related
    names the lexicographic order of fused strings is not monotone in the
    children strings, so every tied string is kept (dual mode allows at most
    6 tables, which bounds this in practice).
    """

    def __init__(self, names, rows, predicates, selectivities2=None,
                 leaf_costs=None, display_endpoints=None, leaf_rows2=None):
        """Plan one connected component.

        leaf_costs optionally maps a component-local table name to the cost
        of scanning that leaf (used for the indivisible cached-result leaf);
        an absent entry costs 0 like an ordinary base table.  In dual mode
        values are (cost1, cost2) pairs, otherwise plain Fractions.

        leaf_rows2 (dual mode only) optionally maps a leaf name to its row
        count under the second scenario when it differs from `rows` (the
        cached-result leaf's observed rows2).

        display_endpoints optionally gives, per predicate in input order, the
        (left, right) names to emit in the output tree — used by the cached
        contraction whose internal pseudo-table token must appear as the
        original endpoint name.
        """
        self.names = list(names)
        self.n = len(names)
        index = {name: i for i, name in enumerate(self.names)}
        self.rows = [rows[name] for name in self.names]
        self.dual = selectivities2 is not None

        def leaf_cost(name):
            if leaf_costs is None or name not in leaf_costs:
                return (Fraction(0), Fraction(0)) if self.dual else Fraction(0)
            return leaf_costs[name]

        # Per-leaf row counts under the second scenario; identical to the
        # first scenario except for the cached-result leaf, whose observed
        # rows2 may differ from rows.
        rows2_override = dict(leaf_rows2) if (self.dual and leaf_rows2) else {}

        selectivity = [[Fraction(1)] * self.n for _ in range(self.n)]
        selectivity2 = (
            [[Fraction(1)] * self.n for _ in range(self.n)] if self.dual else None
        )
        adjacency = [0] * self.n
        self.predicates = []  # (left_bit, right_bit, left, right, sel, sel2)
        for k, (left, right, sel) in enumerate(predicates):
            i, j = index[left], index[right]
            selectivity[i][j] *= sel
            selectivity[j][i] *= sel
            sel2 = None
            if self.dual:
                sel2 = selectivities2[k]
                selectivity2[i][j] *= sel2
                selectivity2[j][i] *= sel2
            adjacency[i] |= 1 << j
            adjacency[j] |= 1 << i
            if display_endpoints is not None:
                show_left, show_right = display_endpoints[k]
            else:
                show_left, show_right = left, right
            self.predicates.append(
                (1 << i, 1 << j, show_left, show_right, sel, sel2)
            )

        size = 1 << self.n
        adjmask = [0] * size
        self.rows_est = [Fraction(0)] * size
        self.rows_est2 = [Fraction(0)] * size if self.dual else None
        self.mask_tables = [None] * size
        for mask in range(1, size):
            low = mask & (-mask)
            i = low.bit_length() - 1
            rest = mask ^ low
            adjmask[mask] = adjmask[rest] | adjacency[i]

            cut = Fraction(1)
            cut2 = Fraction(1) if self.dual else None
            bits = rest
            while bits:
                bit = bits & (-bits)
                j = bit.bit_length() - 1
                cut *= selectivity[i][j]
                if self.dual:
                    cut2 *= selectivity2[i][j]
                bits ^= bit

            leaf_rows_i = Fraction(self.rows[i])
            leaf_rows_i2 = None
            if self.dual:
                leaf_rows_i2 = Fraction(
                    rows2_override.get(self.names[i], self.rows[i])
                )
            if rest:
                # row(i) times the selectivities crossing the i/rest cut
                # times the estimate of rest: each leaf's rows and each
                # internal selectivity contributes exactly once.  For a
                # cached leaf row(i) is its observed cardinality, not the
                # product of its hidden base tables.
                estimate = leaf_rows_i * cut * self.rows_est[rest]
                if self.dual:
                    estimate2 = leaf_rows_i2 * cut2 * self.rows_est2[rest]
            else:
                estimate = leaf_rows_i
                estimate2 = leaf_rows_i2
            self.rows_est[mask] = estimate
            if self.dual:
                self.rows_est2[mask] = estimate2
            self.mask_tables[mask] = sorted(
                self.mask_tables[rest] + [self.names[i]] if rest else [self.names[i]]
            )
        self.adjmask = adjmask
        self.prefix_free = _prefix_free(self.names)

        if self.dual:
            self.pairs = [None] * size
            for mask in range(1, size):
                if mask & (mask - 1) == 0:
                    i = (mask & (-mask)).bit_length() - 1
                    c1, c2 = leaf_cost(self.names[i])
                    self.pairs[mask] = {(c1, c2): {self.names[i]: None}}
                else:
                    self._combine_dual(mask)
        else:
            self.cost = [None] * size
            self.trees = [None] * size
            for mask in range(1, size):
                if mask & (mask - 1) == 0:
                    i = (mask & (-mask)).bit_length() - 1
                    self.cost[mask] = leaf_cost(self.names[i])
                    self.trees[mask] = {self.names[i]: None}
                else:
                    self._combine(mask)

    def _combine(self, mask):
        low = mask & (-mask)
        best = None
        found = {}
        # enumerate each unordered split sub | other exactly once by forcing
        # the lowest table of `mask` into `sub`
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub & low and self.adjmask[sub] & other:
                left_cost, right_cost = self.cost[sub], self.cost[other]
                if left_cost is not None and right_cost is not None:
                    total = left_cost + right_cost + self.rows_est[mask]
                    if best is None or total < best:
                        best = total
                        found = {}
                    if total == best:
                        if self.prefix_free:
                            (left_string,) = self.trees[sub]
                            (right_string,) = self.trees[other]
                            fused = _fuse(left_string, right_string)
                            if not found or fused < next(iter(found)):
                                found = {fused: (sub, left_string, right_string)}
                        else:
                            for left_string in self.trees[sub]:
                                for right_string in self.trees[other]:
                                    found.setdefault(
                                        _fuse(left_string, right_string),
                                        (sub, left_string, right_string),
                                    )
            sub = (sub - 1) & mask
        self.cost[mask] = best
        self.trees[mask] = found

    def _combine_dual(self, mask):
        """Non-dominated (cost1, cost2) pairs of `mask` and their witnesses."""
        low = mask & (-mask)
        found = {}
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub & low and self.adjmask[sub] & other:
                for left_pair, left_strings in self.pairs[sub].items():
                    for right_pair, right_strings in self.pairs[other].items():
                        pair = (
                            left_pair[0] + right_pair[0] + self.rows_est[mask],
                            left_pair[1] + right_pair[1] + self.rows_est2[mask],
                        )
                        bucket = found.setdefault(pair, {})
                        for left_string in left_strings:
                            for right_string in right_strings:
                                bucket.setdefault(
                                    _fuse(left_string, right_string),
                                    (
                                        sub,
                                        left_pair,
                                        left_string,
                                        right_pair,
                                        right_string,
                                    ),
                                )
            sub = (sub - 1) & mask
        # keep only non-dominated pairs: scanning by ascending cost1, a pair
        # is dominated exactly when an earlier pair has cost2 <= its own
        kept = {}
        best_cost2 = None
        for pair in sorted(found):
            if best_cost2 is not None and pair[1] >= best_cost2:
                continue
            best_cost2 = pair[1]
            strings = found[pair]
            if self.prefix_free:
                smallest = min(strings)
                strings = {smallest: strings[smallest]}
            kept[pair] = strings
        self.pairs[mask] = kept

    def best_string(self, mask):
        return min(self.trees[mask])

    def best_dual(self, mask):
        """((cost1, cost2), tree_string) minimizing (max, sum, string)."""
        best = None
        for pair, strings in self.pairs[mask].items():
            candidate = (max(pair), pair[0] + pair[1], min(strings))
            if best is None or candidate < best[0]:
                best = (candidate, pair, candidate[2])
        return best[1], best[2]

    def _join_node(self, mask, sub, children):
        """Assemble the JSON-able dict of the join node splitting off `sub`."""
        effective = []
        for left_bit, right_bit, left, right, sel, sel2 in self.predicates:
            # first effective here: both endpoints inside this subtree and
            # separated by this merge's cut
            if (left_bit & mask) and (right_bit & mask) and (
                bool(left_bit & sub) != bool(right_bit & sub)
            ):
                entry = {
                    "left": left,
                    "right": right,
                    "selectivity": format_fraction(sel),
                }
                if self.dual:
                    entry["selectivity2"] = format_fraction(sel2)
                effective.append(entry)
        node = {"type": "join", "rows": format_fraction(self.rows_est[mask])}
        if self.dual:
            node["rows2"] = format_fraction(self.rows_est2[mask])
        node["tables"] = self.mask_tables[mask]
        node["predicates"] = effective
        node["children"] = children
        return node

    def build(self, mask, tree_string):
        """Materialize the chosen tree as a nested JSON-able dict."""
        witness = self.trees[mask][tree_string]
        if witness is None:
            i = (mask & (-mask)).bit_length() - 1
            return {"type": "table", "name": self.names[i], "rows": self.rows[i]}
        sub, left_string, right_string = witness
        other = mask ^ sub
        child_sub = self.build(sub, left_string)
        child_other = self.build(other, right_string)
        children = (
            [child_sub, child_other]
            if left_string <= right_string
            else [child_other, child_sub]
        )
        return self._join_node(mask, sub, children)

    def build_dual(self, mask, pair, tree_string):
        """Materialize the chosen dual-scenario tree."""
        witness = self.pairs[mask][pair][tree_string]
        if witness is None:
            i = (mask & (-mask)).bit_length() - 1
            return {"type": "table", "name": self.names[i], "rows": self.rows[i]}
        sub, left_pair, left_string, right_pair, right_string = witness
        other = mask ^ sub
        child_sub = self.build_dual(sub, left_pair, left_string)
        child_other = self.build_dual(other, right_pair, right_string)
        children = (
            [child_sub, child_other]
            if left_string <= right_string
            else [child_other, child_sub]
        )
        return self._join_node(mask, sub, children)


def _components(names, predicates):
    adjacency = {name: set() for name in names}
    for left, right, _ in predicates:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen = set()
    result = []
    for name in names:
        if name in seen:
            continue
        stack = [name]
        seen.add(name)
        component = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        result.append(sorted(component))
    result.sort(key=lambda tables: tables[0])
    return result


def _plan_component(tables, rows, predicates, selectivities2=None,
                    leaf_costs=None, display_endpoints=None, leaf_rows2=None):
    """Plan one connected component.

    Returns (cost, cost2, tree_string, tree); cost2 is None unless
    selectivities2 is given.  `rows` must provide the row count of every
    entry of `tables`; `leaf_costs` optionally charges a scan cost for
    individual leaves (see _ComponentPlanner); `display_endpoints`
    optionally rewrites predicate endpoint names in the output tree;
    `leaf_rows2` overrides leaf row counts under the second scenario.
    """
    zero = Fraction(0)
    if len(tables) == 1:
        name = tables[0]
        if leaf_costs is not None and name in leaf_costs:
            if selectivities2 is None:
                cost = leaf_costs[name]
                cost2 = None
            else:
                cost, cost2 = leaf_costs[name]
        else:
            cost, cost2 = zero, zero if selectivities2 is not None else None
        leaf_rows = rows[name]
        tree = {"type": "table", "name": name, "rows": leaf_rows}
        return cost, cost2, name, tree
    planner = _ComponentPlanner(
        tables, rows, predicates, selectivities2, leaf_costs,
        display_endpoints, leaf_rows2,
    )
    full = (1 << len(tables)) - 1
    if selectivities2 is None:
        cost = planner.cost[full]
        if cost is None:  # pragma: no cover - a connected component is always joinable
            raise AssertionError("connected component has no legal join tree")
        tree_string = planner.best_string(full)
        return cost, None, tree_string, planner.build(full, tree_string)
    if not planner.pairs[full]:  # pragma: no cover - same as above
        raise AssertionError("connected component has no legal join tree")
    pair, tree_string = planner.best_dual(full)
    return pair[0], pair[1], tree_string, planner.build_dual(full, pair, tree_string)


def _materialized_leaf(materialized, dual):
    """The JSON-able cached leaf node describing the whole subset."""
    leaf = {
        "type": "materialized",
        "tables": list(materialized["tables"]),
        "rows": materialized["rows"],
        "read_cost": format_fraction(materialized["read_cost"]),
    }
    if dual:
        leaf["rows2"] = materialized["rows2"]
        leaf["read_cost2"] = format_fraction(materialized["read_cost2"])
    return leaf


def _expand_cached_tree(node, token, group, leaf):
    """Replace the contracted token leaf with the materialized leaf, and each
    join node's pseudo-table token with the subset's real, sorted tables."""
    if node.get("type") == "table" and node.get("name") == token:
        return dict(leaf)
    if node["type"] == "join":
        children = [
            _expand_cached_tree(child, token, group, leaf)
            for child in node["children"]
        ]
        rewritten = dict(node)
        if token in node["tables"]:
            rewritten["tables"] = sorted(
                (set(node["tables"]) - {token}) | set(group)
            )
        rewritten["children"] = children
        return rewritten
    return node


def _plan_cached_component(tables, rows, predicates, selectivities2,
                           materialized):
    """Plan the component holding the subset as one indivisible cached leaf.

    Contracts the subset to a single pseudo-table `token`: predicates fully
    inside the subset are dropped (already applied to the observed rows),
    predicates crossing the boundary stay and take effect at their first
    merge, and the token leaf carries the observed rows and pays the cache
    read cost exactly once.
    """
    dual = selectivities2 is not None
    group = set(materialized["tables"])
    token = materialized_token(materialized["tables"], tables)

    kept_predicates = []
    kept_sel2 = []
    display_endpoints = []
    for k, (left, right, sel) in enumerate(predicates):
        if left in group and right in group:
            continue  # covered by the cache: must never be re-applied
        left_c = token if left in group else left
        right_c = token if right in group else right
        kept_predicates.append((left_c, right_c, sel))
        display_endpoints.append((left, right))
        if dual:
            kept_sel2.append(selectivities2[k])
    # selectivities2 reached us aligned with the component-local predicate
    # list, so it is filtered by the same skip above.

    contracted_tables = [t for t in tables if t not in group] + [token]
    contracted_rows = {name: rows[name] for name in tables if name not in group}
    contracted_rows[token] = materialized["rows"]
    if dual:
        leaf_costs = {
            token: (materialized["read_cost"], materialized["read_cost2"])
        }
    else:
        leaf_costs = {token: materialized["read_cost"]}

    cost, cost2, tree_string, tree = _plan_component(
        contracted_tables,
        contracted_rows,
        kept_predicates,
        kept_sel2 if dual else None,
        leaf_costs,
        display_endpoints,
        {token: materialized["rows2"]} if dual else None,
    )
    tree = _expand_cached_tree(
        tree, token, materialized["tables"],
        _materialized_leaf(materialized, dual),
    )
    return cost, cost2, tree_string, tree


def _component_entry(tables, cost, cost2, tree_string, tree):
    """Assemble one component block of the output document."""
    entry = {"tables": tables, "cost": format_fraction(cost)}
    if cost2 is not None:
        entry["cost2"] = format_fraction(cost2)
    entry["tree_string"] = tree_string
    entry["tree"] = tree
    return entry


def recompute_entry_cost(entry, second):
    """Read back the (second-scenario) cost fraction of a finished entry."""
    return Fraction(entry["cost2"] if second else entry["cost"])


def _objective(cost, cost2):
    """Comparison key of a finished plan: single min-cost, or dual lexicographic."""
    if cost2 is None:
        return (cost,)
    return (max(cost, cost2), cost + cost2)


def plan(names, rows, predicates, selectivities2=None, materialized=None):
    """Produce the plan JSON structure for an already-validated problem."""
    dual = selectivities2 is not None
    components = _components(names, predicates)

    group = set(materialized["tables"]) if materialized else None
    target_component = None
    if materialized:
        for component in components:
            if group <= set(component):
                target_component = component
                break
        # validation proved the subset induced graph connected, so its tables
        # share one component
        if target_component is None:  # pragma: no cover - ruled out upstream
            raise AssertionError("materialized subset spans no component")

    recompute_entries = []
    recompute_total = Fraction(0)
    recompute_total2 = Fraction(0) if dual else None
    cached_total = cached_total2 = None
    cached_target = None  # (index, cost, cost2, tree_string, tree)

    for idx, tables in enumerate(components):
        members = set(tables)
        indexes = [k for k, p in enumerate(predicates) if p[0] in members]
        local = [predicates[k] for k in indexes]
        local2 = [selectivities2[k] for k in indexes] if dual else None

        cost, cost2, tree_string, tree = _plan_component(
            tables, rows, local, local2
        )
        recompute_entries.append(
            _component_entry(tables, cost, cost2, tree_string, tree)
        )
        recompute_total += cost
        if dual:
            recompute_total2 += cost2

        if materialized and tables is target_component:
            cached_target = (
                idx,
                *_plan_cached_component(
                    tables, rows, local, local2, materialized
                ),
            )

    if materialized:
        target_idx, c_cost, c_cost2, c_string, c_tree = cached_target
        cached_entries = [dict(entry) for entry in recompute_entries]
        cached_entries[target_idx] = _component_entry(
            components[target_idx], c_cost, c_cost2, c_string, c_tree
        )
        cached_total = recompute_total - recompute_entry_cost(
            recompute_entries[target_idx], False
        ) + c_cost
        if dual:
            cached_total2 = (
                recompute_total2
                - recompute_entry_cost(recompute_entries[target_idx], True)
                + c_cost2
            )

        choose_cached = _objective(cached_total, cached_total2) < _objective(
            recompute_total, recompute_total2
        )  # an exact tie keeps recompute
        chosen_entries = cached_entries if choose_cached else recompute_entries
    else:
        choose_cached = False
        chosen_entries = recompute_entries

    if len(chosen_entries) == 1:
        only = chosen_entries[0]
        result = {"status": "ok", "cost": only["cost"]}
        if dual:
            result["cost2"] = only["cost2"]
        result["tree_string"] = only["tree_string"]
        result["tree"] = only["tree"]
    else:
        result = {"status": "disconnected", "components": chosen_entries}

    if materialized:
        covered = []
        for k, (left, right, sel) in enumerate(predicates):
            if left in group and right in group:
                item = {
                    "left": left,
                    "right": right,
                    "selectivity": format_fraction(sel),
                }
                if dual:
                    item["selectivity2"] = format_fraction(selectivities2[k])
                covered.append(item)
        block = {
            "chosen": "cached" if choose_cached else "recompute",
            "materialized_leaf": _materialized_leaf(materialized, dual),
            "covered_predicates": covered,
            "recompute_cost": format_fraction(recompute_total),
            "cached_cost": format_fraction(cached_total),
        }
        if dual:
            block["recompute_cost2"] = format_fraction(recompute_total2)
            block["cached_cost2"] = format_fraction(cached_total2)
        result["materialized"] = block

    return result


def plan_problem(data):
    """Validate a raw JSON document and plan it."""
    names, rows, predicates, selectivities2, materialized = parse_problem(data)
    return plan(names, rows, predicates, selectivities2, materialized)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

_USAGE = """usage: joinplan.py [problem.json]

Reads a join-planning problem as JSON (from the file argument, or stdin) and
writes the optimal plan as JSON to stdout.  With the optional "selectivities2"
array (one second-scenario selectivity per predicate, 2..6 tables) the plan
minimizes max(cost1, cost2), then cost1 + cost2, then the tree string.  With
the optional "materialized" object (3..6 tables) the recompute plan is also
compared against scanning the cached subset as one indivisible leaf; ties
prefer recompute.  Exit code is 0 for a successful plan (including
disconnected inputs) and 2 for invalid input."""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    try:
        if argv:
            with open(argv[0], "r", encoding="utf-8") as handle:
                raw = handle.read()
        else:
            raw = sys.stdin.read()
    except OSError as exc:
        print(json.dumps({"status": "error", "error": f"cannot read input: {exc}"}))
        return 2
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "error", "error": f"invalid JSON: {exc}"}))
        return 2
    try:
        result = plan_problem(data)
    except InputError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
