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

Tree nodes: leaves are {"type": "table", "name": ..., "rows": <int>}; joins
are {"type": "join", "rows": "p/q", "tables": [...sorted...],
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


def parse_problem(data):
    """Validate the raw JSON document.

    Returns (names, rows, predicates, selectivities2) where names is a list
    of table names in input order, rows maps name -> positive int, predicates
    is a list of (left, right, Fraction) in input order, and selectivities2
    is either None or a list of Fraction aligned with predicates by index.

    Everything is validated here, before any planning happens, so an invalid
    fraction can never lead to a partially computed plan.
    """
    if not isinstance(data, dict):
        raise InputError("the top level must be a JSON object")
    unknown = set(data) - {"tables", "predicates", "selectivities2"}
    if unknown:
        raise InputError(f"unknown top-level keys: {sorted(unknown)}")
    if "tables" not in data:
        raise InputError("missing 'tables'")

    dual = "selectivities2" in data
    max_tables = MAX_TABLES_DUAL if dual else MAX_TABLES

    tables = data["tables"]
    if not isinstance(tables, list):
        raise InputError("'tables' must be a list")
    if not MIN_TABLES <= len(tables) <= max_tables:
        raise InputError(
            f"need between {MIN_TABLES} and {max_tables} tables, "
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

    return names, rows, predicates, selectivities2


# ---------------------------------------------------------------------------
# planner core
# ---------------------------------------------------------------------------

def _fuse(left_string, right_string):
    """Canonical fully-parenthesized string of a join: children byte-sorted."""
    if left_string <= right_string:
        return "(" + left_string + right_string + ")"
    return "(" + right_string + left_string + ")"


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

    def __init__(self, names, rows, predicates, selectivities2=None):
        self.names = list(names)
        self.n = len(names)
        index = {name: i for i, name in enumerate(self.names)}
        self.rows = [rows[name] for name in self.names]
        self.dual = selectivities2 is not None

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
            self.predicates.append((1 << i, 1 << j, left, right, sel, sel2))

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
            estimate = Fraction(self.rows[i])
            estimate2 = Fraction(self.rows[i]) if self.dual else None
            bits = rest
            while bits:
                bit = bits & (-bits)
                j = bit.bit_length() - 1
                estimate *= selectivity[i][j]
                if self.dual:
                    estimate2 *= selectivity2[i][j]
                bits ^= bit
            if rest:
                estimate *= self.rows_est[rest]
                if self.dual:
                    estimate2 *= self.rows_est2[rest]
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
                    self.pairs[mask] = {
                        (Fraction(0), Fraction(0)): {self.names[i]: None}
                    }
                else:
                    self._combine_dual(mask)
        else:
            self.cost = [None] * size
            self.trees = [None] * size
            for mask in range(1, size):
                if mask & (mask - 1) == 0:
                    i = (mask & (-mask)).bit_length() - 1
                    self.cost[mask] = Fraction(0)
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


def _plan_component(tables, rows, predicates, selectivities2=None):
    """Plan one connected component.

    Returns (cost, cost2, tree_string, tree); cost2 is None unless
    selectivities2 is given.
    """
    if len(tables) == 1:
        name = tables[0]
        tree = {"type": "table", "name": name, "rows": rows[name]}
        zero = Fraction(0)
        return zero, zero if selectivities2 is not None else None, name, tree
    planner = _ComponentPlanner(tables, rows, predicates, selectivities2)
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


def plan(names, rows, predicates, selectivities2=None):
    """Produce the plan JSON structure for an already-validated problem."""
    dual = selectivities2 is not None
    components = _components(names, predicates)
    planned = []
    for tables in components:
        members = set(tables)
        indexes = [k for k, p in enumerate(predicates) if p[0] in members]
        local = [predicates[k] for k in indexes]
        local2 = [selectivities2[k] for k in indexes] if dual else None
        cost, cost2, tree_string, tree = _plan_component(tables, rows, local, local2)
        entry = {"tables": tables, "cost": format_fraction(cost)}
        if dual:
            entry["cost2"] = format_fraction(cost2)
        entry["tree_string"] = tree_string
        entry["tree"] = tree
        planned.append(entry)
    if len(planned) == 1:
        only = planned[0]
        result = {"status": "ok", "cost": only["cost"]}
        if dual:
            result["cost2"] = only["cost2"]
        result["tree_string"] = only["tree_string"]
        result["tree"] = only["tree"]
        return result
    return {"status": "disconnected", "components": planned}


def plan_problem(data):
    """Validate a raw JSON document and plan it."""
    names, rows, predicates, selectivities2 = parse_problem(data)
    return plan(names, rows, predicates, selectivities2)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

_USAGE = """usage: joinplan.py [problem.json]

Reads a join-planning problem as JSON (from the file argument, or stdin) and
writes the optimal plan as JSON to stdout.  With the optional "selectivities2"
array (one second-scenario selectivity per predicate, 2..6 tables) the plan
minimizes max(cost1, cost2), then cost1 + cost2, then the tree string.  Exit
code is 0 for a successful plan (including disconnected inputs) and 2 for
invalid input."""


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
