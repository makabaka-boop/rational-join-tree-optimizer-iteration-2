"""Tests for joinplan.

For small graphs every legal binary join tree is enumerated independently
(no per-subset pruning) and the planner's cost and tie-breaking are checked
against that brute force.  The returned tree is also validated structurally:
node row counts, first-effective predicates, child ordering, legality of
every merge, and the total cost.

A second suite supplies "selectivities2" and duels the dual-scenario
objective (minimize max(cost1, cost2), then cost1 + cost2, then the canonical
tree string) against an equally exhaustive enumerator, validates the
per-node "rows2"/"selectivity2" estimates and "cost2" totals, checks that
identical scenarios reproduce the single-scenario plan exactly, and covers
dual-mode input validation and the CLI.
"""

import json
import random
import subprocess
import sys
from functools import lru_cache
from fractions import Fraction
from pathlib import Path

import pytest

import joinplan
from joinplan import InputError, format_fraction, plan_problem

HERE = Path(__file__).resolve().parent
PLANNER = HERE / "joinplan.py"


# ---------------------------------------------------------------------------
# independent brute force (no pruning: keeps every legal tree)
# ---------------------------------------------------------------------------

def fuse(left, right):
    if left <= right:
        return "(" + left + right + ")"
    return "(" + right + left + ")"


def components_of(names, predicates):
    adjacency = {name: set() for name in names}
    for left, right, _ in predicates:
        adjacency[left].add(right)
        adjacency[right].add(left)
    seen, result = set(), []
    for name in names:
        if name in seen:
            continue
        stack, component = [name], []
        seen.add(name)
        while stack:
            node = stack.pop()
            component.append(node)
            for other in adjacency[node]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        result.append(sorted(component))
    return result


def brute_best(names, rows, predicates):
    """Exhaustively enumerate all legal join trees of one component.

    Returns (cost, canonical_tree_string) of the best tree, or None when the
    tables cannot be joined.
    """
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    selectivity = [[Fraction(1)] * n for _ in range(n)]
    adjacent = [[False] * n for _ in range(n)]
    for left, right, sel in predicates:
        i, j = index[left], index[right]
        selectivity[i][j] *= sel
        selectivity[j][i] *= sel
        adjacent[i][j] = adjacent[j][i] = True

    size = 1 << n
    rows_of = [None] * size
    for mask in range(1, size):
        estimate = Fraction(1)
        for i in range(n):
            if mask >> i & 1:
                estimate *= rows[names[i]]
        for i in range(n):
            for j in range(i + 1, n):
                if mask >> i & 1 and mask >> j & 1:
                    estimate *= selectivity[i][j]
        rows_of[mask] = estimate

    def has_edge(sub, other):
        for i in range(n):
            if not (sub >> i & 1):
                continue
            for j in range(n):
                if other >> j & 1 and adjacent[i][j]:
                    return True
        return False

    @lru_cache(maxsize=None)
    def all_trees(mask):
        """All legal trees of `mask` as (cost, canonical string) pairs."""
        if mask & (mask - 1) == 0:
            i = (mask & -mask).bit_length() - 1
            return ((Fraction(0), names[i]),)
        out = []
        low = mask & -mask
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub & low and other and has_edge(sub, other):
                for left_cost, left_str in all_trees(sub):
                    for right_cost, right_str in all_trees(other):
                        out.append(
                            (
                                left_cost + right_cost + rows_of[mask],
                                fuse(left_str, right_str),
                            )
                        )
            sub = (sub - 1) & mask
        return tuple(out)

    candidates = all_trees(size - 1)
    if not candidates:
        return None
    return min(candidates)


def brute_best_dual(names, rows, predicates, selectivities2):
    """Exhaustively enumerate every legal join tree of one component.

    Returns (cost1, cost2, canonical_tree_string) of the tree minimizing
    (max(cost1, cost2), cost1 + cost2, tree_string), or None when the tables
    cannot be joined.  Independent of the planner's Pareto pruning: it keeps
    every tree.
    """
    n = len(names)
    index = {name: i for i, name in enumerate(names)}
    sel1 = [[Fraction(1)] * n for _ in range(n)]
    sel2 = [[Fraction(1)] * n for _ in range(n)]
    adjacent = [[False] * n for _ in range(n)]
    for (left, right, first), second in zip(predicates, selectivities2):
        i, j = index[left], index[right]
        sel1[i][j] *= first
        sel1[j][i] *= first
        sel2[i][j] *= second
        sel2[j][i] *= second
        adjacent[i][j] = adjacent[j][i] = True

    size = 1 << n
    rows1_of = [None] * size
    rows2_of = [None] * size
    for mask in range(1, size):
        base = Fraction(1)
        for i in range(n):
            if mask >> i & 1:
                base *= rows[names[i]]
        estimate1, estimate2 = base, base
        for i in range(n):
            for j in range(i + 1, n):
                if mask >> i & 1 and mask >> j & 1:
                    estimate1 *= sel1[i][j]
                    estimate2 *= sel2[i][j]
        rows1_of[mask] = estimate1
        rows2_of[mask] = estimate2

    def has_edge(sub, other):
        for i in range(n):
            if not (sub >> i & 1):
                continue
            for j in range(n):
                if other >> j & 1 and adjacent[i][j]:
                    return True
        return False

    @lru_cache(maxsize=None)
    def all_trees(mask):
        """All legal trees of `mask` as (cost1, cost2, canonical string)."""
        if mask & (mask - 1) == 0:
            i = (mask & -mask).bit_length() - 1
            return ((Fraction(0), Fraction(0), names[i]),)
        out = []
        low = mask & -mask
        sub = (mask - 1) & mask
        while sub:
            other = mask ^ sub
            if sub & low and other and has_edge(sub, other):
                for c1, c2, left_str in all_trees(sub):
                    for d1, d2, right_str in all_trees(other):
                        out.append(
                            (
                                c1 + d1 + rows1_of[mask],
                                c2 + d2 + rows2_of[mask],
                                fuse(left_str, right_str),
                            )
                        )
            sub = (sub - 1) & mask
        return tuple(out)

    candidates = all_trees(size - 1)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            max(candidate[0], candidate[1]),
            candidate[0] + candidate[1],
            candidate[2],
        ),
    )


# ---------------------------------------------------------------------------
# structural validation of the planner's tree
# ---------------------------------------------------------------------------

def parse_frac(text):
    num, den = text.split("/")
    return Fraction(int(num), int(den))


def walk_tree(node, rows, predicates):
    """Validate a plan tree against the raw problem.

    Returns (tables, canonical_string, internal_cost, used_predicate_indexes).
    """
    if node["type"] == "table":
        assert set(node) == {"type", "name", "rows"}
        assert node["rows"] == rows[node["name"]]
        return {node["name"]}, node["name"], Fraction(0), []

    assert node["type"] == "join"
    assert set(node) == {"type", "rows", "tables", "predicates", "children"}
    assert len(node["children"]) == 2
    left_tables, left_str, left_cost, left_used = walk_tree(
        node["children"][0], rows, predicates
    )
    right_tables, right_str, right_cost, right_used = walk_tree(
        node["children"][1], rows, predicates
    )
    # children ordered by tree-string byte order
    assert left_str <= right_str
    assert left_tables.isdisjoint(right_tables)
    tables = left_tables | right_tables
    assert node["tables"] == sorted(tables)

    # estimated rows: product of base rows times all internal selectivities
    expected = Fraction(1)
    for name in tables:
        expected *= rows[name]
    for left, right, sel in predicates:
        if left in tables and right in tables:
            expected *= sel
    assert parse_frac(node["rows"]) == expected

    # predicates first effective here: exactly the cross predicates, in input order
    cross = [
        k
        for k, (left, right, _) in enumerate(predicates)
        if (left in left_tables and right in right_tables)
        or (left in right_tables and right in left_tables)
    ]
    assert cross, "every merge must have at least one predicate across the cut"
    listed = [
        {"left": predicates[k][0], "right": predicates[k][1],
         "selectivity": format_fraction(predicates[k][2])}
        for k in cross
    ]
    assert node["predicates"] == listed

    canonical = "(" + left_str + right_str + ")"
    return tables, canonical, left_cost + right_cost + expected, left_used + right_used + cross


def check_whole_result(result, names, rows, predicates):
    """Cross-check a planner result against brute force and structure."""
    components = components_of(names, predicates)
    if len(components) == 1:
        assert result["status"] == "ok"
        entries = [(tuple(components[0]), result)]
    else:
        assert result["status"] == "disconnected"
        assert len(result["components"]) == len(components)
        by_tables = {tuple(c["tables"]): c for c in result["components"]}
        entries = [(tuple(sorted(c)), by_tables[tuple(sorted(c))]) for c in components]

    for key, entry in entries:
        tables = list(key)
        members = set(tables)
        local = [p for p in predicates if p[0] in members]
        if len(tables) == 1:
            assert entry["cost"] == "0/1"
            assert entry["tree_string"] == tables[0]
            assert entry["tree"] == {
                "type": "table",
                "name": tables[0],
                "rows": rows[tables[0]],
            }
            continue
        best = brute_best(tables, rows, local)
        assert best is not None
        cost, canonical = best
        assert entry["cost"] == format_fraction(cost)
        assert entry["tree_string"] == canonical
        got_tables, got_str, got_cost, used = walk_tree(entry["tree"], rows, local)
        assert got_tables == members
        assert got_str == entry["tree_string"]
        assert got_cost == cost
        assert sorted(used) == list(range(len(local))), (
            "every predicate must take effect at exactly one node"
        )


def solve(payload):
    return plan_problem(payload)


def make_payload(names, rows, predicates):
    return {
        "tables": [{"name": name, "rows": rows[name]} for name in names],
        "predicates": [
            {"left": left, "right": right, "selectivity": format_fraction(sel)}
            for left, right, sel in predicates
        ],
    }


def make_dual_payload(names, rows, predicates, selectivities2):
    payload = make_payload(names, rows, predicates)
    payload["selectivities2"] = [format_fraction(sel) for sel in selectivities2]
    return payload


# ---------------------------------------------------------------------------
# structural validation of the dual-scenario tree
# ---------------------------------------------------------------------------

def walk_tree_dual(node, rows, predicates, selectivities2):
    """Dual-scenario counterpart of walk_tree.

    Returns (tables, canonical_string, cost1, cost2, used_predicate_indexes).
    """
    if node["type"] == "table":
        assert set(node) == {"type", "name", "rows"}
        assert node["rows"] == rows[node["name"]]
        return {node["name"]}, node["name"], Fraction(0), Fraction(0), []

    assert node["type"] == "join"
    assert set(node) == {
        "type", "rows", "rows2", "tables", "predicates", "children"
    }
    assert len(node["children"]) == 2
    left_tables, left_str, left_c1, left_c2, left_used = walk_tree_dual(
        node["children"][0], rows, predicates, selectivities2
    )
    right_tables, right_str, right_c1, right_c2, right_used = walk_tree_dual(
        node["children"][1], rows, predicates, selectivities2
    )
    # children ordered by tree-string byte order
    assert left_str <= right_str
    assert left_tables.isdisjoint(right_tables)
    tables = left_tables | right_tables
    assert node["tables"] == sorted(tables)

    # estimated rows under both scenarios: base rows times internal selectivities
    base = Fraction(1)
    for name in tables:
        base *= rows[name]
    expected1, expected2 = base, base
    for (left, right, sel), sel2 in zip(predicates, selectivities2):
        if left in tables and right in tables:
            expected1 *= sel
            expected2 *= sel2
    assert parse_frac(node["rows"]) == expected1
    assert parse_frac(node["rows2"]) == expected2

    # predicates first effective here: exactly the cross predicates, input order
    cross = [
        k
        for k, (left, right, _) in enumerate(predicates)
        if (left in left_tables and right in right_tables)
        or (left in right_tables and right in left_tables)
    ]
    assert cross, "every merge must have at least one predicate across the cut"
    listed = [
        {
            "left": predicates[k][0],
            "right": predicates[k][1],
            "selectivity": format_fraction(predicates[k][2]),
            "selectivity2": format_fraction(selectivities2[k]),
        }
        for k in cross
    ]
    assert node["predicates"] == listed

    canonical = "(" + left_str + right_str + ")"
    return (
        tables,
        canonical,
        left_c1 + right_c1 + expected1,
        left_c2 + right_c2 + expected2,
        left_used + right_used + cross,
    )


def check_whole_result_dual(result, names, rows, predicates, selectivities2):
    """Cross-check a dual-scenario result against brute force and structure."""
    components = components_of(names, predicates)
    if len(components) == 1:
        assert result["status"] == "ok"
        assert set(result) == {"status", "cost", "cost2", "tree_string", "tree"}
        entries = [(tuple(components[0]), result)]
    else:
        assert result["status"] == "disconnected"
        assert len(result["components"]) == len(components)
        by_tables = {tuple(c["tables"]): c for c in result["components"]}
        entries = [(tuple(sorted(c)), by_tables[tuple(sorted(c))]) for c in components]

    for key, entry in entries:
        assert "cost" in entry and "cost2" in entry
        tables = list(key)
        members = set(tables)
        indexes = [k for k, p in enumerate(predicates) if p[0] in members]
        local = [predicates[k] for k in indexes]
        local2 = [selectivities2[k] for k in indexes]
        if len(tables) == 1:
            assert entry["cost"] == "0/1"
            assert entry["cost2"] == "0/1"
            assert entry["tree_string"] == tables[0]
            assert entry["tree"] == {
                "type": "table",
                "name": tables[0],
                "rows": rows[tables[0]],
            }
            continue
        best = brute_best_dual(tables, rows, local, local2)
        assert best is not None
        cost1, cost2, canonical = best
        assert entry["cost"] == format_fraction(cost1)
        assert entry["cost2"] == format_fraction(cost2)
        assert entry["tree_string"] == canonical
        got_tables, got_str, got_c1, got_c2, used = walk_tree_dual(
            entry["tree"], rows, local, local2
        )
        assert got_tables == members
        assert got_str == entry["tree_string"]
        assert got_c1 == cost1
        assert got_c2 == cost2
        assert sorted(used) == list(range(len(local))), (
            "every predicate must take effect at exactly one node"
        )


def strip_dual(node):
    """Drop dual-scenario-only keys from a plan tree, yielding single form."""
    if node["type"] == "table":
        return {"type": "table", "name": node["name"], "rows": node["rows"]}
    return {
        "type": "join",
        "rows": node["rows"],
        "tables": list(node["tables"]),
        "predicates": [
            {"left": p["left"], "right": p["right"], "selectivity": p["selectivity"]}
            for p in node["predicates"]
        ],
        "children": [strip_dual(child) for child in node["children"]],
    }


# ---------------------------------------------------------------------------
# deterministic cases
# ---------------------------------------------------------------------------

def test_three_table_chain_exact():
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 200},
            {"name": "C", "rows": 50},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/10"},
            {"left": "B", "right": "C", "selectivity": "1/4"},
        ],
    }
    result = solve(payload)
    assert result["status"] == "ok"
    # rows(AB) = 2000, rows(ABC) = 25000 -> 27000 beats rows(BC) plan (27500)
    assert result["cost"] == "27000/1"
    assert result["tree_string"] == "((AB)C)"
    root = result["tree"]
    assert root["rows"] == "25000/1"
    assert root["predicates"] == [
        {"left": "B", "right": "C", "selectivity": "1/4"}
    ]
    join_ab = root["children"][0]
    assert join_ab["rows"] == "2000/1"
    assert join_ab["predicates"] == [
        {"left": "A", "right": "B", "selectivity": "1/10"}
    ]
    assert join_ab["children"] == [
        {"type": "table", "name": "A", "rows": 100},
        {"type": "table", "name": "B", "rows": 200},
    ]


def test_tie_break_picks_smallest_tree_string():
    # rows(AB) == rows(BC) == 2000 and rows(ABC) == 20000: both orders tie at
    # 22000, and "((AB)C)" < "(A(BC))" byte-wise.
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 200},
            {"name": "C", "rows": 50},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/10"},
            {"left": "B", "right": "C", "selectivity": "1/5"},
        ],
    }
    result = solve(payload)
    assert result["cost"] == "22000/1"
    assert result["tree_string"] == "((AB)C)"


def test_bushy_beats_left_deep():
    rows = {"A": 100, "B": 100, "C": 100, "D": 100}
    predicates = [
        ("A", "B", Fraction(1, 10)),
        ("B", "C", Fraction(1, 10)),
        ("C", "D", Fraction(1, 10)),
    ]
    result = solve(make_payload(["A", "B", "C", "D"], rows, predicates))
    # bushy ((AB)(CD)): 1000 + 1000 + 100000 = 102000
    # left-deep chains:  1000 + 10000 + 100000 = 111000
    assert result["cost"] == "102000/1"
    assert result["tree_string"] == "((AB)(CD))"


def test_multiple_predicates_on_same_pair():
    payload = {
        "tables": [{"name": "A", "rows": 10}, {"name": "B", "rows": 20}],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/2"},
            {"left": "A", "right": "B", "selectivity": "1/3"},
        ],
    }
    result = solve(payload)
    assert result["cost"] == "100/3"  # 10 * 20 * 1/2 * 1/3
    assert result["tree_string"] == "(AB)"
    assert len(result["tree"]["predicates"]) == 2


def test_selectivity_zero_and_one():
    payload = {
        "tables": [{"name": "A", "rows": 10}, {"name": "B", "rows": 20}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "0/1"}],
    }
    result = solve(payload)
    assert result["cost"] == "0/1"
    assert result["tree"]["rows"] == "0/1"

    payload["predicates"] = [{"left": "A", "right": "B", "selectivity": 1}]
    result = solve(payload)
    assert result["cost"] == "200/1"


def test_disconnected_components():
    payload = {
        "tables": [
            {"name": "A", "rows": 10},
            {"name": "B", "rows": 20},
            {"name": "C", "rows": 5},
            {"name": "D", "rows": 7},
            {"name": "E", "rows": 3},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/2"},
            {"left": "C", "right": "D", "selectivity": "1/5"},
        ],
    }
    result = solve(payload)
    assert result["status"] == "disconnected"
    components = {tuple(c["tables"]): c for c in result["components"]}
    assert set(components) == {("A", "B"), ("C", "D"), ("E",)}
    assert components[("A", "B")]["cost"] == "100/1"
    assert components[("A", "B")]["tree_string"] == "(AB)"
    assert components[("C", "D")]["cost"] == "7/1"
    assert components[("E",)]["cost"] == "0/1"
    assert components[("E",)]["tree"] == {"type": "table", "name": "E", "rows": 3}


def test_no_predicates_at_all():
    payload = {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 2}],
        "predicates": [],
    }
    result = solve(payload)
    assert result["status"] == "disconnected"
    assert len(result["components"]) == 2


# ---------------------------------------------------------------------------
# dual-scenario deterministic cases
# ---------------------------------------------------------------------------

def test_dual_compromise_beats_either_single_scenario_optimum():
    # Scenario 1 wants (((AB)C)D) (cost 101010) but that tree costs 1110000 in
    # scenario 2; scenario 2 wants (((CD)B)A) and vice versa.  The bushy tree
    # ((AB)(CD)) costs 110010 in *both* scenarios, giving the smallest maximum
    # although it is optimal under neither single scenario alone.
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 100},
            {"name": "C", "rows": 100},
            {"name": "D", "rows": 100},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/1000"},
            {"left": "B", "right": "C", "selectivity": "1"},
            {"left": "C", "right": "D", "selectivity": "1"},
        ],
        "selectivities2": ["1", "1", "1/1000"],
    }
    result = solve(payload)
    assert result["status"] == "ok"
    assert set(result) == {"status", "cost", "cost2", "tree_string", "tree"}
    assert result["cost"] == "110010/1"
    assert result["cost2"] == "110010/1"
    assert result["tree_string"] == "((AB)(CD))"
    root = result["tree"]
    assert root["rows"] == "100000/1"
    assert root["rows2"] == "100000/1"
    assert root["predicates"] == [
        {"left": "B", "right": "C", "selectivity": "1/1",
         "selectivity2": "1/1"}
    ]
    join_ab, join_cd = root["children"]
    assert join_ab["rows"] == "10/1"
    assert join_ab["rows2"] == "10000/1"
    assert join_ab["predicates"] == [
        {"left": "A", "right": "B", "selectivity": "1/1000",
         "selectivity2": "1/1"}
    ]
    assert join_cd["rows"] == "10000/1"
    assert join_cd["rows2"] == "10/1"
    assert join_cd["predicates"] == [
        {"left": "C", "right": "D", "selectivity": "1/1",
         "selectivity2": "1/1000"}
    ]

    # The single-scenario run of the same graph picks a different tree.
    single = solve({k: v for k, v in payload.items() if k != "selectivities2"})
    assert single["cost"] == "101010/1"
    assert "cost2" not in single
    assert single["tree_string"] == "(((AB)C)D)"


def test_dual_objective_max_then_sum_then_string():
    # Legal trees (4-chain), with their (cost1, cost2) pairs:
    #   (((AB)C)D):  (10200, 30000)
    #   (((BC)A)D):  (10200, 20100)  max 20100, sum 30300
    #   ((A(BC))D):  (20100, 10200)  max 20100, sum 30300
    #   ((D(CB))A):  (30000, 10200)
    #   ((AB)(CD)):  (20100, 20100)  max 20100, sum 40200
    # min max = 20100; the sum rule eliminates the bushy tree; the two
    # survivors tie again and "(((BC)A)D)" wins byte-wise.
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 100},
            {"name": "C", "rows": 100},
            {"name": "D", "rows": 100},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/100"},
            {"left": "B", "right": "C", "selectivity": "1/100"},
            {"left": "C", "right": "D", "selectivity": "1"},
        ],
        "selectivities2": ["1", "1/100", "1/100"],
    }
    result = solve(payload)
    assert result["cost"] == "10200/1"
    assert result["cost2"] == "20100/1"
    assert result["tree_string"] == "(((BC)A)D)"


def test_dual_tied_pair_target_tree_string_decides():
    # The two trees form the pair (101000, 110000) vs (110000, 101000): same
    # maximum and same sum, so the canonical tree string decides.
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 100},
            {"name": "C", "rows": 100},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/10"},
            {"left": "B", "right": "C", "selectivity": "1"},
        ],
        "selectivities2": ["1", "1/10"],
    }
    result = solve(payload)
    assert result["cost"] == "101000/1"
    assert result["cost2"] == "110000/1"
    assert result["tree_string"] == "((AB)C)"


def test_dual_multiple_predicates_same_pair_aligned_by_index():
    # selectivities2 lines up with predicates by input index, not by pair.
    payload = {
        "tables": [{"name": "A", "rows": 10}, {"name": "B", "rows": 20}],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/2"},
            {"left": "A", "right": "B", "selectivity": "1/3"},
        ],
        "selectivities2": ["1/4", "1/5"],
    }
    result = solve(payload)
    assert result["cost"] == "100/3"      # 10 * 20 * 1/2 * 1/3
    assert result["cost2"] == "10/1"      # 10 * 20 * 1/4 * 1/5
    assert result["tree"]["rows"] == "100/3"
    assert result["tree"]["rows2"] == "10/1"
    assert result["tree"]["predicates"] == [
        {"left": "A", "right": "B", "selectivity": "1/2",
         "selectivity2": "1/4"},
        {"left": "A", "right": "B", "selectivity": "1/3",
         "selectivity2": "1/5"},
    ]


def test_dual_selectivity2_zero_one_and_non_reduced():
    payload = {
        "tables": [{"name": "A", "rows": 10}, {"name": "B", "rows": 20}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "1/2"}],
        "selectivities2": [0],
    }
    result = solve(payload)
    assert result["cost"] == "100/1"
    assert result["cost2"] == "0/1"
    assert result["tree"]["rows2"] == "0/1"

    payload["selectivities2"] = [1]
    result = solve(payload)
    assert result["cost2"] == "200/1"

    payload["selectivities2"] = ["2/4"]  # accepted and reduced
    result = solve(payload)
    assert result["cost2"] == "100/1"
    assert result["tree"]["predicates"][0]["selectivity2"] == "1/2"


def test_dual_disconnected_components():
    payload = {
        "tables": [
            {"name": "A", "rows": 10},
            {"name": "B", "rows": 20},
            {"name": "C", "rows": 5},
            {"name": "D", "rows": 7},
            {"name": "E", "rows": 3},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/2"},
            {"left": "C", "right": "D", "selectivity": "1/5"},
        ],
        "selectivities2": ["1/4", "1/3"],
    }
    result = solve(payload)
    assert result["status"] == "disconnected"
    components = {tuple(c["tables"]): c for c in result["components"]}
    assert set(components) == {("A", "B"), ("C", "D"), ("E",)}
    assert components[("A", "B")]["cost"] == "100/1"
    assert components[("A", "B")]["cost2"] == "50/1"
    assert components[("A", "B")]["tree_string"] == "(AB)"
    assert components[("A", "B")]["tree"]["rows2"] == "50/1"
    assert components[("C", "D")]["cost"] == "7/1"
    assert components[("C", "D")]["cost2"] == "35/3"
    assert components[("E",)]["cost"] == "0/1"
    assert components[("E",)]["cost2"] == "0/1"
    assert components[("E",)]["tree"] == {"type": "table", "name": "E", "rows": 3}


def test_dual_empty_selectivities2_with_no_predicates():
    payload = {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 2}],
        "selectivities2": [],
    }
    result = solve(payload)
    assert result["status"] == "disconnected"
    assert len(result["components"]) == 2
    for component in result["components"]:
        assert component["cost"] == "0/1"
        assert component["cost2"] == "0/1"


def test_nine_table_complete_graph_all_ties():
    # Every legal tree costs exactly 8 (all estimates are 1); the planner must
    # stay fast and return the lexicographically smallest canonical string.
    names = [f"T{i}" for i in range(1, 10)]
    rows = {name: 1 for name in names}
    predicates = [
        (a, b, Fraction(1))
        for i, a in enumerate(names)
        for b in names[i + 1:]
    ]
    result = solve(make_payload(names, rows, predicates))
    assert result["status"] == "ok"
    assert result["cost"] == "8/1"
    check_whole_tree_only(result, rows, predicates)


def check_whole_tree_only(result, rows, predicates):
    tables, canonical, cost, used = walk_tree(result["tree"], rows, predicates)
    assert canonical == result["tree_string"]
    assert cost == parse_frac(result["cost"])
    assert sorted(used) == list(range(len(predicates)))
    return tables


def test_nine_table_chain():
    names = [f"T{i}" for i in range(1, 10)]
    rows = {name: 1000 for name in names}
    predicates = [
        (names[i], names[i + 1], Fraction(1, 10)) for i in range(8)
    ]
    result = solve(make_payload(names, rows, predicates))
    assert result["status"] == "ok"
    check_whole_tree_only(result, rows, predicates)


# ---------------------------------------------------------------------------
# randomized duel against the exhaustive enumerator
# ---------------------------------------------------------------------------

SAFE_NAMES = ["A", "B", "C", "D", "E", "F"]
PREFIX_NAMES = ["a", "ab", "abc", "b", "ba", "c"]
SELECTIVITIES = [
    Fraction(1, 2),
    Fraction(1, 10),
    Fraction(3, 7),
    Fraction(2, 5),
    Fraction(1),
    Fraction(0),
]


@pytest.mark.parametrize("seed", range(300))
def test_random_against_brute_force(seed):
    rng = random.Random(seed)
    pool = PREFIX_NAMES if seed % 5 == 0 else SAFE_NAMES
    n = rng.randint(2, 6)
    names = rng.sample(pool, n)
    if seed % 3 == 0:
        rows = {name: rng.randint(1, 3) for name in names}  # force ties
    else:
        rows = {name: rng.randint(1, 10**6) for name in names}
    predicates = []
    for _ in range(rng.randint(0, 2 * n + 2)):
        left, right = rng.sample(names, 2)
        predicates.append((left, right, rng.choice(SELECTIVITIES)))

    result = solve(make_payload(names, rows, predicates))
    check_whole_result(result, names, rows, predicates)


@pytest.mark.parametrize("seed", range(40))
def test_random_connected_dense(seed):
    rng = random.Random(10_000 + seed)
    n = rng.randint(2, 6)
    names = rng.sample(SAFE_NAMES, n)
    rows = {name: rng.randint(1, 1000) for name in names}
    predicates = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.7:
                predicates.append(
                    (names[i], names[j], Fraction(rng.randint(1, 9), 10))
                )
    if not predicates:
        predicates.append((names[0], names[1], Fraction(1, 2)))
    result = solve(make_payload(names, rows, predicates))
    check_whole_result(result, names, rows, predicates)


# ---------------------------------------------------------------------------
# randomized dual-scenario duels
# ---------------------------------------------------------------------------

DUAL_SELECTIVITIES = [
    Fraction(1, 2),
    Fraction(1, 10),
    Fraction(3, 7),
    Fraction(2, 5),
    Fraction(1),
    Fraction(0),
    Fraction(9, 10),
]


@pytest.mark.parametrize("seed", range(300))
def test_random_dual_against_brute_force(seed):
    rng = random.Random(30_000 + seed)
    pool = PREFIX_NAMES if seed % 5 == 0 else SAFE_NAMES
    n = rng.randint(2, 6)
    names = rng.sample(pool, n)
    if seed % 3 == 0:
        rows = {name: rng.randint(1, 3) for name in names}  # force ties
    else:
        rows = {name: rng.randint(1, 10**6) for name in names}
    predicates = []
    for _ in range(rng.randint(0, 2 * n + 2)):
        left, right = rng.sample(names, 2)
        predicates.append((left, right, rng.choice(DUAL_SELECTIVITIES)))
    selectivities2 = [rng.choice(DUAL_SELECTIVITIES) for _ in predicates]

    result = solve(make_dual_payload(names, rows, predicates, selectivities2))
    check_whole_result_dual(result, names, rows, predicates, selectivities2)


@pytest.mark.parametrize("seed", range(40))
def test_random_dual_connected_dense(seed):
    rng = random.Random(40_000 + seed)
    n = rng.randint(2, 6)
    names = rng.sample(SAFE_NAMES, n)
    rows = {name: rng.randint(1, 1000) for name in names}
    predicates = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.7:
                predicates.append(
                    (names[i], names[j], Fraction(rng.randint(1, 9), 10))
                )
    if not predicates:
        predicates.append((names[0], names[1], Fraction(1, 2)))
    selectivities2 = [Fraction(rng.randint(1, 9), 10) for _ in predicates]
    result = solve(make_dual_payload(names, rows, predicates, selectivities2))
    check_whole_result_dual(result, names, rows, predicates, selectivities2)


@pytest.mark.parametrize("seed", range(60))
def test_dual_identical_scenarios_reproduce_single_plan(seed):
    # With selectivities2 identical to the first scenario the dual objective
    # must reproduce the single-scenario plan exactly (tree and tree body).
    rng = random.Random(50_000 + seed)
    pool = PREFIX_NAMES if seed % 5 == 0 else SAFE_NAMES
    n = rng.randint(2, 6)
    names = rng.sample(pool, n)
    rows = {name: rng.randint(1, 10**4) for name in names}
    predicates = []
    for _ in range(rng.randint(0, 2 * n + 2)):
        left, right = rng.sample(names, 2)
        predicates.append((left, right, rng.choice(SELECTIVITIES)))

    payload = make_payload(names, rows, predicates)
    single = solve(payload)
    dual = solve(
        make_dual_payload(names, rows, predicates, [sel for _, _, sel in predicates])
    )
    assert dual["status"] == single["status"]
    if single["status"] == "ok":
        assert dual["cost"] == single["cost"]
        assert dual["cost2"] == single["cost"]
        assert dual["tree_string"] == single["tree_string"]
        assert strip_dual(dual["tree"]) == single["tree"]
    else:
        by_tables = {tuple(c["tables"]): c for c in dual["components"]}
        for component in single["components"]:
            got = by_tables[tuple(component["tables"])]
            assert got["cost"] == component["cost"]
            assert got["cost2"] == component["cost"]
            assert got["tree_string"] == component["tree_string"]
            assert strip_dual(got["tree"]) == component["tree"]


# ---------------------------------------------------------------------------
# input validation
# ---------------------------------------------------------------------------

BAD_PAYLOADS = {
    "not an object": [1, 2, 3],
    "missing tables": {"predicates": []},
    "unknown key": {"tables": [], "bogus": 1},
    "tables not a list": {"tables": {}},
    "too few tables": {"tables": [{"name": "A", "rows": 1}]},
    "too many tables": {
        "tables": [{"name": f"T{i}", "rows": 1} for i in range(10)]
    },
    "duplicate names": {
        "tables": [{"name": "A", "rows": 1}, {"name": "A", "rows": 2}]
    },
    "empty name": {
        "tables": [{"name": "", "rows": 1}, {"name": "B", "rows": 1}]
    },
    "non-ascii name": {
        "tables": [{"name": "é", "rows": 1}, {"name": "B", "rows": 1}]
    },
    "paren in name": {
        "tables": [{"name": "A(", "rows": 1}, {"name": "B", "rows": 1}]
    },
    "rows zero": {
        "tables": [{"name": "A", "rows": 0}, {"name": "B", "rows": 1}]
    },
    "rows negative": {
        "tables": [{"name": "A", "rows": -3}, {"name": "B", "rows": 1}]
    },
    "rows float": {
        "tables": [{"name": "A", "rows": 1.5}, {"name": "B", "rows": 1}]
    },
    "rows string": {
        "tables": [{"name": "A", "rows": "10"}, {"name": "B", "rows": 1}]
    },
    "rows bool": {
        "tables": [{"name": "A", "rows": True}, {"name": "B", "rows": 1}]
    },
    "table extra key": {
        "tables": [
            {"name": "A", "rows": 1, "x": 1},
            {"name": "B", "rows": 1},
        ]
    },
    "predicate unknown table": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "Z", "selectivity": "1/2"}],
    },
    "predicate self loop": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "A", "selectivity": "1/2"}],
    },
    "predicate missing field": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B"}],
    },
    "selectivity above one": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "3/2"}],
    },
    "selectivity zero denominator": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "1/0"}],
    },
    "selectivity negative": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "-1/2"}],
    },
    "selectivity float": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B", "selectivity": 0.5}],
    },
    "selectivity junk": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "abc"}],
    },
    "selectivity int too big": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [{"left": "A", "right": "B", "selectivity": 2}],
    },
    "predicates not a list": {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": {},
    },
}


@pytest.mark.parametrize("payload", BAD_PAYLOADS.values(), ids=BAD_PAYLOADS.keys())
def test_invalid_inputs_raise(payload):
    with pytest.raises(InputError):
        solve(payload)


_BAD_DUAL_TABLES = [
    {"name": "A", "rows": 1},
    {"name": "B", "rows": 1},
]
_BAD_DUAL_PREDICATE = [
    {"left": "A", "right": "B", "selectivity": "1/2"}
]

BAD_DUAL_PAYLOADS = {
    "selectivities2 not a list": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": "1/2",
    },
    "selectivities2 null": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": None,
    },
    "selectivities2 too few": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": [],
    },
    "selectivities2 too many": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": ["1/2", "1/3"],
    },
    "selectivities2 above one": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": ["3/2"],
    },
    "selectivities2 zero denominator": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": ["1/0"],
    },
    "selectivities2 negative": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": ["-1/2"],
    },
    "selectivities2 float": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": [0.5],
    },
    "selectivities2 junk": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": ["abc"],
    },
    "selectivities2 int too big": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": [2],
    },
    "selectivities2 bool": {
        "tables": _BAD_DUAL_TABLES,
        "predicates": _BAD_DUAL_PREDICATE,
        "selectivities2": [True],
    },
    "selectivities2 without predicates": {
        "tables": _BAD_DUAL_TABLES,
        "selectivities2": ["1/2"],
    },
    "dual seven tables rejected": {
        "tables": [{"name": f"T{i}", "rows": 1} for i in range(7)],
        "predicates": [
            {"left": "T0", "right": "T1", "selectivity": "1/2"}
        ],
        "selectivities2": ["1/2"],
    },
    "dual seven tables and no predicates": {
        "tables": [{"name": f"T{i}", "rows": 1} for i in range(7)],
        "selectivities2": [],
    },
}


@pytest.mark.parametrize(
    "payload", BAD_DUAL_PAYLOADS.values(), ids=BAD_DUAL_PAYLOADS.keys()
)
def test_invalid_dual_inputs_raise(payload):
    with pytest.raises(InputError):
        solve(payload)


def test_dual_six_tables_accepted_and_eight_still_single_mode():
    names = [f"T{i}" for i in range(6)]
    rows = {name: 1 for name in names}
    predicates = [
        (names[i], names[i + 1], Fraction(1, 2)) for i in range(5)
    ]
    payload = make_dual_payload(names, rows, predicates, [Fraction(1, 3)] * 5)
    result = solve(payload)
    assert result["status"] == "ok"
    assert result["cost2"]

    # the 9-table single-scenario limit still applies without selectivities2
    names9 = [f"T{i}" for i in range(9)]
    rows9 = {name: 1 for name in names9}
    predicates9 = [
        (names9[i], names9[i + 1], Fraction(1, 2)) for i in range(8)
    ]
    result9 = solve(make_payload(names9, rows9, predicates9))
    assert result9["status"] == "ok"
    assert "cost2" not in result9


def test_non_reduced_selectivity_is_accepted_and_reduced():
    payload = {
        "tables": [{"name": "A", "rows": 10}, {"name": "B", "rows": 10}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "2/4"}],
    }
    result = solve(payload)
    assert result["cost"] == "50/1"
    assert result["tree"]["predicates"][0]["selectivity"] == "1/2"


# ---------------------------------------------------------------------------
# command line interface
# ---------------------------------------------------------------------------

def run_cli(args=(), stdin=""):
    return subprocess.run(
        [sys.executable, str(PLANNER), *args],
        input=stdin,
        capture_output=True,
        text=True,
    )


def test_cli_stdin_roundtrip():
    payload = {
        "tables": [{"name": "A", "rows": 100}, {"name": "B", "rows": 200}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "1/10"}],
    }
    proc = run_cli(stdin=json.dumps(payload))
    assert proc.returncode == 0
    assert json.loads(proc.stdout) == solve(payload)


def test_cli_file_argument(tmp_path):
    payload = {
        "tables": [{"name": "A", "rows": 4}, {"name": "B", "rows": 9}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "1/3"}],
    }
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(payload))
    proc = run_cli(args=[str(path)])
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["cost"] == "12/1"


def test_cli_invalid_json():
    proc = run_cli(stdin="{not json")
    assert proc.returncode == 2
    assert json.loads(proc.stdout)["status"] == "error"


def test_cli_invalid_problem():
    proc = run_cli(stdin=json.dumps({"tables": []}))
    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert body["status"] == "error"
    assert "between" in body["error"]


def test_cli_disconnected_exit_zero():
    payload = {
        "tables": [{"name": "A", "rows": 1}, {"name": "B", "rows": 1}],
        "predicates": [],
    }
    proc = run_cli(stdin=json.dumps(payload))
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["status"] == "disconnected"


def test_cli_dual_stdin_roundtrip():
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 200},
            {"name": "C", "rows": 50},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/10"},
            {"left": "B", "right": "C", "selectivity": "1/4"},
        ],
        "selectivities2": ["1/5", "1"],
    }
    proc = run_cli(stdin=json.dumps(payload))
    assert proc.returncode == 0
    body = json.loads(proc.stdout)
    assert body == solve(payload)
    assert body["tree"]["rows2"]


def test_cli_dual_file_argument(tmp_path):
    payload = {
        "tables": [{"name": "A", "rows": 4}, {"name": "B", "rows": 9}],
        "predicates": [{"left": "A", "right": "B", "selectivity": "1/3"}],
        "selectivities2": ["1/6"],
    }
    path = tmp_path / "problem.json"
    path.write_text(json.dumps(payload))
    proc = run_cli(args=[str(path)])
    assert proc.returncode == 0
    body = json.loads(proc.stdout)
    assert body["cost"] == "12/1"      # 36 * 1/3
    assert body["cost2"] == "6/1"      # 36 * 1/6
    assert body["tree"]["rows2"] == "6/1"


def test_cli_dual_invalid_selectivity2_emits_no_partial_plan():
    payload = {
        "tables": [
            {"name": "A", "rows": 100},
            {"name": "B", "rows": 200},
        ],
        "predicates": [{"left": "A", "right": "B", "selectivity": "1/10"}],
        "selectivities2": ["1/0"],
    }
    proc = run_cli(stdin=json.dumps(payload))
    assert proc.returncode == 2
    # the whole stdout must be exactly one error document, no plan fragments
    body = json.loads(proc.stdout)
    assert body == {"status": "error", "error": body["error"]}
    assert "zero denominator" in body["error"]
    assert "cost" not in body and "cost2" not in body
    assert "tree" not in body and "components" not in body


def test_cli_dual_too_many_tables():
    payload = {
        "tables": [{"name": f"T{i}", "rows": 1} for i in range(7)],
        "predicates": [{"left": "T0", "right": "T1", "selectivity": "1/2"}],
        "selectivities2": ["1/2"],
    }
    proc = run_cli(stdin=json.dumps(payload))
    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert body["status"] == "error"
    assert "6" in body["error"]


def test_cli_dual_wrong_length():
    payload = {
        "tables": [
            {"name": "A", "rows": 1},
            {"name": "B", "rows": 1},
            {"name": "C", "rows": 1},
        ],
        "predicates": [
            {"left": "A", "right": "B", "selectivity": "1/2"},
            {"left": "B", "right": "C", "selectivity": "1/2"},
        ],
        "selectivities2": ["1/2"],
    }
    proc = run_cli(stdin=json.dumps(payload))
    assert proc.returncode == 2
    body = json.loads(proc.stdout)
    assert body["status"] == "error"
    assert "one entry per predicate" in body["error"]
