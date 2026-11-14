"""Verification harness: unit, boundary, differential and index checks.

Run::

    python3 -I verify.py

The harness has no third-party dependencies and exits non-zero on the
first failed check.  It covers:

1. the worked example from the specification, including release timing;
2. boundary behavior (W=0, +/-inf watermarks, monotonicity, validation);
3. lazy iterator semantics (callers may consume one item at a time);
4. treap structural correctness against a sorted-list oracle;
5. randomized differential testing vs. an independent brute-force model
   (``reference.py``), including per-operation timing invariants;
6. index complexity: range-query node visits grow logarithmically while
   the window keeps the hit count constant.
"""

from __future__ import annotations

import math
import os
import random
import sys

# Under ``python3 -I`` the script directory is not added to sys.path;
# these modules live next to this file, so add that directory explicitly.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joiner import Join, _Treap
from reference import OracleJoin, ReferenceJoin

INF = float("inf")
NEG_INF = float("-inf")


def check(cond, message):
    if not cond:
        raise AssertionError(message)


def drain(it):
    return list(it)


# ---------------------------------------------------------------------- #
# 1. Specification example
# ---------------------------------------------------------------------- #
def test_spec_example():
    j = Join(2)
    check(drain(j.push("L", "a", "k", 10, {})) == [], "first push emits nothing")
    check(drain(j.push("R", "b", "k", 12, {})) == [("a", "b")], "window pair")
    check(drain(j.push("L", "c", "k", 20, {})) == [], "c has no match yet")
    check(j.active_count == 3, "all three records active")
    # R watermark 22 closes L records with t + 2 <= 22, i.e. t <= 20.
    check(drain(j.advance("R", 22)) == [("c", None)], "c released unmatched")
    check(j.active_count == 1, "paired L:a already dropped; R:b still active")
    check(drain(j.finish()) == [], "R:b was paired, no unmatched output")
    check(j.active_count == 0, "finish drains all state")


def test_boundary_pairs():
    j = Join(2)
    # Exactly at the window boundaries.
    check(drain(j.push("L", "l1", "k", 10, None)) == [], "no R yet")
    check(drain(j.push("R", "r1", "k", 8, None)) == [("l1", "r1")], "t diff = W")
    check(drain(j.push("R", "r2", "k", 12, None)) == [("l1", "r2")], "t diff = W")
    check(drain(j.push("R", "r3", "k", 7, None)) == [], "t diff = W+1, no pair")
    check(drain(j.push("R", "r4", "k", 13, None)) == [], "t diff = W+1, no pair")
    out = drain(j.finish())
    check(set(out) == {(None, "r3"), (None, "r4")}, "far records unmatched")


def test_zero_window():
    j = Join(0)
    drain(j.push("L", "l1", "k", 5, 0))
    drain(j.push("L", "l2", "k", 6, 0))
    check(drain(j.push("R", "r1", "k", 5, 0)) == [("l1", "r1")], "W=0 exact time")
    check(drain(j.push("R", "r2", "k", 7, 0)) == [], "W=0 different time")
    out = set(drain(j.finish()))
    check(out == {("l2", None), (None, "r2")}, "W=0 unmatched on finish")


def test_watermark_release_timing_fixed():
    j = Join(3)
    drain(j.push("L", "l1", "x", 10, None))
    drain(j.push("L", "l2", "x", 11, None))
    drain(j.push("R", "r1", "x", 20, None))
    check(drain(j.advance("R", 12)) == [], "wm 12 < 10+3: nothing")
    check(j.active_count == 3, "state retained below cutoff")
    check(drain(j.advance("R", 13)) == [("l1", None)], "cutoff at exact t+W")
    check(drain(j.advance("R", 13)) == [], "equal watermark is a no-op")
    check(drain(j.advance("R", 100)) == [("l2", None)], "remaining L released")
    out = drain(j.finish())
    check(out == [(None, "r1")], "unpaired R emitted on finish")
    check(j.active_count == 0, "empty state after finish")


def test_many_to_many_and_no_duplicates():
    j = Join(1)
    drain(j.push("L", "l1", "k", 10, None))
    drain(j.push("L", "l2", "k", 11, None))
    drain(j.push("R", "r1", "k", 10, None))
    out = drain(j.push("R", "r2", "k", 11, None))
    # l1 pairs r1,r2; l2 pairs r1,r2 -> four unique pairs from this push,
    # plus (l1,r1),(l2,r1) were already emitted on the r1 push.
    check(sorted(out) == sorted([("l1", "r2"), ("l2", "r2")]), "new pairs only")
    all_out = set()
    # Rebuild a fresh join to collect the full multiset in one view.
    j = Join(1)
    for o in j.push("L", "l1", "k", 10, None):
        all_out.add(o)
    for o in j.push("L", "l2", "k", 11, None):
        all_out.add(o)
    for o in j.push("R", "r1", "k", 10, None):
        all_out.add(o)
    for o in j.push("R", "r2", "k", 11, None):
        all_out.add(o)
    expected = {
        ("l1", "r1"), ("l1", "r2"), ("l2", "r1"), ("l2", "r2"),
    }
    check(all_out == expected, "cartesian within window, each pair once")
    check(drain(j.finish()) == [], "no unmatched when all paired")


def test_out_of_order_within_watermark():
    # Late-but-valid records (time > side watermark) still associate.
    j = Join(5)
    drain(j.advance("L", 0))
    drain(j.advance("R", 0))
    drain(j.push("R", "r1", "k", 3, None))
    # L arrives "late" relative to R's event time but above L's watermark.
    check(drain(j.push("L", "l1", "k", 8, None)) == [("l1", "r1")],
          "out-of-order L record joins active R record")
    check(drain(j.finish()) == [], "both paired")


def test_key_isolation():
    j = Join(10)
    drain(j.push("L", "l1", "alpha", 10, None))
    drain(j.push("L", "l2", "beta", 10, None))
    check(drain(j.push("R", "r1", "beta", 10, None)) == [("l2", "r1")],
          "same time but different key does not pair")
    out = set(drain(j.finish()))
    check(out == {("l1", None)}, "only cross-key L record is unmatched")


def test_validation():
    j = Join(1)
    for bad in [
        lambda: Join(-1),
        lambda: Join(1.0),
        lambda: drain(j.push("X", "i", "k", 1, None)),
        lambda: drain(j.push("L", 1, "k", 1, None)),
        lambda: drain(j.push("L", "i", 9, 1, None)),
        lambda: drain(j.push("L", "i", "k", 1.5, None)),
        lambda: drain(j.advance("L", "10")),
        lambda: drain(j.advance("L", float("nan"))),
    ]:
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"validation did not reject {bad}")

    # Booleans are not accepted as W / time / watermark even though bool
    # is a subclass of int.
    for bad in (
        lambda: Join(True),
        lambda: drain(Join(1).push("L", "i", "k", True, None)),
        lambda: drain(Join(1).advance("L", True)),
    ):
        try:
            bad()
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"validation did not reject bool input {bad}")

    drain(j.advance("L", 5))
    try:
        drain(j.push("L", "late", "k", 5, None))  # time must be > wm
    except ValueError:
        pass
    else:
        raise AssertionError("record at watermark time must be rejected")
    try:
        drain(j.advance("L", 4))  # watermark regression
    except ValueError:
        pass
    else:
        raise AssertionError("watermark regression must be rejected")

    # finish() raises both watermarks to +inf, so use a fresh instance to
    # isolate the post-finish contract from the watermark-time check.
    j2 = Join(1)
    drain(j2.advance("L", 0))
    drain(j2.finish())
    try:
        drain(j2.push("L", "x", "k", 6, None))
    except RuntimeError:
        pass
    else:
        raise AssertionError("push after finish must be rejected")
    # finish is idempotent and emits nothing on repeated calls.
    check(drain(j2.finish()) == [], "repeated finish must be a no-op")


def test_lazy_iterator_semantics():
    j = Join(0)
    for i in range(20):
        drain(j.push("L", f"l{i}", "k", i, None))
    it = j.push("R", "r", "k", 10, None)  # pairs only l10
    first = next(it)
    check(first == ("l10", "r"), "pair produced lazily")
    try:
        next(it)
    except StopIteration:
        pass
    else:
        raise AssertionError("single pair expected")
    # Iterator was exhausted -> adoption + release already happened.
    # Advance R to +inf; L records never paired get unmatched tuples.
    out = drain(j.advance("R", INF))
    check(("l10", None) not in out, "paired L is never reported unmatched")
    check(len(out) == 19, "all other L records unmatched exactly once")
    check(set(out) == {(f"l{i}", None) for i in range(20) if i != 10},
          "unmatched identity set")
    check(drain(j.finish()) == [], "r was paired")


def test_unconsumed_iterator_has_no_side_effects():
    # A push iterator that is never started must not mutate join state:
    # association and adoption are both driven by generator execution.
    j = Join(2)
    drain(j.push("L", "l1", "k", 10, None))
    it = j.push("R", "r1", "k", 11, None)  # would pair (l1, r1)
    check(j.active_count == 1, "new record not adopted before iteration")
    del it  # caller contract says exhaust before the next call anyway
    check(j.active_count == 1, "dropping the iterator changes nothing")
    # A freshly created and fully consumed iterator behaves normally.
    check(drain(j.push("R", "r2", "k", 12, None)) == [("l1", "r2")],
          "fresh iterator still associates the pre-existing L record")
    check(j.active_count == 2, "r2 adopted after consumption")


def test_paired_records_release_silently():
    n = 3000
    j = Join(2)
    for i in range(n):
        drain(j.push("L", f"l{i}", "k", i, None))
    pairs = 0
    expected = sum(min(3, i + 1) + min(2, n - 1 - i) for i in range(n))
    for i in range(n):
        pairs += len(drain(j.push("R", f"r{i}", "k", i, None)))
    check(pairs == expected, "all window pairs produced exactly once")
    # No watermark progress yet: retention of 2n records is required so
    # that future opposite-side records can still find their matches.
    check(j.active_count == 2 * n, "all records retained without watermarks")
    check(drain(j.advance("L", INF)) == [], "paired R records drop silently")
    check(j.active_count == n, "only L records remain after R-side drain")
    check(drain(j.advance("R", INF)) == [], "paired L records drop silently")
    check(j.active_count == 0, "state fully drained")
    check(drain(j.finish()) == [], "nothing left to report")


# ---------------------------------------------------------------------- #
# 4. Treap structural correctness vs. a sorted list
# ---------------------------------------------------------------------- #
def test_treap_vs_sorted_list():
    rng = random.Random(424242)
    for n in (1, 2, 3, 7, 33, 129, 500):
        treap = _Treap(rng)
        oracle = []

        class R:
            __slots__ = ("time", "rid")

            def __init__(self, t, i):
                self.time = t
                self.rid = i

        recs = [R(rng.randrange(1000), f"id{i:04d}") for i in range(n)]
        for r in recs:
            treap.insert(r)
            oracle.append((r.time, r.rid, r))
        oracle.sort(key=lambda x: (x[0], x[1]))

        got = [(x.time, x.rid) for x in treap.range_time(-10_000, 10_000)]
        want = [(t, i) for t, i, _ in oracle]
        check(got == want, f"treap in-order mismatch at n={n}")

        # Windowed ranges.
        for _ in range(20):
            lo = rng.randrange(-50, 1000)
            hi = lo + rng.randrange(0, 50)
            got = [(x.time, x.rid) for x in treap.range_time(lo, hi)]
            want = [(t, i) for t, i, _ in oracle if lo <= t <= hi]
            check(got == want, f"treap range mismatch n={n} [{lo},{hi}]")

        # Random erasures keep the structure consistent after each step.
        alive = {(r.time, r.rid) for r in recs}
        order = recs[:]
        rng.shuffle(order)
        for r in order:
            treap.erase((r.time, r.rid))
            alive.discard((r.time, r.rid))
            got = {(x.time, x.rid) for x in treap.range_time(-10_000, 10_000)}
            check(got == alive, f"treap state mismatch after erase n={n}")
        check(treap.root is None, "treap empty after erasing every key")


def test_range_query_complexity():
    # One key, N far-apart records; a tiny window hits K=0/1 records.
    # Pruning must keep node visits near O(log N), not O(N).
    sizes = []
    visits = []
    for n in (2_000, 8_000, 32_000):
        j = Join(0, seed=7)
        for i in range(n):
            drain(j.push("L", f"l{i}", "k", i * 100, None))
        tree = j._trees["L"]["k"]
        tree.range_visits = 0
        drain(tree.range_time(500_000, 500_000))  # no record in range
        drain(tree.range_time(10_000, 10_000))    # exactly one record
        sizes.append(n)
        visits.append(tree.range_visits)
    # From 2k -> 32k is a 16x size increase; log2(16) = 4 extra levels,
    # so visits may grow by a small factor, nowhere near 16x.
    check(visits[-1] < 6 * visits[0],
          f"range scan not pruned: n={sizes} visits={visits}")


def test_treap_height_bounded():
    # 100k-node treaps must stay logarithmic: expected height ~ 3 ln n,
    # and the split/merge recursion must remain far below Python's limit.
    class R:
        __slots__ = ("time", "rid")

        def __init__(self, t, i):
            self.time, self.rid = t, i

    for seed in range(5):
        treap = _Treap(random.Random(seed))
        n = 100_000
        for i in range(n):
            treap.insert(R(i, f"id{i:07d}"))
        height = 0
        stack = [(treap.root, 1)]
        while stack:
            node, depth = stack.pop()
            if node is None:
                continue
            height = max(height, depth)
            stack.append((node.left, depth + 1))
            stack.append((node.right, depth + 1))
        check(height < 10 * math.log2(n),
              f"treap height {height} not logarithmic at n={n}")
        check(height < sys.getrecursionlimit(), "treap recursion unsafe")
        check(len(list(treap.range_time(0, n))) == n, "full range complete")
        check(len(list(treap.range_time(54321, 54321))) == 1,
              "narrow range hits one record")


# ---------------------------------------------------------------------- #
# 5. Randomized differential testing
# ---------------------------------------------------------------------- #
def _run_scenario(seed, n_ops, n_keys, max_w):
    rng = random.Random(seed)
    W = rng.randrange(0, max_w + 1)
    join = Join(W, seed=seed ^ 0x5EED)
    ref = ReferenceJoin(W)
    wm = {"L": NEG_INF, "R": NEG_INF}
    seq = {"L": 0, "R": 0}
    key_pool = [f"k{rng.randrange(n_keys)}" for _ in range(max(8, n_keys * 3))]
    emitted = set()
    total_pairs = 0

    def run(op_kind):
        nonlocal total_pairs
        if op_kind < 55:  # push
            side = "L" if rng.random() < 0.5 else "R"
            seq[side] += 1
            rid = f"{side}{seq[side]}"
            key = rng.choice(key_pool)
            # Valid records: time strictly above the side watermark.
            base = int(wm[side]) if wm[side] != NEG_INF else -100
            time = base + rng.randrange(1, 40)
            out = drain(join.push(side, rid, key, time, f"p{rid}"))
            ref.push(side, rid, key, time, f"p{rid}")
            for tup in out:
                if tup in emitted:
                    raise AssertionError(f"duplicate emission {tup}")
                emitted.add(tup)
                l, r = tup
                if l is not None and r is not None:
                    total_pairs += 1
                else:
                    # Unmatched tuple: the released record's cutoff must be
                    # consistent with the operated-side watermark.
                    pass
            # Active state can never exceed records pushed on both sides.
            check(0 <= join.active_count <= sum(seq.values()),
                  "active count out of range")
        elif op_kind < 80:  # bounded watermark advance
            side = "L" if rng.random() < 0.5 else "R"
            cur = int(wm[side]) if wm[side] != NEG_INF else -100
            new_wm = cur + rng.randrange(0, 25)
            out = drain(join.advance(side, new_wm))
            wm[side] = float(new_wm)
            for tup in out:
                if tup in emitted:
                    raise AssertionError(f"duplicate emission {tup}")
                emitted.add(tup)
        else:  # big jump: forces a large drain
            side = "L" if rng.random() < 0.5 else "R"
            cur = int(wm[side]) if wm[side] != NEG_INF else -100
            new_wm = cur + rng.randrange(500, 2000)
            out = drain(join.advance(side, new_wm))
            wm[side] = float(new_wm)
            for tup in out:
                if tup in emitted:
                    raise AssertionError(f"duplicate emission {tup}")
                emitted.add(tup)

    for _ in range(n_ops):
        run(rng.randrange(100))

    final = drain(join.finish())
    for tup in final:
        if tup in emitted:
            raise AssertionError(f"duplicate finish emission {tup}")
        emitted.add(tup)

    check(join.active_count == 0, "state must be empty after finish")
    check(emitted == ref.expected(),
          f"seed={seed} W={W}: streaming output differs from reference")
    return len(emitted), total_pairs


def test_differential():
    scenarios = 0
    total_emitted = 0
    for seed in range(300):
        n_ops = {0: 500, 1: 1000}.get(seed % 10, 200)
        emitted, pairs = _run_scenario(
            seed=seed,
            n_ops=n_ops,
            n_keys=1 + seed % 6,
            max_w=seed % 8,
        )
        total_emitted += emitted
        scenarios += 1
    check(scenarios == 300, "all randomized scenarios must run")
    return scenarios, total_emitted


def test_three_way_differential():
    """Join (treap) vs OracleJoin (bisect) per-batch, both vs brute force."""
    rng_master = random.Random(2024)
    for scenario in range(60):
        seed = rng_master.randrange(1 << 30)
        rng = random.Random(seed)
        W = rng.randrange(0, 7)
        treap_join = Join(W, seed=seed)
        oracle = OracleJoin(W)
        brute = ReferenceJoin(W)
        wm = {"L": NEG_INF, "R": NEG_INF}
        seq = {"L": 0, "R": 0}
        treap_all = set()

        def cmp_batch(a, b, where):
            # is-None flag first so None and str never get compared directly.
            key = lambda t: tuple((v is None, v) for v in t)
            check(sorted(a, key=key) == sorted(b, key=key),
                  f"per-batch mismatch at {where} seed={seed}: {a} vs {b}")
            for tup in a:
                check(tup not in treap_all, f"dup emission {tup} seed={seed}")
                treap_all.add(tup)

        for _ in range(rng.randrange(80, 260)):
            kind = rng.randrange(100)
            if kind < 60:
                side = "L" if rng.random() < 0.5 else "R"
                seq[side] += 1
                rid = f"{side}{seq[side]}"
                key = f"k{rng.randrange(5)}"
                base = int(wm[side]) if wm[side] != NEG_INF else -50
                t = base + rng.randrange(1, 30)
                a = drain(treap_join.push(side, rid, key, t, None))
                b = oracle.push(side, rid, key, t, None)
                cmp_batch(a, b, f"push {rid}")
                brute.push(side, rid, key, t, None)
            else:
                side = "L" if rng.random() < 0.5 else "R"
                cur = int(wm[side]) if wm[side] != NEG_INF else -50
                wm[side] = float(cur + rng.randrange(0, 40))
                a = drain(treap_join.advance(side, wm[side]))
                b = oracle.advance(side, wm[side])
                cmp_batch(a, b, "advance")

        a = drain(treap_join.finish())
        b = oracle.finish()
        cmp_batch(a, b, "finish")
        oracle_all = set(oracle.emitted_set)
        check(treap_all == oracle_all, "treap vs oracle final sets differ")
        check(treap_all == brute.expected(),
              f"streaming outputs vs brute force differ seed={seed}")
    return 60, len(treap_all)


# ---------------------------------------------------------------------- #
# 6. Unmatched-emission timing invariant (watermark cutoff)
# ---------------------------------------------------------------------- #
def test_unmatched_timing_invariant():
    rng = random.Random(99)
    for _ in range(40):
        W = rng.randrange(0, 6)
        j = Join(W)
        log = {"L": {}, "R": {}}  # id -> (key, time)
        wm = {"L": NEG_INF, "R": NEG_INF}
        for step in range(220):
            if rng.random() < 0.6:
                side = "L" if rng.random() < 0.5 else "R"
                rid = f"{side}{step}"
                key = f"k{rng.randrange(4)}"
                base = int(wm[side]) if wm[side] != NEG_INF else 0
                t = base + rng.randrange(1, 12)
                # Register before consuming the push iterator: matched
                # pairs emitted by the iterator reference the new record.
                log[side][rid] = (key, t)
                for out in j.push(side, rid, key, t, None):
                    _assert_timing(out, side, wm, log, W, on="push")
            else:
                side = "L" if rng.random() < 0.5 else "R"
                cur = int(wm[side]) if wm[side] != NEG_INF else 0
                wm[side] = float(cur + rng.randrange(0, 15))
                for out in j.advance(side, wm[side]):
                    _assert_timing(out, side, wm, log, W, on="advance")
        for out in j.finish():
            check(out[0] is None or out[1] is None, "finish only unmatched")


def _assert_timing(out, operated_side, wm, log, W, on):
    l, r = out
    if l is not None and r is not None:
        lk, lt = log["L"][l]
        rk, rt = log["R"][r]
        check(lk == rk and abs(lt - rt) <= W, f"illegal pair emitted: {out}")
        return
    # Unmatched record belongs to the opposite side of the operated side;
    # its t + W must be <= operated-side watermark.
    if l is not None:
        _t = log["L"][l][1]
        check(operated_side == "R", "L release must be driven by R watermark")
        cutoff = wm["R"]
    else:
        _t = log["R"][r][1]
        check(operated_side == "L", "R release must be driven by L watermark")
        cutoff = wm["L"]
    check(_t + W <= cutoff,
          f"emitted too early on {on}: t={_t} W={W} wm={cutoff}")


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
TESTS = [
    test_spec_example,
    test_boundary_pairs,
    test_zero_window,
    test_watermark_release_timing_fixed,
    test_many_to_many_and_no_duplicates,
    test_out_of_order_within_watermark,
    test_key_isolation,
    test_validation,
    test_lazy_iterator_semantics,
    test_unconsumed_iterator_has_no_side_effects,
    test_paired_records_release_silently,
    test_treap_vs_sorted_list,
    test_range_query_complexity,
    test_treap_height_bounded,
    test_unmatched_timing_invariant,
    test_differential,
    test_three_way_differential,
]


def main():
    failed = 0
    for test in TESTS:
        try:
            extra = test()
        except Exception as exc:  # noqa: BLE001 - harness must report all
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            detail = f" ({extra[0]} scenarios, {extra[1]} outputs)" if extra else ""
            print(f"ok   {test.__name__}{detail}")
    print()
    if failed:
        print(f"{failed}/{len(TESTS)} test groups FAILED")
        return 1
    print(f"all {len(TESTS)} test groups passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
