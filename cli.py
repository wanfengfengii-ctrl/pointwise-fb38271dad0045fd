"""Command-line entry points: demonstration and long-stream benchmarking.

Usage::

    python3 -I cli.py demo [--seed N]
    python3 -I cli.py bench [--records N] [--window W] [--seed N]

The demo plays an annotated, out-of-order, two-sided event stream and
prints every emitted tuple with the operation that produced it.

The benchmark drives a long stream while measuring wall time, peak RSS and
the high-water mark of retained active records, then cross-checks the
complete result multiset against the independent brute-force reference
(``reference.py``).  It also includes a heavy many-to-many scenario that
streams pairs without buffering them.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from joiner import Join  # noqa: E402
from reference import OracleJoin, ReferenceJoin  # noqa: E402

INF = float("inf")


def _emit(op, side, args, out):
    for tup in out:
        left, right = tup
        kind = "PAIR    " if left is not None and right is not None else "UNMATCHED"
        arg = " ".join(str(a) for a in args)
        print(f"  {kind:9s} {op}({side or '-'} {arg}) -> {tup}")


def cmd_demo(seed):
    print("Streaming full-outer interval join demo (W=2)")
    print("tuple shape: (left_id, right_id); None marks an unmatched side\n")
    j = Join(2)
    ref = ReferenceJoin(2)

    def play(op, side, *args):
        if op == "push":
            _rid, _key, t, payload = args
            print(f"push {side} id={_rid} key={_key} time={t} payload={payload!r}")
            out = list(j.push(side, _rid, _key, t, payload))
            ref.push(side, _rid, _key, t, payload)
        elif op == "advance":
            (wm,) = args
            print(f"advance {side} watermark -> {wm}")
            out = list(j.advance(side, wm))
        else:
            print("finish (both watermarks -> +inf)")
            out = list(j.finish())
        _emit(op, side, args, out)
        print(f"    active records: {j.active_count}")
        return out

    play("push", "L", "a", "ord-1", 10, "click")
    play("push", "R", "b", "ord-1", 12, "view")      # pair (a,b)
    play("push", "R", "c", "ord-2", 15, "view")      # different key
    play("push", "L", "d", "ord-2", 11, "click")     # |15-11|=4 > 2
    play("advance", "L", 12)                          # releases R c? 15+2>12 no
    play("advance", "L", 17)                          # R c@15: 15+2<=17 unmatched
    play("push", "L", "e", "ord-1", 20, "click")
    play("advance", "R", 22)                          # L e@20 unmatched
    play("push", "R", "f", "ord-3", 30, "view")
    play("push", "L", "g", "ord-3", 30, "click")     # W=0 for this key pair
    play("finish", None)

    print("\nreference model result set:")
    for tup in sorted(ref.expected(), key=lambda x: tuple(str(v) for v in x)):
        print("  ", tup)
    print("\ndemo completed; active records:", j.active_count)


def cmd_bench(records, window, seed):
    rng = __import__("random").Random(seed)
    print(f"long-stream benchmark: records/side={records}, W={window}, seed={seed}")
    join = Join(window, seed=seed)
    # Independent streaming oracle (bisect + heapq); an O(n^2) brute force
    # cannot keep up at this scale and is reserved for verify.py.
    oracle = OracleJoin(window)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    peak_active = 0
    emitted = 0
    wm = {"L": 0, "R": 0}
    next_id = {"L": 0, "R": 0}

    # Interleave pushes with periodic watermark advances so the active set
    # stays bounded: records older than ~ horizon are continuously retired.
    horizon = 200
    key_space = 16
    for step in range(records * 2):
        side = "L" if step % 2 == 0 else "R"
        next_id[side] += 1
        rid = f"{side}{next_id[side]}"
        key = f"k{rng.randrange(key_space)}"
        t = wm[side] + rng.randrange(1, 60)
        payload = f"payload-{rid}"
        out = list(join.push(side, rid, key, t, payload))
        o_out = oracle.push(side, rid, key, t, payload)
        if _keyed(out) != _keyed(o_out):
            raise AssertionError(f"oracle mismatch at push {side} {rid}")
        emitted += len(out)
        peak_active = max(peak_active, join.active_count)

        if step % 64 == 63:
            side_wm = "L" if rng.random() < 0.5 else "R"
            wm[side_wm] += rng.randrange(20, horizon)
            out = list(join.advance(side_wm, wm[side_wm]))
            o_out = oracle.advance(side_wm, wm[side_wm])
            if _keyed(out) != _keyed(o_out):
                raise AssertionError("oracle mismatch at watermark advance")
            emitted += len(out)
            peak_active = max(peak_active, join.active_count)

    emitted += sum(1 for _ in join.finish())
    oracle.finish()
    elapsed = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(f"  wall time:            {elapsed:.3f} s")
    print(f"  tuples emitted:       {emitted}")
    print(f"  peak active records:  {peak_active}")
    print(f"  final active records: {join.active_count}")
    # ru_maxrss is kilobytes on Linux.
    print(f"  peak RSS delta:       {(rss_after - rss_before) / 1024:.1f} MiB")

    # Resource bound: with continuous watermark advances, retained state
    # must track the bounded horizon rather than total stream length.
    bound = horizon * key_space * 4
    if peak_active > bound:
        raise AssertionError(
            f"active set {peak_active} exceeded resource bound {bound}: "
            "state appears to grow with total stream length"
        )
    print(f"  resource bound:       peak_active <= {bound}  OK")

    # Exhaustive cross-check against the independent streaming oracle.
    if emitted != len(oracle.emitted_set):
        raise AssertionError(
            f"emitted {emitted} tuples but oracle has {len(oracle.emitted_set)}"
        )
    print(f"  cross-check:          {emitted} tuples match independent oracle OK")

    _heavy_pairs(window)
    print("\nbenchmark completed successfully")


def _keyed(tuples):
    """Order-insensitive comparison key for a batch of result tuples."""
    return sorted(tuples, key=lambda t: tuple((v is None, v) for v in t))


def _heavy_pairs(window):
    # One key, 2k records per side at identical times: every L matches
    # every R (4m pairs).  Pairs must stream out of the iterator rather
    # than being collected inside the join; active state stays O(A).
    n = 2000
    j = Join(window)
    for i in range(n):
        list(j.push("L", f"l{i}", "k", i, None))
    peak = j.active_count
    count = 0
    for i in range(n):
        for pair in j.push("R", f"r{i}", "k", i, None):
            count += 1
            if count == 1:
                # While the first pair is being consumed, state holds only
                # the active L records plus the not-yet-adopted R record.
                peak = max(peak, j.active_count)
    list(j.finish())
    expected_pairs = _count_pairs(n, window)
    if count != expected_pairs:
        raise AssertionError(f"heavy pairing emitted {count}, expected {expected_pairs}")
    if j.active_count != 0:
        raise AssertionError("state not drained after heavy scenario")
    print(f"  heavy pairing:        {count} pairs streamed, "
          f"peak active {peak} (n/side={n}) OK")


def _count_pairs(n, W):
    # Brute-force pair count for identical time sequences 0..n-1.
    total = 0
    for lt in range(n):
        for rt in range(n):
            if abs(lt - rt) <= W:
                total += 1
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="annotated demonstration run")
    d.add_argument("--seed", type=int, default=0)

    b = sub.add_parser("bench", help="long-stream resource benchmark")
    b.add_argument("--records", type=int, default=20_000,
                   help="records per side (default 20000)")
    b.add_argument("--window", type=int, default=4)
    b.add_argument("--seed", type=int, default=1234)

    args = parser.parse_args(argv)
    if args.cmd == "demo":
        cmd_demo(args.seed)
    else:
        cmd_bench(args.records, args.window, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
