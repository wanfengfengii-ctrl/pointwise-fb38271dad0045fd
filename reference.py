"""Independent brute-force reference model for the interval join.

This module deliberately shares *no* indexing code with ``joiner.py``.  It
provides two oracles built from disjoint data structures:

* :class:`ReferenceJoin` buffers the whole event log and recomputes the
  expected result set with plain nested loops -- the ultimate ground
  truth for small/medium randomized scenarios;
* :class:`OracleJoin` is a second, independently written streaming
  implementation based on :mod:`bisect` sorted lists and :mod:`heapq`,
  used to cross-check long streams where an O(n^2) brute force is
  impractical.

The streaming implementation under test is pointwise correct iff the set
of tuples it emits over a whole run equals the set produced here.
"""

from __future__ import annotations

import bisect
import heapq

INF = float("inf")


class ReferenceJoin:
    """Reference full-outer interval join over a fully buffered event log."""

    def __init__(self, W):
        if not isinstance(W, int) or isinstance(W, bool) or W < 0:
            raise TypeError("W must be a non-negative integer")
        self.W = W
        self.left = []   # (id, key, time, payload)
        self.right = []

    def push(self, side, rid, key, time, payload):
        rec = (rid, key, time, payload)
        if side == "L":
            self.left.append(rec)
        else:
            self.right.append(rec)

    def expected(self):
        """Return the expected result as a *set* of result tuples.

        Every valid cross-side pair appears once; records participating in
        no pair appear once as an unmatched tuple.
        """
        W = self.W
        results = set()
        matched_left = set()
        matched_right = set()
        for lid, lkey, lt, _ in self.left:
            for rid, rkey, rt, _ in self.right:
                if lkey == rkey and abs(lt - rt) <= W:
                    results.add((lid, rid))
                    matched_left.add(lid)
                    matched_right.add(rid)
        for lid, _lkey, _lt, _ in self.left:
            if lid not in matched_left:
                results.add((lid, None))
        for rid, _rkey, _rt, _ in self.right:
            if rid not in matched_right:
                results.add((None, rid))
        return results

    # ------------------------------------------------------------------ #
    # Static oracle helpers reused by the differential harness
    # ------------------------------------------------------------------ #
    @staticmethod
    def pairs_between(left_rec, right_rec, W):
        """All valid pairs between two buffered record lists (nested loop)."""
        out = set()
        for lid, lkey, lt in left_rec:
            for rid, rkey, rt in right_rec:
                if lkey == rkey and abs(lt - rt) <= W:
                    out.add((lid, rid))
        return out


class OracleJoin:
    """Independent streaming oracle using bisect lists and heapq.

    The algorithm mirrors the spec's operation order (associate, then
    release on the opposite watermark) but uses different data structures
    than the treap-based implementation under test, so agreement is a
    genuine cross-check rather than shared-code agreement.
    """

    def __init__(self, W):
        if not isinstance(W, int) or isinstance(W, bool) or W < 0:
            raise TypeError("W must be a non-negative integer")
        self.W = W
        self._wm = {"L": float("-inf"), "R": float("-inf")}
        # side -> key -> sorted list of (time, rid)
        self._idx = {"L": {}, "R": {}}
        # side -> heap of (time + W, time, rid, key)
        self._heap = {"L": [], "R": []}
        self._paired = set()   # (side, rid) of records ever paired
        self._emitted = set()  # all result tuples already yielded
        self._finished = False

    def push(self, side, rid, key, time, payload):
        opp = "R" if side == "L" else "L"
        results = []
        bucket = self._idx[opp].get(key)
        if bucket is not None:
            lo = time - self.W
            hi = time + self.W
            left_i = bisect.bisect_left(bucket, (lo, ""))
            right_i = bisect.bisect_right(bucket, (hi, "\U0010ffff"))
            for ot, orid in bucket[left_i:right_i]:
                tup = (rid, orid) if side == "L" else (orid, rid)
                self._paired.add((side, rid))
                self._paired.add((opp, orid))
                results.append(tup)
        my_bucket = self._idx[side].setdefault(key, [])
        bisect.insort(my_bucket, (time, rid))
        heapq.heappush(self._heap[side], (time + self.W, time, rid, key))
        results.extend(self._release(opp, self._wm[side]))
        return self._unique(results)

    def advance(self, side, watermark):
        results = []
        if watermark != self._wm[side]:
            self._wm[side] = watermark
            opp = "R" if side == "L" else "L"
            results.extend(self._release(opp, watermark))
        return self._unique(results)

    def finish(self):
        if self._finished:
            return []
        self._finished = True
        self._wm["L"] = INF
        self._wm["R"] = INF
        results = self._release("L", INF) + self._release("R", INF)
        return self._unique(results)

    def _release(self, side, cutoff):
        out = []
        heap = self._heap[side]
        idx = self._idx[side]
        while heap and heap[0][0] <= cutoff:
            _exp, time, rid, key = heapq.heappop(heap)
            bucket = idx[key]
            pos = bisect.bisect_left(bucket, (time, rid))
            bucket.pop(pos)
            if not bucket:
                del idx[key]
            if (side, rid) not in self._paired:
                out.append((rid, None) if side == "L" else (None, rid))
        return out

    def _unique(self, tuples):
        # The spec guarantees one emission per pair/unmatched record; the
        # oracle dedups defensively and remembers everything it yielded.
        out = []
        for tup in tuples:
            if tup not in self._emitted:
                self._emitted.add(tup)
                out.append(tup)
        return out

    @property
    def emitted_set(self):
        return set(self._emitted)
