"""Event-time full-outer interval join over two out-of-order streams.

The join is driven explicitly by per-side watermarks::

    j = Join(W)
    for out in j.push(side, rid, key, time, payload): ...
    for out in j.advance(side, watermark): ...
    for out in j.finish(): ...

Every public mutator returns an *iterator* of result tuples
``(left_id, right_id)``:

* a matched pair  -> (left_id, right_id), both non-None;
* an unmatched L  -> (left_id, None);
* an unmatched R  -> (None, right_id).

Semantics (see README.md for the full argument):

* records on the two sides with the same key and ``|tL - tR| <= W`` are
  paired exactly once, many-to-many;
* each operation first performs association, then releases opposite-side
  records whose ``time + W`` is no longer above the operated side's
  watermark; paired records are never released early, and never emit an
  unmatched tuple;
* each record is output at most once as unmatched;
* ``finish()`` is equivalent to advancing both watermarks to +infinity.

State space is O(A) where A is the number of active (not yet released)
records.  Per-key range scans cost O(log(A+2) + K) expected, where K is
the number of pairs produced by that push; no whole-key scan or
recomputation of old pairs is ever performed.

Only the Python standard library is used.
"""

from __future__ import annotations

import heapq
import random

__all__ = ["Join"]

_INF = float("inf")


class _Node:
    """Treap node keyed by ``(event_time, record_id)``.

    Treap priorities are drawn from a private PRNG independent of the key,
    so every shape is an expected balanced BST: height is O(log n) with
    overwhelmingly high probability.
    """

    __slots__ = ("key", "prio", "left", "right", "rec")

    def __init__(self, rec, prio):
        self.key = (rec.time, rec.rid)
        self.rec = rec
        self.prio = prio
        self.left = None
        self.right = None


class _Treap:
    """Minimal randomized BST (treap) with expected O(log n) operations."""

    __slots__ = ("root", "_rng", "range_visits")

    def __init__(self, rng):
        self.root = None
        self._rng = rng
        # Number of BST nodes touched by range queries; used by the
        # complexity checks in the test/bench harness.
        self.range_visits = 0

    def _split(self, root, key):
        if root is None:
            return None, None
        if root.key < key:
            a, b = self._split(root.right, key)
            root.right = a
            return root, b
        a, b = self._split(root.left, key)
        root.left = b
        return a, root

    def _merge(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        if a.prio > b.prio:
            a.right = self._merge(a.right, b)
            return a
        b.left = self._merge(a, b.left)
        return b

    def insert(self, rec):
        node = _Node(rec, self._rng.getrandbits(64))
        if self.root is None:
            self.root = node
            return
        a, b = self._split(self.root, node.key)
        # IDs are unique per side, so the exact key cannot already exist.
        self.root = self._merge(self._merge(a, node), b)

    def erase(self, key):
        """Remove ``key`` (expected O(log n)); the key is assumed present."""
        root = self.root
        parent = None
        cur = root
        while cur.key != key:
            parent = cur
            if key < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        merged = self._merge(cur.left, cur.right)
        if parent is None:
            self.root = merged
        elif parent.left is cur:
            parent.left = merged
        else:
            parent.right = merged

    def range_time(self, lo, hi):
        """Yield records whose event time lies in the closed interval.

        Key order is ``(time, id)``; BST key bounds use the id sentinels
        ``""`` / a maximum sentinel so that *all* ids at the boundary
        times are included.  Subtrees wholly inside / wholly outside the
        window are pruned without descending, giving O(log n + K) visited
        nodes and O(log n) auxiliary stack space -- nothing is buffered.

        Iteration is in-order (lowest key first); the output order of
        matched pairs is not semantically constrained.
        """
        visits = 0
        lo_key = (lo, "")
        hi_key = (hi, _HIGH_ID)
        # In-order iterative traversal with window pruning.  Each stack
        # frame is (node, state): state 0 = unseen, 1 = already in-order.
        stack = [(self.root, 0)]
        while stack:
            node, state = stack.pop()
            if node is None:
                continue
            visits += 1
            if state:
                yield node.rec
                continue
            if node.key < lo_key:
                # Node and its left subtree are too old.
                stack.append((node.right, 0))
            elif node.key > hi_key:
                # Node and its right subtree are too new.
                stack.append((node.left, 0))
            else:
                # In-order: left, node, right.
                stack.append((node.right, 0))
                stack.append((node, 1))
                stack.append((node.left, 0))
        self.range_visits += visits


# Sentinel larger than every str id: valid ids are non-empty strings and
# this tuple is only used as an ordering bound, never stored.
_HIGH_ID = "\U0010ffff"


class _Rec:
    __slots__ = ("side", "rid", "key", "time", "payload", "paired")

    def __init__(self, side, rid, key, time, payload):
        self.side = side
        self.rid = rid
        self.key = key
        self.time = time
        self.payload = payload
        self.paired = False


class Join:
    """Streaming full-outer interval join.

    Parameters
    ----------
    W:
        Non-negative integer join window: same-key cross-side records are
        paired iff ``abs(tL - tR) <= W``.
    seed:
        Optional seed for the internal treap PRNG.  The default is fixed
        so runs are reproducible; the PRNG stream is private to the join.
    """

    def __init__(self, W, seed=0xC0FFEE):
        if not isinstance(W, int) or isinstance(W, bool) or W < 0:
            raise TypeError("W must be a non-negative integer")
        self._W = W
        # side -> watermark
        self._wm = {"L": _NEG_INF, "R": _NEG_INF}
        # side -> key -> treap of active _Rec
        self._trees = {"L": {}, "R": {}}
        # side -> heap of (time, rid, key, _Rec) for every active record
        self._heaps = {"L": [], "R": []}
        self._rng = random.Random(seed)
        self._finished = False
        self._active = 0

    # ------------------------------------------------------------------ #
    # Observability
    # ------------------------------------------------------------------ #
    @property
    def active_count(self):
        """Number of records currently retained in the join state."""
        return self._active

    def watermark(self, side):
        return self._wm[self._check_side(side)]

    # ------------------------------------------------------------------ #
    # Public API -- all mutators return iterators of (left_id, right_id)
    # ------------------------------------------------------------------ #
    def push(self, side, rid, key, time, payload):
        side = self._check_side(side)
        self._check_str(rid, "id")
        self._check_str(key, "key")
        if not isinstance(time, int) or isinstance(time, bool):
            raise TypeError("time must be an integer")
        if self._finished:
            raise RuntimeError("push() after finish()")
        if time <= self._wm[side]:
            raise ValueError(
                "record time must be strictly greater than the side watermark"
            )
        return self._push(side, rid, key, time, payload)

    def advance(self, side, watermark):
        side = self._check_side(side)
        if not isinstance(watermark, (int, float)) or isinstance(watermark, bool):
            raise TypeError("watermark must be an int or float('inf')")
        if watermark != watermark or watermark < self._wm[side]:
            raise ValueError("watermarks may only advance monotonically")
        return self._advance(side, watermark)

    def finish(self):
        return self._finish()

    # ------------------------------------------------------------------ #
    # Core machinery
    # ------------------------------------------------------------------ #
    def _push(self, side, rid, key, time, payload):
        opp = _other(side)
        rec = _Rec(side, rid, key, time, payload)

        # 1) Associate first: range query on the opposite side's treap for
        #    this key.  The treap only contains active records, and window
        #    pruning makes this O(log(A+2) + K).
        tree = self._trees[opp].get(key)
        if tree is not None:
            lo = time - self._W
            hi = time + self._W
            for other in tree.range_time(lo, hi):
                other.paired = True
                rec.paired = True
                if side == "L":
                    yield (rid, other.rid)
                else:
                    yield (other.rid, rid)

        # 2) Adopt the new record in both indexes (per-key treap + global
        #    release heap).
        bucket = self._trees[side]
        t = bucket.get(key)
        if t is None:
            t = _Treap(self._rng)
            bucket[key] = t
        t.insert(rec)
        heapq.heappush(self._heaps[side], (time, rid, key, rec))
        self._active += 1

        # 3) Release opposite-side records closed out by this side's
        #    current watermark: a record at time t can never match a
        #    future record on the operated side once wm >= t + W.
        yield from self._release(opp, self._wm[side])

    def _advance(self, side, watermark):
        # Monotonicity was validated by the wrapper; equal watermarks
        # release nothing.
        if watermark != self._wm[side]:
            self._wm[side] = watermark
            opp = _other(side)
            yield from self._release(opp, watermark)

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        self._wm["L"] = _INF
        self._wm["R"] = _INF
        # Both heaps drain entirely; order between the two sides is not
        # semantically constrained.
        yield from self._release("L", _INF)
        yield from self._release("R", _INF)

    def _release(self, side, cutoff_wm):
        """Release active ``side`` records with ``time + W <= cutoff_wm``.

        Unpaired records emit exactly one unmatched tuple; paired records
        are dropped silently.  The global min-heap makes this a linear
        drain over released records (each record is popped once over the
        whole stream), and the per-key treap keeps a matching erase at
        expected O(log n).
        """
        heap = self._heaps[side]
        trees = self._trees[side]
        while heap:
            time, rid, key, rec = heap[0]
            if time + self._W > cutoff_wm:
                break
            heapq.heappop(heap)
            tree = trees[key]
            tree.erase((time, rid))
            if tree.root is None:
                del trees[key]
            self._active -= 1
            if not rec.paired:
                if side == "L":
                    yield (rid, None)
                else:
                    yield (None, rid)

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_side(side):
        if side != "L" and side != "R":
            raise ValueError("side must be 'L' or 'R'")
        return side

    @staticmethod
    def _check_str(value, name):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")


_NEG_INF = float("-inf")


def _other(side):
    return "R" if side == "L" else "L"
