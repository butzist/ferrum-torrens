#!/usr/bin/env python3
"""Replay a decode route trace against expert-cache policies (R15 S13).

The trace comes from the engine's dev probe: run an eager decode with

    FT_ROUTE_TRACE=/tmp/routes.txt

which writes one line per (step, layer) as `layer id id ...` with layer-local
router ids. This script replays that sequence against the committed strict-LRU
policy (the `_ensure_experts_kernel` contract: one flat slot pool shared by every
layer, this call's hits pinned, misses take the coldest remaining slot, free
slots first) and against the alternatives an offline hot set or a frequency
policy would implement. Its miss counts reproduce `FT_MISS_TRACE` within a
couple of experts per step, so the model of the kernel is exact enough to size a
change before writing a kernel.

Usage: route-trace-sim.py TRACE [TRACE...]
"""
import random
import sys

NUM_EXPERTS = 256
SLOTS = 2048


def load(path):
    steps, cur = [], None
    for line in open(path):
        p = line.split()
        layer, ids = int(p[0]), [int(x) for x in p[1:]]
        if layer == 0:
            cur = []
            steps.append(cur)
        cur.append([layer * NUM_EXPERTS + e for e in ids if e >= 0])
    return steps


class Lru:
    """Strict LRU; `pin` is a static hot set that occupies the first slots."""

    def __init__(self, slots, pin=()):
        self.slots = slots
        self.pin_n = len(pin)
        if self.pin_n > slots:
            raise SystemExit("pin larger than the cache")
        self.pinned = set(pin)
        self.usage = [0] * slots
        self.id_of = [-1] * slots
        self.slot_of = {}
        for i, fid in enumerate(pin):
            self.slot_of[fid] = i
            self.id_of[i] = fid
            self.usage[i] = 1 << 60
        self.step = 0

    def touch(self):
        self.step += 1
        return self.step

    def pick(self, todo, st):
        cand = sorted(
            (self.usage[s], s)
            for s in range(self.pin_n, self.slots)
            if self.usage[s] != st
        )
        if len(cand) < len(todo):
            raise SystemExit(f"empty candidate set: {len(cand)} < {len(todo)}")
        return [s for _, s in cand[: len(todo)]]

    def install(self, todo, take, st):
        for s in take:
            old = self.id_of[s]
            if old >= 0:
                del self.slot_of[old]
        for fid, s in zip(todo, take):
            self.slot_of[fid] = s
            self.id_of[s] = fid
            self.usage[s] = st


class Lfu(Lru):
    """Frequency admission: least-used first, recency as the tiebreak."""

    def __init__(self, slots, horizon=None):
        super().__init__(slots)
        self.freq = [0] * slots
        self.horizon = horizon
        self.last_decay = 0

    def pick(self, todo, st):
        if self.horizon and st - self.last_decay >= self.horizon:
            self.freq = [f // 2 for f in self.freq]
            self.last_decay = st
        cand = sorted(
            (self.freq[s], self.usage[s], s)
            for s in range(self.pin_n, self.slots)
            if self.usage[s] != st
        )
        return [s for _, _, s in cand[: len(todo)]]

    def install(self, todo, take, st):
        for s in take:
            old = self.id_of[s]
            if old >= 0:
                del self.slot_of[old]
            self.freq[s] = 0
        for fid, s in zip(todo, take):
            self.slot_of[fid] = s
            self.id_of[s] = fid
            self.usage[s] = st
            self.freq[s] = 1


class RandomVictim(Lru):
    def __init__(self, slots, seed=7):
        super().__init__(slots)
        self.rng = random.Random(seed)

    def pick(self, todo, st):
        pool = [s for s in range(self.pin_n, self.slots) if self.usage[s] != st]
        return self.rng.sample(pool, len(todo))


def replay(steps, cache, warmup=0):
    per = []
    for i, layers in enumerate(steps):
        miss = 0
        for ids in layers:
            st = cache.touch()
            todo, seen = [], set()
            for fid in ids:
                if fid in seen:
                    continue
                seen.add(fid)
                s = cache.slot_of.get(fid)
                if s is None:
                    todo.append(fid)
                else:
                    cache.usage[s] = st
                    if hasattr(cache, "freq"):
                        cache.freq[s] += 1
            if todo:
                miss += len(todo)
                cache.install(todo, cache.pick(todo, st), st)
        if i >= warmup:
            per.append(miss)
    return per


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def study(path):
    steps = load(path)
    routes = sum(len(ids) for s in steps for ids in s)
    print(f"\n{path}: {len(steps)} steps, {routes} routes")
    base = mean(replay(steps, Lru(SLOTS)))
    print(f"  {'LRU 2048 (committed)':34s} {base:7.2f} miss/step")

    for slots in (SLOTS + 256, SLOTS + 1024):
        m = mean(replay(steps, Lru(slots)))
        print(f"  {f'LRU {slots}':34s} {m:7.2f} {(m / base - 1) * 100:+6.1f}%")

    for name, cache in (
        ("LFU (no decay)", Lfu(SLOTS)),
        ("LFU (decay/4096 calls)", Lfu(SLOTS, 4096)),
        ("random victim", RandomVictim(SLOTS)),
    ):
        m = mean(replay(steps, cache))
        print(f"  {name:34s} {m:7.2f} {(m / base - 1) * 100:+6.1f}%")

    # The pre-collected hot set: the top-P ids of the whole trace, pinned. It is
    # an upper bound (chosen with hindsight), so it bounds any offline list.
    freq = {}
    for s in steps:
        for ids in s:
            for fid in ids:
                freq[fid] = freq.get(fid, 0) + 1
    rank = [fid for fid, _ in sorted(freq.items(), key=lambda kv: -kv[1])]
    for p in (256, 1024, 1792):
        m = mean(replay(steps, Lru(SLOTS, pin=rank[:p])))
        print(f"  {f'pin {p} oracle + LRU':34s} {m:7.2f} {(m / base - 1) * 100:+6.1f}%")

    # What a set collected from the first steps would have scored: the realistic
    # version, scored on the steps after the collection window.
    for warm in (16, 32):
        head = {}
        for s in steps[:warm]:
            for ids in s:
                for fid in ids:
                    head[fid] = head.get(fid, 0) + 1
        rank = [fid for fid, _ in sorted(head.items(), key=lambda kv: -kv[1])]
        cold = mean(replay(steps, Lru(SLOTS), warmup=warm))
        for p in (512, 1024):
            m = mean(replay(steps, Lru(SLOTS, pin=rank[:p]), warmup=warm))
            print(
                f"  {f'pin {p} from {warm} steps':34s} {m:7.2f} "
                f"{(m / cold - 1) * 100:+6.1f}% (vs LRU on the same window)"
            )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for path in sys.argv[1:]:
        study(path)
