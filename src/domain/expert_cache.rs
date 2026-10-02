// MoE expert-slot LRU cache. Pure slot machine mirroring FreeToken's
// `_ensure_experts_hybrid_cpu` (the CPU oracle for the GPU kernel): each expert
// occupies one flat id `layer * num_experts + expert`, `slot_for_id` maps id ->
// slot (-1 = not resident), `id_of_slot` maps slot -> id, `usage[slot]` is the
// last step the slot was touched. Copying the bytes in/out of slots is an
// infrastructure concern (this module only emits the copy plan).

pub struct ExpertCache {
    num_layers: usize,
    num_experts: usize,
    cache_size: usize,
    slot_for_id: Vec<i32>,
    id_of_slot: Vec<i32>,
    usage: Vec<u64>,
    step: u64,
}

/// Plan produced by `ensure_experts`; infra consumes `misses`+`write_slots` to
/// schedule host->slot copies and uses `rewrites` to reindex routed tokens.
pub struct ExpertPlan {
    /// Layer-local expert ids needing a copy this step, ascending.
    pub misses: Vec<u32>,
    /// Aligned with `misses`: the slot each copied expert will land in.
    pub write_slots: Vec<usize>,
    /// Aligned with the caller's routed ids: resident slot or -1 (overflow).
    pub rewrites: Vec<i32>,
}

impl ExpertCache {
    /// `cache_size >= num_experts` is the engine's floor (config re-checks it
    /// against num_experts); the slot machine itself stays well-defined for
    /// smaller caches, which just evict more eagerly.
    pub fn new(num_layers: usize, num_experts: usize, cache_size: usize) -> Self {
        debug_assert!(num_layers * num_experts > 0);
        debug_assert!(cache_size > 0);
        Self {
            num_layers,
            num_experts,
            cache_size,
            slot_for_id: vec![-1; num_layers * num_experts],
            id_of_slot: vec![-1; cache_size],
            usage: vec![0; cache_size],
            step: 0,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    pub fn cache_size(&self) -> usize {
        self.cache_size
    }

    /// Slot an expert currently occupies, or -1.
    pub fn slot_of(&self, layer: usize, expert: u32) -> i32 {
        assert!(layer < self.num_layers && (expert as usize) < self.num_experts);
        self.slot_for_id[layer * self.num_experts + expert as usize]
    }

    /// Make `layer`'s routed experts resident and rewrite them to slot ids.
    pub fn ensure_experts(&mut self, layer: usize, routed: &[u32]) -> ExpertPlan {
        debug_assert!(layer < self.num_layers);
        let id_base = layer * self.num_experts;

        let mut seen: Vec<u32> = Vec::with_capacity(routed.len());
        for &e in routed {
            debug_assert!((e as usize) < self.num_experts);
            if !seen.contains(&e) {
                seen.push(e);
            }
        }

        self.step += 1;
        let step = self.step;

        for &e in &seen {
            let slot = self.slot_for_id[id_base + e as usize];
            if slot >= 0 {
                self.usage[slot as usize] = step;
            }
        }

        let mut missing: Vec<u32> = seen
            .iter()
            .copied()
            .filter(|&e| self.slot_for_id[id_base + e as usize] == -1)
            .collect();
        missing.sort_unstable();

        let rewrites = Vec::with_capacity(routed.len());
        let mut plan = ExpertPlan {
            misses: Vec::new(),
            write_slots: Vec::new(),
            rewrites,
        };

        for &e in &missing {
            let victim = self.pick_victim();
            let old_id = self.id_of_slot[victim];
            if old_id >= 0 {
                self.slot_for_id[old_id as usize] = -1;
            }
            let flat = (id_base + e as usize) as i32;
            self.id_of_slot[victim] = flat;
            self.slot_for_id[id_base + e as usize] = victim as i32;
            self.usage[victim] = step;
            plan.misses.push(e);
            plan.write_slots.push(victim);
        }

        for &e in routed {
            plan.rewrites.push(self.slot_for_id[id_base + e as usize]);
        }
        plan
    }

    /// Drop every expert of `layer` back to "not resident". The GPU slot cache
    /// holds one layer's working set at a time (256 experts x ~2.6 MB fits once
    /// on 8 GB, not forty times), so switching layers has to invalidate the
    /// layer's slots or the adapter would read another layer's bytes.
    pub fn forget_layer(&mut self, layer: usize) {
        assert!(layer < self.num_layers);
        let id_base = layer * self.num_experts;
        for e in 0..self.num_experts {
            let slot = self.slot_for_id[id_base + e];
            if slot >= 0 {
                self.id_of_slot[slot as usize] = -1;
                self.usage[slot as usize] = 0;
                self.slot_for_id[id_base + e] = -1;
            }
        }
    }

    /// argmin (usage, slot): least-recently-used slot wins, ties by lower slot.
    fn pick_victim(&self) -> usize {
        let mut victim = 0usize;
        let mut best = (self.usage[0], 0usize);
        for (s, usage) in self.usage.iter().enumerate().skip(1) {
            let key = (*usage, s);
            if key < best {
                best = key;
                victim = s;
            }
        }
        victim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forget_layer_invalidates_only_that_layer() {
        let mut c = ExpertCache::new(2, 4, 4);
        c.ensure_experts(0, &[0, 1]);
        c.ensure_experts(1, &[0, 1]);
        assert!(c.slot_of(0, 0) >= 0 && c.slot_of(1, 0) >= 0);

        c.forget_layer(0);
        assert_eq!(c.slot_of(0, 0), -1);
        assert_eq!(c.slot_of(0, 1), -1);
        assert!(c.slot_of(1, 0) >= 0, "layer 1 keeps its slots");
        assert_eq!(
            c.id_of_slot
                .iter()
                .filter(|&&id| (0..4).contains(&id))
                .count(),
            0
        );
        assert_eq!(
            c.id_of_slot
                .iter()
                .filter(|&&id| (4..8).contains(&id))
                .count(),
            2
        );

        // Coming back to layer 0 must re-miss everything.
        let p = c.ensure_experts(0, &[0, 1]);
        assert_eq!(p.misses, vec![0, 1]);
        assert!(p.rewrites.iter().all(|&s| s >= 0));
    }

    #[test]
    fn forget_layer_is_idempotent_and_frees_slots_for_the_next_layer() {
        let mut c = ExpertCache::new(2, 3, 3);
        c.ensure_experts(0, &[0, 1, 2]);
        c.forget_layer(0);
        c.forget_layer(0);
        let p = c.ensure_experts(1, &[0, 1, 2]);
        assert_eq!(p.misses, vec![0, 1, 2], "the freed slots are reusable");
        assert_eq!(p.write_slots, vec![0, 1, 2]);
    }

    #[test]
    fn fills_free_slots_first_lowest_slot() {
        let mut c = ExpertCache::new(2, 4, 8);
        assert_eq!((c.num_layers(), c.num_experts(), c.cache_size()), (2, 4, 8));
        let p = c.ensure_experts(0, &[2, 0]);
        // seen = [2, 0]; missing sorted = [0, 2]; victims (usage 0) = slots 0,1.
        assert_eq!(p.misses, vec![0, 2]);
        assert_eq!(p.write_slots, vec![0, 1]);
        assert_eq!(p.rewrites, vec![1, 0]);
        assert_eq!(c.slot_of(0, 0), 0);
        assert_eq!(c.slot_of(0, 2), 1);
        assert_eq!(c.id_of_slot[0], 0, "flat id layer 0, expert 0");
        assert_eq!(c.id_of_slot[1], 2, "flat id layer 0, expert 2");
    }

    #[test]
    fn hits_are_resident_and_noop() {
        let mut c = ExpertCache::new(1, 4, 8);
        c.ensure_experts(0, &[1, 3]);
        let p = c.ensure_experts(0, &[3, 1, 3]);
        assert!(p.misses.is_empty());
        assert_eq!(
            p.rewrites,
            vec![c.slot_of(0, 3), c.slot_of(0, 1), c.slot_of(0, 3)]
        );
        assert!(p.rewrites.iter().all(|&s| s >= 0));
    }

    #[test]
    fn full_cache_within_one_layer_never_misses() {
        // cache_size == num_experts: the engine's floor, so a layer can always
        // hold every expert it routes this step without evicting a live one.
        let mut c = ExpertCache::new(1, 6, 6);
        assert_eq!(
            c.ensure_experts(0, &[0, 1, 2, 3, 4, 5]).write_slots,
            vec![0, 1, 2, 3, 4, 5]
        );
        let p = c.ensure_experts(0, &[5, 2, 0]);
        assert!(p.misses.is_empty());
        assert_eq!(p.rewrites, vec![5, 2, 0]);
    }

    #[test]
    fn lru_victim_is_the_coldest_slot() {
        // num_experts(8) > cache_size(6): the reference tolerates it (free
        // slots have usage 0, so they go first), so a miss can only evict
        // once every slot is warm, and the coldest one dies.
        let mut c = ExpertCache::new(1, 8, 6);
        c.ensure_experts(0, &[0, 1, 2, 3, 4, 5]);
        c.ensure_experts(0, &[5, 0]); // bump 5 and 0 to step 2
        let p = c.ensure_experts(0, &[6]);
        assert_eq!(p.misses, vec![6]);
        assert_eq!(
            p.write_slots,
            vec![1],
            "slot 1 = expert 1, coldest at step 1"
        );
        assert_eq!(c.slot_of(0, 1), -1);
        assert_eq!(c.slot_of(0, 5), 5, "warm expert 5 survives");
    }

    #[test]
    fn same_step_hits_not_evicted_while_colder_guests_are() {
        let mut c = ExpertCache::new(1, 8, 6);
        assert_eq!(
            c.ensure_experts(0, &[0, 1, 2, 3, 4, 5]).write_slots,
            vec![0, 1, 2, 3, 4, 5]
        );
        // Step 2: touch 0,1 again, then bring 6,7. All six slots are warm, so
        // the victims are 2,3 (coldest at step 1): never 0/1, used this step.
        c.ensure_experts(0, &[0, 1]);
        let p = c.ensure_experts(0, &[6, 7]);
        assert_eq!(p.write_slots, vec![2, 3]);
        assert_eq!(c.slot_of(0, 0), 0);
        assert_eq!(c.slot_of(0, 1), 1);
        assert_eq!(c.slot_of(0, 2), -1, "cold expert 2 was evicted");
    }

    #[test]
    fn layer_id_space_is_flat() {
        let mut c = ExpertCache::new(2, 4, 8);
        c.ensure_experts(0, &[0]);
        c.ensure_experts(1, &[0, 0]);
        let e = c.slot_of(0, 0);
        assert_ne!(e, -1);
        let e1 = c.slot_of(1, 0);
        assert_ne!(e1, -1);
        assert_ne!(e, e1, "expert 0 of different layers are different ids");
        let p = c.ensure_experts(1, &[1, 2, 3]);
        assert_eq!(p.rewrites.len(), 3);
        assert!(c.slot_of(1, 0) >= 0 && c.slot_of(1, 1) >= 0);
    }

    #[test]
    fn duplicate_routed_ids_do_not_multiply() {
        let mut c = ExpertCache::new(1, 4, 8);
        let p = c.ensure_experts(0, &[2, 2, 2]);
        assert_eq!(p.misses, vec![2]);
        assert_eq!(p.rewrites, vec![c.slot_of(0, 2); 3]);
    }

    #[test]
    fn cache_smaller_than_experts_still_deterministic() {
        // Engine enforces cache_size >= num_experts; sanity-check beyond that
        // boundary stays a stable LRU round-robin over the slots.
        let mut c = ExpertCache::new(1, 6, 3);
        let p = c.ensure_experts(0, &[5, 4, 3, 2]);
        assert_eq!(p.misses, vec![2, 3, 4, 5]);
        assert_eq!(p.write_slots, vec![0, 1, 2, 0]);
        // expert 2 was evicted by itself; the rewrites are the per-occurrence
        // slots, so the first occurrence of 5 keeps its (later evicted) slot.
        assert_eq!(
            p.rewrites,
            vec![
                c.slot_of(0, 5),
                c.slot_of(0, 4),
                c.slot_of(0, 3),
                c.slot_of(0, 2)
            ]
        );
        assert!(c.slot_of(0, 5) >= 0 && c.slot_of(0, 4) >= 0 && c.slot_of(0, 3) >= 0);
    }
}
