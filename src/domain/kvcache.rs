// Pure KV bookkeeping: the paged pool's free list, per-request page tables, the
// radix prefix tree and the GDN linear-state slot pool. Nothing here touches
// device tensors; infra (kvcache_dev) owns the bytes these indices point at.
//
// Radix semantics mirror FreeToken's `kvcache/radix_cache.py` (node split at a
// partial page-boundary match, unlocked-leaf LRU eviction, ref-counted locks).
// page_size == 1 is the only supported granularity today (config enforces it).

use std::collections::BTreeMap;

use crate::domain::model::{ForwardInput, Model};

pub const SENTINEL_PAGE: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum KvcacheError {
    #[error("kv cache out of pages: wanted {want}, have {have}")]
    OutOfPages { want: usize, have: usize },
    /// A cached forward was asked for something its cache cannot serve, e.g. a
    /// batch, or tokens it was not grown for. `why` says which.
    #[error("cached forward asked for something its cache cannot serve: {why}")]
    MisalignedCache { why: String },
    #[error("not enough evictable pages: wanted {want}, have {have}")]
    NotEnoughEvictable { want: usize, have: usize },
    #[error("no unlocked leaf to evict")]
    NoLeavesEvictable,
}

/// MHA paged pool free list. Page ids are flat indices into the device slabs;
/// the sentinel page id is out of range so dummy rows never alias a real one.
pub struct PagePool {
    page_size: usize,
    total_pages: usize,
    free: Vec<u32>,
}

impl PagePool {
    pub fn new(page_size: usize, total_pages: usize) -> Self {
        Self {
            page_size,
            total_pages,
            free: (0..total_pages as u32).rev().collect(),
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn total_pages(&self) -> usize {
        self.total_pages
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn allocate(&mut self, n: usize) -> Result<Vec<u32>, KvcacheError> {
        if n > self.free.len() {
            return Err(KvcacheError::OutOfPages {
                want: n,
                have: self.free.len(),
            });
        }
        Ok(self.free.split_off(self.free.len() - n))
    }

    pub fn free_pages(&mut self, pages: impl IntoIterator<Item = u32>) {
        for p in pages {
            debug_assert!(p != SENTINEL_PAGE && (p as usize) < self.total_pages);
            debug_assert!(!self.free.contains(&p));
            self.free.push(p);
        }
    }
}

/// One request's token -> page row. Unwritten positions hold the sentinel page
/// (dummy row), so decode does not need the head-of-history padding the MHA
/// pool keeps.
#[derive(Debug, Clone, Default)]
pub struct ReqPageTable {
    row: Vec<u32>,
}

impl ReqPageTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.row.len()
    }

    pub fn is_empty(&self) -> bool {
        self.row.is_empty()
    }

    pub fn push(&mut self, page: u32) {
        self.row.push(page);
    }

    pub fn set(&mut self, pos: usize, page: u32) -> &mut Self {
        if self.row.len() <= pos {
            self.row.resize(pos + 1, SENTINEL_PAGE);
        }
        self.row[pos] = page;
        self
    }

    pub fn page_at(&self, pos: usize) -> u32 {
        self.row.get(pos).copied().unwrap_or(SENTINEL_PAGE)
    }

    pub fn pages(&self) -> &[u32] {
        &self.row
    }
}

type NodeId = usize;

struct RadixNode {
    token_key: Vec<u32>,
    pages: Vec<u32>,
    parent: Option<NodeId>,
    children: BTreeMap<u32, NodeId>,
    refcount: u32,
    timestamp: u64,
}

/// Radix prefix cache holding token keys and the KV pages that back each span.
/// Ref-counted locks protect an active request's path from eviction.
pub struct RadixIndex {
    page_size: usize,
    nodes: Vec<RadixNode>,
    root: NodeId,
    clock: u64,
    evictable: usize,
    protected: usize,
}

pub struct RadixMatch {
    pub node: NodeId,
    pub prefix_len: usize,
}

impl RadixIndex {
    pub fn new(page_size: usize) -> Self {
        debug_assert_eq!(page_size, 1, "only page_size == 1 is implemented");
        let nodes = vec![RadixNode {
            token_key: Vec::new(),
            pages: Vec::new(),
            parent: None,
            children: BTreeMap::new(),
            refcount: 1,
            timestamp: 0,
        }];
        Self {
            page_size,
            nodes,
            root: 0,
            clock: 0,
            evictable: 0,
            protected: 0,
        }
    }

    pub fn evictable(&self) -> usize {
        self.evictable
    }

    pub fn protected(&self) -> usize {
        self.protected
    }

    pub fn insert_prefix(&mut self, token_key: &[u32], pages: &[u32]) -> RadixMatch {
        debug_assert_eq!(token_key.len(), pages.len());
        let insert_len = token_key.len() / self.page_size * self.page_size;
        let (mut node, prefix_len) = self.tree_walk(&token_key[..insert_len]);
        if prefix_len != insert_len {
            let new = self.new_leaf(
                &token_key[prefix_len..insert_len],
                &pages[prefix_len..insert_len],
            );
            self.reparent(node, new);
            self.evictable += new_len(&self.nodes[new]);
            node = new;
        }
        RadixMatch { node, prefix_len }
    }

    pub fn match_prefix(&mut self, token_key: &[u32]) -> RadixMatch {
        let (node, prefix_len) = self.tree_walk(token_key);
        RadixMatch { node, prefix_len }
    }

    pub fn lock(&mut self, node: NodeId) {
        self.adjust_refcount(node, 1);
    }

    pub fn unlock(&mut self, node: NodeId) {
        self.adjust_refcount(node, -1);
    }

    /// Evict `size` pages of unlocked leaves, oldest timestamp first, and
    /// return the freed page ids. Mirrors the reference evict (heap of
    /// unlocked leaves; a parent that becomes an unlocked leaf rejoins).
    pub fn evict(&mut self, size: usize) -> Result<Vec<u32>, KvcacheError> {
        if size == 0 {
            return Ok(Vec::new());
        }
        if size > self.evictable {
            return Err(KvcacheError::NotEnoughEvictable {
                want: size,
                have: self.evictable,
            });
        }
        let mut freed = Vec::new();
        let mut freed_len = 0usize;
        loop {
            if freed_len >= size {
                break;
            }
            let sel = self
                .oldest_unlocked_leaf()
                .ok_or(KvcacheError::NoLeavesEvictable)?;
            freed_len += self.nodes[sel].pages.len();
            self.evictable -= self.nodes[sel].pages.len();
            freed.extend_from_slice(&self.nodes[sel].pages);
            let parent = self.nodes[sel].parent.expect("leaf is never the root");
            let key = first_key(&self.nodes[sel]);
            self.nodes[parent].children.remove(&key);
            self.nodes[sel].parent = None;
        }
        Ok(freed)
    }

    fn reparent(&mut self, parent: NodeId, child: NodeId) {
        let key = first_key(&self.nodes[child]);
        debug_assert!(!self.nodes[parent].children.contains_key(&key));
        self.nodes[child].parent = Some(parent);
        self.nodes[parent].children.insert(key, child);
    }

    fn new_leaf(&mut self, token_key: &[u32], pages: &[u32]) -> NodeId {
        debug_assert!(!token_key.is_empty());
        debug_assert_eq!(token_key.len(), pages.len());
        assert!(
            token_key.len().is_multiple_of(self.page_size),
            "insert_len is page-aligned"
        );
        self.clock += 1;
        self.nodes.push(RadixNode {
            token_key: token_key.to_vec(),
            pages: pages.to_vec(),
            parent: None,
            children: BTreeMap::new(),
            refcount: 0,
            timestamp: self.clock,
        });
        self.nodes.len() - 1
    }

    fn adjust_refcount(&mut self, node: NodeId, delta: i32) {
        let mut cur = Some(node);
        while let Some(n) = cur {
            let len = self.nodes[n].pages.len();
            let rc = self.nodes[n].refcount;
            if rc == 0 && delta > 0 {
                self.evictable -= len;
                self.protected += len;
            } else if rc == 1 && delta < 0 {
                self.evictable += len;
                self.protected -= len;
            }
            self.nodes[n].refcount = (rc as i32 + delta) as u32;
            cur = self.nodes[n].parent;
        }
    }

    fn oldest_unlocked_leaf(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| {
                !n.token_key.is_empty()
                    && n.children.is_empty()
                    && n.refcount == 0
                    && n.parent.is_some()
            })
            .min_by_key(|(_, n)| n.timestamp)
            .map(|(id, _)| id)
    }

    fn tree_walk(&mut self, token_key: &[u32]) -> (NodeId, usize) {
        self.clock += 1;
        let clock = self.clock;
        let mut prefix_len = 0usize;
        let mut node = self.root;
        while prefix_len < token_key.len() {
            let first = token_key[prefix_len];
            let Some(&child) = self.nodes[node].children.get(&first) else {
                return (node, prefix_len);
            };
            node = child;
            let matched = self.match_len(node, &token_key[prefix_len..]);
            let matched = matched / self.page_size * self.page_size;
            prefix_len += matched;
            if matched != self.nodes[node].token_key.len() {
                let split = self.split_at(node, matched);
                self.nodes[split].timestamp = clock;
                return (split, prefix_len);
            }
            self.nodes[node].timestamp = clock;
        }
        (node, prefix_len)
    }

    /// Number of leading token_key entries matching `node`'s key.
    fn match_len(&self, node: NodeId, token_key: &[u32]) -> usize {
        let key = &self.nodes[node].token_key;
        key.iter()
            .zip(token_key)
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Split `node` at `pos` (page boundary), inserting a fresh prefix node
    /// between it and its parent; the suffix keeps being `node`.
    fn split_at(&mut self, node: NodeId, pos: usize) -> NodeId {
        debug_assert!(pos > 0 && pos < self.nodes[node].token_key.len());
        self.clock += 1;
        let parent = self.nodes[node].parent.expect("root is never split");
        let prefix_pages = self.nodes[node].pages[..pos].to_vec();
        let prefix_key = self.nodes[node].token_key[..pos].to_vec();
        self.nodes[node].token_key.drain(..pos);
        self.nodes[node].pages.drain(..pos);
        self.nodes.push(RadixNode {
            token_key: prefix_key,
            pages: prefix_pages,
            parent: Some(parent),
            children: BTreeMap::new(),
            refcount: self.nodes[node].refcount,
            timestamp: self.clock,
        });
        let prefix = self.nodes.len() - 1;
        // The suffix is keyed by its own new first token; the parent keeps
        // pointing at the prefix node (same first token as the old node).
        let suffix_key = first_key(&self.nodes[node]);
        self.nodes[prefix].children.insert(suffix_key, node);
        self.nodes[node].parent = Some(prefix);
        let prefix_key = first_key(&self.nodes[prefix]);
        self.nodes[parent].children.insert(prefix_key, prefix);
        prefix
    }
}

/// page_size == 1: a node's routing key is its first token.
fn first_key(node: &RadixNode) -> u32 {
    node.token_key[0]
}

fn new_len(node: &RadixNode) -> usize {
    node.pages.len()
}

/// GDN linear-state pool bookkeeping: free-slot list of recurrent + conv state
/// rows. The ping-pong/restore dance between live slots and tree snapshots is
/// infra; this type only hands out and reclaims row ids.
pub struct LinearStatePool {
    total: usize,
    free: Vec<u32>,
}

impl LinearStatePool {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            free: (0..total as u32).rev().collect(),
        }
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn available(&self) -> usize {
        self.free.len()
    }

    pub fn allocate(&mut self) -> Option<u32> {
        self.free.pop()
    }

    pub fn release(&mut self, slot: u32) {
        debug_assert!((slot as usize) < self.total && !self.free.contains(&slot));
        self.free.push(slot);
    }
}

/// One running request's cache bookkeeping: the pool location of every token
/// whose KV is stored, and the linear-state slot its GDN layers own.
///
/// Pure indices. The device slabs behind them belong to the compute adapter
/// (`infrastructure::kvcache_dev`), which is handed `filled()` to read and
/// `tail()` to write, so this type stays testable without libtorch.
pub struct SequenceCache {
    /// `pages[i]` is the pool location of the request's token `i`. `page_size`
    /// is 1, so this doubles as the token-location row the radix tree stores.
    pages: Vec<u32>,
    /// How many of `pages` have actually been written this request.
    len: usize,
    state_slot: u32,
}

impl SequenceCache {
    pub fn new(state_slot: u32) -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
            state_slot,
        }
    }

    /// Tokens whose KV and GDN state are already computed.
    pub fn cached(&self) -> usize {
        self.len
    }

    pub fn state_slot(&self) -> u32 {
        self.state_slot
    }

    /// Reserve room for `n` more tokens, drawing pages from the pool. The
    /// caller owns the pool and grows the cache before the forward, because the
    /// forward only writes into the locations it is handed.
    pub fn grow(&mut self, n: usize, pool: &mut PagePool) -> Result<(), KvcacheError> {
        let pages = pool.allocate(n)?;
        self.pages.extend_from_slice(&pages);
        self.len += n;
        Ok(())
    }

    /// Give `n` of this request's pages back, keeping the earlier ones. Used
    /// when a prefix is rolled back to a shorter cached length.
    pub fn truncate(&mut self, n: usize, pool: &mut PagePool) {
        debug_assert!(n <= self.len);
        let keep = self.len - n;
        let tail = self.pages.split_off(keep);
        pool.free_pages(tail);
        self.len = keep;
    }

    /// Hand every page back, e.g. when the request finishes.
    pub fn release(&mut self, pool: &mut PagePool) {
        self.truncate(self.len, pool);
    }

    /// Every filled location, in token order: what attention gathers over.
    pub fn filled(&self) -> &[u32] {
        &self.pages[..self.len]
    }

    /// The `n` locations a step of `n` tokens writes: the tail of `filled()`.
    pub fn tail(&self, n: usize) -> &[u32] {
        debug_assert!(n <= self.len);
        &self.pages[self.len - n..self.len]
    }
}

/// The engine-facing side of the cache: a pool of pages and of GDN state slots,
/// the device slabs behind them included.
///
/// The port keeps tensors out of the domain: a caller learns the page numbers it
/// must reserve, hands them to a cached forward, and the adapter does the rest.
/// Implementation in `infrastructure::kvcache_dev`.
pub trait KvCache: Send + Sync {
    /// A fresh request: an empty page table and a private state slot. Grow it
    /// with `reserve` before the first forward.
    fn begin(&self) -> SequenceCache;

    /// Make room for `n` more tokens and return them as the cache's tail.
    fn reserve(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError>;

    /// Give the request's pages and state slot back.
    fn end(&self, cache: &mut SequenceCache);

    /// Pages the pool can still hand out.
    fn available(&self) -> usize;

    /// Total pages, i.e. the token budget across every live request.
    fn num_pages(&self) -> usize;
}

/// A forward that reads and writes a `SequenceCache` instead of re-running the
/// prefix. Implemented by the compute adapter; the domain owns the contract.
///
/// `cache` must already be grown by `input`'s token count (`reserve`), and
/// `input` must hold exactly one sequence: the cached path has no notion of a
/// batch, since every sequence's KV lives in its own page table.
pub trait CachedModel: Model {
    /// The cache this forward reads and writes, if it has one. The engine takes
    /// it to open sessions; an adapter without a cache returns `None` and falls
    /// back to the stateless forward below.
    fn cache(&self) -> Option<std::sync::Arc<dyn KvCache>> {
        None
    }

    /// Advance `cache` over `input`'s tokens and return the last token's logits.
    ///
    /// The default re-runs the whole prefix and ignores the cache, which is
    /// correct but slow: a model that cannot cache still answers, it just pays
    /// for the prefix again every step.
    fn forward_cached(
        &self,
        input: &ForwardInput,
        cache: &mut SequenceCache,
    ) -> Result<Vec<f32>, KvcacheError> {
        let _ = cache;
        Ok(self.forward(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_pool_hands_out_distinct_pages() {
        let mut p = PagePool::new(1, 4);
        assert_eq!(p.page_size(), 1);
        assert_eq!(p.total_pages(), 4);
        assert_eq!(p.available(), 4);
        let a = p.allocate(2).unwrap();
        let b = p.allocate(2).unwrap();
        assert_eq!(a.len() + b.len(), 4);
        assert!(a.iter().all(|x| !b.contains(x)), "no double hand-out");
        assert_eq!(p.available(), 0);
        assert_eq!(
            p.allocate(1),
            Err(KvcacheError::OutOfPages { want: 1, have: 0 })
        );
        p.free_pages(a);
        assert_eq!(p.available(), 2);
    }

    #[test]
    fn page_table_dummy_rows_point_at_sentinel() {
        let mut t = ReqPageTable::new();
        assert!(t.is_empty());
        assert_eq!(t.page_at(0), SENTINEL_PAGE);
        t.set(3, 7);
        assert_eq!(t.len(), 4, "gap filled with sentinel rows");
        assert_eq!(t.page_at(0), SENTINEL_PAGE);
        assert_eq!(t.page_at(3), 7);
        t.push(9);
        assert_eq!(t.page_at(4), 9);
        assert_eq!(t.page_at(99), SENTINEL_PAGE, "beyond the row is sentinel");
    }

    #[test]
    fn radix_insert_then_match_full_prefix() {
        let mut r = RadixIndex::new(1);
        let m = r.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        assert_eq!(m.prefix_len, 0, "nothing cached yet");
        assert_eq!(r.evictable(), 3);
        let m2 = r.match_prefix(&[1, 2, 3]);
        assert_eq!(m2.prefix_len, 3, "full hit");
    }

    #[test]
    fn radix_match_splits_on_partial_prefix() {
        let mut r = RadixIndex::new(1);
        r.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        // Common prefix [1,2] only: walk must split the [1,2,3] node.
        let m = r.match_prefix(&[1, 2, 9]);
        assert_eq!(m.prefix_len, 2);
        // After the split, matching [1,2] hits the new prefix node, and
        // matching [1,2,3] still hits the original suffix.
        assert_eq!(r.match_prefix(&[1, 2]).prefix_len, 2);
        assert_eq!(r.match_prefix(&[1, 2, 3]).prefix_len, 3);
    }

    #[test]
    fn radix_insert_under_common_prefix() {
        let mut r = RadixIndex::new(1);
        r.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        let m = r.insert_prefix(&[1, 2, 9], &[20, 21, 22]);
        assert_eq!(m.prefix_len, 2, "reuses the cached [1,2] prefix");
        // The split keeps 3 old pages; only the unmatched tail [9] is new.
        assert_eq!(r.evictable(), 4);
        assert_eq!(r.match_prefix(&[1, 2, 9]).prefix_len, 3);
        assert_eq!(r.match_prefix(&[1, 2, 3]).prefix_len, 3);
    }

    #[test]
    fn radix_lock_protects_from_eviction() {
        let mut r = RadixIndex::new(1);
        let m = r.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        r.lock(m.node);
        assert_eq!(r.protected(), 3);
        assert_eq!(r.evictable(), 0);
        assert_eq!(
            r.evict(3),
            Err(KvcacheError::NotEnoughEvictable { want: 3, have: 0 })
        );
        r.unlock(m.node);
        assert_eq!(r.evictable(), 3);
        assert_eq!(r.protected(), 0);
    }

    #[test]
    fn radix_evicts_oldest_unlocked_leaf_first() {
        let mut r = RadixIndex::new(1);
        r.insert_prefix(&[1], &[10]);
        r.insert_prefix(&[2], &[20]);
        r.insert_prefix(&[3], &[30]);
        // Access 2 to bump its recency: 1 is now the oldest leaf.
        assert_eq!(r.match_prefix(&[2]).prefix_len, 1);
        assert_eq!(r.evict(1).unwrap(), vec![10]);
        assert_eq!(r.evictable(), 2);
        assert_eq!(
            r.evict(1).unwrap(),
            vec![30],
            "2 was refreshed, so 3 is next"
        );
    }

    #[test]
    fn radix_evicts_whole_leaves_oldest_first() {
        let mut r = RadixIndex::new(1);
        r.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        r.insert_prefix(&[1, 2, 4], &[20, 21, 22]);
        r.insert_prefix(&[9], &[30]);
        assert_eq!(r.evictable(), 5, "3 old pages + [4] leaf + [9] leaf");
        // The [1,2] split node has two children, so it only becomes a leaf
        // (and thus evictable) once both [3] and [4] are gone.
        assert_eq!(r.evict(1).unwrap(), vec![12], "[3] is the oldest leaf");
        assert_eq!(r.evict(1).unwrap(), vec![22], "then [4]");
        assert_eq!(
            r.evict(1).unwrap(),
            vec![10, 11],
            "the [1,2] parent is next"
        );
        assert_eq!(r.evictable(), 1);
        assert_eq!(r.match_prefix(&[1, 2, 3]).prefix_len, 0, "branch is gone");
        assert_eq!(r.match_prefix(&[9]).prefix_len, 1, "other branch survives");
        assert_eq!(r.evict(1).unwrap(), vec![30]);
        assert_eq!(r.evictable(), 0);
    }

    #[test]
    fn radix_evict_may_overshoot_to_a_whole_leaf() {
        let mut r = RadixIndex::new(1);
        r.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        r.insert_prefix(&[1, 2, 4], &[20, 21, 22]);
        // Asking for 3 pages consumes whole leaves: 1 + 1 + 2 = 4.
        let freed = r.evict(3).unwrap();
        assert_eq!(freed.len(), 4);
        assert_eq!(r.evictable(), 0);
    }

    #[test]
    fn linear_state_pool_recycles_slots() {
        let mut p = LinearStatePool::new(2);
        assert_eq!(p.total(), 2);
        assert_eq!(p.available(), 2);
        let a = p.allocate().unwrap();
        let b = p.allocate().unwrap();
        assert_ne!(a, b);
        assert_eq!(p.available(), 0);
        assert_eq!(p.allocate(), None);
        p.release(a);
        assert_eq!(p.allocate(), Some(a), "freed slot is reused");
    }

    #[test]
    fn growing_a_cache_takes_pages_and_keeps_the_earlier_ones() {
        let mut pool = PagePool::new(1, 4);
        let mut c = SequenceCache::new(0);
        assert_eq!(c.cached(), 0);
        c.grow(3, &mut pool).unwrap();
        let first = c.filled().to_vec();
        c.grow(1, &mut pool).unwrap();
        assert_eq!(c.cached(), 4);
        assert_eq!(
            &c.filled()[..3],
            &first[..],
            "a decode step must not move the prefix"
        );
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn the_tail_is_where_the_next_step_writes() {
        let mut pool = PagePool::new(1, 8);
        let mut c = SequenceCache::new(1);
        c.grow(4, &mut pool).unwrap();
        c.grow(1, &mut pool).unwrap();
        assert_eq!(c.tail(1), &c.filled()[4..5]);
        assert_eq!(c.tail(4), &c.filled()[1..5]);
        assert_eq!(c.state_slot(), 1);
    }

    #[test]
    fn a_cache_cannot_grow_past_the_pool() {
        let mut pool = PagePool::new(1, 2);
        let mut c = SequenceCache::new(0);
        assert_eq!(
            c.grow(3, &mut pool),
            Err(KvcacheError::OutOfPages { want: 3, have: 2 })
        );
        assert_eq!(c.cached(), 0, "a refused grow must not reserve anything");
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn truncating_a_cache_returns_its_tail_to_the_pool() {
        let mut pool = PagePool::new(1, 4);
        let mut c = SequenceCache::new(0);
        c.grow(4, &mut pool).unwrap();
        let kept = c.filled()[..2].to_vec();
        c.truncate(2, &mut pool);
        assert_eq!(c.cached(), 2);
        assert_eq!(c.filled(), &kept[..]);
        assert_eq!(pool.available(), 2);
        c.release(&mut pool);
        assert_eq!(c.cached(), 0);
        assert_eq!(pool.available(), 4, "release gives every page back");
    }
}
