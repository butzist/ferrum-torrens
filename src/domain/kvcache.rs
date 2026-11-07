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
    /// The GDN-state snapshot this node donates, if any. Only `HybridRadixIndex`
    /// sets it; the plain `RadixIndex` leaves it `None`.
    mamba_value: Option<u32>,
    mamba_refcount: u32,
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
            mamba_value: None,
            mamba_refcount: 0,
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
            mamba_value: None,
            mamba_refcount: 0,
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
            mamba_value: None,
            mamba_refcount: 0,
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

/// Hybrid radix index: the prefix tree plus a second currency, an optional
/// GDN-state snapshot (a `LinearStatePool` slot id) donated at a node's END
/// boundary. Mirrors FreeToken's `HybridRadixCache` (decision 68).
///
/// A snapshot is point-attached, not a window: a strict prefix of a
/// snapshot-bearing node is not resumable, so a continuation can only resume the
/// GDN recurrence from a checkpointed boundary. `match_prefix` therefore
/// truncates the reusable length to the deepest node on the matched path that
/// still owns a LIVE snapshot, and `split_at` leaves the snapshot on the suffix
/// half. `evict_mamba` reclaims snapshot slots (tombstoning an internal node in
/// place, keeping its KV); `evict_full` reclaims KV leaves and cascades through
/// the KV-only tombstone leaves an unlink exposes. `inc_lock` pins a snapshot
/// AND the whole root..node KV path (`full_ref >= mamba_ref`).
pub struct HybridRadixIndex {
    page_size: usize,
    nodes: Vec<RadixNode>,
    root: NodeId,
    clock: u64,
    full_evictable: usize,
    full_protected: usize,
    mamba_evictable: usize,
    mamba_protected: usize,
}

/// The result of a hybrid prefix match: the reusable KV locations, truncated to
/// the deepest live snapshot, and the snapshot to restore (`None` = cold).
pub struct HybridMatch {
    pub node: NodeId,
    pub kv_indices: Vec<u32>,
    pub cached_len: usize,
    pub mamba_value: Option<u32>,
}

/// The result of a hybrid insert: how much of the key was already cached, and
/// whether the boundary already owns a snapshot (the caller must free its
/// donated slot; the tree keeps the existing one).
pub struct HybridInsert {
    pub prefix_len: usize,
    pub mamba_exist: bool,
}

/// What a hybrid eviction freed: KV locations and GDN-state slots.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct EvictResult {
    pub kv: Vec<u32>,
    pub mamba_slots: Vec<u32>,
}

impl HybridRadixIndex {
    pub fn new(page_size: usize) -> Self {
        debug_assert_eq!(page_size, 1, "only page_size == 1 is implemented");
        let nodes = vec![RadixNode {
            token_key: Vec::new(),
            pages: Vec::new(),
            parent: None,
            children: BTreeMap::new(),
            refcount: 1,
            timestamp: 0,
            mamba_value: None,
            mamba_refcount: 0,
        }];
        Self {
            page_size,
            nodes,
            root: 0,
            clock: 0,
            full_evictable: 0,
            full_protected: 0,
            mamba_evictable: 0,
            mamba_protected: 0,
        }
    }

    /// KV tokens reclaimable by `evict_full`.
    pub fn full_evictable(&self) -> usize {
        self.full_evictable
    }

    pub fn full_protected(&self) -> usize {
        self.full_protected
    }

    /// Live, unlocked snapshot slots reclaimable by `evict_mamba`.
    pub fn mamba_evictable(&self) -> usize {
        self.mamba_evictable
    }

    pub fn mamba_protected(&self) -> usize {
        self.mamba_protected
    }

    /// Every live snapshot in the tree, locked or not.
    pub fn live_snapshots(&self) -> usize {
        self.mamba_evictable + self.mamba_protected
    }

    /// Insert the committed KV prefix and donate `mamba_value` at the
    /// page-aligned end boundary. Returns how much was already cached and whether
    /// the boundary already owns a live snapshot, in which case the caller must
    /// free its donated slot (dedup).
    pub fn insert(&mut self, token_key: &[u32], pages: &[u32], mamba_value: u32) -> HybridInsert {
        debug_assert_eq!(token_key.len(), pages.len());
        let insert_len = token_key.len() / self.page_size * self.page_size;
        let (mut node, prefix_len) = self.tree_walk(&token_key[..insert_len]);
        if prefix_len != insert_len {
            let new = self.new_leaf(
                &token_key[prefix_len..insert_len],
                &pages[prefix_len..insert_len],
            );
            self.full_evictable += new_len(&self.nodes[new]);
            self.reparent(node, new);
            node = new;
        }
        if node == self.root || self.nodes[node].mamba_value.is_some() {
            return HybridInsert {
                prefix_len,
                mamba_exist: true,
            };
        }
        self.nodes[node].mamba_value = Some(mamba_value);
        if self.nodes[node].mamba_refcount == 0 {
            self.mamba_evictable += 1;
        }
        HybridInsert {
            prefix_len,
            mamba_exist: false,
        }
    }

    /// Match the token prefix and truncate the reusable length to the deepest
    /// node on the path that still owns a live snapshot.
    pub fn match_prefix(&mut self, token_key: &[u32]) -> HybridMatch {
        let (node, _) = self.tree_walk(token_key);
        let mut cur = node;
        let mut end_len = self.path_len(node);
        while cur != self.root {
            if self.nodes[cur].mamba_value.is_some() {
                return HybridMatch {
                    node: cur,
                    kv_indices: self.collect_kv(cur),
                    cached_len: end_len,
                    mamba_value: self.nodes[cur].mamba_value,
                };
            }
            end_len -= new_len(&self.nodes[cur]);
            cur = self.nodes[cur].parent.expect("non-root has a parent");
        }
        HybridMatch {
            node: self.root,
            kv_indices: Vec::new(),
            cached_len: 0,
            mamba_value: None,
        }
    }

    /// Protect a matched node's snapshot and its whole KV path up to the root.
    pub fn inc_lock(&mut self, node: NodeId) {
        if self.nodes[node].mamba_value.is_some() {
            if self.nodes[node].mamba_refcount == 0 {
                self.mamba_evictable -= 1;
                self.mamba_protected += 1;
            }
            self.nodes[node].mamba_refcount += 1;
        }
        let mut cur = node;
        while cur != self.root {
            let len = new_len(&self.nodes[cur]);
            if self.nodes[cur].refcount == 0 {
                self.full_evictable -= len;
                self.full_protected += len;
            }
            self.nodes[cur].refcount += 1;
            cur = self.nodes[cur].parent.expect("non-root has a parent");
        }
    }

    pub fn dec_lock(&mut self, node: NodeId) {
        if self.nodes[node].mamba_value.is_some() && self.nodes[node].mamba_refcount > 0 {
            self.nodes[node].mamba_refcount -= 1;
            if self.nodes[node].mamba_refcount == 0 {
                self.mamba_evictable += 1;
                self.mamba_protected -= 1;
            }
        }
        let mut cur = node;
        while cur != self.root {
            let len = new_len(&self.nodes[cur]);
            debug_assert!(self.nodes[cur].refcount > 0);
            self.nodes[cur].refcount -= 1;
            if self.nodes[cur].refcount == 0 {
                self.full_evictable += len;
                self.full_protected -= len;
            }
            cur = self.nodes[cur].parent.expect("non-root has a parent");
        }
    }

    /// Evict `num_tokens` of KV by LRU over unlocked leaves, freeing each
    /// evicted node's snapshot and cascading through the KV-only tombstone leaves
    /// the unlinks expose. Counts tokens, so it may overshoot to a whole leaf.
    pub fn evict_full(&mut self, num_tokens: usize) -> EvictResult {
        let mut out = EvictResult::default();
        let mut freed = 0usize;
        while freed < num_tokens {
            let Some(node) = self.oldest_unlocked_leaf() else {
                break;
            };
            let len = new_len(&self.nodes[node]);
            freed += len;
            out.kv.extend_from_slice(&self.nodes[node].pages);
            self.full_evictable -= len;
            self.free_node_mamba(node, &mut out.mamba_slots);
            let parent = self.unlink(node);
            let (_, casc) = self.cascade_tombstone_leaves(parent, &mut out.kv);
            freed += casc;
        }
        out
    }

    /// Evict `num` GDN-state snapshots by LRU over unlocked snapshot-bearing
    /// nodes, internal nodes included. An internal node is tombstoned in place
    /// (slot freed, KV kept); a leaf node frees both and unlinks, cascading
    /// through the KV-only tombstone leaves it exposes. Counts snapshots, not
    /// tokens.
    pub fn evict_mamba(&mut self, num: usize) -> EvictResult {
        let mut out = EvictResult::default();
        let mut freed = 0usize;
        while freed < num {
            let Some(node) = self.oldest_unlocked_snapshot() else {
                break;
            };
            let is_leaf = self.nodes[node].children.is_empty();
            if is_leaf && self.nodes[node].refcount == 0 {
                let len = new_len(&self.nodes[node]);
                out.kv.extend_from_slice(&self.nodes[node].pages);
                self.full_evictable -= len;
                self.free_node_mamba(node, &mut out.mamba_slots);
                freed += 1;
                let parent = self.unlink(node);
                self.cascade_tombstone_leaves(parent, &mut out.kv);
            } else {
                self.free_node_mamba(node, &mut out.mamba_slots);
                freed += 1;
            }
        }
        out
    }

    /// Structural invariant: every snapshot-bearing node holds a slot id and
    /// non-negative ref counts. KV/slot conservation is the caller's ledger.
    pub fn check_integrity(&self) {
        debug_assert!(
            self.nodes[self.root].mamba_value.is_none(),
            "the root holds no snapshot"
        );
        for node in &self.nodes {
            if node.mamba_value.is_none() {
                debug_assert_eq!(node.mamba_refcount, 0, "a free node has no mamba lock");
            }
        }
    }

    fn free_node_mamba(&mut self, node: NodeId, out: &mut Vec<u32>) {
        if let Some(slot) = self.nodes[node].mamba_value {
            out.push(slot);
            self.nodes[node].mamba_value = None;
            if self.nodes[node].mamba_refcount == 0 {
                self.mamba_evictable -= 1;
            }
        }
    }

    fn unlink(&mut self, node: NodeId) -> NodeId {
        let parent = self.nodes[node].parent.expect("leaf is never the root");
        let key = first_key(&self.nodes[node]);
        self.nodes[parent].children.remove(&key);
        parent
    }

    /// After a leaf is unlinked, eagerly reclaim the KV-only tombstone leaves it
    /// exposes upward, keeping the "a leaf always carries a live snapshot"
    /// invariant. Returns the highest surviving ancestor and the tokens freed.
    fn cascade_tombstone_leaves(
        &mut self,
        mut parent: NodeId,
        kv: &mut Vec<u32>,
    ) -> (NodeId, usize) {
        let mut freed = 0usize;
        while parent != self.root
            && self.nodes[parent].mamba_value.is_none()
            && self.nodes[parent].children.is_empty()
            && self.nodes[parent].refcount == 0
        {
            let len = new_len(&self.nodes[parent]);
            kv.extend_from_slice(&self.nodes[parent].pages);
            self.full_evictable -= len;
            freed += len;
            parent = self.unlink(parent);
        }
        (parent, freed)
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

    fn oldest_unlocked_snapshot(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.mamba_value.is_some() && n.mamba_refcount == 0)
            .min_by_key(|(_, n)| n.timestamp)
            .map(|(id, _)| id)
    }

    fn path_len(&self, node: NodeId) -> usize {
        let mut total = 0usize;
        let mut cur = node;
        while cur != self.root {
            total += new_len(&self.nodes[cur]);
            cur = self.nodes[cur].parent.expect("non-root has a parent");
        }
        total
    }

    fn collect_kv(&self, node: NodeId) -> Vec<u32> {
        let mut parts: Vec<&[u32]> = Vec::new();
        let mut cur = node;
        while cur != self.root {
            parts.push(&self.nodes[cur].pages);
            cur = self.nodes[cur].parent.expect("non-root has a parent");
        }
        parts.reverse();
        parts.into_iter().flatten().copied().collect()
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
            mamba_value: None,
            mamba_refcount: 0,
        });
        self.nodes.len() - 1
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

    fn match_len(&self, node: NodeId, token_key: &[u32]) -> usize {
        let key = &self.nodes[node].token_key;
        key.iter()
            .zip(token_key)
            .take_while(|(a, b)| a == b)
            .count()
    }

    /// Split `node` at `pos` (page boundary). The new prefix node gets no
    /// snapshot: a snapshot is attached to an end boundary, so only the suffix
    /// (whose boundary is unchanged) keeps it.
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
            mamba_value: None,
            mamba_refcount: 0,
        });
        let prefix = self.nodes.len() - 1;
        let suffix_key = first_key(&self.nodes[node]);
        self.nodes[prefix].children.insert(suffix_key, node);
        self.nodes[node].parent = Some(prefix);
        let prefix_key = first_key(&self.nodes[prefix]);
        self.nodes[parent].children.insert(prefix_key, prefix);
        prefix
    }
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
/// `tail()` to write, so this type stays testable without a GPU runtime.
pub struct SequenceCache {
    /// `pages[i]` is the pool location of the request's token `i`. `page_size`
    /// is 1, so this doubles as the token-location row the radix tree stores.
    pages: Vec<u32>,
    /// How many of `pages` have actually been written this request.
    len: usize,
    state_slot: u32,
    /// How many leading tokens are tree-owned (locked, reused from a hybrid
    /// prefix). `pages[..reused]` are the tree's canonical locations and must
    /// not be freed by this request; only `pages[reused..len]` are its own.
    reused: usize,
    /// The hybrid tree node this request holds locked, if any. Released on
    /// commit or `end`.
    lock: Option<usize>,
    /// The live GDN slot was donated to the tree on commit; `end` must not
    /// return it to the slot pool.
    donated: bool,
}

impl SequenceCache {
    pub fn new(state_slot: u32) -> Self {
        Self {
            pages: Vec::new(),
            len: 0,
            state_slot,
            reused: 0,
            lock: None,
            donated: false,
        }
    }

    /// Tokens whose KV and GDN state are already computed.
    pub fn cached(&self) -> usize {
        self.len
    }

    pub fn state_slot(&self) -> u32 {
        self.state_slot
    }

    /// Tokens of `filled()` reused from a hybrid prefix (tree-owned, locked).
    pub fn reused(&self) -> usize {
        self.reused
    }

    pub fn set_reused(&mut self, n: usize) {
        debug_assert!(n <= self.len);
        self.reused = n;
    }

    pub fn lock(&self) -> Option<usize> {
        self.lock
    }

    pub fn set_lock(&mut self, node: Option<usize>) {
        self.lock = node;
    }

    pub fn take_lock(&mut self) -> Option<usize> {
        self.lock.take()
    }

    pub fn donated(&self) -> bool {
        self.donated
    }

    pub fn set_donated(&mut self, donated: bool) {
        self.donated = donated;
    }

    /// The request's own locations, `filled()[reused..]`: what a commit may free
    /// or hand to the tree.
    pub fn own(&self) -> &[u32] {
        &self.pages[self.reused..self.len]
    }

    /// Point `pages[from..to]` at the tree's canonical locations. A commit that
    /// found `[reused, prefix_len)` already in the tree returns the request's own
    /// (now duplicate) locations to the pool; the request keeps reading this row
    /// while it decodes, so it must be re-pointed at the canonical ones first.
    pub fn repoint(&mut self, from: usize, to: usize, canonical: &[u32]) {
        debug_assert!(from <= to && to <= self.len && to <= canonical.len());
        self.pages[from..to].copy_from_slice(&canonical[from..to]);
    }

    /// Adopt `pages` as the leading, tree-owned (reused) locations: a matched
    /// prefix at admission.
    pub fn adopt_prefix(&mut self, pages: &[u32]) {
        debug_assert!(self.pages.is_empty() && self.len == 0);
        self.pages.extend_from_slice(pages);
        self.len = pages.len();
        self.reused = pages.len();
    }

    /// Drop every page reference (a finished commit has handed all of them to
    /// the tree or freed the duplicates), so `end` frees nothing.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.len = 0;
        self.reused = 0;
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

    /// Give `n` of this request's own pages back, keeping the earlier ones. Used
    /// when a prefix is rolled back to a shorter cached length. Never crosses the
    /// tree-owned `reused` boundary.
    pub fn truncate(&mut self, n: usize, pool: &mut PagePool) {
        debug_assert!(n <= self.len - self.reused);
        let keep = self.len - n;
        let tail = self.pages.split_off(keep);
        pool.free_pages(tail);
        self.len = keep;
    }

    /// Hand the request's own pages back; the tree-owned leading pages are not
    /// this request's to free. Used by `end` for a request that did not commit.
    pub fn release(&mut self, pool: &mut PagePool) {
        let own = self.pages.split_off(self.reused);
        pool.free_pages(own);
        self.len = self.reused;
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

    /// Give back `n` of the request's own tail pages. A speculative verify writes
    /// K/V for every row it proposes and keeps only the accepted prefix, so the
    /// rejected rows' pages must return to the pool before the next step reuses
    /// their positions. Never crosses the tree-owned `reused` boundary.
    fn rollback(&self, cache: &mut SequenceCache, n: usize) -> Result<(), KvcacheError> {
        let _ = (cache, n);
        Err(KvcacheError::MisalignedCache {
            why: "this cache cannot roll back a speculative verify".to_string(),
        })
    }

    /// Pages the pool can still hand out.
    fn available(&self) -> usize;

    /// Total pages, i.e. the token budget across every live request.
    fn num_pages(&self) -> usize;

    /// A hybrid (GDN) cache: try to reuse a cached prefix of `ids`, which is the
    /// prompt without its last token (the forward must still recompute that token
    /// for its logits). On a hit, restore the matched GDN snapshot into `cache`'s
    /// live slot (copy-on-write), point the cache's leading pages at the tree's
    /// canonical KV, lock the matched node, and return the reused length. Zero is
    /// a cold start and leaves `cache` untouched. A cache without GDN snapshots
    /// never reuses, which is the default.
    fn adopt_prefix(&self, cache: &mut SequenceCache, ids: &[u32]) -> usize {
        let _ = (cache, ids);
        0
    }

    /// A hybrid (GDN) cache: commit `ids` (the tokens whose KV and state `cache`
    /// holds) into the prefix tree, donating the request's GDN state. `finished`
    /// distinguishes the mid-request prompt-end commit (the request keeps
    /// decoding, so duplicate pages are re-pointed at the tree's canonical ones)
    /// from the finish commit (the request is done, duplicates are just freed).
    /// A cache without a prefix tree does nothing, which is the default.
    fn commit(&self, cache: &mut SequenceCache, ids: &[u32], finished: bool) {
        let _ = (cache, ids, finished);
    }
}

/// A speculative verify step's answer: the logits and final-norm hidden of every
/// row the caller asked for, in `answers` order.
///
/// The hidden rows are what an MTP draft head consumes (the next draft row pairs
/// a token embedding with the previous row's hidden), so they ride along instead
/// of costing a second pass.
pub struct VerifyOut {
    /// `answers.len() * params.vocab_size` f32.
    pub logits: Vec<f32>,
    /// `answers.len() * params.hidden_size` f32.
    pub hidden: Vec<f32>,
}

/// A draft pass's answer: the final-norm hidden of every row it ran and the
/// argmax token each row predicts.
pub struct Drafted {
    /// `rows * params.hidden_size` f32.
    pub hidden: Vec<f32>,
    /// One token per row: the draft's prediction of the token after it.
    pub tokens: Vec<u32>,
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

    /// How many routed-expert steps this forward has run through its MoE port.
    /// Zero means the experts were widened on the host, which is what the
    /// default says: a forward with no kernel port has nothing to count. Kept
    /// beside `cache()` because both answer "was the kernel in the loop", and
    /// both are what a gate needs to tell the two paths apart.
    fn expert_steps(&self) -> u64 {
        0
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

    /// A speculative verify step: run every row of `input` through the model,
    /// storing K/V for each, and answer each row in `input.answers` (in order).
    ///
    /// The linear-attention state is *not* advanced: the caller decides the
    /// accepted prefix and commits it with [`commit_verify`](Self::commit_verify),
    /// so a rejected row leaves no trace in the recurrence. `cache` must already
    /// be grown by `input`'s token count.
    fn verify_cached(
        &self,
        input: &ForwardInput,
        cache: &SequenceCache,
    ) -> Result<VerifyOut, KvcacheError> {
        let _ = (input, cache);
        Err(KvcacheError::MisalignedCache {
            why: "this model has no speculative verify path".to_string(),
        })
    }

    /// Advance the linear-attention state over the first `rows` rows of the last
    /// [`verify_cached`](Self::verify_cached), so the state matches the accepted
    /// prefix rather than the whole proposed batch.
    fn commit_verify(&self, cache: &SequenceCache, rows: usize) -> Result<(), KvcacheError> {
        let _ = (cache, rows);
        Err(KvcacheError::MisalignedCache {
            why: "this model has no speculative verify path".to_string(),
        })
    }

    /// Whether the checkpoint carries an MTP draft head and the engine has
    /// enabled it. `false` means speculation cannot start, not that it failed.
    fn draft_ready(&self) -> bool {
        false
    }

    /// Run the MTP draft over consecutive labels `from .. from + tokens.len()`.
    ///
    /// Row `i` embeds `tokens[i]` (the token at label `from + i + 1`) together
    /// with `hiddens[i]` (the target's final-norm hidden at label `from + i`) and
    /// answers the token after it. The draft's own K/V is written at the row's
    /// label, so the next call over a later label range attends to this one.
    fn draft_rows(
        &self,
        cache: &SequenceCache,
        tokens: &[u32],
        hiddens: &[f32],
        from: usize,
    ) -> Result<Drafted, KvcacheError> {
        let _ = (cache, tokens, hiddens, from);
        Err(KvcacheError::MisalignedCache {
            why: "this model has no draft head".to_string(),
        })
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

    #[test]
    fn adopting_a_prefix_makes_it_tree_owned_and_pool_free() {
        let mut pool = PagePool::new(1, 8);
        let mut c = SequenceCache::new(0);
        c.adopt_prefix(&[3, 4, 5]);
        assert_eq!(c.cached(), 3);
        assert_eq!(c.reused(), 3);
        assert_eq!(c.filled(), &[3, 4, 5]);
        assert!(
            c.own().is_empty(),
            "an adopted prefix is not the request's own"
        );
        assert_eq!(
            pool.available(),
            8,
            "adopted pages are not drawn from the pool"
        );
        c.grow(2, &mut pool).unwrap();
        assert_eq!(c.cached(), 5);
        assert_eq!(c.own().len(), 2, "own pages are the tail past the match");
        assert_eq!(c.own(), &c.filled()[3..], "own is filled past reused");
        assert_eq!(pool.available(), 6);

        c.release(&mut pool);
        assert_eq!(c.cached(), 3, "release keeps the tree-owned prefix");
        assert_eq!(
            pool.available(),
            8,
            "only the request's own pages come back"
        );
    }

    #[test]
    fn repointing_swaps_own_pages_for_the_canonical_ones() {
        let mut pool = PagePool::new(1, 8);
        let mut c = SequenceCache::new(0);
        c.adopt_prefix(&[3, 4]);
        c.grow(2, &mut pool).unwrap();
        let own = c.own().to_vec();
        c.repoint(2, 4, &[3, 4, 9, 9]);
        assert_eq!(c.filled(), &[3, 4, 9, 9]);
        assert_ne!(
            c.own(),
            &own[..],
            "the duplicate span now names canonical pages"
        );
    }

    #[test]
    fn clear_drops_every_reference_so_end_frees_nothing() {
        let mut pool = PagePool::new(1, 8);
        let mut c = SequenceCache::new(0);
        c.adopt_prefix(&[3, 4]);
        c.grow(2, &mut pool).unwrap();
        let free_before = pool.available();
        c.clear();
        assert_eq!(c.cached(), 0);
        assert_eq!(c.reused(), 0);
        assert_eq!(c.own().len(), 0);
        c.release(&mut pool);
        assert_eq!(
            pool.available(),
            free_before,
            "a committed cache owns no pages, so end frees nothing"
        );
    }

    #[test]
    fn the_lock_and_donation_flags_track_the_request() {
        let mut c = SequenceCache::new(7);
        assert_eq!(c.lock(), None);
        assert!(!c.donated());
        c.set_lock(Some(42));
        c.set_donated(true);
        assert_eq!(c.lock(), Some(42));
        assert!(c.donated());
        assert_eq!(c.take_lock(), Some(42));
        assert_eq!(c.lock(), None, "take_lock clears it");
    }

    /// Build a two-node tree: X=[1,2] with snapshot `mx`, its child Y=[3,4] with
    /// snapshot `my`. Returns (index, mx, my). Adapted from the reference
    /// `two_node_tree`, at page_size 1.
    fn hybrid_two_node_tree() -> (HybridRadixIndex, u32, u32) {
        let mut h = HybridRadixIndex::new(1);
        h.insert(&[1, 2], &[10, 11], 100);
        h.insert(&[1, 2, 3, 4], &[10, 11, 12, 13], 200);
        (h, 100, 200)
    }

    #[test]
    fn hybrid_insert_attaches_snapshot_and_match_restores_it() {
        let mut h = HybridRadixIndex::new(1);
        let ins = h.insert(&[1, 2], &[10, 11], 100);
        assert_eq!(ins.prefix_len, 0, "nothing cached yet");
        assert!(!ins.mamba_exist);
        h.check_integrity();

        let m = h.match_prefix(&[1, 2]);
        assert_eq!(m.cached_len, 2);
        assert_eq!(m.kv_indices, vec![10, 11]);
        assert_eq!(m.mamba_value, Some(100));
        assert_eq!(h.live_snapshots(), 1);
    }

    #[test]
    fn hybrid_deepest_snapshot_wins_and_unmatched_suffix_is_dropped() {
        let (mut h, mx, my) = hybrid_two_node_tree();
        assert_ne!(mx, my);

        let m = h.match_prefix(&[1, 2, 3, 4]);
        assert_eq!((m.cached_len, m.mamba_value), (4, Some(my)));
        // a query running past the end of the tree still resumes from the
        // deepest snapshot
        let m = h.match_prefix(&[1, 2, 3, 4, 5]);
        assert_eq!((m.cached_len, m.mamba_value), (4, Some(my)));
        h.check_integrity();
    }

    #[test]
    fn hybrid_split_leaves_the_snapshot_on_the_suffix_half() {
        let (mut h, mx, my) = hybrid_two_node_tree();

        // Diverges inside Y=[3,4] -> Y splits into [3] + [4]; the prefix half
        // owns no snapshot, so the match truncates back to X's snapshot.
        let m = h.match_prefix(&[1, 2, 3, 5]);
        assert_eq!((m.cached_len, m.mamba_value), (2, Some(mx)));
        h.check_integrity();

        // The suffix half kept my, so the deep hit survives.
        let m = h.match_prefix(&[1, 2, 3, 4]);
        assert_eq!((m.cached_len, m.mamba_value), (4, Some(my)));
        assert_eq!(h.live_snapshots(), 2);
        assert_eq!(h.full_evictable(), 4, "a split moves no KV");
    }

    #[test]
    fn hybrid_prefix_of_a_snapshot_node_is_not_reusable() {
        let mut h = HybridRadixIndex::new(1);
        h.insert(&[1, 2, 3, 4], &[10, 11, 12, 13], 100);
        h.check_integrity();

        // The snapshot is point-attached at the node's end boundary: matching an
        // interior boundary yields nothing at all, not a shortened hit.
        let m = h.match_prefix(&[1, 2]);
        assert_eq!((m.cached_len, m.mamba_value), (0, None));
        assert!(m.kv_indices.is_empty());

        let m = h.match_prefix(&[1, 2, 3, 4]);
        assert_eq!((m.cached_len, m.mamba_value), (4, Some(100)));
        assert_eq!(m.kv_indices, vec![10, 11, 12, 13]);
        assert_eq!(h.full_evictable(), 4);
    }

    #[test]
    fn hybrid_insert_dedups_and_the_caller_frees_the_donated_slot() {
        let mut h = HybridRadixIndex::new(1);
        h.insert(&[1, 2], &[10, 11], 100);
        let ins = h.insert(&[1, 2], &[20, 21], 200);
        assert_eq!(ins.prefix_len, 2);
        assert!(ins.mamba_exist, "the boundary already owns a snapshot");
        assert_eq!(h.live_snapshots(), 1, "still exactly one snapshot");

        let m = h.match_prefix(&[1, 2]);
        assert_eq!(m.mamba_value, Some(100), "the original snapshot is kept");
        assert_eq!(m.kv_indices, vec![10, 11], "the KV stays canonical");
    }

    #[test]
    fn hybrid_insert_stores_whole_boundaries_only() {
        let mut h = HybridRadixIndex::new(1);
        let ins = h.insert(&[], &[], 100);
        assert_eq!((ins.prefix_len, ins.mamba_exist), (0, true));
        assert_eq!(h.full_evictable(), 0);
        assert_eq!(h.live_snapshots(), 0, "the root cannot hold a snapshot");

        let ins = h.insert(&[1, 2], &[10, 11], 100);
        assert_eq!((ins.prefix_len, ins.mamba_exist), (0, false));
        assert_eq!(h.full_evictable(), 2);
        let m = h.match_prefix(&[1, 2]);
        assert_eq!((m.cached_len, m.mamba_value), (2, Some(100)));
    }

    #[test]
    fn hybrid_evict_mamba_tombstones_an_internal_node_and_keeps_its_kv() {
        let (mut h, mx, my) = hybrid_two_node_tree();

        let e = h.evict_mamba(1);
        assert_eq!(e.mamba_slots, vec![mx]);
        assert!(e.kv.is_empty(), "no KV was reclaimed");
        assert_eq!(h.full_evictable(), 4);
        assert_eq!(h.live_snapshots(), 1);

        let m = h.match_prefix(&[1, 2]);
        assert_eq!(
            (m.cached_len, m.mamba_value),
            (0, None),
            "tombstoned boundary"
        );
        let m = h.match_prefix(&[1, 2, 3, 4]);
        assert_eq!((m.cached_len, m.mamba_value), (4, Some(my)));
    }

    #[test]
    fn hybrid_insert_refills_a_tombstoned_node() {
        let (mut h, _, _) = hybrid_two_node_tree();
        h.evict_mamba(1);
        h.check_integrity();

        let ins = h.insert(&[1, 2], &[20, 21], 300);
        assert_eq!((ins.prefix_len, ins.mamba_exist), (2, false), "attaches");
        h.check_integrity();

        let m = h.match_prefix(&[1, 2]);
        assert_eq!((m.cached_len, m.mamba_value), (2, Some(300)));
        assert_eq!(m.kv_indices, vec![10, 11], "KV stays canonical");
        assert_eq!(h.live_snapshots(), 2);
    }

    #[test]
    fn hybrid_evict_mamba_on_a_leaf_frees_kv_and_cascades_through_tombstones() {
        let (mut h, _, _) = hybrid_two_node_tree();
        h.evict_mamba(1); // X -> KV-only tombstone
        let e = h.evict_mamba(1); // Y is now the only snapshot node, and a leaf
        assert_eq!(e.mamba_slots, vec![200]);
        assert_eq!(e.kv, vec![12, 13, 10, 11], "Y then the exposed X");

        assert_eq!(h.full_evictable(), 0);
        assert_eq!(h.live_snapshots(), 0);
        assert_eq!(h.match_prefix(&[1, 2, 3, 4]).cached_len, 0);
    }

    #[test]
    fn hybrid_evict_mamba_counts_snapshots_not_tokens() {
        let mut h = HybridRadixIndex::new(1);
        h.insert(&[1], &[10], 100);
        h.insert(&[1, 2], &[10, 11], 200);
        h.insert(&[1, 2, 3], &[10, 11, 12], 300);
        h.check_integrity();
        assert_eq!(h.live_snapshots(), 3);
        assert_eq!(h.full_evictable(), 3);

        let e = h.evict_mamba(2);
        assert_eq!(e.mamba_slots.len(), 2, "exactly two snapshots");
        assert!(e.kv.is_empty(), "no KV touched");
        assert_eq!(h.full_evictable(), 3);
        assert_eq!(h.live_snapshots(), 1);
    }

    #[test]
    fn hybrid_snapshot_slots_are_conserved_across_eviction_waves() {
        let n: u32 = 8;
        let mut h = HybridRadixIndex::new(1);
        let mut handed: Vec<u32> = Vec::new();
        for k in 1..=n {
            let key: Vec<u32> = (1..=k).collect();
            let pages: Vec<u32> = (0..k).map(|i| 1000 + i).collect();
            let slot = 100 + k;
            h.insert(&key, &pages, slot);
            handed.push(slot);
        }
        h.check_integrity();
        assert_eq!(h.live_snapshots() as u32, n);
        assert_eq!(h.full_evictable() as u32, n);

        let mut freed: Vec<u32> = Vec::new();
        for _ in 0..16 {
            if h.live_snapshots() == 0 {
                break;
            }
            freed.extend(h.evict_mamba(3).mamba_slots);
        }
        assert_eq!(h.live_snapshots(), 0, "evict_mamba converges");
        freed.sort_unstable();
        handed.sort_unstable();
        assert_eq!(freed, handed, "every donated slot comes back exactly once");
        assert_eq!(h.full_evictable(), 0);
        assert_eq!(h.match_prefix(&(1..=n).collect::<Vec<_>>()).cached_len, 0);
    }

    #[test]
    fn hybrid_evict_full_takes_the_leaf_snapshot_and_leaves_the_ancestor_usable() {
        let (mut h, mx, my) = hybrid_two_node_tree();

        let e = h.evict_full(2); // the only unlocked leaf is Y
        assert_eq!(e.mamba_slots, vec![my]);
        assert!(e.kv.contains(&12) && e.kv.contains(&13));
        assert!(!e.mamba_slots.contains(&mx), "X was not reclaimed");
        assert_eq!(h.full_evictable(), 2);
        assert_eq!(h.live_snapshots(), 1);
        h.check_integrity();

        let m = h.match_prefix(&[1, 2, 3, 4]);
        assert_eq!((m.cached_len, m.mamba_value), (2, Some(mx)));
    }

    #[test]
    fn hybrid_evict_full_cascades_through_an_exposed_tombstone_leaf() {
        let (mut h, _, _) = hybrid_two_node_tree();
        h.evict_mamba(1); // X -> tombstone
        h.evict_full(2); // evicting Y exposes X and reclaims it too

        assert_eq!(h.full_evictable(), 0);
        assert_eq!(h.live_snapshots(), 0);
        assert_eq!(h.match_prefix(&[1, 2, 3, 4]).cached_len, 0);
    }

    #[test]
    fn hybrid_lock_pins_the_snapshot_and_the_whole_kv_path() {
        let (mut h, mx, _my) = hybrid_two_node_tree();
        let held = h.match_prefix(&[1, 2, 3, 4]).node;
        h.inc_lock(held);
        h.check_integrity();

        // Y is locked, X is internal -> nothing is evictable.
        let e = h.evict_full(4);
        assert_eq!(e, EvictResult::default());
        assert_eq!(h.full_evictable(), 0, "all 4 pages are protected now");

        // X's snapshot is unprotected -> tombstoned in place.
        assert_eq!(h.evict_mamba(1).mamba_slots, vec![mx]);
        // Y's snapshot is pinned by the lock.
        assert!(h.evict_mamba(1).mamba_slots.is_empty());
        h.check_integrity();

        h.dec_lock(held);
        assert_eq!(h.full_evictable(), 4);
        let e = h.evict_full(2); // Y is evictable again; X cascades behind it
        assert!(!e.mamba_slots.is_empty());
    }
}
