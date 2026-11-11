//! `FERRUM_DUP`: dev-only duplicate-launch hook for captured-step attribution.
//!
//! A captured decode step has no per-stage time. CUDA events recorded inside a
//! stream capture read back as capture-time host deltas, and neither `nsys` nor
//! `ncu` works in this build (OPTIMIZATION.md R19), so the only attribution left
//! is marginal cost: relaunch one stage a second time with identical inputs and
//! output buffers and read the step-time delta. That is how R18 priced the
//! shared expert.
//!
//! `FERRUM_DUP=<group>[,<group>...]` is parsed once at construction into `Dup`,
//! so an unset variable is a constant `false` branch in each stage and costs no
//! `std::env::var` per call or per layer. A duplicated launch writes the same
//! bytes into the same buffers, so the numerics are unchanged; stateful stages
//! (`gdn_conv`, `gdn_recur`) belong to no group and cannot be selected.

/// One stage group a duplicate can be asked for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Group {
    /// `res_norm`, `post_norm`, `gdn_tail_norm`.
    Norms,
    /// The attention projections (`attn_proj`).
    AttnProj,
    /// The attention kernel itself (`attn_sdpa`); the one-token KV store is
    /// not part of the duplicate.
    AttnSdpa,
    /// The attention output projection (`attn_out`).
    AttnOut,
    /// `gdn_qkvz`, `gdn_ba`, `gdn_tail_gate`, `gdn_tail_proj`.
    GdnProj,
    /// The LM head GEMV.
    Head,
    /// The router's gate matmul + top-k.
    Router,
    /// The routed experts' grouped GEMV; the miss copy is never duplicated.
    Routed,
}

impl Group {
    /// Every group, in `name` order.
    pub const ALL: [Group; 8] = [
        Group::Norms,
        Group::AttnProj,
        Group::AttnSdpa,
        Group::AttnOut,
        Group::GdnProj,
        Group::Head,
        Group::Router,
        Group::Routed,
    ];

    /// The `FERRUM_DUP` spelling of this group.
    pub fn name(self) -> &'static str {
        match self {
            Group::Norms => "norms",
            Group::AttnProj => "attn_proj",
            Group::AttnSdpa => "attn_sdpa",
            Group::AttnOut => "attn_out",
            Group::GdnProj => "gdn_proj",
            Group::Head => "head",
            Group::Router => "router",
            Group::Routed => "routed",
        }
    }

    /// The group `name` spells, or `None` when it spells none of them.
    pub fn parse(name: &str) -> Option<Self> {
        Group::ALL.into_iter().find(|g| g.name() == name)
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// The set of groups `FERRUM_DUP` asked to duplicate.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Dup(u8);

impl Dup {
    /// The empty set: nothing is duplicated, which is the committed path.
    pub const NONE: Dup = Dup(0);

    /// Read `FERRUM_DUP` once. Call this from a constructor, never from a
    /// per-layer path.
    pub fn from_env() -> Self {
        Self::parse(&std::env::var("FERRUM_DUP").unwrap_or_default())
    }

    /// Parse a comma-separated group list. Empty items are skipped and an
    /// unknown name is reported and ignored, so a typo cannot silently turn
    /// into a measurement of nothing.
    pub fn parse(spec: &str) -> Self {
        let mut bits = 0u8;
        for item in spec.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            match Group::parse(item) {
                Some(g) => bits |= g.bit(),
                None => eprintln!(
                    "[dup] FERRUM_DUP: unknown group {item:?}, known: {}",
                    Group::ALL.map(Group::name).join(",")
                ),
            }
        }
        Self(bits)
    }

    /// Whether `group` is in the set.
    pub fn has(self, group: Group) -> bool {
        self.0 & group.bit() != 0
    }

    /// Whether nothing is in the set.
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The groups in the set, in `Group::ALL` order.
    pub fn groups(self) -> impl Iterator<Item = Group> {
        Group::ALL.into_iter().filter(move |g| self.has(*g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unset_or_empty_spec_duplicates_nothing() {
        assert!(Dup::parse("").is_empty());
        assert!(Dup::parse("   ").is_empty());
        assert!(Dup::parse(",").is_empty());
        assert!(Dup::NONE.is_empty());
        assert_eq!(Dup::default(), Dup::NONE);
        assert!(Group::ALL.into_iter().all(|g| !Dup::NONE.has(g)));
    }

    #[test]
    fn every_name_parses_to_its_group_and_round_trips() {
        for group in Group::ALL {
            let dup = Dup::parse(group.name());
            assert!(dup.has(group), "{} did not select itself", group.name());
            assert_eq!(Group::parse(group.name()), Some(group));
            assert_eq!(dup.groups().collect::<Vec<_>>(), vec![group]);
        }
    }

    #[test]
    fn a_list_selects_exactly_the_named_groups() {
        let dup = Dup::parse("norms, routed");
        assert!(dup.has(Group::Norms));
        assert!(dup.has(Group::Routed));
        assert!(!dup.has(Group::Head));
        assert!(!dup.has(Group::AttnProj));
        assert!(!dup.is_empty());
        assert_eq!(
            dup.groups().collect::<Vec<_>>(),
            vec![Group::Norms, Group::Routed]
        );
    }

    #[test]
    fn the_whole_list_selects_every_group() {
        let spec = Group::ALL.map(Group::name).join(",");
        let dup = Dup::parse(&spec);
        assert!(Group::ALL.into_iter().all(|g| dup.has(g)));
        assert_eq!(dup.groups().collect::<Vec<_>>(), Group::ALL.to_vec());
    }

    #[test]
    fn an_unknown_name_is_ignored_and_leaves_the_rest_intact() {
        let dup = Dup::parse("head,not_a_group,router");
        assert!(dup.has(Group::Head));
        assert!(dup.has(Group::Router));
        assert_eq!(dup.groups().count(), 2);
    }
}
