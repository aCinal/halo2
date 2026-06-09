//! Hash-map and ordered-map types used internally.
//!
//! With the `std` feature enabled, [`HashMap`] and [`HashSet`] are thin
//! re-exports of [`std::collections`].  Without `std` they are backed by
//! `hashbrown` with a deterministic FNV-1a hasher (no OS randomness required).
//!
//! [`VecMap`] is an insertion-order-preserving map that replaces
//! `indexmap::IndexMap` and is available in both configurations.

// --------------------------------------------------------------------------
// HashMap / HashSet
// --------------------------------------------------------------------------

#[cfg(feature = "std")]
pub(crate) use std::collections::{HashMap, HashSet};

#[cfg(not(feature = "std"))]
pub(crate) use self::nostd::{HashMap, HashSet};

#[cfg(not(feature = "std"))]
mod nostd {
    /// FNV-1a hasher — deterministic, no OS randomness required.
    pub(crate) struct FnvHasher(u64);

    impl Default for FnvHasher {
        fn default() -> Self {
            FnvHasher(0xcbf29ce484222325)
        }
    }

    impl core::hash::Hasher for FnvHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            for &byte in bytes {
                self.0 ^= byte as u64;
                self.0 = self.0.wrapping_mul(0x100000001b3);
            }
        }
    }

    type FnvBuildHasher = core::hash::BuildHasherDefault<FnvHasher>;

    pub(crate) type HashMap<K, V> = hashbrown::HashMap<K, V, FnvBuildHasher>;
    pub(crate) type HashSet<T> = hashbrown::HashSet<T, FnvBuildHasher>;
}

// --------------------------------------------------------------------------
// VecMap — insertion-order-preserving map (replaces indexmap::IndexMap)
// Available in both std and no_std builds.
// --------------------------------------------------------------------------

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Insertion-order-preserving map backed by `Vec`.
pub(crate) struct VecMap<K, V>(Vec<(K, V)>);

impl<K, V> VecMap<K, V> {
    pub(crate) fn new() -> Self {
        VecMap(Vec::new())
    }

    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.0.iter_mut().map(|(k, v)| (&*k, v))
    }

    pub(crate) fn get(&self, key: &K) -> Option<&V> where K: Eq {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub(crate) fn get_mut(&mut self, key: &K) -> Option<&mut V> where K: Eq {
        self.0.iter_mut().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) where K: Eq {
        for (k, v) in &mut self.0 {
            if *k == key {
                *v = value;
                return;
            }
        }
        self.0.push((key, value));
    }

    pub(crate) fn entry(&mut self, key: K) -> VecMapEntry<'_, K, V> where K: Eq {
        if let Some(pos) = self.0.iter().position(|(k, _)| *k == key) {
            VecMapEntry::Occupied(OccupiedEntry { map: self, pos })
        } else {
            VecMapEntry::Vacant(VacantEntry { map: self, key })
        }
    }
}

impl<K, V> Default for VecMap<K, V> {
    fn default() -> Self { VecMap(Vec::new()) }
}

impl<K, V> IntoIterator for VecMap<K, V> {
    type Item = (K, V);
    type IntoIter = <Vec<(K, V)> as IntoIterator>::IntoIter;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub(crate) enum VecMapEntry<'a, K, V> {
    Occupied(OccupiedEntry<'a, K, V>),
    Vacant(VacantEntry<'a, K, V>),
}

impl<'a, K, V> VecMapEntry<'a, K, V> {
    pub(crate) fn or_insert_with(self, f: impl FnOnce() -> V) -> &'a mut V {
        match self {
            VecMapEntry::Occupied(e) => &mut e.map.0[e.pos].1,
            VecMapEntry::Vacant(e) => {
                let value = f();
                e.map.0.push((e.key, value));
                let last = e.map.0.len() - 1;
                &mut e.map.0[last].1
            }
        }
    }
}

pub(crate) struct OccupiedEntry<'a, K, V> {
    map: &'a mut VecMap<K, V>,
    pos: usize,
}

pub(crate) struct VacantEntry<'a, K, V> {
    map: &'a mut VecMap<K, V>,
    key: K,
}
