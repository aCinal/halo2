//! Hash-map types used internally.
//!
//! With the `std` feature enabled these are thin re-exports of
//! [`std::collections`].  Without `std` they are backed by `hashbrown` with
//! a deterministic FNV-1a hasher (no OS randomness required).

#[cfg(feature = "std")]
pub(crate) use std::collections::HashSet;

#[cfg(not(feature = "std"))]
pub(crate) use self::nostd::HashSet;

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

    pub(crate) type HashSet<T> = hashbrown::HashSet<T, FnvBuildHasher>;
}
