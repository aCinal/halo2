//! An interface for dealing with the kinds of parallel computations involved in
//! `halo2`. It's currently just a (very!) thin wrapper around `rayon` but may
//! be extended in the future to allow for various parallelism strategies.

#[cfg(all(
    feature = "multicore",
    target_arch = "wasm32",
    not(target_feature = "atomics")
))]
compile_error!(
    "The multicore feature flag is not supported on wasm32 architectures without atomics"
);

// --- Multicore (rayon-backed) path ---------------------------------------

// `maybe_rayon` itself references `std` unconditionally (e.g. `std::marker::PhantomData`,
// `std::slice::ChunksExact`), so it cannot be imported at all in a no_std build.  The
// `multicore` feature already implies `std` (see `Cargo.toml`), so gating every
// `maybe_rayon` import on `#[cfg(feature = "multicore")]` is sufficient.
//
// `IndexedParallelIterator` and `IntoParallelIterator` are only used by the batch
// verifier (`plonk/verifier/batch.rs`), so they are further gated on `feature = "batch"`.

#[cfg(feature = "multicore")]
pub use maybe_rayon::{
    current_num_threads,
    iter::ParallelIterator,
    join, scope,
};

#[cfg(all(feature = "multicore", feature = "batch"))]
pub use maybe_rayon::iter::{IndexedParallelIterator, IntoParallelIterator};

// --- Sequential (no_std-compatible) path ---------------------------------
//
// Each item below is a drop-in replacement for the rayon symbol of the same name.
// The strategy is uniform: mirror rayon's trait/function signatures exactly so
// that call sites compile unchanged, then implement every method as a plain
// sequential operation.

/// Returns 1, the single-thread equivalent of `rayon::current_num_threads`.
#[cfg(not(feature = "multicore"))]
pub fn current_num_threads() -> usize {
    1
}

/// Sequential stand-in for `rayon::iter::IntoParallelIterator`.
///
/// Blanket-implemented for every `IntoIterator`, so any collection that
/// supports `.into_iter()` also supports `.into_par_iter()` — the latter just
/// returns the same sequential iterator.
///
/// Only needed when the `batch` feature is enabled, since no non-batch code
/// calls `.into_par_iter()`.
#[cfg(all(not(feature = "multicore"), feature = "batch"))]
pub trait IntoParallelIterator {
    type Item;
    type Iter: Iterator<Item = Self::Item>;
    fn into_par_iter(self) -> Self::Iter;
}

#[cfg(all(not(feature = "multicore"), feature = "batch"))]
impl<I: IntoIterator> IntoParallelIterator for I {
    type Item = I::Item;
    type Iter = I::IntoIter;
    fn into_par_iter(self) -> Self::Iter {
        self.into_iter()
    }
}

/// Sequential stand-in for `rayon::iter::ParallelIterator`.
///
/// Declared as a supertrait of `Iterator` with no additional methods, and
/// blanket-implemented for all `Iterator`s.  This means every adapter that
/// call sites chain on a `ParallelIterator` (`.map`, `.enumerate`, `.rev`, …)
/// simply resolves to the corresponding `Iterator` method — no `#[cfg]`
/// annotations needed at the call site.
#[cfg(not(feature = "multicore"))]
pub trait ParallelIterator: Iterator {}

#[cfg(not(feature = "multicore"))]
impl<I: Iterator> ParallelIterator for I {}

/// Sequential stand-in for `rayon::iter::IndexedParallelIterator`.
///
/// Like `ParallelIterator`, this is a marker supertrait over `Iterator` that
/// exists solely so trait bounds using `IndexedParallelIterator` compile in
/// no_std mode.  Only needed with the `batch` feature.
#[cfg(all(not(feature = "multicore"), feature = "batch"))]
pub trait IndexedParallelIterator: core::iter::Iterator {}

/// Sequential stand-in for `rayon::join`.
///
/// Rayon's `join` runs two closures potentially in parallel on the thread pool.
/// The sequential version simply runs them one after another on the same thread.
#[cfg(not(feature = "multicore"))]
pub fn join<A, B, RA, RB>(oper_a: A, oper_b: B) -> (RA, RB)
where
    A: FnOnce() -> RA,
    B: FnOnce() -> RB,
{
    (oper_a(), oper_b())
}

/// Sequential stand-in for `rayon::Scope`.
///
/// Rayon's `Scope` keeps track of in-flight tasks and ensures they all finish
/// before `scope()` returns.  The sequential version needs no bookkeeping: every
/// `spawn` runs immediately inline, so there is nothing to wait for.
///
/// The `PhantomData<Cell<&'scope ()>>` makes `Scope` *invariant* over `'scope`,
/// matching rayon's variance.  This is required for the `BODY: 'scope` bound on
/// `spawn` to correctly prevent closures from capturing short-lived references.
#[cfg(not(feature = "multicore"))]
pub struct Scope<'scope> {
    _marker: core::marker::PhantomData<core::cell::Cell<&'scope ()>>,
}

/// Sequential stand-in for `rayon::scope`.
///
/// Creates a dummy `Scope` and calls `op` with it.  Because `spawn` executes
/// closures immediately, all work is complete by the time `op` returns.
#[cfg(not(feature = "multicore"))]
pub fn scope<'scope, OP, R>(op: OP) -> R
where
    OP: FnOnce(&Scope<'scope>) -> R,
{
    op(&Scope { _marker: core::marker::PhantomData })
}

#[cfg(not(feature = "multicore"))]
impl<'scope> Scope<'scope> {
    /// Runs `body` immediately on the current thread rather than spawning it.
    ///
    /// Rayon would hand `body` to a worker thread; here it simply executes inline.
    /// The signature — including `Send` and the nested `&Scope` argument — is kept
    /// identical to rayon's so that closures written for rayon compile unchanged.
    pub fn spawn<BODY>(&self, body: BODY)
    where
        BODY: FnOnce(&Scope<'scope>) + Send,
    {
        body(self)
    }
}

/// Sequential drop-in for `maybe_rayon::prelude` — provides `par_iter_mut()` etc.
/// on slices/vecs so that call sites need no `#[cfg]` annotations.
///
/// Rayon's prelude injects `.par_iter()` / `.par_iter_mut()` onto common
/// collection types via extension traits.  The stubs below mirror those traits
/// and delegate straight to `.iter()` / `.iter_mut()`, so the same method-call
/// syntax works in both builds.
#[cfg(not(feature = "multicore"))]
pub mod prelude {
    pub trait IntoParallelRefMutIterator<'data> {
        type Iter: Iterator<Item = &'data mut Self::Item>;
        type Item: 'data + ?Sized;
        fn par_iter_mut(&'data mut self) -> Self::Iter;
    }

    impl<'data, T: 'data + Send> IntoParallelRefMutIterator<'data> for [T] {
        type Iter = core::slice::IterMut<'data, T>;
        type Item = T;
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            self.iter_mut()
        }
    }

    impl<'data, T: 'data + Send> IntoParallelRefMutIterator<'data> for alloc::vec::Vec<T> {
        type Iter = core::slice::IterMut<'data, T>;
        type Item = T;
        fn par_iter_mut(&'data mut self) -> Self::Iter {
            self.iter_mut()
        }
    }

    pub trait IntoParallelRefIterator<'data> {
        type Iter: Iterator<Item = &'data Self::Item>;
        type Item: 'data + ?Sized;
        fn par_iter(&'data self) -> Self::Iter;
    }

    impl<'data, T: 'data + Sync> IntoParallelRefIterator<'data> for [T] {
        type Iter = core::slice::Iter<'data, T>;
        type Item = T;
        fn par_iter(&'data self) -> Self::Iter {
            self.iter()
        }
    }

    impl<'data, T: 'data + Sync> IntoParallelRefIterator<'data> for alloc::vec::Vec<T> {
        type Iter = core::slice::Iter<'data, T>;
        type Item = T;
        fn par_iter(&'data self) -> Self::Iter {
            self.iter()
        }
    }
}

// --- Shared trait implementations ----------------------------------------

#[cfg(feature = "batch")]
pub trait TryFoldAndReduce<T, E> {
    /// Implements `iter.try_fold().try_reduce()` for `rayon::iter::ParallelIterator`,
    /// falling back on `Iterator::try_fold` when the `multicore` feature flag is
    /// disabled.
    /// The `try_fold_and_reduce` function can only be called by a iter with
    /// `Result<T, E>` item type because the `fold_op` must meet the trait
    /// bounds of both `try_fold` and `try_reduce` from rayon.
    fn try_fold_and_reduce(
        self,
        identity: impl Fn() -> T + Send + Sync,
        fold_op: impl Fn(T, Result<T, E>) -> Result<T, E> + Send + Sync,
    ) -> Result<T, E>;
}

#[cfg(all(feature = "multicore", feature = "batch"))]
impl<T, E, I> TryFoldAndReduce<T, E> for I
where
    T: Send + Sync,
    E: Send + Sync,
    I: maybe_rayon::iter::ParallelIterator<Item = Result<T, E>>,
{
    fn try_fold_and_reduce(
        self,
        identity: impl Fn() -> T + Send + Sync,
        fold_op: impl Fn(T, Result<T, E>) -> Result<T, E> + Send + Sync,
    ) -> Result<T, E> {
        self.try_fold(&identity, &fold_op)
            .try_reduce(&identity, |a, b| fold_op(a, Ok(b)))
    }
}

#[cfg(all(not(feature = "multicore"), feature = "batch"))]
impl<T, E, I> TryFoldAndReduce<T, E> for I
where
    I: core::iter::Iterator<Item = Result<T, E>>,
{
    fn try_fold_and_reduce(
        mut self,
        identity: impl Fn() -> T + Send + Sync,
        fold_op: impl Fn(T, Result<T, E>) -> Result<T, E> + Send + Sync,
    ) -> Result<T, E> {
        self.try_fold(identity(), fold_op)
    }
}

pub(crate) trait TheBestReduce {
    type Item;

    /// Combines the best of `std::iter` and `rayon` reductions.
    ///
    /// With `multicore`: delegates to `rayon::ParallelIterator::reduce`, which takes
    /// an identity *function* (called per thread) and always returns `T` (wrapped
    /// in `Some` here to produce a uniform `Option<T>` return type).
    ///
    /// Without `multicore`: delegates to `Iterator::reduce`, which takes only the
    /// combining function and returns `None` for an empty iterator.  The identity
    /// argument is ignored because there is only ever one "thread" of work.
    fn the_best_reduce(
        self,
        identity: impl Fn() -> Self::Item + Send + Sync,
        op: impl Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    ) -> Option<Self::Item>;
}

#[cfg(feature = "multicore")]
impl<I> TheBestReduce for I
where
    I: maybe_rayon::iter::ParallelIterator,
{
    type Item = <Self as maybe_rayon::iter::ParallelIterator>::Item;

    fn the_best_reduce(
        self,
        identity: impl Fn() -> Self::Item + Send + Sync,
        op: impl Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    ) -> Option<Self::Item> {
        Some(self.reduce(identity, op))
    }
}

#[cfg(not(feature = "multicore"))]
impl<I> TheBestReduce for I
where
    I: core::iter::Iterator,
{
    type Item = <Self as core::iter::Iterator>::Item;

    fn the_best_reduce(
        self,
        _: impl Fn() -> Self::Item + Send + Sync,
        f: impl Fn(Self::Item, Self::Item) -> Self::Item + Send + Sync,
    ) -> Option<Self::Item> {
        self.reduce(f)
    }
}
