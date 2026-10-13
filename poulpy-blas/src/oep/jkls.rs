use poulpy_ckks::{CKKSCtBounds, CKKSResult as Result, SetCKKSInfos};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Extension point of the JKLS kernel, one output block of the prepared product: the reference
/// [`crate::reference::BlasMatMulReference::blas_matmul_prepared_reference`] runs one [`Self::blas_jkls_block_impl`]
/// per block of `res`, so both `blas_matmul` and `blas_matmul_prepared` reach it.
///
/// # Safety
///
/// [`Self::blas_jkls_block_impl`] reproduces [`crate::reference::BlasMatMulReference::blas_jkls_block_reference`] bit
/// for bit, digits and CKKS metadata: `res = sum_t lhs[t] rhs[t]` as poulpy-ckks's `ckks_dot_product_ct` computes it
/// for pairs that share one layout per side.
///
/// **Fixed:**
/// - each term is one core tensor product of `(lhs[t], rhs[t])` at the offset poulpy-ckks's ct x ct rule gives for
///   `res`, rounded on its own at the tensor width `max(lhs.k, rhs.k)`; summing unrounded products, as
///   `cnv_apply_dft_sum` would, changes bits;
/// - the terms are summed exactly, the sum normalized at that width before the relinearization reads it, `res` stamped
///   with the rule's budget and scale, the smallest label and the joined slot kind, then relinearized once into `res`
///   with the tensor key at that width; `res` comes back canonical, without a noise tag.
///
/// **Free (bit-identical):** the order, grouping and split of the terms over workers, materialized or streamed;
/// normalizing partial sums, or leaving them unnormalized until the DFT that reads the sum: normalization preserves the
/// value, so its placement never changes the bits; caching the operands or their convolution forms; normalizing an
/// operand read through a DFT. Anything else changes bits.
///
/// It is called with `lhs.len() == rhs.len() >= 1` pairs, every `lhs` of one layout and every `rhs` of another, on
/// inputs the prepared product's validator accepted, at a radix of 62 at most, which checked the tensor key at that
/// width and of their rank. Within the scratch [`Self::blas_jkls_block_tmp_bytes_impl`] reports for `res`, `lhs[0]`,
/// `rhs[0]`, the pair count and the key, it returns `Ok`, or poulpy-ckks's budget error before any write to `res`; with
/// less scratch it panics before any write, as the reference does. Any other `Err` is a backend fault (a validator gap
/// or a device failure), and `res` is then unspecified. The reference prepared product calls it for the blocks of `res`
/// concurrently on a parallel executor, each call on its own block and scratch slice. The pairs and the key are `Sync`,
/// so an implementation may split one block's terms over workers.
pub unsafe trait BlasJklsImpl: Backend {
    /// Scratch bytes of [`Self::blas_jkls_block_impl`] into a block of layout `res` from `terms` pairs of layouts `lhs`
    /// and `rhs`, with a tensor key of layout `tsk`.
    fn blas_jkls_block_tmp_bytes_impl<R, L, Rh, T>(
        module: &Module<Self>,
        res: &R,
        lhs: &L,
        rhs: &Rh,
        terms: usize,
        tsk: &T,
    ) -> usize
    where
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos;

    /// The JKLS kernel: `res = sum_t lhs[t] rhs[t]`, relinearized once. See the trait's `# Safety`.
    fn blas_jkls_block_impl<R, L, Rh, T>(
        module: &Module<Self>,
        res: &mut R,
        lhs: &[&L],
        rhs: &[&Rh],
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        L: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        Rh: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        T: GetTensorKey<Self> + Sync;
}

/// Selects [`crate::reference::BlasMatMulReference::blas_jkls_block_reference`] as the JKLS kernel of `$be`.
#[macro_export]
macro_rules! impl_blas_jkls_reference {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasJklsImpl for $be {
            fn blas_jkls_block_tmp_bytes_impl<R, L, Rh, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &R,
                lhs: &L,
                rhs: &Rh,
                terms: usize,
                tsk: &T,
            ) -> usize
            where
                R: ::poulpy_ckks::CKKSCtBounds,
                L: ::poulpy_ckks::CKKSCtBounds,
                Rh: ::poulpy_ckks::CKKSCtBounds,
                T: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasMatMulReference::blas_jkls_block_tmp_bytes_reference(module, res, lhs, rhs, terms, tsk)
            }

            fn blas_jkls_block_impl<R, L, Rh, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                lhs: &[&L],
                rhs: &[&Rh],
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                L: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + Sync,
                Rh: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + Sync,
                T: ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
            {
                $crate::reference::BlasMatMulReference::blas_jkls_block_reference(module, res, lhs, rhs, tsk, scratch)
            }
        }
    };
}
