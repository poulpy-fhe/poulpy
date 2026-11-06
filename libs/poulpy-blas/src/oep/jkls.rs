use poulpy_ckks::{
    CKKSCtBounds, CKKSResult as Result, SetCKKSInfos,
    api::LinearTransformationPrepared,
    layouts::{CKKSLeftOperandPrepared, CKKSRightOperandPrepared},
};
use poulpy_core::layouts::{GGLWEInfos, GLWEToBackendMut, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey};
use poulpy_hal::layouts::{Backend, Module, ScratchArena};

/// Extension point of the JKLS kernels. [`Self::blas_jkls_block`] is one output block of the prepared product: the
/// references
/// [`BlasMatMulReference::blas_matmul_prepared_reference`](crate::reference::BlasMatMulReference::blas_matmul_prepared_reference)
/// and [`BlasMatMulReference::blas_matmul_reference`](crate::reference::BlasMatMulReference::blas_matmul_reference) run
/// one per block of `res`, so both `blas_matmul` and `blas_matmul_prepared` reach it, whatever the hints of the
/// operands. [`Self::blas_jkls_pt_unit`] is one unit of a product by public matrices under
/// [`BlasPtAlgorithm::JklsStored`](crate::layouts::BlasPtAlgorithm::JklsStored) and
/// [`BlasPtAlgorithm::JklsGalois`](crate::layouts::BlasPtAlgorithm::JklsGalois): the references of
/// `blas_matmul_pt_right` and `blas_matmul_pt_left` run one per block row (column) of `res`.
///
/// # Safety
/// Each method reproduces its reference in [`BlasJklsReference`](crate::reference::BlasJklsReference) bit for bit,
/// within the scratch its `_tmp_bytes` reports; with less scratch it panics before any write, as the reference does.
/// Anything its doc does not list as free changes bits.
///
/// The memory, layout, synchrony and aliasing rules of `poulpy-hal/docs/backend_safety_contract.md` apply.
pub unsafe trait BlasJklsImpl: Backend {
    /// Scratch bound of [`Self::blas_jkls_block`] into a block of layout `res` from `terms` pairs prepared from terms
    /// of layouts `lhs` and `rhs`, with a tensor key of layout `tsk`; may report less than the reference, never less
    /// than the body takes. It reports for `res`, `lhs[0]`, `rhs[0]`, the pair count and the key.
    fn blas_jkls_block_tmp_bytes<R, L, Rh, T>(module: &Module<Self>, terms: usize, res: &R, lhs: &L, rhs: &Rh, tsk: &T) -> usize
    where
        R: CKKSCtBounds,
        L: CKKSCtBounds,
        Rh: CKKSCtBounds,
        T: GGLWEInfos;

    /// The JKLS kernel over prepared terms, relinearized once. Reproduces
    /// [`blas_jkls_block_reference`](crate::reference::BlasJklsReference::blas_jkls_block_reference) bit for bit,
    /// digits and CKKS metadata, within [`Self::blas_jkls_block_tmp_bytes`]: `res = sum_t lhs[t] rhs[t]` as
    /// poulpy-ckks's `ckks_dot_product_ct` computes it for the terms the pairs were prepared from, which share one
    /// layout per side. The pairs are terms prepared once by poulpy-ckks's `ckks_left_operand_prepare_alloc` and
    /// `ckks_right_operand_prepare_alloc`, which report the terms they were prepared from and give their convolution operands through
    /// `cnv()`: the kernel transforms no operand.
    ///
    /// **Fixed:**
    ///
    /// - the circuit, one-pass where poulpy-ckks's `ckks_mul_into` takes its one-pass product: rank 1, one `base2k`
    ///   across `res`, the terms and the tensor key, and an offset of at least `base2k`; two-pass otherwise;
    /// - one-pass: poulpy-core's `glwe_mul_prepared_relinearize_sum` over the pairs, in slice order, at the offset
    ///   poulpy-ckks's ct x ct rule gives for `res` and the width it stamps, with the tensor key at the tensor width
    ///   `max(lhs.k, rhs.k)`: every pair's products are added in the transform domain, in that order, and only the
    ///   key-switched column of the sum comes back to digits before the one final normalization; an FFT backend adds in
    ///   floating point, so the order of the pairs is part of the bits; then `res` is stamped with the rule's budget
    ///   and scale, the smallest label and the joined slot kind;
    /// - two-pass: each term is one poulpy-core `glwe_tensor_apply_prepared` of `(lhs[t], rhs[t])` at that offset,
    ///   rounded on its own at the tensor width (summing unrounded products, as `cnv_apply_dft_sum` would, changes
    ///   bits); the terms are summed exactly, the sum normalized at that width before the relinearization reads it,
    ///   `res` stamped as above, then relinearized once into `res` with the tensor key at that width; like any lazy
    ///   addition, the sum must fit a 64-bit digit, which the caller's parameters ensure;
    /// - either way, `res` comes back canonical, without a noise tag.
    ///
    /// **Free (bit-identical):** streaming the pairs; caching other forms of the operands. On the two-pass circuit,
    /// also the order and grouping of the terms, computed in turn or concurrently, and normalizing partial sums or
    /// leaving them unnormalized until the DFT that reads the sum: normalization preserves the value, so its placement
    /// never changes the bits. On the one-pass circuit, the order and grouping of the pairs are free only on an NTT
    /// backend, whose transform-domain additions are exact. Anything else changes bits.
    ///
    /// It is called with `lhs.len() == rhs.len() >= 1` pairs, every `lhs` prepared from terms of one layout and every
    /// `rhs` from terms of another, on inputs the prepared product's validator accepted, at a radix of 62 at most,
    /// which checked the tensor key at that width and of their rank. Within the scratch
    /// [`Self::blas_jkls_block_tmp_bytes`] reports for `res`, `lhs[0]`, `rhs[0]`, the pair count and the key, it
    /// returns `Ok`, or poulpy-ckks's budget error before any write to `res`; with less scratch it panics before any
    /// write, as the reference does. Any other `Err` is a backend fault (a validator gap or a device failure), and
    /// `res` is then unspecified. The reference products call it once per block of `res`, block after block, row by row
    /// or column by column, on one scratch. The pairs and the key are `Sync`, so an implementation may compute one
    /// block's terms concurrently, within the one operation. A backend holds the terms in its own representation
    /// through its poulpy-ckks preparations, the storage of `CKKSLeftOperandPrepared<Self>` and `CKKSRightOperandPrepared<Self>`.
    fn blas_jkls_block<R, T>(
        module: &Module<Self>,
        res: &mut R,
        lhs: &[&CKKSLeftOperandPrepared<Self>],
        rhs: &[&CKKSRightOperandPrepared<Self>],
        tsk: &T,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        R: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos,
        T: GetTensorKey<Self> + Sync;

    /// Scratch bound of [`Self::blas_jkls_pt_unit`] into `outputs` blocks of layout `res` from sources of layout `src`
    /// through `first` and `masks`, by plaintexts prepared like `plaintext`, with rotation keys of layout `keys`; may
    /// report less than the reference, never less than the body takes. It reports for `res` and `src` at their widths
    /// whatever their allocations.
    #[allow(clippy::too_many_arguments)]
    fn blas_jkls_pt_unit_tmp_bytes<R, S, K>(
        module: &Module<Self>,
        outputs: usize,
        res: &R,
        src: &S,
        first: &LinearTransformationPrepared<Self>,
        masks: &[&LinearTransformationPrepared<Self>],
        plaintext: &CKKSRightOperandPrepared<Self>,
        keys: &K,
    ) -> usize
    where
        R: CKKSCtBounds,
        S: CKKSCtBounds,
        K: GGLWEInfos;

    /// One unit of a product by public matrices under `JklsStored` or `JklsGalois`. Reproduces
    /// [`blas_jkls_pt_unit_reference`](crate::reference::BlasJklsReference::blas_jkls_pt_unit_reference) bit for bit,
    /// digits and CKKS metadata, within [`Self::blas_jkls_pt_unit_tmp_bytes`].
    ///
    /// The terms of source `l` are `per = masks.len() + 1` blocks: term `0` is `first` evaluated from `sources[l]` by the
    /// map hook, then copied by poulpy-ckks's `ckks_copy` into a block of the terms' layout, and term `k >= 1` is
    /// `masks[k - 1]` evaluated by the map hook from that same evaluation of `first`. Each term is prepared by
    /// `ckks_left_operand_prepare_alloc` and multiplied by `plaintexts[o * sources.len() + l][k % len]` read through `galois[k]`.
    /// Block `o` of `res` receives poulpy-ckks's `ckks_dot_product_prepared_pt_vec_dft` of the first source's terms,
    /// `ckks_dot_product_prepared_pt_vec_dft_add` of each next source's, in order, into one transform-domain accumulator,
    /// then `ckks_dot_product_prepared_pt_vec_from_dft`: `ckks_dot_product_prepared_pt_vec` over every term, sources in order
    /// and terms in order within a source, normalized once.
    ///
    /// **Fixed:** the circuit above; on an FFT backend, which adds in floating point, the order of the terms and their
    /// grouping by source are part of the bits.
    ///
    /// **Free (bit-identical):** streaming the sources, keeping one accumulator per block of `res`; the order of the
    /// blocks of `res`; the hint of the term preparations; computing within the unit concurrently; on an NTT backend,
    /// whose transform-domain additions are exact, the order and grouping of the terms, a term at a time included.
    /// Anything else changes bits.
    ///
    /// It is called with at least one source and one block of `res`, at least one mask, `plaintexts` holding
    /// `res.len() * sources.len()` non-empty slices and `galois` one odd element per term of a source, on inputs the
    /// product's validator accepted, which checked the rotation keys of `first` and `masks` at the widths the map hook
    /// looks them up. Within that scratch it returns `Ok`, or poulpy-ckks's budget error before any write to `res`; with
    /// less scratch it panics before any write, as the reference does. Any other `Err` is a backend fault, and `res` is
    /// then unspecified. The caller stamps the label of every block of `res` afterwards.
    #[allow(clippy::too_many_arguments)]
    fn blas_jkls_pt_unit<D, S, K>(
        module: &Module<Self>,
        res: &mut [&mut D],
        sources: &[&S],
        first: &LinearTransformationPrepared<Self>,
        masks: &[&LinearTransformationPrepared<Self>],
        plaintexts: &[&[CKKSRightOperandPrepared<Self>]],
        galois: &[i64],
        keys: &K,
        scratch: &mut ScratchArena<'_, Self>,
    ) -> Result<()>
    where
        D: GLWEToBackendMut<Self> + CKKSCtBounds + SetCKKSInfos + Send,
        S: GLWEToBackendRef<Self> + CKKSCtBounds + Sync,
        K: GetAutomorphismKey<Self> + GetTensorKey<Self> + Sync;
}

/// Selects the references of [`BlasJklsReference`](crate::reference::BlasJklsReference) as the JKLS kernels of `$be`.
#[macro_export]
macro_rules! impl_blas_jkls_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::BlasJklsImpl for $be {
            fn blas_jkls_block_tmp_bytes<R, L, Rh, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                terms: usize,
                res: &R,
                lhs: &L,
                rhs: &Rh,
                tsk: &T,
            ) -> usize
            where
                R: ::poulpy_ckks::CKKSCtBounds,
                L: ::poulpy_ckks::CKKSCtBounds,
                Rh: ::poulpy_ckks::CKKSCtBounds,
                T: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasJklsReference::blas_jkls_block_tmp_bytes_reference(module, terms, res, lhs, rhs, tsk)
            }

            fn blas_jkls_block<R, T>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut R,
                lhs: &[&::poulpy_ckks::layouts::CKKSLeftOperandPrepared<Self>],
                rhs: &[&::poulpy_ckks::layouts::CKKSRightOperandPrepared<Self>],
                tsk: &T,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                R: ::poulpy_core::layouts::GLWEToBackendMut<Self> + ::poulpy_ckks::CKKSCtBounds + ::poulpy_ckks::SetCKKSInfos,
                T: ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
            {
                $crate::reference::BlasJklsReference::blas_jkls_block_reference(module, res, lhs, rhs, tsk, scratch)
            }

            fn blas_jkls_pt_unit_tmp_bytes<R, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                outputs: usize,
                res: &R,
                src: &S,
                first: &::poulpy_ckks::api::LinearTransformationPrepared<Self>,
                masks: &[&::poulpy_ckks::api::LinearTransformationPrepared<Self>],
                plaintext: &::poulpy_ckks::layouts::CKKSRightOperandPrepared<Self>,
                keys: &K,
            ) -> usize
            where
                R: ::poulpy_ckks::CKKSCtBounds,
                S: ::poulpy_ckks::CKKSCtBounds,
                K: ::poulpy_core::layouts::GGLWEInfos,
            {
                $crate::reference::BlasJklsReference::blas_jkls_pt_unit_tmp_bytes_reference(
                    module, outputs, res, src, first, masks, plaintext, keys,
                )
            }

            fn blas_jkls_pt_unit<D, S, K>(
                module: &::poulpy_hal::layouts::Module<Self>,
                res: &mut [&mut D],
                sources: &[&S],
                first: &::poulpy_ckks::api::LinearTransformationPrepared<Self>,
                masks: &[&::poulpy_ckks::api::LinearTransformationPrepared<Self>],
                plaintexts: &[&[::poulpy_ckks::layouts::CKKSRightOperandPrepared<Self>]],
                galois: &[i64],
                keys: &K,
                scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, Self>,
            ) -> ::poulpy_ckks::CKKSResult<()>
            where
                D: ::poulpy_core::layouts::GLWEToBackendMut<Self>
                    + ::poulpy_ckks::CKKSCtBounds
                    + ::poulpy_ckks::SetCKKSInfos
                    + Send,
                S: ::poulpy_core::layouts::GLWEToBackendRef<Self> + ::poulpy_ckks::CKKSCtBounds + Sync,
                K: ::poulpy_core::layouts::GetAutomorphismKey<Self> + ::poulpy_core::layouts::GetTensorKey<Self> + Sync,
            {
                $crate::reference::BlasJklsReference::blas_jkls_pt_unit_reference(
                    module, res, sources, first, masks, plaintexts, galois, keys, scratch,
                )
            }
        }
    };
}
