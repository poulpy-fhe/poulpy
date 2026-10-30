//! The summed product of prepared GLWE ciphertexts by prepared plaintexts, each read through a Galois element.

use poulpy_hal::{
    api::{
        CnvOps, ModuleN, ScratchArenaTakeVecZnx, ScratchArenaTakeVecZnxBig, ScratchArenaTakeVecZnxDft, VecZnxAddAssign,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftAddAssign, VecZnxDftBytesOf, VecZnxIdftApply,
        VecZnxIdftApplyTmpBytes, VecZnxIdftNormalize, VecZnxIdftNormalizeTmpBytes, VecZnxNormalize, VecZnxNormalizeTmpBytes,
        VecZnxRsh, VecZnxRshTmpBytes,
    },
    layouts::{
        Backend, CnvDftAccTerm, CnvPVecLToBackendRef, CnvPVecRToBackendRef, Module, ScratchArena, VecZnxBigToBackendRef,
        VecZnxDftBackendMut, VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendMut, VecZnxToBackendRef,
        vec_znx_backend_ref_from_mut,
    },
};

use crate::layouts::{Base2K, Degree, GLWEInfos, GLWEToBackendMut, operand_degree};

use crate::gadget::cnv_offset_to_limb_offset;

/// The reference algorithm of [`GLWEMulPreparedPlainSumImpl`](crate::oep::GLWEMulPreparedPlainSumImpl), which
/// `impl_glwe_mul_prepared_plain_sum_reference_full!` forwards to and an override can call.
pub trait GLWEMulPreparedPlainSumReference<BE: Backend> {
    fn glwe_mul_prepared_plain_sum_tmp_bytes_reference<R>(&self, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum_reference<R, AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>;

    fn glwe_mul_prepared_plain_sum_add_tmp_bytes_reference<R>(&self, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum_add_reference<R, AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>;

    fn glwe_mul_prepared_plain_sum_dft_tmp_bytes_reference(&self, res_size: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum_dft_reference<AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>;

    fn glwe_mul_prepared_plain_sum_dft_add_tmp_bytes_reference(&self, res_size: usize, a_size: usize, b_size: usize) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn glwe_mul_prepared_plain_sum_dft_add_reference<AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>;
}

impl<BE: Backend> GLWEMulPreparedPlainSumReference<BE> for Module<BE>
where
    Self: ModuleN
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + CnvOps<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxIdftNormalize<BE>
        + VecZnxIdftNormalizeTmpBytes
        + VecZnxNormalize<BE>
        + VecZnxNormalizeTmpBytes
        + VecZnxAddAssign<BE>
        + VecZnxRsh<BE>
        + VecZnxRshTmpBytes,
{
    /// One column's sum, inverse transform and normalization: the call transforms no operand.
    fn glwe_mul_prepared_plain_sum_tmp_bytes_reference<R>(&self, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos,
    {
        let n = operand_degree(self.n(), &[res.n()]);
        let res_dft_size = a_size + b_size;
        self.vec_znx_dft_bytes_of(n, 1, res_dft_size)
            + self.cnv_apply_dft_sum_tmp_bytes(0, res_dft_size, a_size, b_size).max(
                self.vec_znx_big_bytes_of(n, 1, res_dft_size)
                    + self
                        .vec_znx_idft_apply_tmp_bytes()
                        .max(self.vec_znx_big_normalize_tmp_bytes()),
            )
    }

    fn glwe_mul_prepared_plain_sum_reference<R, AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>,
    {
        const OP: &str = "glwe_mul_prepared_plain_sum";
        let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
        check_terms::<BE, _, _>(OP, n, cols, a, a_size, b, b_size, galois);
        assert!(
            scratch.available() >= self.glwe_mul_prepared_plain_sum_tmp_bytes_reference(res, a_size, b_size),
            "glwe_mul_prepared_plain_sum: scratch below its query"
        );

        res.set_canonical(true);
        let (res_base2k, res_k, base2k) = (res.base2k().as_usize(), res.k().as_usize(), base2k.as_usize());
        let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, base2k);
        let res_dft_size = a_size + b_size - cnv_offset_hi;
        let mut terms = Vec::with_capacity(a.len());
        for col in 0..cols {
            column_terms(&mut terms, col, a, b, galois);
            let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(n, 1, res_dft_size);
            {
                let mut res_dft_backend = res_dft.to_backend_mut();
                self.cnv_apply_dft_sum(cnv_offset_hi, &mut res_dft_backend, 0, &terms, &mut scratch_1);
            }
            let (mut res_big, mut scratch_2) = scratch_1.take_vec_znx_big_scratch(n, 1, res_dft_size);
            self.vec_znx_idft_apply(&mut res_big, 0, &res_dft.to_backend_ref(), 0, &mut scratch_2.borrow());
            let mut res_backend = res.to_backend_mut();
            self.vec_znx_big_normalize(
                &mut res_backend.data,
                res_base2k,
                res_k,
                cnv_offset_lo,
                col,
                &res_big.to_backend_ref(),
                base2k,
                0,
                &mut scratch_2.borrow(),
            );
        }
    }

    /// One column's sum, on at least one limb more than `res`, then either the shifted column of `res` and the fused
    /// inverse transform, or the lossless inverse transform, `res` added, and the normalization.
    fn glwe_mul_prepared_plain_sum_add_tmp_bytes_reference<R>(&self, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: GLWEInfos,
    {
        let n = operand_degree(self.n(), &[res.n()]);
        let (res_dft_size, addend_size) = ((a_size + b_size).max(res.size() + 1), res.size() + 1);
        let sum_size = (res_dft_size + 1).max(res.size());
        let shifted = BE::scratch_aligned(BE::vec_znx_bytes_of(n, 1, addend_size))
            + self
                .vec_znx_rsh_tmp_bytes(addend_size)
                .max(self.vec_znx_idft_normalize_tmp_bytes(res.size(), res_dft_size));
        let lossless = BE::scratch_aligned(BE::vec_znx_bytes_of(n, 1, sum_size))
            + self
                .vec_znx_idft_normalize_tmp_bytes(sum_size, res_dft_size)
                .max(self.vec_znx_normalize_tmp_bytes());
        BE::scratch_aligned(self.vec_znx_dft_bytes_of(n, 1, res_dft_size))
            + self
                .cnv_apply_dft_sum_tmp_bytes(0, res_dft_size, a_size, b_size)
                .max(shifted)
                .max(lossless)
    }

    fn glwe_mul_prepared_plain_sum_add_reference<R, AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut R,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEInfos,
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>,
    {
        const OP: &str = "glwe_mul_prepared_plain_sum_add";
        let (n, cols) = (res.n().as_usize(), res.rank().as_usize() + 1);
        check_terms::<BE, _, _>(OP, n, cols, a, a_size, b, b_size, galois);
        assert!(res.base2k() == base2k, "{OP}: res must be at the radix of the operands");
        assert!(
            scratch.available() >= self.glwe_mul_prepared_plain_sum_add_tmp_bytes_reference(res, a_size, b_size),
            "glwe_mul_prepared_plain_sum_add: scratch below its query"
        );

        let (res_k, base2k) = (res.k().as_usize(), base2k.as_usize());
        let (cnv_offset_hi, cnv_offset_lo) = cnv_offset_to_limb_offset(cnv_offset, base2k);
        // Limbs past the product are zero, and the fused inverse transform reads only as many limbs of its addend.
        let res_dft_size = (a_size + b_size - cnv_offset_hi).max(res.size() + 1);
        let mut terms = Vec::with_capacity(a.len());
        for col in 0..cols {
            column_terms(&mut terms, col, a, b, galois);
            let (mut res_dft, mut scratch_1) = scratch.borrow().take_vec_znx_dft_scratch(n, 1, res_dft_size);
            self.cnv_apply_dft_sum(cnv_offset_hi, &mut res_dft.to_backend_mut(), 0, &terms, &mut scratch_1);
            if cnv_offset_lo >= 0 {
                // `res` enters the fused inverse transform scaled by `2^-cnv_offset_lo`: fewer bits than a limb, exact
                // on one more limb.
                let (mut addend, mut scratch_2) = scratch_1.take_vec_znx_scratch(n, 1, res.size() + 1);
                {
                    let res_backend = res.to_backend_mut();
                    let res_data = vec_znx_backend_ref_from_mut::<BE>(&res_backend.data);
                    let bits = cnv_offset_lo as usize;
                    self.vec_znx_rsh(
                        base2k,
                        bits,
                        &mut addend.to_backend_mut(),
                        0,
                        &res_data,
                        col,
                        &mut scratch_2.borrow(),
                    );
                }
                let mut res_backend = res.to_backend_mut();
                self.vec_znx_idft_normalize(
                    &mut res_backend.data,
                    base2k,
                    res_k,
                    cnv_offset_lo,
                    col,
                    &mut res_dft.to_backend_mut(),
                    0,
                    base2k,
                    Some((&addend.to_backend_ref(), 0)),
                    &mut scratch_2.borrow(),
                );
            } else {
                // Scaled down, the sum carries bits above the torus that no addend holds: it is rounded losslessly,
                // one limb wider, then `res` is added and the total rounded once.
                let sum_size = (res_dft_size + 1).max(res.size());
                let (mut sum, mut scratch_2) = scratch_1.take_vec_znx_scratch(n, 1, sum_size);
                self.vec_znx_idft_normalize(
                    &mut sum.to_backend_mut(),
                    base2k,
                    sum_size * base2k,
                    cnv_offset_lo,
                    0,
                    &mut res_dft.to_backend_mut(),
                    0,
                    base2k,
                    None,
                    &mut scratch_2.borrow(),
                );
                {
                    let res_backend = res.to_backend_mut();
                    let res_data = vec_znx_backend_ref_from_mut::<BE>(&res_backend.data);
                    self.vec_znx_add_assign(&mut sum.to_backend_mut(), 0, &res_data, col);
                }
                let mut res_backend = res.to_backend_mut();
                self.vec_znx_normalize(
                    &mut res_backend.data,
                    base2k,
                    res_k,
                    0,
                    col,
                    &sum.to_backend_ref(),
                    base2k,
                    0,
                    &mut scratch_2.borrow(),
                );
            }
        }
        res.set_canonical(true);
    }

    /// The convolution sums, written straight into `res`.
    fn glwe_mul_prepared_plain_sum_dft_tmp_bytes_reference(&self, res_size: usize, a_size: usize, b_size: usize) -> usize {
        self.cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size)
    }

    fn glwe_mul_prepared_plain_sum_dft_reference<AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>,
    {
        const OP: &str = "glwe_mul_prepared_plain_sum_dft";
        let (n, cols) = (operand_degree(self.n(), &[Degree(res.n() as u32)]), res.cols());
        check_terms::<BE, _, _>(OP, n, cols, a, a_size, b, b_size, galois);
        assert!(
            scratch.available() >= self.glwe_mul_prepared_plain_sum_dft_tmp_bytes_reference(res.size(), a_size, b_size),
            "glwe_mul_prepared_plain_sum_dft: scratch below its query"
        );

        let (cnv_offset_hi, _) = cnv_offset_to_limb_offset(cnv_offset, base2k.as_usize());
        let mut terms = Vec::with_capacity(a.len());
        for col in 0..cols {
            column_terms(&mut terms, col, a, b, galois);
            self.cnv_apply_dft_sum(cnv_offset_hi, res, col, &terms, &mut scratch.borrow());
        }
    }

    /// A one-column sum of `res_size` limbs, then the convolution sum into it.
    fn glwe_mul_prepared_plain_sum_dft_add_tmp_bytes_reference(&self, res_size: usize, a_size: usize, b_size: usize) -> usize {
        BE::scratch_aligned(self.vec_znx_dft_bytes_of(self.n(), 1, res_size))
            + self.cnv_apply_dft_sum_tmp_bytes(0, res_size, a_size, b_size)
    }

    fn glwe_mul_prepared_plain_sum_dft_add_reference<AP, BP>(
        &self,
        cnv_offset: usize,
        res: &mut VecZnxDftBackendMut<'_, BE>,
        a: &[&AP],
        a_size: usize,
        b: &[&BP],
        b_size: usize,
        galois: &[i64],
        base2k: Base2K,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        AP: CnvPVecLToBackendRef<BE>,
        BP: CnvPVecRToBackendRef<BE>,
    {
        const OP: &str = "glwe_mul_prepared_plain_sum_dft_add";
        let (n, cols, size) = (operand_degree(self.n(), &[Degree(res.n() as u32)]), res.cols(), res.size());
        check_terms::<BE, _, _>(OP, n, cols, a, a_size, b, b_size, galois);
        assert!(
            scratch.available() >= self.glwe_mul_prepared_plain_sum_dft_add_tmp_bytes_reference(size, a_size, b_size),
            "glwe_mul_prepared_plain_sum_dft_add: scratch below its query"
        );

        let (cnv_offset_hi, _) = cnv_offset_to_limb_offset(cnv_offset, base2k.as_usize());
        let mut terms = Vec::with_capacity(a.len());
        for col in 0..cols {
            column_terms(&mut terms, col, a, b, galois);
            let (mut sum, mut rest) = scratch.borrow().take_vec_znx_dft_scratch(n, 1, size);
            self.cnv_apply_dft_sum(cnv_offset_hi, &mut sum.to_backend_mut(), 0, &terms, &mut rest);
            self.vec_znx_dft_add_assign(res, col, &sum.to_backend_ref(), 0);
        }
    }
}

/// The refusals of the summed products, `op` naming the public operation, for a result of degree `n` and `cols` columns.
#[allow(clippy::too_many_arguments)]
fn check_terms<BE, AP, BP>(op: &str, n: usize, cols: usize, a: &[&AP], a_size: usize, b: &[&BP], b_size: usize, galois: &[i64])
where
    BE: Backend,
    AP: CnvPVecLToBackendRef<BE>,
    BP: CnvPVecRToBackendRef<BE>,
{
    assert!(
        !a.is_empty() && b.len() == a.len() && galois.len() == a.len(),
        "{op}: a must be non-empty, and b and galois as long"
    );
    assert!(galois.iter().all(|&g| g & 1 == 1), "{op}: every Galois element must be odd");
    assert!(
        a.iter()
            .map(|ap| ap.to_backend_ref())
            .all(|ap| (ap.n(), ap.cols(), ap.size()) == (n, cols, a_size)),
        "{op}: every a[t] must be of the degree of res, with rank + 1 columns of a_size limbs"
    );
    assert!(
        b.iter().map(|bp| bp.to_backend_ref()).all(|bp| {
            bp.cols() == 1
                && bp.size() == b_size
                && bp.n().is_power_of_two()
                && bp.n() >= BE::MIN_DEGREE
                && n.is_multiple_of(bp.n())
        }),
        "{op}: every b[t] must be one column of b_size limbs, of a degree dividing that of res"
    );
}

/// Refills `terms` with the convolution terms of column `col`.
fn column_terms<'a, BE, AP, BP>(terms: &mut Vec<CnvDftAccTerm<'a, BE>>, col: usize, a: &'a [&AP], b: &'a [&BP], galois: &[i64])
where
    BE: Backend,
    AP: CnvPVecLToBackendRef<BE>,
    BP: CnvPVecRToBackendRef<BE>,
{
    terms.clear();
    terms.extend(a.iter().zip(b).zip(galois).map(|((ap, bp), &b_galois)| CnvDftAccTerm {
        a: ap.to_backend_ref(),
        a_col: col,
        b: bp.to_backend_ref(),
        b_col: 0,
        b_galois,
    }));
}

/// Forwards every method of [`GLWEMulPreparedPlainSumReference`] to its HAL-based reference implementation.
#[macro_export]
macro_rules! impl_glwe_mul_prepared_plain_sum_reference_full {
    ($be:ty) => {
        unsafe impl $crate::oep::GLWEMulPreparedPlainSumImpl for $be {
    fn glwe_mul_prepared_plain_sum_tmp_bytes<R>(module: &::poulpy_hal::layouts::Module<$be>, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_tmp_bytes_reference::<R>(module, res, a_size, b_size)
        }

    fn glwe_mul_prepared_plain_sum<R, AP, BP>(module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut R, a: &[&AP], a_size: usize, b: &[&BP], b_size: usize, galois: &[i64], base2k: $crate::layouts::Base2K, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>)
    where
        R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
        AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
        BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_reference::<R, AP, BP>(module, cnv_offset, res, a, a_size, b, b_size, galois, base2k, scratch)
        }

    fn glwe_mul_prepared_plain_sum_add_tmp_bytes<R>(module: &::poulpy_hal::layouts::Module<$be>, res: &R, a_size: usize, b_size: usize) -> usize
    where
        R: $crate::layouts::GLWEInfos {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_add_tmp_bytes_reference::<R>(module, res, a_size, b_size)
        }

    fn glwe_mul_prepared_plain_sum_add<R, AP, BP>(module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut R, a: &[&AP], a_size: usize, b: &[&BP], b_size: usize, galois: &[i64], base2k: $crate::layouts::Base2K, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>)
    where
        R: $crate::layouts::GLWEToBackendMut<$be> + $crate::layouts::GLWEInfos,
        AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
        BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_add_reference::<R, AP, BP>(module, cnv_offset, res, a, a_size, b, b_size, galois, base2k, scratch)
        }

    fn glwe_mul_prepared_plain_sum_dft_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>, res_size: usize, a_size: usize, b_size: usize) -> usize {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_dft_tmp_bytes_reference(module, res_size, a_size, b_size)
        }

    fn glwe_mul_prepared_plain_sum_dft<AP, BP>(module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>, a: &[&AP], a_size: usize, b: &[&BP], b_size: usize, galois: &[i64], base2k: $crate::layouts::Base2K, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>)
    where
        AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
        BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_dft_reference::<AP, BP>(module, cnv_offset, res, a, a_size, b, b_size, galois, base2k, scratch)
        }

    fn glwe_mul_prepared_plain_sum_dft_add_tmp_bytes(module: &::poulpy_hal::layouts::Module<$be>, res_size: usize, a_size: usize, b_size: usize) -> usize {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_dft_add_tmp_bytes_reference(module, res_size, a_size, b_size)
        }

    fn glwe_mul_prepared_plain_sum_dft_add<AP, BP>(module: &::poulpy_hal::layouts::Module<$be>, cnv_offset: usize, res: &mut ::poulpy_hal::layouts::VecZnxDftBackendMut<'_, $be>, a: &[&AP], a_size: usize, b: &[&BP], b_size: usize, galois: &[i64], base2k: $crate::layouts::Base2K, scratch: &mut ::poulpy_hal::layouts::ScratchArena<'_, $be>)
    where
        AP: ::poulpy_hal::layouts::CnvPVecLToBackendRef<$be>,
        BP: ::poulpy_hal::layouts::CnvPVecRToBackendRef<$be> {
            <::poulpy_hal::layouts::Module<$be> as $crate::reference::operations::GLWEMulPreparedPlainSumReference<$be>>::glwe_mul_prepared_plain_sum_dft_add_reference::<AP, BP>(module, cnv_offset, res, a, a_size, b, b_size, galois, base2k, scratch)
        }
        }
    };
}
