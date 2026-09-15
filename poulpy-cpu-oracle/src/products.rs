use crate::{FFT64Oracle, NTT4x30Oracle};
use poulpy_hal::{
    layouts::*,
    oep::{HalConvolutionImpl, HalSvpImpl, HalVmpImpl},
};

trait ProductArithmetic: Backend<ZnxWord = i64> {
    fn forward(module: &Module<Self>, res: &mut [Self::DftWord], a: &[i64]);
    fn multiply_accumulate(res: &mut [Self::DftWord], a: &[Self::DftWord], b: &[Self::DftWord]);
    fn multiply_assign(res: &mut [Self::DftWord], a: &[Self::DftWord]);
}

impl ProductArithmetic for FFT64Oracle {
    fn forward(module: &Module<Self>, res: &mut [f64], a: &[i64]) {
        use crate::reference::fft64::module::FFTModuleHandle;
        for (r, &x) in res.iter_mut().zip(a) {
            *r = x as f64;
        }
        module.get_fft_table().execute(res);
    }
    fn multiply_accumulate(res: &mut [f64], a: &[f64], b: &[f64]) {
        let m = res.len() / 2;
        for i in 0..m {
            res[i] += a[i] * b[i] - a[i + m] * b[i + m];
            res[i + m] += a[i] * b[i + m] + a[i + m] * b[i];
        }
    }
    fn multiply_assign(res: &mut [f64], a: &[f64]) {
        let m = res.len() / 2;
        for i in 0..m {
            let (re, im) = (res[i], res[i + m]);
            res[i] = re * a[i] - im * a[i + m];
            res[i + m] = re * a[i + m] + im * a[i];
        }
    }
}

impl ProductArithmetic for NTT4x30Oracle {
    fn forward(module: &Module<Self>, res: &mut [Self::DftWord], a: &[i64]) {
        use crate::reference::ntt4x30::{NttDFTExecute, NttFromZnx64, vec_znx_dft::NttModuleHandle};
        let res = bytemuck::cast_slice_mut(res);
        Self::ntt_from_znx64(res, a);
        Self::ntt_dft_execute(module.get_ntt_table(), res);
    }
    fn multiply_accumulate(res: &mut [Self::DftWord], a: &[Self::DftWord], b: &[Self::DftWord]) {
        use crate::reference::ntt4x30::primes::{PrimeSet, Primes30};
        let res: &mut [u64] = bytemuck::cast_slice_mut(res);
        let a: &[u64] = bytemuck::cast_slice(a);
        let b: &[u64] = bytemuck::cast_slice(b);
        for (i, r) in res.iter_mut().enumerate() {
            let q = Primes30::Q[i % 4] as u128;
            *r = ((*r as u128 + a[i] as u128 * b[i] as u128) % q) as u64;
        }
    }
    fn multiply_assign(res: &mut [Self::DftWord], a: &[Self::DftWord]) {
        use crate::reference::ntt4x30::primes::{PrimeSet, Primes30};
        let res: &mut [u64] = bytemuck::cast_slice_mut(res);
        let a: &[u64] = bytemuck::cast_slice(a);
        for (i, r) in res.iter_mut().enumerate() {
            *r = (*r as u128 * a[i] as u128 % Primes30::Q[i % 4] as u128) as u64;
        }
    }
}

// Prepared polynomials retain ordinary DFT order. Matrices retain MatZnx order.
macro_rules! impl_products {
    ($backend:ty, $big:ty) => {
        unsafe impl HalSvpImpl for $backend {
            fn svp_prepare(
                module: &Module<Self>,
                res: &mut SvpPPolBackendMut<'_, Self>,
                res_col: usize,
                a: &ScalarZnxBackendRef<'_, Self>,
                a_col: usize,
            ) {
                assert_eq!(res.n(), a.n());
                Self::forward(module, res.at_mut(res_col, 0), a.at(a_col, 0));
            }
            fn svp_ppol_copy(
                _module: &Module<Self>,
                res: &mut SvpPPolBackendMut<'_, Self>,
                res_col: usize,
                a: &SvpPPolBackendRef<'_, Self>,
                a_col: usize,
            ) {
                assert_eq!(res.hint(), a.hint());
                res.at_mut(res_col, 0).copy_from_slice(a.at(a_col, 0));
            }
            fn svp_apply_dft_to_dft(
                _module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &SvpPPolBackendRef<'_, Self>,
                a_col: usize,
                b: &VecZnxDftBackendRef<'_, Self>,
                b_col: usize,
            ) {
                for j in 0..res.size() {
                    let out = res.at_mut(res_col, j);
                    out.fill(Default::default());
                    if j < b.size() {
                        Self::multiply_accumulate(out, a.at(a_col, 0), b.at(b_col, j));
                    }
                }
            }
            fn svp_apply_dft_to_dft_assign(
                _module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &SvpPPolBackendRef<'_, Self>,
                a_col: usize,
            ) {
                for j in 0..res.size() {
                    Self::multiply_assign(res.at_mut(res_col, j), a.at(a_col, 0));
                }
            }
        }
        unsafe impl HalVmpImpl for $backend {
            fn vmp_prepare_tmp_bytes(
                _module: &Module<Self>,
                _rows: usize,
                _cols_in: usize,
                _cols_out: usize,
                _size: usize,
            ) -> usize {
                0
            }
            fn vmp_prepare(
                module: &Module<Self>,
                res: &mut VmpPMatBackendMut<'_, Self>,
                a: &MatZnxBackendRef<'_, Self>,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                assert_eq!(
                    (res.n(), res.rows(), res.cols_in(), res.cols_out(), res.size()),
                    (a.n(), a.rows(), a.cols_in(), a.cols_out(), a.size())
                );
                let n = module.n();
                for (out, input) in res.raw_mut().chunks_exact_mut(n).zip(a.raw().chunks_exact(n)) {
                    Self::forward(module, out, input);
                }
            }
            fn vmp_apply_dft_to_dft_tmp_bytes(
                _module: &Module<Self>,
                _res_size: usize,
                _a_size: usize,
                _b_rows: usize,
                _b_cols_in: usize,
                _b_cols_out: usize,
                _b_size: usize,
            ) -> usize {
                0
            }
            fn vmp_apply_dft_to_dft(
                module: &Module<Self>,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                a: &VecZnxDftBackendRef<'_, Self>,
                b: &VmpPMatBackendRef<'_, Self>,
                limb_offset: usize,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                assert_eq!(res.n(), a.n());
                assert_eq!(res.n(), b.n());
                assert_eq!(a.cols(), b.cols_in());
                assert_eq!(res.cols(), b.cols_out());
                let n = module.n();
                for limb in 0..res.size() {
                    for col in 0..res.cols() {
                        let out = res.at_mut(col, limb);
                        out.fill(Default::default());
                        if let Some(k) = limb.checked_add(limb_offset).filter(|&k| k < b.size()) {
                            for row in 0..a.size().min(b.rows()) {
                                for input in 0..a.cols() {
                                    let index =
                                        ((row * b.cols_in() + input) * b.size() * b.cols_out() + k * b.cols_out() + col) * n;
                                    Self::multiply_accumulate(out, a.at(input, row), &b.raw()[index..index + n]);
                                }
                            }
                        }
                    }
                }
            }
            fn vmp_extract_selected_rows(
                _module: &Module<Self>,
                res: &mut VmpPMatBackendMut<'_, Self>,
                a: &VmpPMatBackendRef<'_, Self>,
                first_row: usize,
                row_step: usize,
            ) {
                crate::reference::vmp_select::assert_extractable(res, a, first_row, row_step);
                let width = res.n() * res.cols_out() * res.size();
                let src_width = a.n() * a.cols_out() * a.size();
                for row in 0..res.rows() {
                    for input in 0..res.cols_in() {
                        let dst = (row * res.cols_in() + input) * width;
                        let src = ((first_row + row * row_step) * a.cols_in() + input) * src_width;
                        res.raw_mut()[dst..dst + width].copy_from_slice(&a.raw()[src..src + width]);
                    }
                }
            }
            fn vmp_zero(_module: &Module<Self>, res: &mut VmpPMatBackendMut<'_, Self>) {
                res.raw_mut().fill(Default::default());
            }
        }
        unsafe impl HalConvolutionImpl for $backend {
            fn cnv_prepare_left_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
                0
            }
            fn cnv_prepare_left(
                module: &Module<Self>,
                res: &mut CnvPVecLBackendMut<'_, Self>,
                a: &VecZnxBackendRef<'_, Self>,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                assert_eq!(res.n(), a.n());
                assert_dense(a, "cnv_prepare");
                assert_eq!(res.cols(), a.cols());
                for j in 0..res.size() {
                    for col in 0..res.cols() {
                        let n = res.n();
                        let index = (j * res.cols() + col) * n;
                        let out = &mut res.raw_mut()[index..index + n];
                        out.fill(Default::default());
                        if j < a.size() {
                            Self::forward(module, out, a.at(col, j));
                        }
                    }
                }
            }
            fn cnv_prepare_right_tmp_bytes(_module: &Module<Self>, _res_size: usize, _a_size: usize) -> usize {
                0
            }
            fn cnv_prepare_right(
                module: &Module<Self>,
                res: &mut CnvPVecRBackendMut<'_, Self>,
                a: &VecZnxBackendRef<'_, Self>,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                assert_eq!(res.n(), a.n());
                assert_dense(a, "cnv_prepare");
                assert_eq!(res.cols(), a.cols());
                for j in 0..res.size() {
                    for col in 0..res.cols() {
                        let n = res.n();
                        let index = (j * res.cols() + col) * n;
                        let out = &mut res.raw_mut()[index..index + n];
                        out.fill(Default::default());
                        if j < a.size() {
                            Self::forward(module, out, a.at(col, j));
                        }
                    }
                }
            }
            fn cnv_apply_dft_tmp_bytes(
                _module: &Module<Self>,
                _cnv_offset: usize,
                _res_size: usize,
                _a_size: usize,
                _b_size: usize,
            ) -> usize {
                0
            }
            fn cnv_apply_dft(
                _module: &Module<Self>,
                cnv_offset: usize,
                res: &mut VecZnxDftBackendMut<'_, Self>,
                res_col: usize,
                a: &CnvPVecLBackendRef<'_, Self>,
                a_col: usize,
                b: &CnvPVecRBackendRef<'_, Self>,
                b_col: usize,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                assert_eq!(res.n(), a.n());
                assert_eq!(res.n(), b.n());
                let bound = a.size().saturating_add(b.size()).saturating_sub(1);
                for k in 0..res.size() {
                    let out = res.at_mut(res_col, k);
                    out.fill(Default::default());
                    if let Some(k) = k.checked_add(cnv_offset).filter(|&k| k < bound) {
                        for i in 0..a.size().min(k + 1) {
                            let j = k - i;
                            if j < b.size() {
                                Self::multiply_accumulate(out, a.at(a_col, i), b.at(b_col, j));
                            }
                        }
                    }
                }
            }
            fn cnv_by_const_apply_tmp_bytes(
                _module: &Module<Self>,
                _cnv_offset: usize,
                _res_size: usize,
                _a_size: usize,
                _b_size: usize,
            ) -> usize {
                0
            }
            fn cnv_by_const_apply(
                _module: &Module<Self>,
                cnv_offset: usize,
                res: &mut VecZnxBigBackendMut<'_, Self>,
                res_col: usize,
                a: &VecZnxBackendRef<'_, Self>,
                a_col: usize,
                b: &VecZnxBackendRef<'_, Self>,
                b_col: usize,
                b_coeff: usize,
                _scratch: &mut ScratchArena<'_, Self>,
            ) {
                assert_dense(res, "cnv_by_const_apply");
                assert_dense(a, "cnv_by_const_apply");
                assert_dense(b, "cnv_by_const_apply");
                assert_eq!(res.n(), a.n());
                let bound = a.size().saturating_add(b.size()).saturating_sub(1);
                for k in 0..res.size() {
                    let out = res.at_mut(res_col, k);
                    out.fill(0);
                    if let Some(k) = k.checked_add(cnv_offset).filter(|&k| k < bound) {
                        for i in 0..a.size().min(k + 1) {
                            let j = k - i;
                            if j < b.size() {
                                let scalar = b.at(b_col, j)[b_coeff] as $big;
                                for (r, &x) in out.iter_mut().zip(a.at(a_col, i)) {
                                    *r = r.wrapping_add((x as $big).wrapping_mul(scalar));
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}
impl_products!(FFT64Oracle, i64);
impl_products!(NTT4x30Oracle, i128);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::ntt4x30::primes::{PrimeSet, Primes30};

    #[test]
    fn ntt_products_reduce_each_prime_lane() {
        let minus_one = Primes30::Q.map(|q| u64::from(q - 1));
        let mut result = minus_one;
        NTT4x30Oracle::multiply_accumulate(
            bytemuck::cast_slice_mut(&mut result),
            bytemuck::cast_slice(&minus_one),
            bytemuck::cast_slice(&minus_one),
        );
        assert_eq!(result, [0; 4]);
        result = minus_one;
        NTT4x30Oracle::multiply_assign(bytemuck::cast_slice_mut(&mut result), bytemuck::cast_slice(&minus_one));
        assert_eq!(result, [1; 4]);
    }
}
