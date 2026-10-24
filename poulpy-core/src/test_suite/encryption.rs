use std::panic::{AssertUnwindSafe, catch_unwind};

use poulpy_hal::{
    api::{ModuleNew, ScratchOwnedAlloc},
    layouts::{Backend, Module, ScratchOwned},
    source::Source,
};

use super::decryption::panic_message;
use crate::{GLWEEncryptSk, api::GLWESecretPreparedFactory, layouts::ModuleCoreAlloc};

/// Secret-key encryption rejects mismatched operand degrees and degrees above the module in all builds.
pub fn test_glwe_encrypt_degree_mismatch<BE: Backend>()
where
    Module<BE>: ModuleNew<BE> + GLWEEncryptSk<BE> + GLWESecretPreparedFactory<BE>,
{
    let n = 2 * BE::MIN_DEGREE.max(16);
    let module = Module::<BE>::new(n as u64);
    let mut scratch = ScratchOwned::<BE>::alloc(0);
    let mut xe = Source::new([1; 32]);
    let mut xa = Source::new([2; 32]);

    for other_n in [n / 2, n * 2] {
        let other = Module::<BE>::new(other_n as u64);
        for (ct_module, pt_module, sk_module, message) in [
            (&other, &module, &module, "operand degrees differ"),
            (&module, &other, &module, "operand degrees differ"),
            (&module, &module, &other, "operand degrees differ"),
            (&other, &other, &other, "operand degree exceeds the module degree"),
        ]
        .into_iter()
        // Operands sharing a smaller degree are valid.
        .take(if other_n > n { 4 } else { 3 })
        {
            let mut ct = ct_module.glwe_alloc(8usize.into(), 8usize.into(), 1usize.into());
            let pt = pt_module.glwe_plaintext_alloc(8usize.into(), 8usize.into());
            let sk = sk_module.glwe_secret_prepared_alloc(1usize.into());
            let err = catch_unwind(AssertUnwindSafe(|| {
                module.glwe_encrypt_sk(&mut ct, &pt, &sk, &mut xe, &mut xa, &mut scratch.arena());
            }))
            .expect_err("encryption accepted mismatched degrees");
            assert!(panic_message(&*err).contains(message));

            if ct_module.n() != n || sk_module.n() != n {
                let err = catch_unwind(AssertUnwindSafe(|| {
                    module.glwe_encrypt_zero_sk(&mut ct, &sk, &mut xe, &mut xa, &mut scratch.arena());
                }))
                .expect_err("zero encryption accepted mismatched degrees");
                assert!(panic_message(&*err).contains(message));
            }
        }
    }
}

/// The plaintext of gadget row `row` lands `pt` on bit `(row + 1) * k_digit`
/// for digit widths on and off the limb boundaries.
///
/// It must equal, bit for bit, an independent placement: the scalar added into
/// the limb holding that bit, normalized at the key radix and shifted up to the
/// bit, which for `k_digit = dsize * base2k` is the old limb placement. Where
/// the vector fits an `i128`, the recomposed value is also checked exactly.
/// The selected column is fully overwritten and the other is untouched.
pub fn test_gadget_row_plaintext<BE: crate::test_suite::noise::TestBackend>(
    params: &poulpy_hal::test_suite::TestParams,
    module: &Module<BE>,
) where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + poulpy_hal::api::ScratchOwnedBorrow<BE>,
{
    use poulpy_hal::{
        api::{ScratchOwnedBorrow, VecZnxAddScalarAssign, VecZnxNormalize, VecZnxNormalizeTmpBytes},
        layouts::{ScalarZnx, ZnxView, ZnxViewMut, vec_znx_alloc_zeroed, vec_znx_backend_mut, vec_znx_backend_ref},
        test_suite::scalar_znx_backend_ref,
    };

    let n: usize = module.n();
    let b: usize = params.base2k;
    let mut source = Source::new([5; 32]);
    // Column 0 holds small digits, column 1 digits near the normalization domain.
    let mut pt: ScalarZnx<BE::OwnedBuf, BE::ZnxWord> = module.scalar_znx_alloc(n, 2);
    for x in pt.at_mut(0, 0) {
        *x = source.next_i64().rem_euclid(7) - 3;
    }
    for x in pt.at_mut(1, 0) {
        *x = source.next_i64() >> 2;
    }
    let mut scratch = ScratchOwned::<BE>::alloc(module.vec_znx_normalize_tmp_bytes());
    let dnum: usize = 3;
    for k_digit in [1, b - 1, b, b + 1, 2 * b, 2 * b + 3, 3 * b - 2] {
        // Rows without an offset, and rows an unaligned and a one-limb-plus offset lower.
        for (extra, k_offset) in [0, 1, b / 2 + 1].into_iter().flat_map(|e| [(e, 0), (e, 3), (e, b + 2)]) {
            let k: usize = k_offset + dnum * k_digit + extra;
            let size: usize = k.div_ceil(b);
            for (row, pt_col, res_col) in (0..dnum).flat_map(|r| [(r, 0, 0), (r, 1, 1), (r, 0, 1)]) {
                let position: usize = k_offset + (row + 1) * k_digit;
                let garbage = |v: &mut poulpy_hal::layouts::VecZnx<BE::OwnedBuf, BE::ZnxWord>, seed: u8| {
                    let mut s = Source::new([seed; 32]);
                    for col in 0..2 {
                        for j in 0..size {
                            v.at_mut(col, j).iter_mut().for_each(|x| *x = s.next_i64());
                        }
                    }
                };
                let mut res = vec_znx_alloc_zeroed::<BE>(n, 2, size);
                garbage(&mut res, 9);
                crate::gadget::gadget_row_plaintext(
                    module,
                    &mut vec_znx_backend_mut::<BE>(&mut res),
                    res_col,
                    b,
                    k,
                    row,
                    k_digit,
                    k_offset,
                    &scalar_znx_backend_ref::<BE>(&pt),
                    pt_col,
                    &mut scratch.borrow(),
                );

                let limb: usize = position.div_ceil(b) - 1;
                let mut placed = vec_znx_alloc_zeroed::<BE>(n, 1, limb + 1);
                module.vec_znx_add_scalar_assign(
                    &mut vec_znx_backend_mut::<BE>(&mut placed),
                    0,
                    limb,
                    &scalar_znx_backend_ref::<BE>(&pt),
                    pt_col,
                );
                let mut want = vec_znx_alloc_zeroed::<BE>(n, 2, size);
                garbage(&mut want, 9);
                module.vec_znx_normalize(
                    &mut vec_znx_backend_mut::<BE>(&mut want),
                    b,
                    k,
                    ((limb + 1) * b - position) as i64,
                    res_col,
                    &vec_znx_backend_ref::<BE>(&placed),
                    b,
                    0,
                    &mut scratch.borrow(),
                );
                assert!(
                    res == want,
                    "gadget row placement differs: base2k={b} k_digit={k_digit} k_offset={k_offset} k={k} row={row} pt_col={pt_col} res_col={res_col}"
                );

                if pt_col == 0 && size * b <= 120 {
                    let width: u32 = (size * b) as u32;
                    for i in 0..n {
                        let value: i128 = (0..size)
                            .map(|j| (res.at(res_col, j)[i] as i128) << (width as usize - (j + 1) * b))
                            .sum();
                        let exact: i128 = (pt.at(0, 0)[i] as i128) << (width as usize - position);
                        assert_eq!(
                            (value - exact).rem_euclid(1i128 << width),
                            0,
                            "gadget row value differs: base2k={b} k_digit={k_digit} k={k} row={row} coeff={i}"
                        );
                    }
                }
            }
        }
    }
}
