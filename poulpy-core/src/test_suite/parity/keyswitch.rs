//! Key-switch parity: the gadget digit loop, compared across backends.

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxAlloc, VecZnxDftAlloc, VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftCopy,
        VecZnxDftZero, VmpApplyDftToDft, VmpApplyDftToDftAccumulate, VmpApplyDftToDftAccumulateTmpBytes,
        VmpApplyDftToDftTmpBytes, VmpPMatAlloc, VmpPrepare, VmpPrepareTmpBytes,
    },
    layouts::{
        FillUniform, HostBytesBackend, HostDataMut, HostDataRef, MatZnx, MatZnxToBackendRef, Module, ScratchOwned, VecZnx,
        VecZnxDftToBackendMut, VecZnxDftToBackendRef, VecZnxToBackendRef, VmpPMatToBackendMut, VmpPMatToBackendRef,
    },
    source::Source,
    test_suite::{TestParams, upload_mat_znx, upload_vec_znx},
};

use crate::{
    GGLWEKeyswitch, GLWEKeyswitch,
    api::TransferInto,
    default::keyswitching::GGLWEProductDefault,
    layouts::{
        Base2K, Degree, Dnum, Dsize, GGLWELayout, GLWELayout, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
        gadget_product_limbs, prepared::GGLWEPreparedFactory,
    },
    layouts::{GGLWEPrepared, resolve_gglwe_key_use},
    oep::GGLWEProductDigitsStridedImpl,
    test_suite::parity::{ParityBackend, ParityShapes, ref_gglwe, ref_glwe},
};

/// Checks a backend's interleaved-digit product against the Core definition.
pub fn test_gglwe_product_digits_strided<BE>(module: &Module<BE>, base2k: usize)
where
    BE: poulpy_hal::test_suite::TestBackend + GGLWEProductDigitsStridedImpl<BE>,
    BE::OwnedBuf: HostDataMut,
    Module<BE>: VecZnxAlloc<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftCopy<BE>
        + VecZnxDftZero<BE>
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftAccumulate<BE>
        + VmpApplyDftToDftTmpBytes
        + VmpApplyDftToDftAccumulateTmpBytes
        + VmpPMatAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let mut source = Source::new([2u8; 32]);
    let cases: [(usize, usize, usize, usize); 6] = [
        (2, 1, 2, 4),
        (2, 2, 1, 5),
        (3, 1, 1, 7),
        (3, 2, 2, 2),
        (2, 1, 1, 1),
        (3, 2, 1, 8),
    ];

    for (dsize, cols_in, cols_out, a_size, sparse) in cases.into_iter().flat_map(|case| {
        let (dsize, cols_in, cols_out, a_size) = case;
        [
            (dsize, cols_in, cols_out, a_size, false),
            (dsize, cols_in, cols_out, a_size, true),
        ]
    }) {
        let rows = a_size.div_ceil(dsize);
        let size_out = a_size;
        let product_terms = module.n().saturating_mul(rows).saturating_mul(dsize).saturating_mul(cols_in);
        let product_limbs = gadget_product_limbs(Base2K(base2k as u32), product_terms);
        let default_tmp = crate::default::keyswitching::glwe::gglwe_product_digits_strided_tmp_bytes_default(
            module, size_out, cols_in, a_size, dsize, rows, cols_in, cols_out, size_out,
        );
        let backend_tmp = BE::gglwe_product_digits_strided_tmp_bytes(
            module, size_out, cols_in, a_size, dsize, rows, cols_in, cols_out, size_out,
        );
        let mut scratch = ScratchOwned::<BE>::alloc(
            default_tmp
                .max(backend_tmp)
                .max(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size_out)),
        );

        let mut a = module.vec_znx_alloc(cols_in, a_size);
        a.fill_uniform(base2k, &mut source);
        let mut a_dft = module.vec_znx_dft_alloc(cols_in, a_size);
        for col in 0..cols_in {
            let a = <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(&a);
            module.vec_znx_dft_apply(1, 0, &mut a_dft.to_backend_mut(), col, &a, col);
        }
        if sparse && a_size > 1 {
            let mut a_dft = a_dft.to_backend_mut();
            let mut prefix = a_dft.with_limb_range_mut(0, a_size - 1);
            for col in 0..cols_in {
                module.vec_znx_dft_zero(&mut prefix, col);
            }
        }

        let mut mat = module.mat_znx_alloc(rows, cols_in, cols_out, size_out);
        mat.fill_uniform(base2k, &mut source);
        let mut pmat = module.vmp_pmat_alloc(rows, cols_in, cols_out, size_out);
        let mat = <MatZnx<BE::OwnedBuf, i64> as MatZnxToBackendRef<BE>>::to_backend_ref(&mat);
        module.vmp_prepare(&mut pmat.to_backend_mut(), &mat, &mut scratch.borrow());

        let mut want = module.vec_znx_dft_alloc(cols_out, size_out);
        let sentinel = vec![1u8; BE::len_bytes(&want.data)];
        BE::copy_from_host(&mut want.data, &sentinel);
        crate::default::keyswitching::glwe::gglwe_product_digits_strided_default(
            module,
            &mut want.to_backend_mut(),
            &a_dft.to_backend_ref(),
            dsize,
            product_limbs,
            &pmat.to_backend_ref(),
            &mut scratch.borrow(),
        );

        let mut have = module.vec_znx_dft_alloc(cols_out, size_out);
        BE::copy_from_host(&mut have.data, &sentinel);
        BE::gglwe_product_digits_strided(
            module,
            &mut have.to_backend_mut(),
            &a_dft.to_backend_ref(),
            dsize,
            product_limbs,
            &pmat.to_backend_ref(),
            &mut scratch.borrow(),
        );

        let want = BE::to_host_bytes(&want.data);
        let have = BE::to_host_bytes(&have.data);
        assert_ne!(want, sentinel, "reference VMP did not overwrite the destination");
        assert_eq!(
            have, want,
            "strided VMP mismatch for dsize={dsize}, a_size={a_size}, sparse={sparse}"
        );
    }
}

/// The selected product on a fine parent equals the plain product on a key
/// built from exactly the selected rows and limb prefixes.
///
/// This is the coarsening oracle: it pins the row map `(i + 1) * s - 1`, the
/// `logical_work_size` limb bound, and the derived `k_aux`.
pub fn test_gglwe_product_dft_selected<BE>(module: &Module<BE>, base2k: usize)
where
    BE: poulpy_hal::test_suite::TestBackend<ZnxWord = i64>,
    Module<BE>: GGLWEProductDefault<BE>
        + VecZnxDftAlloc<BE>
        + VecZnxDftApply<BE>
        + VmpPMatAlloc<BE>
        + VmpPrepare<BE>
        + VmpPrepareTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let mut source = Source::new([3u8; 32]);
    let n: u32 = module.n() as u32;
    // Inputs and the oracle are built here and uploaded, so the test does not
    // require the backend's own buffers to be host-resident.
    let host: Module<HostBytesBackend> = Module::<HostBytesBackend>::new(module.n() as u64);
    // (parent dsize, parent dnum, coarsening factor, input limbs)
    let cases: [(u32, u32, u32, usize); 5] = [(1, 8, 2, 4), (1, 12, 4, 6), (2, 8, 2, 5), (2, 6, 3, 6), (4, 4, 2, 8)];

    for (dsize, dnum, s, input_size) in cases {
        let parent = GGLWELayout {
            n: Degree(n),
            base2k: Base2K(base2k as u32),
            dnum: Dnum(dnum),
            dsize: Dsize(dsize),
            k_aux: TorusPrecision(dsize * base2k as u32 + n.ilog2()),
            rank_in: Rank(1),
            rank_out: Rank(1),
        };
        let input_k = TorusPrecision(input_size as u32 * base2k as u32);
        let effective_dsize = Dsize(dsize * s);
        let use_ = *resolve_gglwe_key_use(&parent, input_k, effective_dsize)
            .expect("valid layout")
            .expect("parent realizes the coarsening")
            .active()
            .expect("positive precision");
        let logical = use_.logical_layout;
        let (cols_in, cols_out) = (parent.rank_in.as_usize(), (parent.rank_out + 1).as_usize());
        let (rows, size) = (parent.dnum.as_usize(), parent.max_size());
        let (sel_rows, sel_size) = (logical.dnum.as_usize(), use_.logical_work_size);
        // The oracle runs the same product over a dense key holding exactly the
        // selected rows, so its bound is that key used natively.
        let dense_use = *resolve_gglwe_key_use(&logical, input_k, logical.dsize)
            .expect("valid layout")
            .expect("a dense key realizes its own decomposition")
            .active()
            .expect("positive precision");

        let mut scratch = ScratchOwned::<BE>::alloc(
            module
                .gglwe_product_dft_tmp_bytes_default(sel_size, input_size, &use_)
                .max(module.gglwe_product_dft_tmp_bytes_default(sel_size, input_size, &dense_use))
                .max(module.vmp_prepare_tmp_bytes(rows, cols_in, cols_out, size)),
        );

        // Built on the host and uploaded, so a device backend runs this too.
        let mut a_host = host.vec_znx_alloc(cols_in, input_size);
        a_host.fill_uniform(base2k, &mut source);
        let a = upload_vec_znx::<BE>(&a_host);
        let mut a_dft = module.vec_znx_dft_alloc(cols_in, input_size);
        for col in 0..cols_in {
            let a = <VecZnx<BE::OwnedBuf, BE::ZnxWord> as VecZnxToBackendRef<BE>>::to_backend_ref(&a);
            module.vec_znx_dft_apply(1, 0, &mut a_dft.to_backend_mut(), col, &a, col);
        }

        let mut mat = host.mat_znx_alloc(rows, cols_in, cols_out, size);
        mat.fill_uniform(base2k, &mut source);

        // Oracle: the selected rows and limb prefixes, densely prepared.
        let (mat_ncols, sel_ncols) = (cols_out * size, cols_out * sel_size);
        let mut sel = host.mat_znx_alloc(sel_rows, cols_in, cols_out, sel_size);
        for i in 0..sel_rows {
            for c in 0..cols_in {
                let src_row: usize = ((i + 1) * s as usize - 1) * cols_in + c;
                let dst_row: usize = i * cols_in + c;
                for col in 0..sel_ncols {
                    let (src, dst) = (
                        module.n() * (src_row * mat_ncols + col),
                        module.n() * (dst_row * sel_ncols + col),
                    );
                    sel.raw_mut()[dst..dst + module.n()].copy_from_slice(&mat.raw()[src..src + module.n()]);
                }
            }
        }

        let mut parent_pmat = module.vmp_pmat_alloc(rows, cols_in, cols_out, size);
        module.vmp_prepare(
            &mut parent_pmat.to_backend_mut(),
            &<MatZnx<BE::OwnedBuf, i64> as MatZnxToBackendRef<BE>>::to_backend_ref(&upload_mat_znx::<BE>(&mat)),
            &mut scratch.borrow(),
        );
        let mut sel_pmat = module.vmp_pmat_alloc(sel_rows, cols_in, cols_out, sel_size);
        module.vmp_prepare(
            &mut sel_pmat.to_backend_mut(),
            &<MatZnx<BE::OwnedBuf, i64> as MatZnxToBackendRef<BE>>::to_backend_ref(&upload_mat_znx::<BE>(&sel)),
            &mut scratch.borrow(),
        );

        let parent_key = GGLWEPrepared {
            data: parent_pmat.to_backend_ref(),
            k_aux: parent.k_aux,
            base2k: parent.base2k,
            dsize: parent.dsize,
        };
        let sel_key = GGLWEPrepared {
            data: sel_pmat.to_backend_ref(),
            k_aux: logical.k_aux,
            base2k: logical.base2k,
            dsize: logical.dsize,
        };

        let mut want = module.vec_znx_dft_alloc(cols_out, sel_size);
        module.gglwe_product_dft_default(
            &mut want.to_backend_mut(),
            &a_dft.to_backend_ref(),
            &sel_key,
            &dense_use,
            1,
            &mut scratch.borrow(),
        );
        let mut have = module.vec_znx_dft_alloc(cols_out, sel_size);
        module.gglwe_product_dft_default(
            &mut have.to_backend_mut(),
            &a_dft.to_backend_ref(),
            &parent_key,
            &use_,
            1,
            &mut scratch.borrow(),
        );

        assert_eq!(
            BE::to_host_bytes(&have.data),
            BE::to_host_bytes(&want.data),
            "selected product mismatch for dsize={dsize} dnum={dnum} s={s} input_size={input_size}"
        );

        // A key whose stored shape disagrees with the bound it was resolved from
        // is rejected before dispatch. `dense_use` is contiguous, so nothing
        // extracts and no delegate check stands between this and the kernel:
        // without the guard the parent's extra rows would silently be read as
        // if they were the dense key's.
        let mismatched = GGLWEPrepared {
            data: parent_pmat.to_backend_ref(),
            k_aux: logical.k_aux,
            base2k: logical.base2k,
            dsize: logical.dsize,
        };
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut out = module.vec_znx_dft_alloc(cols_out, sel_size);
            module.gglwe_product_dft_default(
                &mut out.to_backend_mut(),
                &a_dft.to_backend_ref(),
                &mismatched,
                &dense_use,
                1,
                &mut scratch.borrow(),
            );
        }));
        std::panic::set_hook(hook);
        assert!(
            caught.is_err(),
            "a prepared key whose shape disagrees with its bound was accepted"
        );
    }
}

/// Key layout for a `rank_in -> rank_out` switch covering `k` bits of input.
fn key_layout(n: u32, base2k: usize, k: usize, dsize: usize, rank_in: usize, rank_out: usize) -> GGLWELayout {
    let dnum = k.div_ceil(base2k * dsize);
    GGLWELayout {
        n: Degree(n),
        base2k: Base2K(base2k as u32),
        dnum: Dnum(dnum as u32),
        dsize: Dsize(dsize as u32),
        k_aux: TorusPrecision((dsize * base2k) as u32),
        rank_in: Rank(rank_in as u32),
        rank_out: Rank(rank_out as u32),
    }
}

/// `glwe_keyswitch` agrees with the reference backend byte-for-byte.
///
/// Sweeps `dsize` and both ranks, so every branch of the digit loop is compared:
/// the `dsize == 1` short circuit, the `di == 0` overwriting pass, and the
/// narrowed accumulating passes above it.
pub fn test_glwe_keyswitch_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEKeyswitch<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: GLWEKeyswitch<BT> + GGLWEPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k_in = 4 * base2k + 1;
    let mut source = Source::new([7u8; 32]);

    for &rank_in in &shapes.ranks {
        for &rank_out in &shapes.ranks {
            for dsize in shapes.dsizes(k_in, base2k) {
                let a_infos = GLWELayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    k: TorusPrecision(k_in as u32),
                    rank: Rank(rank_in as u32),
                };
                let res_infos = GLWELayout {
                    n: Degree(n),
                    base2k: Base2K(base2k as u32),
                    k: TorusPrecision((k_in + base2k * dsize) as u32),
                    rank: Rank(rank_out as u32),
                };
                let key_infos = key_layout(n, base2k, k_in, dsize, rank_in, rank_out);

                let a_ref = ref_glwe(module_ref, &a_infos, &mut source);
                let key_ref_coeffs = ref_gglwe(module_ref, &key_infos, &mut source);

                let mut a_test = module_test.glwe_alloc_from_infos(&a_infos);
                a_ref.transfer_into(&mut a_test);
                let mut key_test_coeffs = module_test.gglwe_alloc_from_infos(&key_infos);
                key_ref_coeffs.transfer_into(&mut key_test_coeffs);

                let mut res_ref = module_ref.glwe_alloc_from_infos(&res_infos);
                let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);

                let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                    module_ref
                        .gglwe_prepare_tmp_bytes(&key_infos)
                        .max(module_ref.glwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
                );
                let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                    module_test
                        .gglwe_prepare_tmp_bytes(&key_infos)
                        .max(module_test.glwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
                );

                let mut key_ref = module_ref.gglwe_prepared_alloc_from_infos(&key_infos);
                module_ref.gglwe_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
                let mut key_test = module_test.gglwe_prepared_alloc_from_infos(&key_infos);
                module_test.gglwe_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

                module_ref.glwe_keyswitch(&mut res_ref, &a_ref, &key_ref, &mut scratch_ref.borrow());
                module_test.glwe_keyswitch(&mut res_test, &a_test, &key_test, &mut scratch_test.borrow());

                let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
                res_test.transfer_into(&mut have);
                assert_eq!(
                    res_ref, have,
                    "glwe_keyswitch: rank_in={rank_in} rank_out={rank_out} dsize={dsize} k_in={k_in}"
                );
            }
        }
    }
}

/// `glwe_keyswitch_assign` agrees with the reference backend byte-for-byte.
pub fn test_glwe_keyswitch_assign_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GLWEKeyswitch<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: GLWEKeyswitch<BT> + GGLWEPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([11u8; 32]);

    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(k, base2k) {
            let res_infos = GLWELayout {
                n: Degree(n),
                base2k: Base2K(base2k as u32),
                k: TorusPrecision(k as u32),
                rank: Rank(rank as u32),
            };
            let key_infos = key_layout(n, base2k, k, dsize, rank, rank);

            let mut res_ref = ref_glwe(module_ref, &res_infos, &mut source);
            let key_ref_coeffs = ref_gglwe(module_ref, &key_infos, &mut source);

            let mut res_test = module_test.glwe_alloc_from_infos(&res_infos);
            res_ref.transfer_into(&mut res_test);
            let mut key_test_coeffs = module_test.gglwe_alloc_from_infos(&key_infos);
            key_ref_coeffs.transfer_into(&mut key_test_coeffs);

            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                module_ref
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_ref.glwe_keyswitch_tmp_bytes(&res_infos, &res_infos, &key_infos)),
            );
            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                module_test
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_test.glwe_keyswitch_tmp_bytes(&res_infos, &res_infos, &key_infos)),
            );

            let mut key_ref = module_ref.gglwe_prepared_alloc_from_infos(&key_infos);
            module_ref.gglwe_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
            let mut key_test = module_test.gglwe_prepared_alloc_from_infos(&key_infos);
            module_test.gglwe_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

            module_ref.glwe_keyswitch_assign(&mut res_ref, &key_ref, &mut scratch_ref.borrow());
            module_test.glwe_keyswitch_assign(&mut res_test, &key_test, &mut scratch_test.borrow());

            let mut have = module_ref.glwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, have, "glwe_keyswitch_assign: rank={rank} dsize={dsize} k={k}");
        }
    }
}

/// `gglwe_keyswitch` agrees with the reference backend byte-for-byte.
pub fn test_gglwe_keyswitch_parity<BR, BT>(
    params: &TestParams,
    shapes: &ParityShapes,
    module_ref: &Module<BR>,
    module_test: &Module<BT>,
) where
    BR: ParityBackend,
    BT: ParityBackend,
    BR::OwnedBuf: HostDataMut,
    Module<BR>: GGLWEKeyswitch<BR> + GGLWEPreparedFactory<BR>,
    Module<BT>: GGLWEKeyswitch<BT> + GGLWEPreparedFactory<BT>,
    ScratchOwned<BR>: ScratchOwnedAlloc<BR> + ScratchOwnedBorrow<BR>,
    ScratchOwned<BT>: ScratchOwnedAlloc<BT> + ScratchOwnedBorrow<BT>,
{
    assert_eq!(module_ref.n(), module_test.n());

    let n = module_ref.n() as u32;
    let base2k = params.base2k;
    let k = 4 * base2k + 1;
    let mut source = Source::new([23u8; 32]);

    for &rank in &shapes.ranks {
        for dsize in shapes.dsizes(k, base2k) {
            let a_infos = key_layout(n, base2k, k, 1, rank, rank);
            let res_infos = key_layout(n, base2k, k, 1, rank, rank);
            let key_infos = key_layout(n, base2k, k, dsize, rank, rank);

            let a_ref = ref_gglwe(module_ref, &a_infos, &mut source);
            let key_ref_coeffs = ref_gglwe(module_ref, &key_infos, &mut source);

            let mut a_test = module_test.gglwe_alloc_from_infos(&a_infos);
            a_ref.transfer_into(&mut a_test);
            let mut key_test_coeffs = module_test.gglwe_alloc_from_infos(&key_infos);
            key_ref_coeffs.transfer_into(&mut key_test_coeffs);

            let mut res_ref = module_ref.gglwe_alloc_from_infos(&res_infos);
            let mut res_test = module_test.gglwe_alloc_from_infos(&res_infos);

            let mut scratch_ref: ScratchOwned<BR> = ScratchOwned::alloc(
                module_ref
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_ref.gglwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
            );
            let mut scratch_test: ScratchOwned<BT> = ScratchOwned::alloc(
                module_test
                    .gglwe_prepare_tmp_bytes(&key_infos)
                    .max(module_test.gglwe_keyswitch_tmp_bytes(&res_infos, &a_infos, &key_infos)),
            );

            let mut key_ref = module_ref.gglwe_prepared_alloc_from_infos(&key_infos);
            module_ref.gglwe_prepare(&mut key_ref, &key_ref_coeffs, &mut scratch_ref.borrow());
            let mut key_test = module_test.gglwe_prepared_alloc_from_infos(&key_infos);
            module_test.gglwe_prepare(&mut key_test, &key_test_coeffs, &mut scratch_test.borrow());

            module_ref.gglwe_keyswitch(&mut res_ref, &a_ref, &key_ref, &mut scratch_ref.borrow());
            module_test.gglwe_keyswitch(&mut res_test, &a_test, &key_test, &mut scratch_test.borrow());

            let mut have = module_ref.gglwe_alloc_from_infos(&res_infos);
            res_test.transfer_into(&mut have);
            assert_eq!(res_ref, have, "gglwe_keyswitch: rank={rank} dsize={dsize} k={k}");
        }
    }
}
