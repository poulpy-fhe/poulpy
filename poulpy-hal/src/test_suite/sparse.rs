//! Sparse operands: a degree-`n` operand, `n` a power-of-two divisor of
//! the module degree `N`, stands for its ring embedding `switch_ring_{n->N}`.
//! Every sparse-capable slot must give the result the embedded dense operand
//! gives, bit for bit in the coefficient domain and after normalization in the
//! big domain.

use crate::{
    api::{
        ModuleN, ScratchOwnedAlloc, VecZnxAdd, VecZnxAddAssign, VecZnxBigAdd, VecZnxBigAddAssign, VecZnxBigAddSmall,
        VecZnxBigAddSmallAssign, VecZnxBigAlloc, VecZnxBigFromSmall, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes,
        VecZnxBigSub, VecZnxBigSubAssign, VecZnxBigSubNegateAssign, VecZnxBigSubSmallA, VecZnxBigSubSmallAssign,
        VecZnxBigSubSmallB, VecZnxBigSubSmallNegateAssign, VecZnxSub, VecZnxSubAssign, VecZnxSubNegateAssign, VecZnxSwitchRing,
    },
    layouts::{
        FillUniform, Module, ScratchOwned, VecZnx, VecZnxBackendMut, VecZnxBackendRef, VecZnxBig, VecZnxBigBackendMut,
        VecZnxBigBackendRef, VecZnxBigOwned, VecZnxBigToBackendMut, VecZnxBigToBackendRef, VecZnxOwned,
    },
    source::Source,
    test_suite::{TestBackend, TestParams, download_vec_znx, upload_vec_znx, vec_znx_backend_mut, vec_znx_backend_ref},
};

/// The sparse degrees exercised under a module of degree `n`: `n/2` down to
/// `n/16`, never below 8 (the CPU floor).
fn sparse_degrees(n: usize) -> Vec<usize> {
    (1..=4).map(|g| n >> g).filter(|&d| d >= 8).collect()
}

/// The (a_size, b_size, res_size) triples: overlap, longer operand, shorter destination.
const SIZES: [(usize, usize, usize); 3] = [(2, 3, 4), (3, 2, 2), (1, 4, 3)];

/// `switch_ring_{a.n() -> n}(a)` on the backend; a copy when `a` is dense.
fn embed<BE: TestBackend>(module: &Module<BE>, n: usize, a: &VecZnxOwned<i64>) -> VecZnx<BE::OwnedBuf, i64>
where
    Module<BE>: ModuleN + VecZnxSwitchRing<BE>,
{
    let a_be = upload_vec_znx::<BE>(a);
    let mut res = upload_vec_znx::<BE>(&VecZnx::alloc(n, a.cols(), a.size()));
    for col in 0..a.cols() {
        module.vec_znx_switch_ring(
            &mut vec_znx_backend_mut::<BE>(&mut res),
            col,
            &vec_znx_backend_ref::<BE>(&a_be),
            col,
        );
    }
    res
}

fn random_host(n: usize, cols: usize, size: usize, base2k: usize, source: &mut Source) -> VecZnxOwned<i64> {
    let mut a = VecZnx::alloc(n, cols, size);
    a.fill_uniform(base2k, source);
    a
}

/// Runs `op(res, a, b)` once with the embedded dense operands and once with the
/// operands as given, and asserts the two results equal.
fn check_binary<BE: TestBackend>(
    module: &Module<BE>,
    n: usize,
    what: &str,
    res_size: usize,
    a: &VecZnxOwned<i64>,
    b: &VecZnxOwned<i64>,
    op: impl Fn(&mut VecZnxBackendMut<'_, BE>, usize, &VecZnxBackendRef<'_, BE>, usize, &VecZnxBackendRef<'_, BE>, usize),
) where
    Module<BE>: ModuleN + VecZnxSwitchRing<BE>,
{
    let cols = a.cols();
    let (a_dense, b_dense) = (embed(module, n, a), embed(module, n, b));
    let (a_be, b_be) = (upload_vec_znx::<BE>(a), upload_vec_znx::<BE>(b));
    let mut want = upload_vec_znx::<BE>(&VecZnx::alloc(n, cols, res_size));
    let mut have = upload_vec_znx::<BE>(&VecZnx::alloc(n, cols, res_size));
    for col in 0..cols {
        op(
            &mut vec_znx_backend_mut::<BE>(&mut want),
            col,
            &vec_znx_backend_ref::<BE>(&a_dense),
            col,
            &vec_znx_backend_ref::<BE>(&b_dense),
            col,
        );
        op(
            &mut vec_znx_backend_mut::<BE>(&mut have),
            col,
            &vec_znx_backend_ref::<BE>(&a_be),
            col,
            &vec_znx_backend_ref::<BE>(&b_be),
            col,
        );
    }
    assert_eq!(
        download_vec_znx::<BE>(&want),
        download_vec_znx::<BE>(&have),
        "{what}: a.n()={} b.n()={} sizes=({}, {}, {res_size})",
        a.n(),
        b.n(),
        a.size(),
        b.size()
    );
}

/// In-place forms: `res` (degree `n`, random) receives `a` once embedded and
/// once as given; the two destinations must agree.
#[allow(clippy::too_many_arguments)]
fn check_assign<BE: TestBackend>(
    module: &Module<BE>,
    n: usize,
    what: &str,
    res_size: usize,
    a: &VecZnxOwned<i64>,
    base2k: usize,
    source: &mut Source,
    op: impl Fn(&mut VecZnxBackendMut<'_, BE>, usize, &VecZnxBackendRef<'_, BE>, usize),
) where
    Module<BE>: ModuleN + VecZnxSwitchRing<BE>,
{
    let cols = a.cols();
    let seed = random_host(n, cols, res_size, base2k, source);
    let a_dense = embed(module, n, a);
    let a_be = upload_vec_znx::<BE>(a);
    let mut want = upload_vec_znx::<BE>(&seed);
    let mut have = upload_vec_znx::<BE>(&seed);
    for col in 0..cols {
        op(
            &mut vec_znx_backend_mut::<BE>(&mut want),
            col,
            &vec_znx_backend_ref::<BE>(&a_dense),
            col,
        );
        op(
            &mut vec_znx_backend_mut::<BE>(&mut have),
            col,
            &vec_znx_backend_ref::<BE>(&a_be),
            col,
        );
    }
    assert_eq!(
        download_vec_znx::<BE>(&want),
        download_vec_znx::<BE>(&have),
        "{what}: a.n()={} a.size()={} res_size={res_size}",
        a.n(),
        a.size()
    );
}

/// `vec_znx_add`, `sub` and their in-place forms with a degree-`n` operand in
/// each sparse-capable slot equal the same operation on the embedded operand.
pub fn test_vec_znx_sparse_add_sub<BE: TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxSwitchRing<BE>
        + VecZnxAdd<BE>
        + VecZnxAddAssign<BE>
        + VecZnxSub<BE>
        + VecZnxSubAssign<BE>
        + VecZnxSubNegateAssign<BE>,
{
    let n = params.n;
    let base2k = params.base2k;
    let cols = 2;
    let mut source = Source::new([7u8; 32]);
    for sparse_n in sparse_degrees(n) {
        for (a_size, b_size, res_size) in SIZES {
            for (a_n, b_n) in [(sparse_n, n), (n, sparse_n), (sparse_n, sparse_n), (sparse_n, sparse_n / 2)] {
                if b_n < 8 {
                    continue;
                }
                let a = random_host(a_n, cols, a_size, base2k, &mut source);
                let b = random_host(b_n, cols, b_size, base2k, &mut source);
                check_binary(module, n, "vec_znx_add", res_size, &a, &b, |r, rc, x, xc, y, yc| {
                    module.vec_znx_add(r, rc, x, xc, y, yc)
                });
                check_binary(module, n, "vec_znx_sub", res_size, &a, &b, |r, rc, x, xc, y, yc| {
                    module.vec_znx_sub(r, rc, x, xc, y, yc)
                });
            }
            let a = random_host(sparse_n, cols, a_size, base2k, &mut source);
            check_assign(
                module,
                n,
                "vec_znx_add_assign",
                res_size,
                &a,
                base2k,
                &mut source,
                |r, rc, x, xc| module.vec_znx_add_assign(r, rc, x, xc),
            );
            check_assign(
                module,
                n,
                "vec_znx_sub_assign",
                res_size,
                &a,
                base2k,
                &mut source,
                |r, rc, x, xc| module.vec_znx_sub_assign(r, rc, x, xc),
            );
            check_assign(
                module,
                n,
                "vec_znx_sub_negate_assign",
                res_size,
                &a,
                base2k,
                &mut source,
                |r, rc, x, xc| module.vec_znx_sub_negate_assign(r, rc, x, xc),
            );
        }
    }

    // A degree above the destination's is not an embedding and is rejected.
    let too_big = random_host(n << 1, cols, 1, base2k, &mut source);
    let too_big_be = upload_vec_znx::<BE>(&too_big);
    let mut res = upload_vec_znx::<BE>(&VecZnx::alloc(n, cols, 1));
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        module.vec_znx_add_assign(
            &mut vec_znx_backend_mut::<BE>(&mut res),
            0,
            &vec_znx_backend_ref::<BE>(&too_big_be),
            0,
        );
    }))
    .is_err();
    assert!(panicked, "vec_znx_add_assign accepted an operand of degree 2N");
}

/// A degree-`a.n()` `VecZnxBig` holding `a`, through `vec_znx_big_from_small`
/// at that degree.
fn big_at_own_degree<BE: TestBackend>(module: &Module<BE>, a: &VecZnxOwned<i64>) -> VecZnxBigOwned<BE>
where
    Module<BE>: VecZnxBigFromSmall<BE>,
{
    let a_be = upload_vec_znx::<BE>(a);
    let mut big: VecZnxBigOwned<BE> = VecZnxBig::alloc(a.n(), a.cols(), a.size());
    for col in 0..a.cols() {
        module.vec_znx_big_from_small(&mut big.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&a_be), col);
    }
    big
}

/// The embedded operand as a degree-`n` `VecZnxBig`.
fn big_embedded<BE: TestBackend>(module: &Module<BE>, n: usize, a: &VecZnxOwned<i64>) -> VecZnxBigOwned<BE>
where
    Module<BE>: ModuleN + VecZnxSwitchRing<BE> + VecZnxBigAlloc<BE> + VecZnxBigFromSmall<BE>,
{
    let dense = embed(module, n, a);
    let mut big: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(n, a.cols(), a.size());
    for col in 0..a.cols() {
        module.vec_znx_big_from_small(&mut big.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&dense), col);
    }
    big
}

fn normalized<BE: TestBackend>(
    module: &Module<BE>,
    base2k: usize,
    big: &VecZnxBigOwned<BE>,
    scratch: &mut ScratchOwned<BE>,
) -> VecZnxOwned<i64>
where
    Module<BE>: ModuleN + VecZnxBigNormalize<BE>,
{
    let size = big.size();
    let mut out = upload_vec_znx::<BE>(&VecZnx::alloc(big.n(), big.cols(), size));
    for col in 0..big.cols() {
        module.vec_znx_big_normalize(
            &mut vec_znx_backend_mut::<BE>(&mut out),
            base2k,
            size * base2k,
            0,
            col,
            &big.to_backend_ref(),
            base2k,
            col,
            &mut scratch.arena(),
        );
    }
    download_vec_znx::<BE>(&out)
}

type BigBin<'m, BE> =
    &'m dyn Fn(&mut VecZnxBigBackendMut<'_, BE>, usize, &VecZnxBigBackendRef<'_, BE>, usize, &VecZnxBigBackendRef<'_, BE>, usize);
type BigAssign<'m, BE> = &'m dyn Fn(&mut VecZnxBigBackendMut<'_, BE>, usize, &VecZnxBigBackendRef<'_, BE>, usize);
type SmallAssign<'m, BE> = &'m dyn Fn(&mut VecZnxBigBackendMut<'_, BE>, usize, &VecZnxBackendRef<'_, BE>, usize);

/// The `VecZnxBig` add and sub families with a degree-`n` operand in each
/// sparse-capable slot, Big or small, equal the same operation on the embedded
/// operand after normalization.
pub fn test_vec_znx_big_sparse_add_sub<BE: TestBackend>(params: &TestParams, module: &Module<BE>)
where
    Module<BE>: ModuleN
        + VecZnxSwitchRing<BE>
        + VecZnxBigAlloc<BE>
        + VecZnxBigFromSmall<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigAdd<BE>
        + VecZnxBigAddAssign<BE>
        + VecZnxBigAddSmall<BE>
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigSub<BE>
        + VecZnxBigSubAssign<BE>
        + VecZnxBigSubNegateAssign<BE>
        + VecZnxBigSubSmallA<BE>
        + VecZnxBigSubSmallAssign<BE>
        + VecZnxBigSubSmallB<BE>
        + VecZnxBigSubSmallNegateAssign<BE>,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE>,
{
    let n = params.n;
    let base2k = params.base2k;
    let cols = 2;
    let mut source = Source::new([9u8; 32]);
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(module.vec_znx_big_normalize_tmp_bytes());

    let big_bin: [(&str, BigBin<'_, BE>); 2] = [
        ("vec_znx_big_add", &|r, rc, x, xc, y, yc| {
            module.vec_znx_big_add(r, rc, x, xc, y, yc)
        }),
        ("vec_znx_big_sub", &|r, rc, x, xc, y, yc| {
            module.vec_znx_big_sub(r, rc, x, xc, y, yc)
        }),
    ];
    let big_assign: [(&str, BigAssign<'_, BE>); 3] = [
        ("vec_znx_big_add_assign", &|r, rc, x, xc| {
            module.vec_znx_big_add_assign(r, rc, x, xc)
        }),
        ("vec_znx_big_sub_assign", &|r, rc, x, xc| {
            module.vec_znx_big_sub_assign(r, rc, x, xc)
        }),
        ("vec_znx_big_sub_negate_assign", &|r, rc, x, xc| {
            module.vec_znx_big_sub_negate_assign(r, rc, x, xc)
        }),
    ];
    let small_assign: [(&str, SmallAssign<'_, BE>); 3] = [
        ("vec_znx_big_add_small_assign", &|r, rc, x, xc| {
            module.vec_znx_big_add_small_assign(r, rc, x, xc)
        }),
        ("vec_znx_big_sub_small_assign", &|r, rc, x, xc| {
            module.vec_znx_big_sub_small_assign(r, rc, x, xc)
        }),
        ("vec_znx_big_sub_small_negate_assign", &|r, rc, x, xc| {
            module.vec_znx_big_sub_small_negate_assign(r, rc, x, xc)
        }),
    ];

    for sparse_n in sparse_degrees(n) {
        for (a_size, b_size, res_size) in SIZES {
            for (a_n, b_n) in [(sparse_n, n), (n, sparse_n)] {
                let a = random_host(a_n, cols, a_size, base2k, &mut source);
                let b = random_host(b_n, cols, b_size, base2k, &mut source);
                let (a_dense, b_dense) = (big_embedded(module, n, &a), big_embedded(module, n, &b));
                let (a_own, b_own) = (big_at_own_degree(module, &a), big_at_own_degree(module, &b));
                let (a_small, b_small) = (upload_vec_znx::<BE>(&a), upload_vec_znx::<BE>(&b));
                let (a_small_dense, b_small_dense) = (embed(module, n, &a), embed(module, n, &b));

                for (what, op) in big_bin.iter() {
                    let mut want: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, cols, res_size);
                    let mut have: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, cols, res_size);
                    for col in 0..cols {
                        op(
                            &mut want.to_backend_mut(),
                            col,
                            &a_dense.to_backend_ref(),
                            col,
                            &b_dense.to_backend_ref(),
                            col,
                        );
                        op(
                            &mut have.to_backend_mut(),
                            col,
                            &a_own.to_backend_ref(),
                            col,
                            &b_own.to_backend_ref(),
                            col,
                        );
                    }
                    assert_eq!(
                        normalized(module, base2k, &want, &mut scratch),
                        normalized(module, base2k, &have, &mut scratch),
                        "{what}: a.n()={a_n} b.n()={b_n} sizes=({a_size}, {b_size}, {res_size})"
                    );
                }

                // Big with a small operand: add_small and sub_small_b take the small `b`, sub_small_a the small `a`.
                let mut want: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, cols, res_size);
                let mut have: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, cols, res_size);
                for col in 0..cols {
                    module.vec_znx_big_add_small(
                        &mut want.to_backend_mut(),
                        col,
                        &a_dense.to_backend_ref(),
                        col,
                        &vec_znx_backend_ref::<BE>(&b_small_dense),
                        col,
                    );
                    module.vec_znx_big_add_small(
                        &mut have.to_backend_mut(),
                        col,
                        &a_own.to_backend_ref(),
                        col,
                        &vec_znx_backend_ref::<BE>(&b_small),
                        col,
                    );
                }
                assert_eq!(
                    normalized(module, base2k, &want, &mut scratch),
                    normalized(module, base2k, &have, &mut scratch),
                    "vec_znx_big_add_small: a.n()={a_n} b.n()={b_n}"
                );
                for col in 0..cols {
                    module.vec_znx_big_sub_small_a(
                        &mut want.to_backend_mut(),
                        col,
                        &vec_znx_backend_ref::<BE>(&a_small_dense),
                        col,
                        &b_dense.to_backend_ref(),
                        col,
                    );
                    module.vec_znx_big_sub_small_a(
                        &mut have.to_backend_mut(),
                        col,
                        &vec_znx_backend_ref::<BE>(&a_small),
                        col,
                        &b_own.to_backend_ref(),
                        col,
                    );
                }
                assert_eq!(
                    normalized(module, base2k, &want, &mut scratch),
                    normalized(module, base2k, &have, &mut scratch),
                    "vec_znx_big_sub_small_a: a.n()={a_n} b.n()={b_n}"
                );
                for col in 0..cols {
                    module.vec_znx_big_sub_small_b(
                        &mut want.to_backend_mut(),
                        col,
                        &a_dense.to_backend_ref(),
                        col,
                        &vec_znx_backend_ref::<BE>(&b_small_dense),
                        col,
                    );
                    module.vec_znx_big_sub_small_b(
                        &mut have.to_backend_mut(),
                        col,
                        &a_own.to_backend_ref(),
                        col,
                        &vec_znx_backend_ref::<BE>(&b_small),
                        col,
                    );
                }
                assert_eq!(
                    normalized(module, base2k, &want, &mut scratch),
                    normalized(module, base2k, &have, &mut scratch),
                    "vec_znx_big_sub_small_b: a.n()={a_n} b.n()={b_n}"
                );
            }

            // In-place forms: the destination is a random degree-N Big, the operand is sparse.
            let a = random_host(sparse_n, cols, a_size, base2k, &mut source);
            let seed = random_host(n, cols, res_size, base2k, &mut source);
            let a_dense = big_embedded(module, n, &a);
            let a_own = big_at_own_degree(module, &a);
            for (what, op) in big_assign.iter() {
                let mut want = big_embedded(module, n, &seed);
                let mut have = big_embedded(module, n, &seed);
                for col in 0..cols {
                    op(&mut want.to_backend_mut(), col, &a_dense.to_backend_ref(), col);
                    op(&mut have.to_backend_mut(), col, &a_own.to_backend_ref(), col);
                }
                assert_eq!(
                    normalized(module, base2k, &want, &mut scratch),
                    normalized(module, base2k, &have, &mut scratch),
                    "{what}: a.n()={sparse_n} sizes=({a_size}, {res_size})"
                );
            }
            let a_small = upload_vec_znx::<BE>(&a);
            let a_small_dense = embed(module, n, &a);

            // The basis promotion is a sparse-capable slot too: the degree-n
            // operand and its embedding promote to the same Big.
            let mut want: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, cols, res_size);
            let mut have: VecZnxBigOwned<BE> = module.vec_znx_big_alloc(params.n, cols, res_size);
            for col in 0..cols {
                module.vec_znx_big_from_small(
                    &mut want.to_backend_mut(),
                    col,
                    &vec_znx_backend_ref::<BE>(&a_small_dense),
                    col,
                );
                module.vec_znx_big_from_small(&mut have.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&a_small), col);
            }
            assert_eq!(
                normalized(module, base2k, &want, &mut scratch),
                normalized(module, base2k, &have, &mut scratch),
                "vec_znx_big_from_small: a.n()={sparse_n} sizes=({a_size}, {res_size})"
            );
            for (what, op) in small_assign.iter() {
                let mut want = big_embedded(module, n, &seed);
                let mut have = big_embedded(module, n, &seed);
                for col in 0..cols {
                    op(
                        &mut want.to_backend_mut(),
                        col,
                        &vec_znx_backend_ref::<BE>(&a_small_dense),
                        col,
                    );
                    op(&mut have.to_backend_mut(), col, &vec_znx_backend_ref::<BE>(&a_small), col);
                }
                assert_eq!(
                    normalized(module, base2k, &want, &mut scratch),
                    normalized(module, base2k, &have, &mut scratch),
                    "{what}: a.n()={sparse_n} sizes=({a_size}, {res_size})"
                );
            }
        }
    }
}
