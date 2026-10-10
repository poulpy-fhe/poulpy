//! Cleartext grid model, keys and decryption checks shared by every functional test.

use std::{
    collections::{HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
};

use poulpy_ckks::{
    CKKSCompositionError, CKKSInfos, CKKSMeta, CKKSModuleInfos, CKKSResult, SetCKKSInfos,
    api::LinearTransformationPrepared,
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextSharedModule, TestScalar, assert_decrypt_precision, ckks_encrypt},
        parity::helpers::{Snapshot, snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{
    GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, TorusPrecision,
    prepared::{
        GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
        GLWESecretPrepared, GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef, GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::NegacyclicFFT,
    layouts::{Backend, Data, HostBytesBackend, HostDataMut, HostDataRef, Module, Ring, ScratchArena, ZnxWord},
};

use crate::layouts::{BlasCiphertext, BlasCiphertextOwned, BlasGridInfos, grid::split_into_blocks};

/// `log2` of the block dimension every functional test uses.
pub const LOG_D: usize = 2;

/// The block dimension every functional test uses.
pub const D: usize = 1 << LOG_D;

/// One complex `rows d x cols d` matrix per lane: lane `s`, entry `(r, c)` at `(s * rows * d + r) * cols * d + c`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub rows: usize,
    pub cols: usize,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl Grid {
    /// `lanes` matrices of `rows x cols` blocks with entry `(s, r, c) = f(s, r, c)`.
    pub fn from_fn(rows: usize, cols: usize, lanes: usize, f: impl Fn(usize, usize, usize) -> (f64, f64)) -> Grid {
        let (h, w) = (rows * D, cols * D);
        let (re, im) = (0..lanes * h * w).map(|t| f(t / (h * w), (t / w) % h, t % w)).unzip();
        Grid { rows, cols, re, im }
    }

    /// Deterministic test matrices, entries in `[-1, 1]`.
    pub fn model(rows: usize, cols: usize, lanes: usize, seed: u64) -> Grid {
        let value = |t: usize, salt: u64| {
            let x = (t as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503).wrapping_add(salt))
                % 2001;
            x as f64 / 1000.0 - 1.0
        };
        let w = cols * D;
        Grid::from_fn(rows, cols, lanes, |s, r, c| {
            let t = (s * rows * D + r) * w + c;
            (value(t, 0), value(t, 7))
        })
    }

    /// Number of matrices.
    pub fn lanes(&self) -> usize {
        self.re.len() / (self.rows * self.cols * D * D)
    }

    /// Entry `(r, c)` of matrix `s`.
    pub fn at(&self, s: usize, r: usize, c: usize) -> (f64, f64) {
        let t = (s * self.rows * D + r) * self.cols * D + c;
        (self.re[t], self.im[t])
    }

    /// Per-lane complex product `self * other`.
    pub fn matmul(&self, other: &Grid) -> Grid {
        Grid::from_fn(self.rows, other.cols, self.lanes(), |s, r, c| {
            (0..self.cols * D).fold((0.0, 0.0), |(acc_re, acc_im), l| {
                let ((ar, ai), (br, bi)) = (self.at(s, r, l), other.at(s, l, c));
                (acc_re + ar * br - ai * bi, acc_im + ar * bi + ai * br)
            })
        })
    }

    /// Per-lane product by the real public matrix `b(s mod count, row, col)` of `cols` block columns.
    pub fn matmul_public(&self, cols: usize, count: usize, b: impl Fn(usize, usize, usize) -> f64) -> Grid {
        Grid::from_fn(self.rows, cols, self.lanes(), |s, r, c| {
            (0..self.cols * D).fold((0.0, 0.0), |(acc_re, acc_im), l| {
                let ((ar, ai), w) = (self.at(s, r, l), b(s % count, l, c));
                (acc_re + ar * w, acc_im + ai * w)
            })
        })
    }

    /// Per-lane transpose.
    pub fn transpose(&self) -> Grid {
        Grid::from_fn(self.cols, self.rows, self.lanes(), |s, r, c| self.at(s, c, r))
    }

    /// Row `r` of every matrix replaced by row `r + k` modulo the height.
    pub fn rotate_rows(&self, k: usize) -> Grid {
        Grid::from_fn(self.rows, self.cols, self.lanes(), |s, r, c| {
            self.at(s, (r + k) % (self.rows * D), c)
        })
    }

    /// Column `c` of every matrix replaced by column `c + k` modulo the width.
    pub fn rotate_cols(&self, k: usize) -> Grid {
        Grid::from_fn(self.rows, self.cols, self.lanes(), |s, r, c| {
            self.at(s, r, (c + k) % (self.cols * D))
        })
    }

    /// Lane `s` replaced by lane `s + r` modulo the lane count.
    pub fn rotate_lanes(&self, r: i64) -> Grid {
        let lanes = self.lanes() as i64;
        Grid::from_fn(self.rows, self.cols, self.lanes(), |s, row, c| {
            self.at((s as i64 + r).rem_euclid(lanes) as usize, row, c)
        })
    }

    /// Every lane replaced by the sum over lanes.
    pub fn sum_lanes(&self) -> Grid {
        Grid::from_fn(self.rows, self.cols, self.lanes(), |_, r, c| {
            (0..self.lanes()).fold((0.0, 0.0), |(acc_re, acc_im), s| {
                let (re, im) = self.at(s, r, c);
                (acc_re + re, acc_im + im)
            })
        })
    }

    /// Real and imaginary slot vectors of every block, row-major, each `slots` long.
    pub fn blocks(&self, slots: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
        let re = split_into_blocks(self.rows, self.cols, LOG_D, slots, self.lanes(), |s, r, c| self.at(s, r, c).0);
        let im = split_into_blocks(self.rows, self.cols, LOG_D, slots, self.lanes(), |s, r, c| self.at(s, r, c).1);
        re.into_iter().zip(im).collect()
    }
}

fn to_f<F: TestScalar>(v: &[f64]) -> Vec<F> {
    v.iter().map(|x| F::from_f64(*x).unwrap()).collect()
}

/// Encrypts every block of `grid` at torus width `k`.
pub fn encrypt_grid<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
    encoder: &ReferenceEncoder<E>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    grid: &Grid,
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasCiphertextOwned<BE>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
{
    let blocks = grid
        .blocks(params.n / 2)
        .iter()
        .map(|(re, im)| {
            ckks_encrypt(
                params,
                module,
                host_module,
                encoder,
                sk,
                k,
                &to_f::<F>(re),
                &to_f::<F>(im),
                scratch,
            )
        })
        .collect();
    BlasCiphertext::from_blocks(blocks, LOG_D, grid.rows, grid.cols).unwrap()
}

/// Decrypts every block of `ct` and checks it against `want` at the block's precision; every functional test ends here.
pub fn assert_grid_decrypt<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    ct: &BlasCiphertextOwned<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    want: &Grid,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    assert_eq!(
        (ct.log_d(), ct.rows(), ct.cols()),
        (LOG_D, want.rows, want.cols),
        "{label}: grid shape"
    );
    for (b, (block, (re, im))) in ct.blocks().iter().zip(want.blocks(params.n / 2)).enumerate() {
        assert_decrypt_precision::<BE, F, E>(
            &format!("{label} block {b}"),
            params,
            module,
            encoder,
            block,
            sk,
            &to_f::<F>(&re),
            &to_f::<F>(&im),
            scratch,
        );
    }
}

/// A copy of `grid` with the metadata of every block changed by `edit`, its torus width unchanged.
pub fn relabel<DG: Data, W: ZnxWord, R: Ring>(
    grid: &BlasCiphertext<DG, W, R>,
    edit: impl Fn(&mut CKKSMeta),
) -> BlasCiphertext<DG, W, R>
where
    CKKSCiphertext<DG, W, R>: Clone,
{
    let mut grid = grid.clone();
    for block in grid.blocks_mut() {
        let mut meta = block.meta();
        edit(&mut meta);
        block.set_meta(meta);
    }
    grid
}

/// Copies of `grid`, by label, that an operation rejects in its place: read at the next dimension, its last block one
/// limb narrower than the others, labelled one slot so that no block fits, and a scale wider than its blocks. `grid` has
/// at least two blocks.
pub fn rejected_grids<BE>(module: &Module<BE>, grid: &BlasCiphertextOwned<BE>) -> [(&'static str, BlasCiphertextOwned<BE>); 4]
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
{
    let (log_d, rows, cols, layout) = (grid.log_d(), grid.rows(), grid.cols(), grid.block_layout());
    assert!(rows * cols > 1, "one block cannot be uneven");
    let mut uneven = grid.clone().into_blocks();
    *uneven.last_mut().unwrap() =
        module.ckks_ciphertext_alloc(layout.base2k(), (layout.k().as_usize() - layout.base2k().as_usize()).into());
    let one_slot = module.ckks_max_slots().ilog2() as usize;
    [
        (
            "dimension",
            BlasCiphertext::from_blocks(grid.clone().into_blocks(), log_d + 1, rows, cols).unwrap(),
        ),
        ("uneven", BlasCiphertext::from_blocks(uneven, log_d, rows, cols).unwrap()),
        ("one slot", relabel(grid, |meta| meta.log_sparsity = one_slot)),
        ("scale", relabel(grid, |meta| meta.log_delta = layout.k().as_usize() + 1)),
    ]
}

/// Snapshots of every block, compared before and after a rejected call.
pub fn grid_snapshot<BE, DG>(grid: &BlasCiphertext<DG, BE::ZnxWord, BE::Ring>) -> Vec<Snapshot>
where
    BE: Backend<ZnxWord = i64>,
    DG: Data,
    CKKSCiphertext<DG, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    grid.blocks().iter().map(snapshot::<BE, _>).collect()
}

/// The rejection a guard test expects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    /// `CKKSCompositionError::InvalidPlan`.
    InvalidPlan,
    /// `CKKSCompositionError::MissingAutomorphismKey`.
    MissingKey,
    /// `CKKSCompositionError::MissingRelinearizationKey`.
    MissingTensorKey,
}

/// Runs `call` on `target` and asserts it fails with `rejection` raised by `op` and leaves `target` unchanged.
pub fn assert_guard<BE, DG>(
    label: &str,
    op: &str,
    rejection: Rejection,
    target: &mut BlasCiphertext<DG, BE::ZnxWord, BE::Ring>,
    call: impl FnOnce(&mut BlasCiphertext<DG, BE::ZnxWord, BE::Ring>) -> CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    DG: Data,
    CKKSCiphertext<DG, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    assert_guard_on::<BE, _, _>(label, op, rejection, target, |grid| grid, call);
}

/// [`assert_guard`] on a value holding a grid, read through `grid`.
pub fn assert_guard_on<BE, X, DG>(
    label: &str,
    op: &str,
    rejection: Rejection,
    target: &mut X,
    grid: impl Fn(&X) -> &BlasCiphertext<DG, BE::ZnxWord, BE::Ring>,
    call: impl FnOnce(&mut X) -> CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    DG: Data,
    CKKSCiphertext<DG, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    let before = grid_snapshot::<BE, _>(grid(target));
    let e = call(target).expect_err(label);
    let raised = match e.composition() {
        Some(CKKSCompositionError::InvalidPlan { plan, .. }) => Some((Rejection::InvalidPlan, *plan)),
        Some(CKKSCompositionError::MissingAutomorphismKey { op, .. }) => Some((Rejection::MissingKey, *op)),
        Some(CKKSCompositionError::MissingRelinearizationKey { op, .. }) => Some((Rejection::MissingTensorKey, *op)),
        _ => None,
    };
    assert_eq!(raised, Some((rejection, op)), "{label}: {op} raised {e}");
    assert!(
        grid_snapshot::<BE, _>(grid(target)) == before,
        "{label}: {op} changed its output"
    );
}

/// Runs `call` on `target` with a scratch one byte short of `bytes`, its query, and asserts it panics before `target`
/// changes.
pub fn assert_short_scratch<BE, DG>(
    op: &str,
    target: &mut BlasCiphertext<DG, BE::ZnxWord, BE::Ring>,
    bytes: usize,
    call: impl FnOnce(&mut BlasCiphertext<DG, BE::ZnxWord, BE::Ring>, &mut ScratchArena<'_, BE>) -> CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    DG: Data,
    CKKSCiphertext<DG, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    assert_short_scratch_on::<BE, _, _>(op, target, |grid| grid, bytes, call);
}

/// [`assert_short_scratch`] on a value holding a grid, read through `grid`.
pub fn assert_short_scratch_on<BE, X, DG>(
    op: &str,
    target: &mut X,
    grid: impl Fn(&X) -> &BlasCiphertext<DG, BE::ZnxWord, BE::Ring>,
    bytes: usize,
    call: impl FnOnce(&mut X, &mut ScratchArena<'_, BE>) -> CKKSResult<()>,
) where
    BE: Backend<ZnxWord = i64>,
    DG: Data,
    CKKSCiphertext<DG, BE::ZnxWord, BE::Ring>: GLWEToBackendRef<BE>,
{
    let before = grid_snapshot::<BE, _>(grid(target));
    let run = catch_unwind(AssertUnwindSafe(|| with_scratch::<BE, _>(bytes - 1, |s| call(target, s))));
    assert!(run.is_err(), "{op} ran one byte short of its scratch query");
    assert!(
        grid_snapshot::<BE, _>(grid(target)) == before,
        "{op} changed its output before panicking"
    );
}

/// Prepared rotation keys by Galois element plus an optional tensor key.
pub struct TestKeys<BE: Backend> {
    pub atks: HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    pub tsk: Option<GLWETensorKeyPrepared<BE::OwnedBuf, BE>>,
}

impl<BE: Backend> GetAutomorphismKey<BE> for TestKeys<BE>
where
    GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
{
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        self.atks.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend> GetTensorKey<BE> for TestKeys<BE>
where
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn get_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        self.tsk
            .as_ref()
            .ok_or_else(|| poulpy_core::CoreError::GGLWEKeyUse {
                op: "get_tensor_key",
                detail: "this test holds no tensor key".into(),
            })?
            .get_tensor_key(k)
    }
}

impl<BE: Backend> TestKeys<BE> {
    /// These keys, answering only at the listed widths: a rotation key at `(Galois element, width)` in `atks`, the
    /// tensor key at the widths in `tsks`.
    pub fn at_widths(
        &self,
        atks: impl IntoIterator<Item = (i64, usize)>,
        tsks: impl IntoIterator<Item = usize>,
    ) -> StrictKeys<'_, BE> {
        StrictKeys {
            keys: self,
            atks: atks.into_iter().collect(),
            tsks: tsks.into_iter().collect(),
        }
    }
}

/// [`TestKeys`] answering only at the widths [`TestKeys::at_widths`] lists.
pub struct StrictKeys<'a, BE: Backend> {
    keys: &'a TestKeys<BE>,
    atks: HashSet<(i64, usize)>,
    tsks: HashSet<usize>,
}

fn not_at_width(op: &'static str, detail: String) -> poulpy_core::CoreError {
    poulpy_core::CoreError::GGLWEKeyUse { op, detail }
}

impl<BE: Backend> GetAutomorphismKey<BE> for StrictKeys<'_, BE>
where
    GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>: GLWEAutomorphismKeyPreparedToBackendRef<BE>,
{
    fn lookup_automorphism_key(
        &self,
        p: i64,
        k: TorusPrecision,
    ) -> poulpy_core::Result<GLWEAutomorphismKeyPreparedBackendRef<'_, BE>> {
        if !self.atks.contains(&(p, k.as_usize())) {
            return Err(not_at_width(
                "get_automorphism_key",
                format!("no key for p={p} at k={}", k.as_usize()),
            ));
        }
        self.keys.lookup_automorphism_key(p, k)
    }
}

impl<BE: Backend> GetTensorKey<BE> for StrictKeys<'_, BE>
where
    GLWETensorKeyPrepared<BE::OwnedBuf, BE>: GLWETensorKeyPreparedToBackendRef<BE>,
{
    fn get_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
        if !self.tsks.contains(&k.as_usize()) {
            return Err(not_at_width("get_tensor_key", format!("no tensor key at k={}", k.as_usize())));
        }
        self.keys.get_tensor_key(k)
    }
}

/// The rotation keys evaluating `lt` from a ciphertext of width `src_k` into a result of width `res_k` looks up, as
/// `(Galois element, width)`: the baby steps at `src_k`, the non-empty giant steps at `res_k`.
pub fn lt_key_uses<BE: Backend>(
    module: &Module<BE>,
    lt: &LinearTransformationPrepared<BE>,
    src_k: usize,
    res_k: usize,
) -> Vec<(i64, usize)>
where
    Module<BE>: CKKSModuleInfos,
{
    let babies = lt.baby_steps().iter().map(|&rotation| (rotation, src_k));
    let giants = lt
        .giant_steps
        .iter()
        .filter(|gs| !gs.diagonals.is_empty())
        .map(|gs| (gs.rot, res_k));
    babies
        .chain(giants)
        .map(|(rotation, k)| (module.ckks_galois_element(rotation), k))
        .collect()
}

/// The context at `params.prec()` plus a rotation key for each of its Galois elements.
#[allow(clippy::type_complexity)]
pub fn context_and_keys<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> (
    crate::layouts::BlasContext<BE>,
    HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
)
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>
        + poulpy_ckks::api::CKKSEncodingOps<BE, F>
        + poulpy_ckks::api::CKKSLinearTransformationOps<BE>
        + poulpy_hal::api::CnvPVecAlloc<BE>,
    F: TestScalar + poulpy_core::layouts::DiagonalArithmetic,
{
    use poulpy_hal::layouts::CyclotomicOrder;
    let ctx =
        crate::layouts::BlasContextUnprepared::encode::<F>(module, LOG_D, params.base2k.into(), params.prec().into(), scratch)
            .unwrap()
            .prepare(module, scratch);
    let atks = ctx
        .galois_elements(module.cyclotomic_order())
        .into_iter()
        .map(|p| {
            (
                p,
                poulpy_ckks::test_suite::helpers::gen_atk(params, module, p, sk_raw, scratch),
            )
        })
        .collect();
    (ctx, atks)
}

/// Contexts an operation on `module` rejects, by label: one encoded on a module of half the degree, one at a radix one
/// below `params.base2k`. `atks` gains a key for every rotation they take on `module`, so no missing key stops an
/// operation from evaluating them.
pub fn foreign_contexts<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    atks: &mut HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> [(&'static str, crate::layouts::BlasContext<BE>); 2]
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>
        + poulpy_ckks::api::CKKSEncodingOps<BE, F>
        + poulpy_ckks::api::CKKSLinearTransformationOps<BE>
        + poulpy_hal::api::CnvPVecAlloc<BE>,
    F: TestScalar + poulpy_core::layouts::DiagonalArithmetic,
{
    use poulpy_hal::layouts::CyclotomicOrder;
    let half = Module::<BE>::new(params.n as u64 / 2);
    let mut encode = |module: &Module<BE>, base2k: usize| {
        crate::layouts::BlasContextUnprepared::encode::<F>(module, LOG_D, base2k.into(), params.prec().into(), scratch)
            .unwrap()
            .prepare(module, scratch)
    };
    let contexts = [
        ("degree", encode(&half, params.base2k)),
        ("base2k", encode(module, params.base2k - 1)),
    ];
    for p in contexts
        .iter()
        .flat_map(|(_, ctx)| ctx.galois_elements(module.cyclotomic_order()))
    {
        atks.entry(p)
            .or_insert_with(|| poulpy_ckks::test_suite::helpers::gen_atk(params, module, p, sk_raw, scratch));
    }
    contexts
}
