//! Cleartext grid model, keys and decryption checks shared by every functional test.

use std::{
    collections::{HashMap, HashSet},
    panic::{AssertUnwindSafe, catch_unwind},
};

use poulpy_ckks::{
    CKKSCompositionError, CKKSInfos, CKKSLayout, CKKSModuleInfos, CKKSResult, CoeffsMeta, GLWEPlaintextInfos, GLWEPlaintextMeta,
    Scale, SetCKKSInfos, SetGLWEPlaintextInfos, SlotsKind,
    api::{CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSMulOps, LinearTransformationPrepared},
    layouts::{CKKSCiphertext, CKKSModuleAlloc},
    test_suite::{
        CKKSTestParams,
        helpers::{TestContextBackend, TestContextSharedModule, TestScalar, assert_decrypt_precision_at_log_delta, ckks_encrypt},
        parity::helpers::{Snapshot, snapshot, with_scratch},
        reference_encoder::ReferenceEncoder,
    },
};
use poulpy_core::layouts::{
    Base2K, Degree, DiagonalArithmetic, GLWELayout, GLWEToBackendRef, GetAutomorphismKey, GetTensorKey, LWEInfos, Rank,
    TorusPrecision,
    prepared::{
        GLWEAutomorphismKeyPrepared, GLWEAutomorphismKeyPreparedBackendRef, GLWEAutomorphismKeyPreparedToBackendRef,
        GLWESecretPrepared, GLWETensorKeyPrepared, GLWETensorKeyPreparedBackendRef, GLWETensorKeyPreparedToBackendRef,
    },
};
use poulpy_hal::{
    api::NegacyclicFFT,
    layouts::{
        Backend, Data, HostBytesBackend, HostDataMut, HostDataRef, Module, PrepareHint, Ring, ScratchArena, ScratchOwned, ZnxWord,
    },
};

use crate::{
    api::{BlasContextPreparedFactory, BlasEncodingOps, BlasMatMulOps, BlasPtOperandPreparedFactory},
    layouts::{
        BlasCiphertext, BlasCiphertextOwned, BlasContext, BlasContextInfos, BlasGrid, BlasGridInfos, BlasGridMut, BlasLayout,
        BlasMatMulLeft, BlasMatMulOperandInfos, BlasMatMulOperandPreparedFactory, BlasMatMulRight, BlasModuleAlloc, BlasPlan,
        BlasPtAlgorithm, BlasPtOperand, BlasPtSide, blas_matmul_left_operand_layout, blas_matmul_right_operand_layout,
        ciphertext::layout_of, grid::split_into_blocks,
    },
    oep::{BlasEncodingImpl, BlasMatMulImpl},
};

/// Both preparation hints: hot, `OneShot`, then cold, `Reuse`.
pub const HINTS: [PrepareHint; 2] = [PrepareHint::OneShot, PrepareHint::Reuse];

/// [`BlasMatMulOps::blas_matmul_prepare_left`] of `a` under `hint`, into an operand allocated from the infos
/// [`blas_matmul_left_operand_layout`] gives.
pub fn matmul_left_operand<BE: BlasMatMulImpl>(
    module: &Module<BE>,
    a: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    keys: &(impl GetAutomorphismKey<BE> + Sync),
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<BlasMatMulLeft<BE>> {
    let mut res = module.blas_matmul_left_alloc_from_infos(&blas_matmul_left_operand_layout(a, ctx, hint));
    module.blas_matmul_prepare_left(&mut res, a, ctx, keys, scratch)?;
    Ok(res)
}

/// [`BlasMatMulOps::blas_matmul_prepare_right`] of `b` under `hint`, into an operand allocated from the infos
/// [`blas_matmul_right_operand_layout`] gives.
pub fn matmul_right_operand<BE: BlasMatMulImpl>(
    module: &Module<BE>,
    b: &impl BlasGrid<BE>,
    ctx: &BlasContext<BE>,
    keys: &(impl GetAutomorphismKey<BE> + Sync),
    hint: PrepareHint,
    scratch: &mut ScratchArena<'_, BE>,
) -> CKKSResult<BlasMatMulRight<BE>> {
    let mut res = module.blas_matmul_right_alloc_from_infos(&blas_matmul_right_operand_layout(b, ctx, hint));
    module.blas_matmul_prepare_right(&mut res, b, ctx, keys, scratch)?;
    Ok(res)
}

/// The four hint mixes of a prepared product's operands, `(left, right)`.
pub const MIXES: [(PrepareHint, PrepareHint); 4] = [
    (PrepareHint::OneShot, PrepareHint::OneShot),
    (PrepareHint::OneShot, PrepareHint::Reuse),
    (PrepareHint::Reuse, PrepareHint::OneShot),
    (PrepareHint::Reuse, PrepareHint::Reuse),
];

/// Both algorithms of the products by public matrices: diagonal, then JKLS.
pub const PT_ALGORITHMS: [BlasPtAlgorithm; 2] = [BlasPtAlgorithm::Diagonal, BlasPtAlgorithm::Jkls];

/// The paths that put JKLS's masks on the ciphertext: stored plaintexts, then plaintexts read through Galois elements.
pub const MASKED_PT_ALGORITHMS: [BlasPtAlgorithm; 2] = [BlasPtAlgorithm::JklsStored, BlasPtAlgorithm::JklsGalois];

/// One complex `rows d x cols d` matrix per lane, `d = 2^log_d`: lane `s`, entry `(r, c)` at
/// `(s * rows * d + r) * cols * d + c`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub log_d: usize,
    pub rows: usize,
    pub cols: usize,
    pub re: Vec<f64>,
    pub im: Vec<f64>,
}

impl Grid {
    /// `lanes` matrices of `rows x cols` blocks of dimension `2^log_d` with entry `(s, r, c) = f(s, r, c)`.
    pub fn from_fn(log_d: usize, rows: usize, cols: usize, lanes: usize, f: impl Fn(usize, usize, usize) -> (f64, f64)) -> Grid {
        let (h, w) = (rows << log_d, cols << log_d);
        let (re, im) = (0..lanes * h * w).map(|t| f(t / (h * w), (t / w) % h, t % w)).unzip();
        Grid {
            log_d,
            rows,
            cols,
            re,
            im,
        }
    }

    /// Deterministic test matrices, entries dyadic, in `[-1, 1]`, so that every model value is exact at any scalar.
    pub fn model(log_d: usize, rows: usize, cols: usize, lanes: usize, seed: u64) -> Grid {
        let value = |t: usize, salt: u64| {
            let x = (t as u64)
                .wrapping_mul(2654435761)
                .wrapping_add(seed.wrapping_mul(40503).wrapping_add(salt))
                % 2001;
            x as f64 / 1024.0 - 1.0
        };
        let (h, w) = (rows << log_d, cols << log_d);
        Grid::from_fn(log_d, rows, cols, lanes, |s, r, c| {
            let t = (s * h + r) * w + c;
            (value(t, 0), value(t, 7))
        })
    }

    /// The block dimension `d`.
    pub fn d(&self) -> usize {
        1 << self.log_d
    }

    /// Number of matrices.
    pub fn lanes(&self) -> usize {
        self.re.len() / (self.rows * self.cols * self.d() * self.d())
    }

    /// Entry `(r, c)` of matrix `s`.
    pub fn at(&self, s: usize, r: usize, c: usize) -> (f64, f64) {
        let t = (s * self.rows * self.d() + r) * self.cols * self.d() + c;
        (self.re[t], self.im[t])
    }

    /// Per-lane complex product `self * other`.
    pub fn matmul(&self, other: &Grid) -> Grid {
        Grid::from_fn(self.log_d, self.rows, other.cols, self.lanes(), |s, r, c| {
            (0..self.cols * self.d()).fold((0.0, 0.0), |(acc_re, acc_im), l| {
                let ((ar, ai), (br, bi)) = (self.at(s, r, l), other.at(s, l, c));
                (acc_re + ar * br - ai * bi, acc_im + ar * bi + ai * br)
            })
        })
    }

    /// Per-lane product by the real public matrix `b(s mod count, row, col)` of `cols` block columns.
    pub fn matmul_public(&self, cols: usize, count: usize, b: impl Fn(usize, usize, usize) -> f64) -> Grid {
        Grid::from_fn(self.log_d, self.rows, cols, self.lanes(), |s, r, c| {
            (0..self.cols * self.d()).fold((0.0, 0.0), |(acc_re, acc_im), l| {
                let ((ar, ai), w) = (self.at(s, r, l), b(s % count, l, c));
                (acc_re + ar * w, acc_im + ai * w)
            })
        })
    }

    /// Per-lane product by the real public matrix `b(s mod count, row, col)` of `rows` block rows, on the left.
    pub fn public_matmul(&self, rows: usize, count: usize, b: impl Fn(usize, usize, usize) -> f64) -> Grid {
        Grid::from_fn(self.log_d, rows, self.cols, self.lanes(), |s, r, c| {
            (0..self.rows * self.d()).fold((0.0, 0.0), |(acc_re, acc_im), l| {
                let (w, (ar, ai)) = (b(s % count, r, l), self.at(s, l, c));
                (acc_re + w * ar, acc_im + w * ai)
            })
        })
    }

    /// Per-lane transpose.
    pub fn transpose(&self) -> Grid {
        Grid::from_fn(self.log_d, self.cols, self.rows, self.lanes(), |s, r, c| self.at(s, c, r))
    }

    /// Row `r` of every matrix replaced by row `r + k` modulo the height.
    pub fn rotate_rows(&self, k: usize) -> Grid {
        Grid::from_fn(self.log_d, self.rows, self.cols, self.lanes(), |s, r, c| {
            self.at(s, (r + k) % (self.rows * self.d()), c)
        })
    }

    /// Column `c` of every matrix replaced by column `c + k` modulo the width.
    pub fn rotate_cols(&self, k: usize) -> Grid {
        Grid::from_fn(self.log_d, self.rows, self.cols, self.lanes(), |s, r, c| {
            self.at(s, r, (c + k) % (self.cols * self.d()))
        })
    }

    /// Lane `s` replaced by lane `s + r` modulo the lane count.
    pub fn rotate_lanes(&self, r: i64) -> Grid {
        let lanes = self.lanes() as i64;
        Grid::from_fn(self.log_d, self.rows, self.cols, self.lanes(), |s, row, c| {
            self.at((s as i64 + r).rem_euclid(lanes) as usize, row, c)
        })
    }

    /// Every lane replaced by the sum over lanes.
    pub fn sum_lanes(&self) -> Grid {
        Grid::from_fn(self.log_d, self.rows, self.cols, self.lanes(), |_, r, c| {
            (0..self.lanes()).fold((0.0, 0.0), |(acc_re, acc_im), s| {
                let (re, im) = self.at(s, r, c);
                (acc_re + re, acc_im + im)
            })
        })
    }

    /// Real and imaginary slot vectors of every block, row-major, each `slots` long.
    pub fn blocks(&self, slots: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
        let (log_d, lanes) = (self.log_d, self.lanes());
        let re = split_into_blocks(self.rows, self.cols, log_d, slots, lanes, |s, r, c| self.at(s, r, c).0);
        let im = split_into_blocks(self.rows, self.cols, log_d, slots, lanes, |s, r, c| self.at(s, r, c).1);
        re.into_iter().zip(im).collect()
    }
}

/// `v` at scalar `F`.
pub(crate) fn to_f<F: TestScalar>(v: &[f64]) -> Vec<F> {
    v.iter().map(|x| F::from_f64(*x).unwrap()).collect()
}

/// Encrypts every block of `grid` at torus width `k` and labels it `log_sparsity`, which the grid records the batch of:
/// `grid` holds at least `slots >> log_sparsity` matrices, repeated over the slots.
pub fn encrypt_grid<BE, F, E>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend<BE::Ring>>,
    encoder: &ReferenceEncoder<E>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    k: usize,
    log_sparsity: usize,
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
            let mut block = ckks_encrypt(
                params,
                module,
                host_module,
                encoder,
                sk,
                k,
                &to_f::<F>(re),
                &to_f::<F>(im),
                scratch,
            );
            block.set_log_sparsity(log_sparsity);
            block
        })
        .collect();
    BlasCiphertext::from_blocks(blocks, grid.log_d, grid.rows, grid.cols).unwrap()
}

/// A grid of `rows x cols` blocks of dimension `2^log_d` at torus width `k`, at the radix and rank of `params`, recording
/// the dense batch.
pub fn alloc_grid<BE: Backend>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    log_d: usize,
    rows: usize,
    cols: usize,
    k: usize,
) -> BlasCiphertextOwned<BE>
where
    Module<BE>: CKKSModuleAlloc<BE>,
{
    let block = CKKSLayout {
        glwe_layout: GLWELayout {
            k: k.into(),
            ..params.glwe_layout()
        },
        meta: GLWEPlaintextMeta::default(),
    };
    module.blas_ciphertext_alloc_from_infos(&BlasLayout::new(block, log_d, rows, cols))
}

/// The layout of a `rows x cols` result whose blocks take `a`'s at torus width `k` and slot kind `slots`.
fn result_layout(a: &impl BlasGridInfos, rows: usize, cols: usize, k: usize, slots: SlotsKind) -> BlasLayout {
    let mut block = a.block_layout();
    block.glwe_layout.k = k.into();
    block.meta.slots = slots;
    BlasLayout::new(block, a.log_d(), rows, cols)
}

/// The metadata a copy or a rotation of `a` gives a `rows x cols` result written into blocks of width `res_k`: `a`'s
/// layout at `min(res_k, a.k)`.
pub fn copied_layout(a: &impl BlasGridInfos, rows: usize, cols: usize, res_k: usize) -> BlasLayout {
    let block = a.block_layout();
    result_layout(a, rows, cols, res_k.min(block.k().as_usize()), block.slots())
}

/// The metadata a map of scale `scale`, a context map or a public operand, on `a` gives a `rows x cols` result written
/// into blocks of width `res_k`: `a`'s layout at `min(res_k, a.k - scale)`, its slots joined with the maps' `Real`.
pub fn mapped_layout(a: &impl BlasGridInfos, rows: usize, cols: usize, res_k: usize, scale: usize) -> BlasLayout {
    let block = a.block_layout();
    let k = res_k.min(block.k().as_usize() - scale);
    result_layout(a, rows, cols, k, block.slots().join(SlotsKind::Real))
}

/// The metadata the JKLS product of `a` and `b`, through maps of scale `p`, gives a `rows x cols` result written into
/// blocks of width `res_k`. Its terms sit at `a.k - 2p` on the left and, on the right, `b.k - p` for a replicated `b`
/// and `b.k - 2p` otherwise, so: poulpy-ckks's ct x ct rule on those widths,
/// `min(res_k, min(wa - da, wb - db) - max(da, db) + min(da, db))` for scales `da` and `db`, at the smaller scale and
/// label, and the slots of both joined with the maps' `Real`.
pub fn product_layout<R: Ring>(
    a: &impl BlasGridInfos,
    b: &impl BlasGridInfos,
    rows: usize,
    cols: usize,
    res_k: usize,
    p: usize,
) -> BlasLayout {
    let (la, lb) = (a.block_layout(), b.block_layout());
    let (da, db) = (la.log_delta(), lb.log_delta());
    let right = if crate::layouts::replicated::<R>(b) { p } else { 2 * p };
    let (wa, wb) = (la.k().as_usize() - 2 * p, lb.k().as_usize() - right);
    let budget = (wa - da).min(wb - db) - da.max(db);
    let slots = la.slots().join(lb.slots()).join(SlotsKind::Real);
    let mut layout = result_layout(a, rows, cols, res_k.min(da.min(db) + budget), slots);
    layout.block.meta.scale = Scale::Log(da.min(db));
    layout.block.meta.log_sparsity = la.log_sparsity().min(lb.log_sparsity());
    layout
}

/// Decrypts every block of `ct` and checks it against `want`; every functional test ends here. First the metadata
/// against `layout`, from [`copied_layout`], [`mapped_layout`] or [`product_layout`]: the grid's shape, every block's
/// label, layout and empty noise tag, and the grid's record of the batch the label proves. Then the values, at the
/// smaller of `layout`'s `log_delta` and that of `params.prec()`, the scale the maps and public matrices `ct` went
/// through were encoded at ([`at_scale`]): a map below its input's scale bounds the result's precision with its own,
/// although the result keeps its input's scale. A result left unwritten or stamped at another scale fails.
pub fn assert_grid_decrypt<BE, F, E>(
    label: &str,
    params: &CKKSTestParams,
    module: &Module<BE>,
    encoder: &ReferenceEncoder<E>,
    ct: &BlasCiphertextOwned<BE>,
    sk: &GLWESecretPrepared<BE::OwnedBuf, BE>,
    want: &Grid,
    layout: &BlasLayout,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
    F: TestScalar,
    E: NegacyclicFFT<F>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
{
    let shape = (layout.log_d, layout.rows, layout.cols);
    assert_eq!((ct.log_d(), ct.rows(), ct.cols()), shape, "{label}: grid shape");
    assert_eq!((want.log_d, want.rows, want.cols), shape, "{label}: model shape");
    let want_block = layout.block;
    for (b, block) in ct.blocks().iter().enumerate() {
        assert_eq!(block.log_sparsity(), want_block.log_sparsity(), "{label} block {b}: label");
        assert_eq!(layout_of(block), want_block, "{label} block {b}: layout");
        assert!(block.noise().is_none(), "{label} block {b}: kept a noise tag");
    }
    let batch = ((params.n / 2) >> want_block.log_sparsity()) >> (2 * layout.log_d);
    assert_eq!(ct.mat_slots(), Some(batch), "{label}: batch record");
    let log_prec = want_block.log_delta().min(params.prec().log_delta());
    for (b, (block, (re, im))) in ct.blocks().iter().zip(want.blocks(params.n / 2)).enumerate() {
        assert_decrypt_precision_at_log_delta::<BE, F, E>(
            &format!("{label} block {b}"),
            params,
            module,
            encoder,
            block,
            sk,
            &to_f::<F>(&re),
            &to_f::<F>(&im),
            log_prec,
            scratch,
        );
    }
}

/// The top-left `rows x cols` blocks of `grid`, with its batch record.
pub fn corner<BE: Backend>(grid: &BlasCiphertextOwned<BE>, rows: usize, cols: usize) -> BlasCiphertextOwned<BE>
where
    BlasCiphertextOwned<BE>: BlasGridMut<BE> + Clone,
{
    let width = grid.cols();
    let blocks = grid
        .clone()
        .into_blocks()
        .into_iter()
        .enumerate()
        .filter(|(t, _)| t / width < rows && t % width < cols)
        .map(|(_, block)| block)
        .collect();
    let mut corner = BlasCiphertext::from_blocks(blocks, grid.log_d(), rows, cols).unwrap();
    if let Some(mat_slots) = grid.mat_slots() {
        corner.set_mat_slots(mat_slots);
    }
    corner
}

/// A copy of `grid` with the metadata of every block changed by `edit`, its torus width unchanged.
pub fn relabel<DG: Data, W: ZnxWord, R: Ring>(
    grid: &BlasCiphertext<DG, W, R>,
    edit: impl Fn(&mut GLWEPlaintextMeta),
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
/// limb narrower than the others, labelled one slot so that it records no matrix, recording half the dense batch at
/// the dense label, as after adding a dense plaintext to a sparse batch, and a scale wider than its blocks. `grid` has
/// at least two blocks.
pub fn rejected_grids<BE>(module: &Module<BE>, grid: &BlasCiphertextOwned<BE>) -> [(&'static str, BlasCiphertextOwned<BE>); 5]
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
{
    let (log_d, rows, cols, layout) = (grid.log_d(), grid.rows(), grid.cols(), grid.block_layout());
    assert!(rows * cols > 1, "one block cannot be uneven");
    let mut uneven = grid.clone().into_blocks();
    let mut narrow = layout;
    narrow.glwe_layout.k = (layout.k().as_usize() - layout.base2k().as_usize()).into();
    *uneven.last_mut().unwrap() = module.ckks_ciphertext_alloc_from_infos(&narrow);
    let labelled = |log_sparsity: usize| {
        let blocks = relabel(grid, |meta| meta.log_sparsity = log_sparsity).into_blocks();
        BlasCiphertext::from_blocks(blocks, log_d, rows, cols).unwrap()
    };
    [
        (
            "dimension",
            BlasCiphertext::from_blocks(grid.clone().into_blocks(), log_d + 1, rows, cols).unwrap(),
        ),
        ("uneven", BlasCiphertext::from_blocks(uneven, log_d, rows, cols).unwrap()),
        ("one slot", labelled(module.ckks_max_slots().ilog2() as usize)),
        ("batch", relabel(&labelled(1), |meta| meta.log_sparsity = 0)),
        (
            "scale",
            relabel(grid, |meta| meta.scale = Scale::Log(layout.k().as_usize() + 1)),
        ),
    ]
}

/// Fresh grids of the dimensions and block layout of `grid`, by label, that an operation rejects beside it: at twice its
/// degree, one radix lower and one rank higher. Each records the dense batch of its own degree.
pub fn foreign_layouts<BE>(module: &Module<BE>, grid: &impl BlasGridInfos) -> [(&'static str, BlasCiphertextOwned<BE>); 3]
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE>,
{
    let alloc = |edit: fn(&mut GLWELayout)| {
        let mut block = grid.block_layout();
        edit(&mut block.glwe_layout);
        module.blas_ciphertext_alloc_from_infos(&BlasLayout::new(block, grid.log_d(), grid.rows(), grid.cols()))
    };
    [
        ("degree", alloc(|layout| layout.n = Degree(2 * layout.n.0))),
        ("base2k", alloc(|layout| layout.base2k = Base2K(layout.base2k.0 - 1))),
        ("rank", alloc(|layout| layout.rank = Rank(layout.rank.0 + 1))),
    ]
}

/// `grid` read as `rows x cols` blocks whatever it holds: a [`BlasGrid`] whose blocks do not tile the shape it reports,
/// which every operation rejects before `res` changes.
pub struct Misshapen<G> {
    pub grid: G,
    pub rows: usize,
    pub cols: usize,
}

impl<G: BlasGridInfos> BlasGridInfos for Misshapen<G> {
    fn log_d(&self) -> usize {
        self.grid.log_d()
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn block_layout(&self) -> CKKSLayout {
        self.grid.block_layout()
    }

    fn mat_slots(&self) -> Option<usize> {
        self.grid.mat_slots()
    }
}

impl<BE: Backend, G: BlasGrid<BE>> BlasGrid<BE> for Misshapen<G> {
    type Block = G::Block;

    fn blocks(&self) -> &[Self::Block] {
        self.grid.blocks()
    }
}

impl<BE: Backend, G: BlasGridMut<BE>> BlasGridMut<BE> for Misshapen<G> {
    fn blocks_mut(&mut self) -> &mut [Self::Block] {
        self.grid.blocks_mut()
    }

    fn set_mat_slots(&mut self, mat_slots: usize) {
        self.grid.set_mat_slots(mat_slots);
    }
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
    /// `CKKSCompositionError::InvalidOperand`.
    InvalidOperand,
    /// `CKKSCompositionError::MissingAutomorphismKey`.
    MissingKey,
    /// `CKKSCompositionError::MissingRelinearizationKey`.
    MissingTensorKey,
    /// A poulpy-ckks budget error, `InsufficientHomomorphicCapacity` or `MultiplicationPrecisionUnderflow`, raised by
    /// the poulpy-ckks operation named in its place, inside the implementation.
    Budget,
}

/// Runs `call` on `target` and asserts it fails with `rejection` raised by `op` and leaves `target` unchanged, its batch
/// record included. A budget error names the poulpy-ckks operation that raised it.
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
    let state = |target: &X| (grid(target).mat_slots(), grid_snapshot::<BE, _>(grid(target)));
    let before = state(target);
    assert_rejected(label, op, rejection, call(target));
    assert!(state(target) == before, "{label}: {op} changed its output");
}

/// Asserts `result` is the failure `rejection` raised by `op`. A budget error names the poulpy-ckks operation that
/// raised it.
pub fn assert_rejected<T>(label: &str, op: &str, rejection: Rejection, result: CKKSResult<T>) {
    let Err(e) = result else {
        panic!("{label}: {op} succeeded");
    };
    let raised = match e.composition() {
        Some(CKKSCompositionError::InvalidOperand { op, .. }) => Some((Rejection::InvalidOperand, *op)),
        Some(CKKSCompositionError::MissingAutomorphismKey { op, .. }) => Some((Rejection::MissingKey, *op)),
        Some(CKKSCompositionError::MissingRelinearizationKey { op, .. }) => Some((Rejection::MissingTensorKey, *op)),
        Some(
            CKKSCompositionError::InsufficientHomomorphicCapacity { op, .. }
            | CKKSCompositionError::MultiplicationPrecisionUnderflow { op, .. },
        ) => Some((Rejection::Budget, *op)),
        _ => None,
    };
    assert_eq!(raised, Some((rejection, op)), "{label}: {op} raised {e}");
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
    let before = grid_snapshot::<BE, _>(target);
    assert_short_scratch_panics::<BE, _>(op, bytes, |s| call(target, s));
    assert!(
        grid_snapshot::<BE, _>(target) == before,
        "{op} changed its output before panicking"
    );
}

/// Runs `call` with a scratch one byte short of `bytes`, its query, and asserts it panics on its scratch. A zero-byte
/// query has no short scratch to test.
pub fn assert_short_scratch_panics<BE: Backend, T>(op: &str, bytes: usize, call: impl FnOnce(&mut ScratchArena<'_, BE>) -> T) {
    let Some(short) = bytes.checked_sub(1) else {
        return;
    };
    assert_panics(&format!("{op} one byte short of its scratch query"), "scratch", || {
        with_scratch::<BE, _>(short, call)
    });
}

/// The public operand of `plan` on side `S`, `rows x cols` blocks of `count` matrices read through `entry`, at
/// `coeffs_meta`, allocated, encoded and prepared on `module`.
pub fn pt_operand<BE, F, S>(
    module: &Module<BE>,
    plan: BlasPlan,
    rows: usize,
    cols: usize,
    count: usize,
    entry: impl Fn(usize, usize, usize) -> F,
    base2k: Base2K,
    coeffs_meta: CoeffsMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasPtOperand<BE, S>
where
    BE: Backend + BlasEncodingImpl,
    Module<BE>:
        CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CKKSLinearTransformationOps<BE> + CKKSMulOps<BE>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
    S: BlasPtSide,
{
    let mut op = module.blas_pt_operand_alloc::<S>(plan, rows, cols, count, base2k, coeffs_meta);
    module.blas_pt_operand_encode_into(&mut op, entry, scratch).unwrap();
    let mut prepared = module.blas_pt_operand_prepared_alloc_from_infos(&op);
    module.blas_pt_operand_prepare(&mut prepared, &op, scratch);
    prepared
}

/// Panics unless `call` panics with a message that contains `expected`.
pub fn assert_panics<T>(label: &str, expected: &str, call: impl FnOnce() -> T) {
    let Err(payload) = catch_unwind(AssertUnwindSafe(call)) else {
        panic!("{label}: the call did not panic");
    };
    let message = payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or_default();
    assert!(
        message.contains(expected),
        "{label}: panicked with {message:?}, not {expected:?}"
    );
}

/// An owned scratch of `bytes` poisoned with `0xA5`, so an operation that reads a temporary before writing it fails.
pub fn poisoned_scratch<BE: Backend>(bytes: usize) -> ScratchOwned<BE> {
    ScratchOwned {
        data: BE::from_host_bytes(&vec![0xA5; bytes]),
        _phantom: std::marker::PhantomData,
    }
}

/// What a prepared operand, or any grid, reports through [`BlasGridInfos`]: its layout and its batch record.
pub fn grid_infos(g: &impl BlasGridInfos) -> (BlasLayout, Option<usize>) {
    (
        BlasLayout::new(g.block_layout(), g.log_d(), g.rows(), g.cols()),
        g.mat_slots(),
    )
}

/// What a preparation of `a` under `hint` reports, from [`grid_infos`]: hot, `a`'s own; cold, `a`'s dimensions and
/// batch at the layout of its terms, `terms` ([`mapped_layout`] of its maps' scales); then `hint` and `a`'s block
/// layout as the source's, which [`operand_infos`] reads off an operand.
pub fn want_operand_infos(
    a: &impl BlasGridInfos,
    hint: PrepareHint,
    terms: BlasLayout,
) -> (BlasLayout, Option<usize>, PrepareHint, CKKSLayout) {
    let (own, batch) = grid_infos(a);
    let grid = match hint {
        PrepareHint::OneShot => own,
        PrepareHint::Reuse => terms,
    };
    (grid, batch, hint, a.block_layout())
}

/// What a prepared operand reports through [`BlasMatMulOperandInfos`]: [`grid_infos`], its hint and its source's block
/// layout.
pub fn operand_infos(g: &impl BlasMatMulOperandInfos) -> (BlasLayout, Option<usize>, PrepareHint, CKKSLayout) {
    let (layout, batch) = grid_infos(g);
    (layout, batch, g.hint(), g.source_block_layout())
}

/// The grid `.0` read as a prepared operand of hint `.1`, its block layout reported as its source's too: operand infos
/// no preparation makes, for the prepared product's validator.
pub struct OperandInfos<'a, G>(pub &'a G, pub PrepareHint);

impl<G: BlasGridInfos> BlasGridInfos for OperandInfos<'_, G> {
    fn log_d(&self) -> usize {
        self.0.log_d()
    }

    fn rows(&self) -> usize {
        self.0.rows()
    }

    fn cols(&self) -> usize {
        self.0.cols()
    }

    fn block_layout(&self) -> CKKSLayout {
        self.0.block_layout()
    }

    fn mat_slots(&self) -> Option<usize> {
        self.0.mat_slots()
    }
}

impl<G: BlasGridInfos> BlasMatMulOperandInfos for OperandInfos<'_, G> {
    fn hint(&self) -> PrepareHint {
        self.1
    }

    fn source_block_layout(&self) -> CKKSLayout {
        self.0.block_layout()
    }
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
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
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
    fn lookup_tensor_key(&self, k: TorusPrecision) -> poulpy_core::Result<GLWETensorKeyPreparedBackendRef<'_, BE>> {
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

/// `params` with `prec()` at scale `log_delta`, its budget unchanged: given to [`blas_context`] or to an encoding of
/// public matrices, it encodes them at that scale, and given to [`assert_grid_decrypt`], it checks a result through
/// them at their precision.
pub fn at_scale(params: &CKKSTestParams, log_delta: usize) -> CKKSTestParams {
    CKKSTestParams {
        prec_meta: GLWEPlaintextMeta {
            scale: Scale::Log(log_delta),
            ..params.prec_meta
        },
        ..*params
    }
}

/// Scales of maps and public matrices other than the inputs' scale, `params.prec()`'s: a quarter of it below and a
/// quarter above. An operation consumes its maps' scale, whatever its inputs'.
pub fn other_scales(params: &CKKSTestParams) -> [usize; 2] {
    let delta = params.prec().log_delta();
    [delta - delta / 4, delta + delta / 4]
}

/// The context of `plan` at `coeffs_meta` on `module`, allocated, encoded and prepared.
pub fn encoded_context<BE, F>(
    module: &Module<BE>,
    plan: BlasPlan,
    base2k: Base2K,
    coeffs_meta: CoeffsMeta,
    scratch: &mut ScratchArena<'_, BE>,
) -> BlasContext<BE>
where
    BE: Backend + BlasEncodingImpl,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F> + BlasEncodingOps<BE, F> + CKKSLinearTransformationOps<BE>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    let mut ctx = module.blas_context_alloc(plan, base2k, coeffs_meta);
    module.blas_context_encode_into(&mut ctx, scratch).unwrap();
    let mut prepared = module.blas_context_prepared_alloc_from_infos(&ctx);
    module.blas_context_prepare(&mut prepared, &ctx, scratch);
    prepared
}

/// The context of `2^log_d x 2^log_d` blocks at `params.prec()`.
pub fn blas_context<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    log_d: usize,
    scratch: &mut ScratchArena<'_, BE>,
) -> crate::layouts::BlasContext<BE>
where
    BE: TestContextBackend + BlasEncodingImpl,
    Module<BE>: TestContextSharedModule<BE>
        + poulpy_ckks::api::CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + poulpy_ckks::api::CKKSLinearTransformationOps<BE>
        + poulpy_hal::api::CnvPVecAlloc<BE>,
    F: TestScalar + poulpy_core::layouts::DiagonalArithmetic,
{
    encoded_context::<BE, F>(
        module,
        BlasPlan::new(log_d),
        params.base2k.into(),
        params.prec().into(),
        scratch,
    )
}

/// A rotation key for each of `elements`, and no other.
pub fn rotation_keys<BE>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    elements: impl IntoIterator<Item = i64>,
    scratch: &mut ScratchArena<'_, BE>,
) -> HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>
where
    BE: TestContextBackend,
    Module<BE>: TestContextSharedModule<BE>,
{
    elements
        .into_iter()
        .map(|p| {
            (
                p,
                poulpy_ckks::test_suite::helpers::gen_atk(params, module, p, sk_raw, scratch),
            )
        })
        .collect()
}

/// [`blas_context`] plus a rotation key for each Galois element of its plan.
#[allow(clippy::type_complexity)]
pub fn context_and_keys<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    log_d: usize,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    scratch: &mut ScratchArena<'_, BE>,
) -> (
    crate::layouts::BlasContext<BE>,
    HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
)
where
    BE: TestContextBackend + BlasEncodingImpl,
    Module<BE>: TestContextSharedModule<BE>
        + poulpy_ckks::api::CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + poulpy_ckks::api::CKKSLinearTransformationOps<BE>
        + poulpy_hal::api::CnvPVecAlloc<BE>,
    F: TestScalar + poulpy_core::layouts::DiagonalArithmetic,
{
    use poulpy_hal::layouts::CyclotomicOrder;
    let ctx = blas_context::<BE, F>(params, module, log_d, scratch);
    let atks = rotation_keys(
        params,
        module,
        sk_raw,
        ctx.plan().galois_elements(module.cyclotomic_order()),
        scratch,
    );
    (ctx, atks)
}

/// The Galois elements the encoded `maps` rotate by on a ring of cyclotomic order `order`, sorted and deduplicated.
pub fn map_galois_elements<'a, BE: Backend + 'a>(
    maps: impl IntoIterator<Item = &'a LinearTransformationPrepared<BE>>,
    order: i64,
) -> Vec<i64> {
    let mut elements: Vec<i64> = maps.into_iter().flat_map(|lt| lt.galois_elements(order)).collect();
    elements.sort_unstable();
    elements.dedup();
    elements
}

/// Contexts of `plan` an operation on `module` rejects, by label: one encoded on a module of twice the degree, one at a
/// radix one below `params.base2k`. `atks` gains a key for every rotation they take on `module`, so no missing key stops
/// an operation from evaluating them.
pub fn foreign_contexts<BE, F>(
    params: &CKKSTestParams,
    module: &Module<BE>,
    plan: BlasPlan,
    sk_raw: &poulpy_core::layouts::BackendGLWESecret<BE>,
    atks: &mut HashMap<i64, GLWEAutomorphismKeyPrepared<BE::OwnedBuf, BE>>,
    scratch: &mut ScratchArena<'_, BE>,
) -> [(&'static str, crate::layouts::BlasContext<BE>); 2]
where
    BE: TestContextBackend + BlasEncodingImpl,
    Module<BE>: TestContextSharedModule<BE>
        + poulpy_ckks::api::CKKSEncodingOps<BE, F>
        + BlasEncodingOps<BE, F>
        + poulpy_ckks::api::CKKSLinearTransformationOps<BE>
        + poulpy_hal::api::CnvPVecAlloc<BE>,
    F: TestScalar + poulpy_core::layouts::DiagonalArithmetic,
{
    use poulpy_hal::layouts::CyclotomicOrder;
    let double = Module::<BE>::new(2 * params.n as u64);
    let mut encode =
        |module: &Module<BE>, base2k: usize| encoded_context::<BE, F>(module, plan, base2k.into(), params.prec().into(), scratch);
    let contexts = [
        ("degree", encode(&double, params.base2k)),
        ("base2k", encode(module, params.base2k - 1)),
    ];
    for p in contexts
        .iter()
        .flat_map(|(_, ctx)| map_galois_elements(ctx.maps(), module.cyclotomic_order()))
    {
        atks.entry(p)
            .or_insert_with(|| poulpy_ckks::test_suite::helpers::gen_atk(params, module, p, sk_raw, scratch));
    }
    contexts
}
