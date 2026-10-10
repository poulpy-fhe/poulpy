//! The per-matrix maps encoded once, at a caller-chosen precision.

use anyhow::{Context, Result, ensure};
use poulpy_ckks::{
    CKKSInfos, CoeffsMeta, SlotsKind,
    api::{
        CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSModuleInfos, LinearTransformation,
        LinearTransformationPrepared, optimal_bsgs_giant_step,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
    reference::ckks_encode_linear_transformation_from_diagonals,
};
use poulpy_core::layouts::{Base2K, DiagonalArithmetic, Diagonals, LWEInfos, LinearTransformationStrategy};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena},
};

use super::{BlasGridInfos, ciphertext::mat_slots_in, maps};

type Encoded<BE> = LinearTransformation<CKKSPlaintextOwned<BE>>;

/// Encodes a real map with an empty imaginary part, its diagonals marked real; `bsgs` picks the cost-optimal giant step,
/// otherwise one giant step per diagonal.
fn encode_map<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    mut coeffs_meta: CoeffsMeta,
    diagonals: Diagonals<F>,
    bsgs: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Encoded<BE>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    let strategy = if bsgs {
        LinearTransformationStrategy::Bsgs {
            giant_step: optimal_bsgs_giant_step(diagonals.indexes(), diagonals.slots()),
        }
    } else {
        LinearTransformationStrategy::Direct
    };
    let im = Diagonals::new(diagonals.slots());
    coeffs_meta.meta.slots = SlotsKind::Real;
    ckks_encode_linear_transformation_from_diagonals(
        module,
        base2k,
        coeffs_meta,
        &ComplexDiagonals::new(diagonals, im),
        strategy,
        false,
        scratch,
    )
}

fn ensure_fits<BE: Backend>(module: &Module<BE>, log_d: usize) -> Result<()> {
    ensure!(log_d >= 1, "matrices must be at least 2 x 2");
    ensure!(
        log_d <= module.ckks_max_slots().ilog2() as usize / 2,
        "2^{log_d} x 2^{log_d} matrices do not fit the module's slots"
    );
    Ok(())
}

fn prepare_lt<BE>(module: &Module<BE>, lt: &Encoded<BE>, scratch: &mut ScratchArena<'_, BE>) -> LinearTransformationPrepared<BE>
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
{
    let first = lt.first_diagonal_plaintext().expect("every encoded map has a diagonal");
    let mut prepared = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &lt.index(), first);
    module.ckks_prepare_linear_transformation_rhs(&mut prepared, lt, scratch);
    prepared
}

fn prepare_tmp_bytes<BE>(module: &Module<BE>, lt: &Encoded<BE>) -> usize
where
    BE: Backend,
    Module<BE>: CKKSLinearTransformationOps<BE>,
{
    module.ckks_prepare_linear_transformation_rhs_tmp_bytes(
        lt.first_diagonal_plaintext().expect("every encoded map has a diagonal"),
    )
}

/// The JKLS maps of one matrix dimension and their one-diagonal halves, encoded and not yet prepared.
pub struct BlasContextUnprepared<BE: Backend> {
    log_d: usize,
    slots: usize,
    sigma: Encoded<BE>,
    tau: Encoded<BE>,
    transpose: Encoded<BE>,
    rotate_rows: Vec<Encoded<BE>>,
    rotate_cols: Vec<Encoded<BE>>,
    rows_upper: Vec<Encoded<BE>>,
    rows_lower: Vec<Encoded<BE>>,
    cols_left: Vec<Encoded<BE>>,
    cols_right: Vec<Encoded<BE>>,
}

impl<BE: Backend> BlasContextUnprepared<BE> {
    /// Encodes every map for `2^log_d x 2^log_d` matrices at `coeffs_meta`. Operations reject the context on a module
    /// whose slot count differs from `module`'s, and on inputs whose blocks are not at `base2k`.
    ///
    /// `scratch` must hold the backend's encoding of one `d^2`-slot diagonal (`ckks_reim_tmp_bytes(d * d)` plus the
    /// encoder's own needs); operations size their scratch assuming the diagonals' `coeffs_meta.k` is not wider than the
    /// operand ciphertexts.
    pub fn encode<F>(
        module: &Module<BE>,
        log_d: usize,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
    {
        ensure_fits(module, log_d)?;
        let slots = module.ckks_max_slots();
        let d = 1usize << log_d;
        let sigma = encode_map(module, base2k, coeffs_meta, maps::sigma::<F>(slots, log_d), true, scratch)?;
        let tau = encode_map(module, base2k, coeffs_meta, maps::tau::<F>(slots, log_d), true, scratch)?;
        let transpose = encode_map(module, base2k, coeffs_meta, maps::transpose::<F>(slots, log_d), true, scratch)?;
        let mut direct = |map: fn(usize, usize, usize) -> Diagonals<F>| -> Result<Vec<Encoded<BE>>> {
            (1..d)
                .map(|k| encode_map(module, base2k, coeffs_meta, map(slots, log_d, k), false, scratch))
                .collect()
        };
        Ok(Self {
            log_d,
            slots,
            sigma,
            tau,
            transpose,
            rotate_rows: direct(maps::rotate_rows::<F>)?,
            rotate_cols: direct(maps::rotate_cols::<F>)?,
            rows_upper: direct(maps::rows_upper::<F>)?,
            rows_lower: direct(maps::rows_lower::<F>)?,
            cols_left: direct(maps::cols_left::<F>)?,
            cols_right: direct(maps::cols_right::<F>)?,
        })
    }

    /// Scratch bytes [`Self::prepare`] needs.
    pub fn prepare_tmp_bytes(&self, module: &Module<BE>) -> usize
    where
        Module<BE>: CKKSLinearTransformationOps<BE>,
    {
        [&self.sigma, &self.tau, &self.transpose]
            .into_iter()
            .chain(&self.rotate_rows)
            .chain(&self.rotate_cols)
            .chain(&self.rows_upper)
            .chain(&self.rows_lower)
            .chain(&self.cols_left)
            .chain(&self.cols_right)
            .map(|lt| prepare_tmp_bytes(module, lt))
            .max()
            .unwrap_or(0)
    }

    /// Prepares every map into the resident form operations evaluate.
    pub fn prepare(self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>) -> BlasContext<BE>
    where
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    {
        let mut prep = |lt: &Encoded<BE>| prepare_lt(module, lt, scratch);
        BlasContext {
            log_d: self.log_d,
            slots: self.slots,
            sigma: prep(&self.sigma),
            tau: prep(&self.tau),
            transpose: prep(&self.transpose),
            rotate_rows: self.rotate_rows.iter().map(&mut prep).collect(),
            rotate_cols: self.rotate_cols.iter().map(&mut prep).collect(),
            rows_upper: self.rows_upper.iter().map(&mut prep).collect(),
            rows_lower: self.rows_lower.iter().map(&mut prep).collect(),
            cols_left: self.cols_left.iter().map(&mut prep).collect(),
            cols_right: self.cols_right.iter().map(&mut prep).collect(),
        }
    }
}

/// The prepared JKLS maps of one matrix dimension and their one-diagonal halves.
pub struct BlasContext<BE: Backend> {
    log_d: usize,
    slots: usize,
    sigma: LinearTransformationPrepared<BE>,
    tau: LinearTransformationPrepared<BE>,
    transpose: LinearTransformationPrepared<BE>,
    rotate_rows: Vec<LinearTransformationPrepared<BE>>,
    rotate_cols: Vec<LinearTransformationPrepared<BE>>,
    rows_upper: Vec<LinearTransformationPrepared<BE>>,
    rows_lower: Vec<LinearTransformationPrepared<BE>>,
    cols_left: Vec<LinearTransformationPrepared<BE>>,
    cols_right: Vec<LinearTransformationPrepared<BE>>,
}

impl<BE: Backend> BlasContext<BE> {
    /// `log2` of the matrix dimension.
    pub fn log_d(&self) -> usize {
        self.log_d
    }

    /// Slots of the module the maps were encoded on.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Radix of the maps' diagonals.
    pub fn base2k(&self) -> Base2K {
        self.sigma
            .first_diagonal_plaintext()
            .expect("every encoded map has a diagonal")
            .base2k()
    }

    /// JKLS sigma.
    pub fn sigma(&self) -> &LinearTransformationPrepared<BE> {
        &self.sigma
    }

    /// JKLS tau.
    pub fn tau(&self) -> &LinearTransformationPrepared<BE> {
        &self.tau
    }

    /// Per-matrix transpose.
    pub fn transpose(&self) -> &LinearTransformationPrepared<BE> {
        &self.transpose
    }

    /// psi^k, for `1 <= k < d`.
    pub fn rotate_rows(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        &self.rotate_rows[k - 1]
    }

    /// phi^k, for `1 <= k < d`.
    pub fn rotate_cols(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        &self.rotate_cols[k - 1]
    }

    /// The rows of psi^k that stay in the block, for `1 <= k < d`.
    pub fn rows_upper(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        &self.rows_upper[k - 1]
    }

    /// The rows of psi^k that wrap around the block, for `1 <= k < d`.
    pub fn rows_lower(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        &self.rows_lower[k - 1]
    }

    /// The columns of phi^k that stay in the block, for `1 <= k < d`.
    pub fn cols_left(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        &self.cols_left[k - 1]
    }

    /// The columns of phi^k that wrap around the block, for `1 <= k < d`.
    pub fn cols_right(&self, k: usize) -> &LinearTransformationPrepared<BE> {
        &self.cols_right[k - 1]
    }

    /// Galois elements of every rotation the maps use.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let mut elements: Vec<i64> = [&self.sigma, &self.tau, &self.transpose]
            .into_iter()
            .chain(&self.rotate_rows)
            .chain(&self.rotate_cols)
            .chain(&self.rows_upper)
            .chain(&self.rows_lower)
            .chain(&self.cols_left)
            .chain(&self.cols_right)
            .flat_map(|lt| lt.galois_elements(cyclotomic_order))
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }
}

/// A grid of right products by public matrices, encoded and not yet prepared.
pub struct BlasPtOperandUnprepared<BE: Backend> {
    log_d: usize,
    slots: usize,
    rows: usize,
    cols: usize,
    count: usize,
    blocks: Vec<Encoded<BE>>,
}

impl<BE: Backend> BlasPtOperandUnprepared<BE> {
    /// Encodes block `(l, j)` as `A -> A B_lj` at `coeffs_meta`, matrix slot `s` reading public matrix `s mod count`
    /// through `entry(m, row, col)`, `m < count`, `row < rows * d`, `col < cols * d`. Operations reject the operand on a
    /// module whose slot count differs from `module`'s, and on inputs whose blocks are not at `base2k`.
    ///
    /// `scratch` must hold the backend's encoding of one `count d^2`-slot diagonal (`ckks_reim_tmp_bytes(count * d * d)`
    /// plus the encoder's own needs); operations size their scratch assuming the diagonals' `coeffs_meta.k` is not wider
    /// than the operand ciphertexts.
    #[allow(clippy::too_many_arguments)]
    pub fn encode<F>(
        module: &Module<BE>,
        log_d: usize,
        rows: usize,
        cols: usize,
        count: usize,
        entry: impl Fn(usize, usize, usize) -> F,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
    {
        ensure_fits(module, log_d)?;
        ensure!(rows > 0 && cols > 0, "a grid has at least one block");
        let (slots, d) = (module.ckks_max_slots(), 1usize << log_d);
        let mat_slots = slots >> (2 * log_d);
        ensure!(
            count > 0 && mat_slots.is_multiple_of(count),
            "{count} public matrices do not divide the {mat_slots} matrix slots"
        );
        let blocks = (0..rows.checked_mul(cols).context("rows * cols overflows usize")?)
            .map(|t| {
                let (l, j) = (t / cols, t % cols);
                let diagonals = maps::right_mul(slots, log_d, count, |m, r, c| entry(m, l * d + r, j * d + c));
                encode_map(module, base2k, coeffs_meta, diagonals, true, scratch)
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            log_d,
            slots,
            rows,
            cols,
            count,
            blocks,
        })
    }

    /// Scratch bytes [`Self::prepare`] needs.
    pub fn prepare_tmp_bytes(&self, module: &Module<BE>) -> usize
    where
        Module<BE>: CKKSLinearTransformationOps<BE>,
    {
        self.blocks.iter().map(|lt| prepare_tmp_bytes(module, lt)).max().unwrap_or(0)
    }

    /// Prepares every block into the resident form operations evaluate.
    pub fn prepare(self, module: &Module<BE>, scratch: &mut ScratchArena<'_, BE>) -> BlasPtOperand<BE>
    where
        Module<BE>: CKKSLinearTransformationOps<BE> + CnvPVecAlloc<BE>,
    {
        BlasPtOperand {
            log_d: self.log_d,
            slots: self.slots,
            rows: self.rows,
            cols: self.cols,
            count: self.count,
            blocks: self.blocks.iter().map(|lt| prepare_lt(module, lt, scratch)).collect(),
        }
    }
}

/// A prepared grid of right products by public matrices.
pub struct BlasPtOperand<BE: Backend> {
    log_d: usize,
    slots: usize,
    rows: usize,
    cols: usize,
    count: usize,
    blocks: Vec<LinearTransformationPrepared<BE>>,
}

impl<BE: Backend> BlasPtOperand<BE> {
    /// `log2` of the block dimension.
    pub fn log_d(&self) -> usize {
        self.log_d
    }

    /// Slots of the module the blocks were encoded on.
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// Radix of the blocks' diagonals.
    pub fn base2k(&self) -> Base2K {
        self.blocks[0]
            .first_diagonal_plaintext()
            .expect("every encoded map has a diagonal")
            .base2k()
    }

    /// Number of block rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of block columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of distinct public matrices, matrix slot `s` using matrix `s mod count`.
    pub fn count(&self) -> usize {
        self.count
    }

    /// The prepared map `A -> A B_lj` of block `(l, j)`.
    pub fn block(&self, l: usize, j: usize) -> &LinearTransformationPrepared<BE> {
        assert!(
            l < self.rows && j < self.cols,
            "block ({l}, {j}) is outside a {} x {} grid",
            self.rows,
            self.cols
        );
        self.at(l, j)
    }

    /// Block `(l, j)` without the bounds assertion, for the references' loops after validation.
    pub(crate) fn at(&self, l: usize, j: usize) -> &LinearTransformationPrepared<BE> {
        &self.blocks[l * self.cols + j]
    }

    /// Galois elements of every rotation the blocks use.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        let mut elements: Vec<i64> = self
            .blocks
            .iter()
            .flat_map(|lt| lt.galois_elements(cyclotomic_order))
            .collect();
        elements.sort_unstable();
        elements.dedup();
        elements
    }
}

/// Galois elements `blas_sum` needs on a grid of layout `infos`, none when one block does not fit the slots.
pub fn sum_galois_elements<BE: Backend>(module: &Module<BE>, infos: &impl BlasGridInfos) -> Vec<i64> {
    let mat_slots = mat_slots_in(module.ckks_max_slots(), infos.log_d(), infos.block_layout().log_sparsity());
    sum_rotations(infos.log_d(), mat_slots)
        .map(|rotation| module.ckks_galois_element(rotation))
        .collect()
}

/// The rotations of `blas_sum` on `2^log_d x 2^log_d` blocks holding `mat_slots` matrices: `2^i d^2` for
/// `i < log2(mat_slots)`.
pub(crate) fn sum_rotations(log_d: usize, mat_slots: usize) -> impl Iterator<Item = i64> {
    (0..mat_slots.max(1).ilog2()).map(move |i| 1i64 << (2 * log_d + i as usize))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layouts::{BlasLayout, BlasModuleAlloc};
    use poulpy_ckks::test_suite::{BASE19_PARAMS_F64, helpers::alloc_scratch};
    use poulpy_cpu_oracle::FFT64Oracle;
    use poulpy_hal::{api::ScratchOwnedBorrow, layouts::CyclotomicOrder};

    fn module() -> Module<FFT64Oracle> {
        Module::<FFT64Oracle>::new(BASE19_PARAMS_F64.n as u64)
    }

    #[test]
    fn context_lists_the_rotations_of_every_map() {
        let module = module();
        let params = BASE19_PARAMS_F64;
        let mut scratch = alloc_scratch(&params, &module);
        let unprepared =
            BlasContextUnprepared::encode::<f64>(&module, 2, params.base2k.into(), params.prec().into(), &mut scratch.borrow())
                .unwrap();
        let ctx = unprepared.prepare(&module, &mut scratch.borrow());
        assert_eq!(ctx.log_d(), 2);
        let elements = ctx.galois_elements(module.cyclotomic_order());
        // phi^1 rotates by 1 and by 1 - d.
        assert!(elements.contains(&module.ckks_galois_element(1)));
        assert!(elements.contains(&module.ckks_galois_element(1 - 4)));
    }

    #[test]
    fn context_rejects_matrices_wider_than_the_slots() {
        let module = module();
        let params = BASE19_PARAMS_F64;
        let mut scratch = alloc_scratch(&params, &module);
        // 128 slots hold no 16 x 16 matrix; 2^40 x 2^40 has more entries than a usize counts.
        for log_d in [4, 40] {
            let res = BlasContextUnprepared::encode::<f64>(
                &module,
                log_d,
                params.base2k.into(),
                params.prec().into(),
                &mut scratch.borrow(),
            );
            assert!(res.is_err());
        }
    }

    #[test]
    fn sum_keys_plan_from_a_layout() {
        let module = module();
        let ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        let mut layout = BlasLayout {
            block: ct.block_layout(),
            log_d: 2,
            rows: 2,
            cols: 2,
        };
        // 128 slots hold 8 matrix slots of 4 x 4 blocks: rotations by 16, 32 and 64.
        let want: Vec<i64> = [16, 32, 64].map(|r| module.ckks_galois_element(r)).into();
        assert_eq!(sum_galois_elements(&module, &layout), want);
        assert_eq!(sum_galois_elements(&module, &ct), want);
        layout.block.meta.log_sparsity = 1;
        assert_eq!(sum_galois_elements(&module, &layout), want[..2]);
        // No block fits, or a shift is past the word width: no key, no panic.
        for log_sparsity in [64, 65] {
            layout.block.meta.log_sparsity = log_sparsity;
            assert!(
                sum_galois_elements(&module, &layout).is_empty(),
                "log_sparsity={log_sparsity}"
            );
        }
        layout.block.meta.log_sparsity = 0;
        for log_d in [4, 40, usize::MAX] {
            layout.log_d = log_d;
            assert!(sum_galois_elements(&module, &layout).is_empty(), "log_d={log_d}");
        }
    }

    #[test]
    fn pt_operand_is_a_grid_of_maps() {
        let module = module();
        let params = BASE19_PARAMS_F64;
        let mut scratch = alloc_scratch(&params, &module);
        let entry = |m: usize, r: usize, c: usize| ((m + r * 3 + c) % 5) as f64 - 2.0;
        let encode = |count: usize, scratch: &mut ScratchArena<'_, FFT64Oracle>| {
            BlasPtOperandUnprepared::encode(
                &module,
                2,
                3,
                2,
                count,
                entry,
                params.base2k.into(),
                params.prec().into(),
                scratch,
            )
        };
        // 128 slots hold 8 matrix slots: 3 and 0 public matrices do not divide them.
        for count in [0, 3] {
            assert!(encode(count, &mut scratch.borrow()).is_err());
        }
        let b = encode(2, &mut scratch.borrow())
            .unwrap()
            .prepare(&module, &mut scratch.borrow());
        assert_eq!((b.log_d(), b.rows(), b.cols(), b.count()), (2, 3, 2, 2));
        let order = module.cyclotomic_order();
        assert_eq!(b.galois_elements(order), b.block(2, 1).galois_elements(order));
    }
}
