//! The plan of the circuit, its per-operation key sets, and the per-matrix maps encoded once, at a caller-chosen
//! precision.

use std::collections::BTreeSet;

use anyhow::{Context, Result, ensure};
use poulpy_ckks::{
    CoeffsMeta, GLWEPlaintextInfos, SlotsKind,
    api::{
        CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncodingScalar, CKKSLinearTransformationOps, CKKSModuleInfos,
        LinearTransformation, LinearTransformationPrepared, optimal_bsgs_giant_step,
    },
    layouts::{CKKSModuleAlloc, CKKSPlaintextOwned, ComplexDiagonals},
    reference::ckks_encode_linear_transformation_from_diagonals,
};
use poulpy_core::layouts::{
    Base2K, DiagonalArithmetic, Diagonals, LWEInfos, LinearTransformationLayout, LinearTransformationStrategy,
};
use poulpy_hal::{
    api::CnvPVecAlloc,
    layouts::{Backend, Module, ScratchArena, galois_elements_from_rotations},
};

use super::{BlasGridInfos, ciphertext::mat_slots_in, maps};

type Encoded<BE> = LinearTransformation<CKKSPlaintextOwned<BE>>;

/// The block dimension and the schedule of every map: what a client and a server agree on, and the one source of the
/// rotation keys each operation looks up.
///
/// Shift maps (phi^k, psi^k and their halves) are evaluated `Direct`, one giant step per diagonal; sigma, tau, the
/// transpose and the products by public matrices take the cost-optimal BSGS giant step. The key sets take the ring's
/// cyclotomic order, a power of two whose quarter is the slot count, and build the maps on the host without encoding
/// them, so a client plans its keys with no module, context or public matrix. A pipeline needs the union of its
/// operations' sets, [`sum_galois_elements`] for each grid it sums, and a tensor key if and only if it calls
/// `blas_matmul` or `blas_matmul_prepared`. Every set is empty when a block does not fit the slots.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlasPlan {
    log_d: usize,
}

impl BlasPlan {
    /// The circuit on `2^log_d x 2^log_d` blocks.
    pub fn new(log_d: usize) -> Self {
        Self { log_d }
    }

    /// `log2` of the block dimension.
    pub fn log_d(&self) -> usize {
        self.log_d
    }

    /// The schedule of `map`: `Direct` for a shift map, the cost-optimal BSGS giant step otherwise.
    pub(crate) fn strategy<F>(&self, map: &Diagonals<F>, shift: bool) -> LinearTransformationStrategy {
        if shift {
            LinearTransformationStrategy::Direct
        } else {
            LinearTransformationStrategy::Bsgs {
                giant_step: optimal_bsgs_giant_step(map.indexes(), map.slots()),
            }
        }
    }

    /// Galois elements of every map a context encodes, which cover every operation's set but `blas_rotate`'s.
    pub fn galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            let mut list = self.matmul_maps(slots);
            list.push((maps::transpose(slots, self.log_d), false));
            for k in 1..1 << self.log_d {
                list.extend(self.shift_maps(slots, k, true));
                list.extend(self.shift_maps(slots, k, false));
            }
            list
        })
    }

    /// Galois elements `blas_transpose` looks up.
    pub fn transpose_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| vec![(maps::transpose(slots, self.log_d), false)])
    }

    /// Galois elements `blas_rotate_rows` by `k` looks up, on a grid of any height; none when `d` divides `k`.
    pub fn rotate_rows_galois_elements(&self, cyclotomic_order: i64, k: usize) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.shift_maps(slots, k, true))
    }

    /// Galois elements `blas_rotate_cols` by `k` looks up, on a grid of any width; none when `d` divides `k`.
    pub fn rotate_cols_galois_elements(&self, cyclotomic_order: i64, k: usize) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.shift_maps(slots, k, false))
    }

    /// Galois elements `blas_rotate` by `r` matrix slots looks up; none when the matrix slot count divides `r`.
    pub fn rotate_galois_elements(&self, cyclotomic_order: i64, r: i64) -> Vec<i64> {
        let Some(slots) = self.slots(cyclotomic_order) else {
            return Vec::new();
        };
        let mat_slots = (slots >> (2 * self.log_d)) as i64;
        galois_elements_from_rotations([r.rem_euclid(mat_slots) << (2 * self.log_d)], cyclotomic_order)
    }

    /// Galois elements `blas_matmul`, `blas_matmul_prepare_left` and `blas_matmul_prepare_right` look up, which cover
    /// `blas_matmul_prepared`'s.
    pub fn matmul_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| self.matmul_maps(slots))
    }

    /// Galois elements `blas_matmul_pt` looks up, for public matrices of any values, grid shape and count: the right
    /// product stores a diagonal at every offset `l - c`, whatever the entries.
    pub fn matmul_pt_galois_elements(&self, cyclotomic_order: i64) -> Vec<i64> {
        self.elements(cyclotomic_order, |slots| {
            vec![(maps::right_mul(slots, self.log_d, 1, |_, _, _| 1.0), false)]
        })
    }

    /// The slot count of a ring of cyclotomic order `cyclotomic_order`, if a block fits it.
    fn slots(&self, cyclotomic_order: i64) -> Option<usize> {
        let slots = usize::try_from(cyclotomic_order / 4).ok().filter(|&slots| slots > 0)?;
        (self.log_d <= slots.ilog2() as usize / 2).then_some(slots)
    }

    /// The Galois elements of the maps `maps(slots)` lists, each flagged a shift map, scheduled as the context encodes
    /// them; none when a block does not fit the slots.
    fn elements(&self, cyclotomic_order: i64, maps: impl FnOnce(usize) -> Vec<(Diagonals<f64>, bool)>) -> Vec<i64> {
        let Some(slots) = self.slots(cyclotomic_order) else {
            return Vec::new();
        };
        let mut elements = BTreeSet::new();
        for (map, shift) in maps(slots) {
            let layout = LinearTransformationLayout {
                indexes: map.indexes(),
                slots,
                strategy: self.strategy(&map, shift),
            };
            elements.extend(layout.galois_elements(cyclotomic_order));
        }
        elements.into_iter().collect()
    }

    /// Sigma and tau, then phi^k and psi^k for `1 <= k < d`.
    fn matmul_maps(&self, slots: usize) -> Vec<(Diagonals<f64>, bool)> {
        let log_d = self.log_d;
        let mut list = vec![(maps::sigma(slots, log_d), false), (maps::tau(slots, log_d), false)];
        for k in 1..1 << log_d {
            list.push((maps::rotate_cols(slots, log_d, k), true));
            list.push((maps::rotate_rows(slots, log_d, k), true));
        }
        list
    }

    /// With `k' = k mod d`: psi^k' and its two halves (`rows`), else phi^k' and its two halves; none for `k' = 0`, a
    /// copy.
    fn shift_maps(&self, slots: usize, k: usize, rows: bool) -> Vec<(Diagonals<f64>, bool)> {
        let (log_d, k) = (self.log_d, k % (1 << self.log_d));
        if k == 0 {
            return Vec::new();
        }
        let family = if rows {
            [maps::rotate_rows, maps::rows_upper, maps::rows_lower]
        } else {
            [maps::rotate_cols, maps::cols_left, maps::cols_right]
        };
        family.into_iter().map(|map| (map(slots, log_d, k), true)).collect()
    }
}

/// Encodes a real map with an empty imaginary part, its diagonals marked real, scheduled by `plan` as a shift map
/// (`shift`) or not.
fn encode_map<BE, F>(
    module: &Module<BE>,
    base2k: Base2K,
    mut coeffs_meta: CoeffsMeta,
    diagonals: Diagonals<F>,
    plan: &BlasPlan,
    shift: bool,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<Encoded<BE>>
where
    BE: Backend,
    Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
    F: DiagonalArithmetic + CKKSEncodingScalar,
{
    let strategy = plan.strategy(&diagonals, shift);
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
    plan: BlasPlan,
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
    /// Scratch bytes [`Self::encode`] needs for a plan at `log_d`: the encoding of one `d^2`-slot diagonal.
    pub fn encode_tmp_bytes<F>(module: &Module<BE>, log_d: usize) -> usize
    where
        Module<BE>: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
    {
        <Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(
            module,
            1usize.checked_shl(2 * log_d as u32).expect("d^2 overflows usize"),
        )
    }

    /// Encodes every map of `plan` at `coeffs_meta`, each scheduled as `plan` schedules it. Operations reject the context
    /// on a module whose slot count differs from `module`'s, and on inputs whose blocks are not at `base2k`.
    ///
    /// `scratch` must hold [`Self::encode_tmp_bytes`]; operations size their scratch assuming the diagonals'
    /// `coeffs_meta.k` is not wider than the operand ciphertexts.
    pub fn encode<F>(
        module: &Module<BE>,
        plan: BlasPlan,
        base2k: Base2K,
        coeffs_meta: CoeffsMeta,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        Module<BE>: CKKSModuleAlloc<BE> + CKKSEncodingOps<BE, F>,
        F: DiagonalArithmetic + CKKSEncodingScalar,
    {
        let log_d = plan.log_d();
        ensure_fits(module, log_d)?;
        let slots = module.ckks_max_slots();
        let d = 1usize << log_d;
        let mut encode =
            |diagonals: Diagonals<F>, shift: bool| encode_map(module, base2k, coeffs_meta, diagonals, &plan, shift, scratch);
        let sigma = encode(maps::sigma::<F>(slots, log_d), false)?;
        let tau = encode(maps::tau::<F>(slots, log_d), false)?;
        let transpose = encode(maps::transpose::<F>(slots, log_d), false)?;
        let mut shifts = |map: fn(usize, usize, usize) -> Diagonals<F>| -> Result<Vec<Encoded<BE>>> {
            (1..d).map(|k| encode(map(slots, log_d, k), true)).collect()
        };
        Ok(Self {
            plan,
            slots,
            sigma,
            tau,
            transpose,
            rotate_rows: shifts(maps::rotate_rows::<F>)?,
            rotate_cols: shifts(maps::rotate_cols::<F>)?,
            rows_upper: shifts(maps::rows_upper::<F>)?,
            rows_lower: shifts(maps::rows_lower::<F>)?,
            cols_left: shifts(maps::cols_left::<F>)?,
            cols_right: shifts(maps::cols_right::<F>)?,
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
            plan: self.plan,
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
    plan: BlasPlan,
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
    /// The plan the maps were encoded from.
    pub fn plan(&self) -> &BlasPlan {
        &self.plan
    }

    /// `log2` of the matrix dimension.
    pub fn log_d(&self) -> usize {
        self.plan.log_d()
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
    /// Scratch bytes [`Self::encode`] needs at `log_d` and `count`: the encoding of one `count d^2`-slot diagonal.
    pub fn encode_tmp_bytes<F>(module: &Module<BE>, log_d: usize, count: usize) -> usize
    where
        Module<BE>: CKKSEncodingOps<BE, F>,
        F: CKKSEncodingScalar,
    {
        <Module<BE> as CKKSEncodingHostOps<BE, F>>::ckks_reim_tmp_bytes(
            module,
            1usize
                .checked_shl(2 * log_d as u32)
                .and_then(|d2| d2.checked_mul(count))
                .expect("count d^2 overflows usize"),
        )
    }

    /// Encodes block `(l, j)` as `A -> A B_lj` at `coeffs_meta`, matrix slot `s` reading public matrix `s mod count`
    /// through `entry(m, row, col)`, `m < count`, `row < rows * d`, `col < cols * d`. Operations reject the operand on a
    /// module whose slot count differs from `module`'s, and on inputs whose blocks are not at `base2k`.
    ///
    /// `scratch` must hold [`Self::encode_tmp_bytes`]; operations size their scratch assuming the diagonals'
    /// `coeffs_meta.k` is not wider than the operand ciphertexts.
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
                encode_map(module, base2k, coeffs_meta, diagonals, &BlasPlan::new(log_d), false, scratch)
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
}

/// Galois elements `blas_sum` needs on a grid of layout `infos`: at its recorded batch, or for a layout without one
/// the batch its label proves; none when one block does not fit the slots.
pub fn sum_galois_elements<BE: Backend>(module: &Module<BE>, infos: &impl BlasGridInfos) -> Vec<i64> {
    let mat_slots = infos
        .mat_slots()
        .unwrap_or_else(|| mat_slots_in(module.ckks_max_slots(), infos.log_d(), infos.block_layout().log_sparsity()));
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
    use poulpy_ckks::SetGLWEPlaintextInfos;
    use poulpy_hal::layouts::HostBytesBackend;

    #[test]
    fn sum_keys_plan_from_a_layout() {
        let module = Module::<HostBytesBackend>::new(256);
        let mut ct = module.blas_ciphertext_alloc(2, 1, 1, 19usize.into(), 60usize.into());
        let mut layout = BlasLayout::new(ct.block_layout(), 2, 2, 2);
        // 128 slots hold 8 matrix slots of 4 x 4 blocks: rotations by 16, 32 and 64.
        let want: Vec<i64> = [16, 32, 64].map(|r| module.ckks_galois_element(r)).into();
        assert_eq!(sum_galois_elements(&module, &layout), want);
        assert_eq!(sum_galois_elements(&module, &ct), want);
        layout.block.meta.log_sparsity = 1;
        assert_eq!(sum_galois_elements(&module, &layout), want[..2]);
        // A grid plans from its record, not from its label.
        ct.block_mut(0, 0).set_log_sparsity(1);
        assert_eq!(sum_galois_elements(&module, &ct), want);
        ct.mat_slots = 2;
        assert_eq!(sum_galois_elements(&module, &ct), want[..1]);
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

    /// The key sets on 128 slots (cyclotomic order 512), by hand: a shift map rotates by each of its diagonals, psi^k'
    /// by `k' d` and `k' d - d^2`, phi^k' by `k'` and `k' - d`; BSGS rotates by the baby and giant parts of the others'
    /// diagonals, at the cost-optimal giant step. At `d = 2` those parts are the diagonals themselves. At `d = 4` the
    /// transpose's diagonals, the multiples of 3 from -9 to 9, split at giant step 6; sigma's and the right product's,
    /// -3 to 3, at 125; tau's, the multiples of 4 from -12 to 12, at 116.
    #[test]
    fn plan_sets_by_hand() {
        const ORDER: i64 = 512;
        let rotations = |r: &[i64]| galois_elements_from_rotations(r.iter().copied(), ORDER);

        let plan = BlasPlan::new(1);
        assert_eq!(plan.log_d(), 1);
        assert_eq!(plan.transpose_galois_elements(ORDER), rotations(&[1, 127]));
        assert_eq!(plan.matmul_pt_galois_elements(ORDER), rotations(&[1, 127]));
        assert_eq!(plan.matmul_galois_elements(ORDER), rotations(&[1, 2, 126, 127]));
        assert_eq!(plan.galois_elements(ORDER), rotations(&[1, 2, 126, 127]));
        for k in [1, 3] {
            assert_eq!(plan.rotate_rows_galois_elements(ORDER, k), rotations(&[2, 126]), "k={k}");
            assert_eq!(plan.rotate_cols_galois_elements(ORDER, k), rotations(&[1, 127]), "k={k}");
        }
        for k in [0, 2] {
            assert!(plan.rotate_rows_galois_elements(ORDER, k).is_empty(), "k={k}");
            assert!(plan.rotate_cols_galois_elements(ORDER, k).is_empty(), "k={k}");
        }
        // 32 matrix slots.
        for (r, rotation) in [(1, 4), (-1, 124), (33, 4)] {
            assert_eq!(plan.rotate_galois_elements(ORDER, r), rotations(&[rotation]), "r={r}");
        }
        assert!(plan.rotate_galois_elements(ORDER, 32).is_empty());

        let plan = BlasPlan::new(2);
        assert_eq!(plan.transpose_galois_elements(ORDER), rotations(&[2, 3, 5, 6, 114, 120]));
        assert_eq!(plan.matmul_pt_galois_elements(ORDER), rotations(&[1, 2, 3, 125]));
        // Sigma, tau, then phi^k and psi^k for k = 1, 2, 3.
        let matmul = [1, 2, 3, 125, 4, 8, 12, 116, 1, 125, 2, 126, 3, 127, 4, 116, 8, 120, 12, 124];
        assert_eq!(plan.matmul_galois_elements(ORDER), rotations(&matmul));
        let every: Vec<i64> = matmul.into_iter().chain([2, 3, 5, 6, 114, 120]).collect();
        assert_eq!(plan.galois_elements(ORDER), rotations(&every));
        for k in 1..8 {
            let k1 = (k % 4) as i64;
            let (rows, cols) = (
                plan.rotate_rows_galois_elements(ORDER, k),
                plan.rotate_cols_galois_elements(ORDER, k),
            );
            if k1 == 0 {
                assert!(rows.is_empty() && cols.is_empty(), "k={k}");
            } else {
                assert_eq!(rows, rotations(&[4 * k1, 128 + 4 * k1 - 16]), "k={k}");
                assert_eq!(cols, rotations(&[k1, 128 + k1 - 4]), "k={k}");
            }
        }
        // 8 matrix slots.
        for (r, rotation) in [(1, 16), (-1, 112), (9, 16)] {
            assert_eq!(plan.rotate_galois_elements(ORDER, r), rotations(&[rotation]), "r={r}");
        }
        assert!(plan.rotate_galois_elements(ORDER, 8).is_empty());

        // A block wider than the slots, or a dimension past the word width: no key, no panic.
        for log_d in [4, 40, usize::MAX] {
            let plan = BlasPlan::new(log_d);
            let sets = [
                plan.galois_elements(ORDER),
                plan.transpose_galois_elements(ORDER),
                plan.rotate_rows_galois_elements(ORDER, 1),
                plan.rotate_cols_galois_elements(ORDER, 1),
                plan.rotate_galois_elements(ORDER, 1),
                plan.matmul_galois_elements(ORDER),
                plan.matmul_pt_galois_elements(ORDER),
            ];
            assert!(sets.iter().all(Vec::is_empty), "log_d={log_d}");
        }
    }
}
