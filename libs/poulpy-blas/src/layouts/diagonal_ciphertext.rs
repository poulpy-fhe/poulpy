//! Diagonal ciphertexts: square matrices over the slots of one ciphertext encrypted in diagonal encoding, which appends of
//! a row or a column update in place, and their prepared form, which products read resident.

use poulpy_ckks::{CKKSInfos, GLWEPlaintextInfos};
use poulpy_ckks::{
    CKKSLayout,
    api::{
        CKKSModuleInfos, Diagonal, GiantStep, LinearTransformation, LinearTransformationLayout, LinearTransformationPrepared,
        LtDiagonalMeta, PreparedDiagonal,
    },
    layouts::{CKKSCiphertextOwned, CKKSModuleAlloc},
};
use poulpy_core::{
    DiagonalProd, GLWEBytesOf,
    layouts::{GLWEInfos, GetDegree, LWEInfos},
};
use poulpy_hal::layouts::{PrepareHint, galois_elements_from_rotations};
use std::collections::BTreeSet;

use super::encoding::{DiagonalShape, ring_slots, skeleton};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf},
    layouts::Backend,
};

/// A diagonal ciphertext's diagonals and their schedule: diagonal `t` of the `S x S` matrix over the `S` slots, `D_t[i] =
/// M[i][i + t]`, for every `t` of `lt.indexes`, held as a ciphertext of layout `ct` and grouped as `lt.strategy`
/// schedules them.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct BlasDiagonalCiphertextLayout {
    /// Layout of every diagonal ciphertext: the frame an append's products write, the width of the appended
    /// ciphertext less the mask's scale, at its scale.
    pub ct: CKKSLayout,
    /// The diagonals the matrix holds, the slot count and the schedule.
    pub lt: LinearTransformationLayout,
}

impl BlasDiagonalCiphertextLayout {
    /// A matrix of diagonal ciphertexts of layout `ct`, holding and scheduling the diagonals of `lt`.
    pub fn new(ct: CKKSLayout, lt: LinearTransformationLayout) -> Self {
        Self { ct, lt }
    }
}

/// A diagonal ciphertext's layout, read from a layout or a matrix.
pub trait BlasDiagonalCiphertextInfos {
    /// A snapshot of these infos.
    fn blas_diagonal_ciphertext_layout(&self) -> BlasDiagonalCiphertextLayout;
}

impl<T: BlasDiagonalCiphertextInfos + ?Sized> BlasDiagonalCiphertextInfos for &T {
    fn blas_diagonal_ciphertext_layout(&self) -> BlasDiagonalCiphertextLayout {
        (**self).blas_diagonal_ciphertext_layout()
    }
}

impl<T: BlasDiagonalCiphertextInfos + ?Sized> BlasDiagonalCiphertextInfos for &mut T {
    fn blas_diagonal_ciphertext_layout(&self) -> BlasDiagonalCiphertextLayout {
        (**self).blas_diagonal_ciphertext_layout()
    }
}

impl BlasDiagonalCiphertextInfos for BlasDiagonalCiphertextLayout {
    fn blas_diagonal_ciphertext_layout(&self) -> BlasDiagonalCiphertextLayout {
        self.clone()
    }
}

/// A square matrix over the slots of one ciphertext in diagonal encoding, beside the block grids of
/// [`BlasCiphertext`](super::BlasCiphertext): its diagonals as ciphertexts in BSGS storage, entry `(b, g)` of giant
/// step `g` holding `rot(D_(b + g), -g)`. It is allocated zero; `blas_append_row_diagonal` and
/// `blas_append_col_diagonal` write a row or a column into it, and a product reads it streamed, each diagonal prepared
/// at its use, or prepared once into a [`BlasDiagonalCiphertextPrepared`].
pub struct BlasDiagonalCiphertext<BE: Backend> {
    layout: BlasDiagonalCiphertextLayout,
    pub(crate) diagonals: LinearTransformation<CKKSCiphertextOwned<BE>>,
}

/// A [`BlasDiagonalCiphertext`] prepared for products: its diagonals resident, in the backend's prepared form, filled
/// by `blas_diagonal_ciphertext_prepare`.
pub struct BlasDiagonalCiphertextPrepared<BE: Backend> {
    layout: BlasDiagonalCiphertextLayout,
    pub(crate) diagonals: LinearTransformationPrepared<BE>,
}

impl<BE: Backend> BlasDiagonalCiphertextInfos for BlasDiagonalCiphertext<BE> {
    fn blas_diagonal_ciphertext_layout(&self) -> BlasDiagonalCiphertextLayout {
        self.layout.clone()
    }
}

impl<BE: Backend> BlasDiagonalCiphertextInfos for BlasDiagonalCiphertextPrepared<BE> {
    fn blas_diagonal_ciphertext_layout(&self) -> BlasDiagonalCiphertextLayout {
        self.layout.clone()
    }
}

impl<BE: Backend> BlasDiagonalCiphertext<BE> {
    /// The diagonal ciphertexts, in BSGS storage.
    pub fn diagonals(&self) -> &LinearTransformation<CKKSCiphertextOwned<BE>> {
        &self.diagonals
    }

    /// The diagonal ciphertexts an append writes, keeping their schedule and layout.
    pub fn diagonals_mut(&mut self) -> &mut LinearTransformation<CKKSCiphertextOwned<BE>> {
        &mut self.diagonals
    }
}

impl<BE: Backend> BlasDiagonalCiphertextPrepared<BE> {
    /// The prepared diagonals, in BSGS storage.
    pub fn diagonals(&self) -> &LinearTransformationPrepared<BE> {
        &self.diagonals
    }

    /// The prepared diagonals a preparation fills, keeping their schedule and layout.
    pub fn diagonals_mut(&mut self) -> &mut LinearTransformationPrepared<BE> {
        &mut self.diagonals
    }
}

/// A diagonal ciphertext as a product reads it: its diagonals unprepared, prepared at each use (streamed), or prepared
/// (resident), with the same bits.
pub trait BlasDiagonalCiphertextMap<BE: Backend>: BlasDiagonalCiphertextInfos + Sync {
    /// The form of a diagonal.
    type Diagonal: DiagonalProd<BE> + LtDiagonalMeta + Sync;

    /// Whether the diagonals are ciphertexts a product prepares at each use.
    const STREAMED: bool;

    /// The diagonals, in BSGS storage.
    fn map(&self) -> &LinearTransformation<Self::Diagonal>;
}

impl<BE: Backend> BlasDiagonalCiphertextMap<BE> for BlasDiagonalCiphertext<BE>
where
    CKKSCiphertextOwned<BE>: DiagonalProd<BE> + Sync,
{
    type Diagonal = CKKSCiphertextOwned<BE>;
    const STREAMED: bool = true;

    fn map(&self) -> &LinearTransformation<CKKSCiphertextOwned<BE>> {
        &self.diagonals
    }
}

impl<BE: Backend> BlasDiagonalCiphertextMap<BE> for BlasDiagonalCiphertextPrepared<BE> {
    type Diagonal = PreparedDiagonal<BE::OwnedBuf, BE>;
    const STREAMED: bool = false;

    fn map(&self) -> &LinearTransformationPrepared<BE> {
        &self.diagonals
    }
}

/// The axis of an append: a row reads baby rotations of the appended ciphertext, a column the opposites of the giant
/// rotations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AppendAxis {
    Row,
    Col,
}

impl AppendAxis {
    pub(crate) fn op(self) -> &'static str {
        match self {
            Self::Row => "blas_append_row_diagonal",
            Self::Col => "blas_append_col_diagonal",
        }
    }
}

/// The rotations of the appended ciphertext an append along `axis` makes on a matrix of schedule `lt`, ascending and
/// without repeats: the baby steps its diagonals read for a row, the opposites of its non-empty giant steps for a
/// column.
pub(crate) fn append_rotations<P>(axis: AppendAxis, lt: &LinearTransformation<P>) -> Vec<i64> {
    let steps = lt.giant_steps.iter().filter(|gs| !gs.diagonals.is_empty());
    let rotations: BTreeSet<i64> = match axis {
        AppendAxis::Row => steps.flat_map(|gs| gs.diagonals.iter().map(|d| d.baby)).collect(),
        AppendAxis::Col => steps.map(|gs| -gs.rot).collect(),
    };
    rotations.into_iter().collect()
}

/// Panics unless `layout` fits `module`: its slots the module's, at least one diagonal, each in `0..slots`, diagonal
/// ciphertexts of the module's degree, of rank at least 1, at a radix in `1..=62`.
pub(crate) fn assert_fits(module: &(impl CKKSModuleInfos + GetDegree), layout: &BlasDiagonalCiphertextLayout) {
    let slots = layout.lt.slots;
    assert!(slots == module.ckks_max_slots(), "the matrix's slots are not the module's");
    assert!(
        !layout.lt.indexes.is_empty() && layout.lt.indexes.iter().all(|&t| (0..slots as i64).contains(&t)),
        "the matrix holds no diagonal, or one outside its slots"
    );
    assert!(
        layout.ct.n() == module.ring_degree(),
        "the matrix's diagonals are not of the module's degree"
    );
    assert!(layout.ct.rank().as_usize() >= 1, "the matrix's diagonals are not ciphertexts");
    assert!(
        (1..=62).contains(&layout.ct.base2k().as_usize()),
        "the matrix's radix is outside 1..=62"
    );
}

/// The matrix's prepared diagonals as the planning queries read them: its schedule, every diagonal a ciphertext of the
/// matrix's diagonal layout.
pub(crate) fn shape(layout: &BlasDiagonalCiphertextLayout) -> LinearTransformation<DiagonalShape> {
    let ct = layout.ct;
    let diagonal = DiagonalShape {
        k: ct.k(),
        log_scale: ct.log_delta(),
        slots: ct.slots(),
        log_sparsity: ct.log_sparsity(),
        ciphertext_k: Some(ct.k()),
    };
    skeleton(&layout.lt.index(), diagonal)
}

/// Galois elements [`blas_matvec_diagonal`](crate::api::BlasMatVecOps::blas_matvec_diagonal) looks up for a matrix of `infos` on a ring of
/// cyclotomic order `cyclotomic_order`: the baby and giant rotations of its schedule. Empty when the ring's slots are not
/// the matrix's.
pub fn matvec_diagonal_galois_elements(cyclotomic_order: i64, infos: &impl BlasDiagonalCiphertextInfos) -> Vec<i64> {
    let layout = infos.blas_diagonal_ciphertext_layout();
    if ring_slots(cyclotomic_order) != Some(layout.lt.slots) {
        return Vec::new();
    }
    layout.lt.galois_elements(cyclotomic_order)
}

/// Galois elements [`blas_append_row_diagonal`](crate::api::BlasMatVecOps::blas_append_row_diagonal) looks up for a matrix of `infos` on
/// a ring of cyclotomic order `cyclotomic_order`: the rotations of the appended ciphertext by the baby steps its
/// diagonals read, the identity left out. Empty when the ring's slots are not the matrix's.
pub fn append_row_diagonal_galois_elements(cyclotomic_order: i64, infos: &impl BlasDiagonalCiphertextInfos) -> Vec<i64> {
    append_galois_elements(cyclotomic_order, infos, AppendAxis::Row)
}

/// Galois elements [`blas_append_col_diagonal`](crate::api::BlasMatVecOps::blas_append_col_diagonal) looks up for a matrix of `infos` on
/// a ring of cyclotomic order `cyclotomic_order`: the rotations of the appended ciphertext by the opposites of its giant
/// steps, the identity left out. Empty when the ring's slots are not the matrix's.
pub fn append_col_diagonal_galois_elements(cyclotomic_order: i64, infos: &impl BlasDiagonalCiphertextInfos) -> Vec<i64> {
    append_galois_elements(cyclotomic_order, infos, AppendAxis::Col)
}

fn append_galois_elements(cyclotomic_order: i64, infos: &impl BlasDiagonalCiphertextInfos, axis: AppendAxis) -> Vec<i64> {
    let layout = infos.blas_diagonal_ciphertext_layout();
    if ring_slots(cyclotomic_order) != Some(layout.lt.slots) {
        return Vec::new();
    }
    let slots = layout.lt.slots as i64;
    let rotations = append_rotations(axis, &shape(&layout))
        .into_iter()
        .map(|r| r.rem_euclid(slots));
    galois_elements_from_rotations(rotations, cyclotomic_order)
}

/// [`super::BlasModuleAlloc::blas_diagonal_ciphertext_alloc_from_infos`]: zero diagonals of the layout's metadata.
pub(crate) fn alloc<BE, M>(module: &M, layout: &BlasDiagonalCiphertextLayout) -> BlasDiagonalCiphertext<BE>
where
    BE: Backend,
    M: CKKSModuleAlloc<BE> + GetDegree,
{
    assert_fits(module, layout);
    let plan = layout.lt.index();
    let diagonals = LinearTransformation {
        baby_steps: plan.baby_steps.clone(),
        rotations: plan.rotations,
        giant_steps: plan
            .giant_steps
            .iter()
            .zip(&plan.index)
            .map(|(&rot, babies)| GiantStep {
                rot,
                diagonals: babies
                    .iter()
                    .map(|&baby| Diagonal {
                        baby,
                        plaintext: module.ckks_ciphertext_alloc_from_infos(&layout.ct),
                    })
                    .collect(),
            })
            .collect(),
    };
    BlasDiagonalCiphertext {
        layout: layout.clone(),
        diagonals,
    }
}

/// [`crate::api::BlasDiagonalCiphertextPreparedFactory::blas_diagonal_ciphertext_prepared_alloc_from_infos`]: prepared
/// diagonals of zero carrying the layout's metadata.
pub(crate) fn alloc_prepared<BE, M>(module: &M, layout: &BlasDiagonalCiphertextLayout) -> BlasDiagonalCiphertextPrepared<BE>
where
    BE: Backend,
    M: CKKSModuleInfos + CnvPVecAlloc<BE> + GetDegree,
{
    assert_fits(module, layout);
    let mut diagonals = LinearTransformationPrepared::<BE>::alloc_prepared_from_index(module, &layout.lt.index(), &layout.ct);
    diagonals.set_plaintext_meta(layout.ct.meta);
    BlasDiagonalCiphertextPrepared {
        layout: layout.clone(),
        diagonals,
    }
}

/// The diagonals a matrix of `layout` holds.
fn count(layout: &BlasDiagonalCiphertextLayout) -> usize {
    layout.lt.index().index.iter().map(Vec::len).sum()
}

/// [`super::BlasBytesOf::blas_diagonal_ciphertext_bytes_of_from_infos`]: one ciphertext of the diagonals' layout per
/// diagonal.
pub(crate) fn bytes_of<BE: Backend>(module: &impl GLWEBytesOf<BE>, layout: &BlasDiagonalCiphertextLayout) -> usize {
    count(layout)
        .checked_mul(module.glwe_bytes_of_from_infos(&layout.ct))
        .expect("the diagonal ciphertext's bytes overflow usize")
}

/// [`crate::api::BlasDiagonalCiphertextPreparedFactory::blas_diagonal_ciphertext_prepared_bytes_of_from_infos`]: one
/// prepared right operand of the diagonals' layout per diagonal.
pub(crate) fn prepared_bytes_of(module: &impl CnvPVecBytesOf, layout: &BlasDiagonalCiphertextLayout) -> usize {
    let ct = layout.ct;
    let diagonal = module.cnv_pvec_right_bytes_of(ct.n().as_usize(), ct.rank().as_usize() + 1, ct.size(), PrepareHint::Reuse);
    count(layout)
        .checked_mul(diagonal)
        .expect("the prepared diagonal ciphertext's bytes overflow usize")
}

#[cfg(test)]
mod tests {
    use poulpy_ckks::{
        Scale,
        api::{LinearTransformationStrategy, ckks_linear_transformation_output_layout},
        layouts::CKKSModuleAlloc,
    };
    use poulpy_hal::layouts::{CyclotomicOrder, HostBytesBackend, Module};

    use super::*;
    use crate::layouts::{
        BlasVectorLayout, BlasVectorShape, blas_append_col_diagonal_key_uses, blas_append_row_diagonal_key_uses,
        blas_matvec_diagonal_key_uses, ciphertext::layout_of,
    };

    fn layout(
        module: &Module<HostBytesBackend>,
        indexes: Vec<i64>,
        strategy: LinearTransformationStrategy,
    ) -> BlasDiagonalCiphertextLayout {
        let mut ct = layout_of(&module.ckks_ciphertext_alloc(19usize.into(), 60usize.into()));
        ct.meta.scale = Scale::Log(20);
        BlasDiagonalCiphertextLayout::new(
            ct,
            LinearTransformationLayout {
                indexes,
                slots: module.ckks_max_slots(),
                strategy,
            },
        )
    }

    fn elements(module: &Module<HostBytesBackend>, rotations: impl IntoIterator<Item = i64>) -> Vec<i64> {
        let mut elements: Vec<i64> = rotations.into_iter().map(|r| module.ckks_galois_element(r)).collect();
        elements.sort_unstable();
        elements
    }

    /// A row append rotates by the baby steps its diagonals read, a column by the opposites of the giant steps: of the
    /// diagonals `0..8` at giant step 4, rows by 1 to 3 and columns by -4.
    #[test]
    fn append_key_sets_follow_the_schedule() {
        let module = Module::<HostBytesBackend>::new(64);
        let order = module.cyclotomic_order();
        let m = layout(
            &module,
            (0..8).collect(),
            LinearTransformationStrategy::Bsgs { giant_step: 4 },
        );
        assert_eq!(append_row_diagonal_galois_elements(order, &m), elements(&module, 1..4));
        assert_eq!(append_col_diagonal_galois_elements(order, &m), elements(&module, [-4]));
        let diagonal = layout(&module, vec![0], LinearTransformationStrategy::Direct);
        assert!(append_row_diagonal_galois_elements(order, &diagonal).is_empty());
        assert!(append_col_diagonal_galois_elements(order, &diagonal).is_empty());
        assert!(matvec_diagonal_galois_elements(order, &diagonal).is_empty());
        let foreign = 2 * order;
        assert!(append_row_diagonal_galois_elements(foreign, &m).is_empty());
        assert!(append_col_diagonal_galois_elements(foreign, &m).is_empty());
        assert!(matvec_diagonal_galois_elements(foreign, &m).is_empty());
    }

    /// Under every strategy, the key-use queries list the elements of the matching key sets, the product's baby steps at
    /// the input's width and its giant steps at the result's.
    #[test]
    fn key_uses_list_the_key_sets() {
        let module = Module::<HostBytesBackend>::new(64);
        let order = module.cyclotomic_order();
        let (slots, k) = (module.ckks_max_slots(), 60usize);
        let strategies = [
            LinearTransformationStrategy::Bsgs { giant_step: 4 },
            LinearTransformationStrategy::BsgsPowersOfTwo { giant_step: 4 },
            LinearTransformationStrategy::Direct,
        ];
        for strategy in strategies {
            let m = layout(&module, (0..11).collect(), strategy);
            let v = BlasVectorLayout::new(m.ct, BlasVectorShape::new(slots, slots, 0, 1));
            let mut v_wide = v;
            v_wide.ct.glwe_layout.k = (k + 20).into();
            let uses = blas_matvec_diagonal_key_uses(&module, &v, &m, &v_wide);
            let mut listed: Vec<i64> = uses.rotations.iter().map(|&(p, _)| p).collect();
            listed.sort_unstable();
            listed.dedup();
            assert_eq!(listed, matvec_diagonal_galois_elements(order, &m), "{strategy:?}: product");
            let giant_k = ckks_linear_transformation_output_layout(&shape(&m), v_wide.ct, v.ct.k()).k();
            assert!(
                uses.rotations
                    .iter()
                    .all(|&(_, width)| width == v_wide.ct.k() || width == giant_k),
                "{strategy:?}: widths"
            );
            for (uses, set) in [
                (
                    blas_append_row_diagonal_key_uses(&module, &m, &v),
                    append_row_diagonal_galois_elements(order, &m),
                ),
                (
                    blas_append_col_diagonal_key_uses(&module, &m, &v),
                    append_col_diagonal_galois_elements(order, &m),
                ),
            ] {
                let listed: Vec<i64> = uses.rotations.iter().map(|&(p, _)| p).collect();
                assert_eq!(listed, set, "{strategy:?}: append");
                assert!(
                    uses.rotations.iter().all(|&(_, width)| width == v.ct.k()),
                    "{strategy:?}: append widths"
                );
            }
        }
    }
}
