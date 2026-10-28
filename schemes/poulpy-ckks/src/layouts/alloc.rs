use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWEPlaintextLayout, GetDegree, LWEInfos, LinearTransformation, LinearTransformationDiagonal,
    LinearTransformationGiantStep, LinearTransformationLayout, LinearTransformationPrepared, ModuleCoreAlloc, Rank,
    TorusPrecision,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf},
    layouts::{Backend, Module, PrepareHint},
};

use crate::{CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, api::CKKSModuleInfos};

use super::{
    CKKSCiphertext, CKKSCiphertextOwned, CKKSPlaintext, CKKSPlaintextOwned, CKKSPreparedLeft, CKKSPreparedRight, DFTMatrix,
    DFTMatrixFactors, DFTMatrixPrepared,
};

/// CKKS container allocation on a backend module.
///
/// Every method is default-bodied over the [`ModuleCoreAlloc`], [`CnvPVecAlloc`] and [`CnvPVecBytesOf`] supertraits, so
/// the blanket impl for `Module<BE>` is empty: the whole constructor matrix is
/// two primitive shapes (ciphertext with explicit rank, plaintext with explicit
/// degree) plus thin conveniences over them, and the prepared operands and DFT matrices with their byte counts.
pub trait CKKSModuleAlloc<BE: Backend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CnvPVecAlloc<BE> + CnvPVecBytesOf + CKKSModuleInfos
{
    /// Allocates a ciphertext with `infos`' layout **and** its CKKS metadata
    /// (`log_delta`, `log_sparsity`), mirroring
    /// [`Self::ckks_plaintext_alloc_from_infos`]. Use
    /// [`Self::ckks_ciphertext_alloc_from_glwe_infos`] when only a GLWE layout
    /// is available (fresh default metadata).
    fn ckks_ciphertext_alloc_from_infos<A>(&self, infos: &A) -> CKKSCiphertextOwned<BE>
    where
        A: GLWEInfos + CKKSInfos,
    {
        CKKSCiphertext::from_inner(self.glwe_alloc_from_infos(infos), infos.meta())
    }

    /// Allocates a default-meta ciphertext from a bare GLWE layout; the name
    /// makes the metadata drop explicit.
    fn ckks_ciphertext_alloc_from_glwe_infos<A>(&self, infos: &A) -> CKKSCiphertextOwned<BE>
    where
        A: GLWEInfos,
    {
        CKKSCiphertext::from_inner(self.glwe_alloc_from_infos(infos), GLWEPlaintextMeta::default())
    }

    /// Allocates a default-meta ciphertext of the given `rank`.
    fn ckks_ciphertext_alloc_with_rank(&self, base2k: Base2K, k: TorusPrecision, rank: Rank) -> CKKSCiphertextOwned<BE> {
        CKKSCiphertext::from_inner(self.glwe_alloc(base2k, k, rank), GLWEPlaintextMeta::default())
    }

    /// Rank-1 convenience over [`Self::ckks_ciphertext_alloc_with_rank`].
    fn ckks_ciphertext_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSCiphertextOwned<BE> {
        self.ckks_ciphertext_alloc_with_rank(base2k, k, Rank(1))
    }

    fn ckks_plaintext_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintextOwned<BE>
    where
        A: CKKSInfos,
    {
        let mut pt = self.ckks_plaintext_alloc(infos.n(), infos.base2k(), infos.k());
        pt.set_meta(infos.meta());
        pt
    }

    /// Allocates a default-meta plaintext sized to `k` over `base2k`. The semantic
    /// [`GLWEPlaintextMeta`] is not needed to size the buffer — set it afterwards with
    /// [`SetCKKSInfos::set_meta`] (the `_from_infos` variants do this for you).
    fn ckks_plaintext_alloc(&self, n: Degree, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE> {
        // `k` is the effective torus width (`log_delta + log_budget`); the buffer
        // auto-sizes to `ceil(k / base2k)` limbs, so the integer-poly storage spans
        // `max_k` while `k` records the meaningful precision. The semantic meta is
        // independent of sizing and defaults here; callers set it via `set_meta`.
        CKKSPlaintext::from_inner(
            self.glwe_plaintext_alloc_from_infos(&GLWEPlaintextLayout { n, base2k, k }),
            GLWEPlaintextMeta::default(),
        )
    }

    /// Coefficient-count convenience over [`Self::ckks_plaintext_alloc`].
    fn ckks_pt_coeffs_alloc(&self, coeff_count: usize, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE> {
        self.ckks_plaintext_alloc(coeff_count.into(), base2k, k)
    }

    /// A plaintext of `slots` slots, at the degree that holds them: `2 * slots` on the standard ring, `slots` on the
    /// invariant ring; [`CKKSModuleInfos::ckks_max_slots`] gives one at the ring degree.
    ///
    /// Panics when `slots` is not a power of two, exceeds `ckks_max_slots`, or needs a degree below the backend's
    /// minimum.
    fn ckks_pt_vec_alloc(&self, slots: usize, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE>
    where
        Self: GetDegree,
    {
        let max_slots = self.ckks_max_slots();
        assert!(
            slots.is_power_of_two() && slots <= max_slots,
            "ckks_pt_vec_alloc: the slot count must be a power of two up to the module's slots"
        );
        let n = slots * (self.ring_degree().as_usize() / max_slots);
        assert!(
            n >= BE::MIN_DEGREE,
            "ckks_pt_vec_alloc: the slot count needs a degree below the backend's minimum"
        );
        self.ckks_plaintext_alloc(n.into(), base2k, k)
    }

    /// A transform of `layout`'s schedule, a zero plaintext of `infos` at every diagonal; `infos`' degree must be the
    /// encoder's, that of a plaintext of the diagonals' period in slots. It is the schedule
    /// [`ComplexDiagonals::build_transform`](crate::layouts::ComplexDiagonals::build_transform) produces for that
    /// layout, so the receiver of an encoding into a transform.
    fn ckks_linear_transformation_alloc_from_infos<A>(
        &self,
        layout: &LinearTransformationLayout,
        infos: &A,
    ) -> LinearTransformation<CKKSPlaintextOwned<BE>>
    where
        A: CKKSInfos,
    {
        transform_of(layout, || self.ckks_plaintext_alloc_from_infos(infos))
    }

    /// A transform of `layout`'s schedule whose diagonals are ciphertexts of `infos` prepared under `hint`, the
    /// receiver of [`CKKSMulOps::ckks_prepare_right_into`](crate::api::CKKSMulOps::ckks_prepare_right_into) at every
    /// diagonal for
    /// [`CKKSLinearTransformationOps::ckks_eval_ct_linear_transformation_into`](crate::api::CKKSLinearTransformationOps::ckks_eval_ct_linear_transformation_into):
    /// the slot of giant step `rot` and baby rotation `baby` takes diagonal `rot + baby` rotated by `-rot` slots.
    fn ckks_ct_linear_transformation_alloc_from_infos<A>(
        &self,
        layout: &LinearTransformationLayout,
        infos: &A,
        hint: PrepareHint,
    ) -> LinearTransformation<CKKSPreparedRight<BE>>
    where
        A: GLWEInfos,
    {
        transform_of(layout, || self.ckks_prepared_right_alloc_from_infos(infos, hint))
    }

    /// Bytes of a left operand [`Self::ckks_prepared_left_alloc_from_infos`] allocates for `infos` under `hint`.
    fn ckks_prepared_left_bytes_of_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        A: GLWEInfos,
    {
        self.bytes_of_cnv_pvec_left(infos.n().as_usize(), infos.rank().as_usize() + 1, infos.size(), hint)
    }

    /// Bytes of a right operand [`Self::ckks_prepared_right_alloc_from_infos`] allocates for `infos` under `hint`.
    fn ckks_prepared_right_bytes_of_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> usize
    where
        A: GLWEInfos,
    {
        self.bytes_of_cnv_pvec_right(infos.n().as_usize(), infos.rank().as_usize() + 1, infos.size(), hint)
    }

    /// A left operand for a ciphertext of `infos`, on its `ceil(k / base2k)` limbs and `rank + 1` columns, in the
    /// representation `hint` selects, as [`CKKSMulOps::ckks_prepare_right`](crate::api::CKKSMulOps::ckks_prepare_right)
    /// describes it, which [`CKKSMulOps::ckks_prepare_left_into`](crate::api::CKKSMulOps::ckks_prepare_left_into) fills.
    /// Its contents are unspecified until then, and its infos report `infos` and no metadata.
    fn ckks_prepared_left_alloc_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> CKKSPreparedLeft<BE>
    where
        A: GLWEInfos,
    {
        let size = infos.size();
        CKKSPreparedLeft {
            prep: self.cnv_pvec_left_alloc(infos.n().as_usize(), infos.rank().as_usize() + 1, size, hint),
            size,
            layout: infos.glwe_layout(),
            meta: None,
        }
    }

    /// As [`Self::ckks_prepared_left_alloc_from_infos`], a right operand, which
    /// [`CKKSMulOps::ckks_prepare_right_into`](crate::api::CKKSMulOps::ckks_prepare_right_into) fills.
    fn ckks_prepared_right_alloc_from_infos<A>(&self, infos: &A, hint: PrepareHint) -> CKKSPreparedRight<BE>
    where
        A: GLWEInfos,
    {
        let size = infos.size();
        CKKSPreparedRight {
            prep: self.cnv_pvec_right_alloc(infos.n().as_usize(), infos.rank().as_usize() + 1, size, hint),
            size,
            layout: infos.glwe_layout(),
            meta: None,
        }
    }

    /// Bytes of the prepared matrix [`Self::ckks_dft_matrix_prepared_alloc_from_infos`] allocates for `dft`.
    ///
    /// # Panics
    ///
    /// As [`Self::ckks_dft_matrix_prepared_alloc_from_infos`], and when the byte count overflows `usize`.
    fn ckks_dft_matrix_prepared_bytes_of_from_infos<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    ) -> usize
    where
        P: LWEInfos,
    {
        dft.factor_operands().iter().fold(0usize, |bytes, factor| {
            let first = factor
                .first_diagonal_plaintext()
                .expect("ckks_dft_matrix_prepared_bytes_of_from_infos: every factor has a diagonal");
            let diagonals: usize = factor.index().index.iter().map(Vec::len).sum();
            diagonals
                .checked_mul(self.bytes_of_cnv_pvec_right(first.n().as_usize(), 1, first.size(), PrepareHint::Reuse))
                .and_then(|factor_bytes| bytes.checked_add(factor_bytes))
                .expect("ckks_dft_matrix_prepared_bytes_of_from_infos: the byte count overflows usize")
        })
    }

    /// A prepared matrix of `dft`'s plan and factor schedules, each diagonal at the degree, radix and width of the
    /// factor's first diagonal, which [`CKKSDFTOps::ckks_prepare_dft_matrix`](crate::api::CKKSDFTOps::ckks_prepare_dft_matrix)
    /// fills. Its diagonals are unspecified until then.
    ///
    /// # Panics
    ///
    /// When a factor of `dft` has no diagonal.
    fn ckks_dft_matrix_prepared_alloc_from_infos<Dir, Fmt, P>(
        &self,
        dft: &DFTMatrix<BE, Dir, Fmt, LinearTransformation<P>>,
    ) -> DFTMatrixPrepared<BE, Dir, Fmt>
    where
        Self: Sized,
        P: LWEInfos,
    {
        let factors = dft
            .factor_operands()
            .iter()
            .map(|factor| {
                let first = factor
                    .first_diagonal_plaintext()
                    .expect("ckks_dft_matrix_prepared_alloc_from_infos: every factor has a diagonal");
                LinearTransformationPrepared::<BE>::alloc_prepared_from_index(self, &factor.index(), first)
            })
            .collect();
        DFTMatrix::from_factors(DFTMatrixFactors::new(dft.plan().clone(), factors))
    }
}

impl<BE: Backend> CKKSModuleAlloc<BE> for Module<BE> where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}

/// A transform of `layout`'s schedule with a diagonal of `diagonal()` at every slot.
fn transform_of<P>(layout: &LinearTransformationLayout, mut diagonal: impl FnMut() -> P) -> LinearTransformation<P> {
    let plan = layout.index();
    LinearTransformation {
        giant_steps: plan
            .giant_steps
            .iter()
            .zip(&plan.index)
            .map(|(&rot, babies)| LinearTransformationGiantStep {
                rot,
                diagonals: babies
                    .iter()
                    .map(|&baby| LinearTransformationDiagonal {
                        baby,
                        plaintext: diagonal(),
                    })
                    .collect(),
            })
            .collect(),
        baby_steps: plan.baby_steps,
        rotations: plan.rotations,
    }
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{
        Diagonals, GLWELayout, LWEInfos, LinearTransformationLayout, LinearTransformationStrategy, Rank, TorusPrecision,
    };
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    use super::*;
    use crate::{CKKSLayout, Scale, SlotsKind, layouts::ComplexDiagonals};

    /// A transform allocated from a layout has the schedule `build_transform` encodes into, with a zero plaintext of
    /// the given infos at every diagonal, at the degree of the diagonals' period.
    #[test]
    fn transform_alloc_is_the_encoding_receiver() {
        let module = Module::<HostBytesBackend>::new(64);
        let mut cd = ComplexDiagonals::new(Diagonals::new(32), Diagonals::new(32));
        cd.re.set(0, vec![1.0, 2.0]);
        cd.im.set(0, vec![0.5]);
        cd.re.set(1, vec![3.0]);
        cd.re.set(5, vec![-1.0, 4.0, 2.0, 0.5]);
        cd.im.set(7, vec![1.0, -2.0]);
        let strategy = LinearTransformationStrategy::Bsgs { giant_step: 4 };
        let layout = LinearTransformationLayout {
            indexes: cd.indexes(),
            slots: cd.slots(),
            strategy,
        };
        let n = module.ckks_pt_vec_alloc(cd.period(), Base2K(12), TorusPrecision(24)).n();
        let infos = CKKSLayout {
            glwe_layout: GLWELayout {
                n,
                base2k: Base2K(12),
                k: TorusPrecision(24),
                rank: Rank(0),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(10),
                slots: SlotsKind::Real,
                log_sparsity: 0,
            },
        };
        let built = cd.build_transform(strategy, |_, _| module.ckks_plaintext_alloc_from_infos(&infos));
        let alloc = module.ckks_linear_transformation_alloc_from_infos(&layout, &infos);
        let schedule = |t: &LinearTransformation<_>| -> Vec<(i64, Vec<i64>)> {
            t.giant_steps
                .iter()
                .map(|gs| (gs.rot, gs.diagonals.iter().map(|d| d.baby).collect()))
                .collect()
        };
        assert_eq!((&alloc.baby_steps, schedule(&alloc)), (&built.baby_steps, schedule(&built)));
        assert!(
            alloc
                .giant_steps
                .iter()
                .flat_map(|gs| &gs.diagonals)
                .all(|d| d.plaintext.n() == n && d.plaintext.meta() == infos.meta)
        );
    }
}
