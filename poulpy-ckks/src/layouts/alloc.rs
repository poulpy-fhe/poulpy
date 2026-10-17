use poulpy_core::layouts::{
    Base2K, Degree, GLWEInfos, GLWEPlaintextLayout, GetDegree, LinearTransformation, LinearTransformationDiagonal,
    LinearTransformationGiantStep, LinearTransformationLayout, ModuleCoreAlloc, Rank, TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::{CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, api::CKKSModuleInfos};

use super::{CKKSCiphertext, CKKSCiphertextOwned, CKKSPlaintext, CKKSPlaintextOwned};

/// CKKS container allocation on a backend module.
///
/// Every method is default-bodied over the [`ModuleCoreAlloc`] supertrait, so
/// the blanket impl for `Module<BE>` is empty: the whole constructor matrix is
/// two primitive shapes (ciphertext with explicit rank, plaintext with explicit
/// degree) plus thin conveniences over them.
pub trait CKKSModuleAlloc<BE: Backend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + CKKSModuleInfos
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

    /// Full-ring-degree convenience over [`Self::ckks_plaintext_alloc`].
    fn ckks_pt_vec_alloc(&self, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE>
    where
        Self: GetDegree,
    {
        self.ckks_plaintext_alloc(self.ring_degree(), base2k, k)
    }

    /// Degree of the compact plaintext for `slots` slots: `slots * N / ckks_max_slots` (`2 * slots` on the standard
    /// ring, `slots` on the invariant ring), raised to the backend's minimum degree and capped at the ring degree.
    fn ckks_compact_degree(&self, slots: usize) -> Degree
    where
        Self: GetDegree,
    {
        assert!(slots.is_power_of_two(), "a compact plaintext holds a power-of-two slot count");
        let ring = self.ring_degree().as_usize();
        slots
            .checked_mul(ring / self.ckks_max_slots())
            .expect("slot count overflow")
            .max(BE::MIN_DEGREE)
            .min(ring)
            .into()
    }

    /// Allocates the compact plaintext for `slots` slots, at [`Self::ckks_compact_degree`]. Its value under the ring
    /// embedding is `M(X^(N/n))` for the degree-`n` polynomial `M` it stores; `slots == ckks_max_slots` gives the dense
    /// plaintext.
    fn ckks_pt_vec_alloc_compact(&self, slots: usize, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE>
    where
        Self: GetDegree,
    {
        self.ckks_plaintext_alloc(self.ckks_compact_degree(slots), base2k, k)
    }

    /// A transform of `layout`'s schedule, a zero plaintext of `infos` at every diagonal; `infos`' degree must be the
    /// encoder's, for example `ckks_compact_degree(period)` for compact diagonals. It is the schedule
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
                            plaintext: self.ckks_plaintext_alloc_from_infos(infos),
                        })
                        .collect(),
                })
                .collect(),
            baby_steps: plan.baby_steps,
        }
    }
}

impl<BE: Backend> CKKSModuleAlloc<BE> for Module<BE> where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}

#[cfg(test)]
mod tests {
    use poulpy_core::layouts::{
        GLWELayout, LWEInfos, LinearTransformationLayout, LinearTransformationStrategy, Rank, TorusPrecision,
    };
    use poulpy_hal::layouts::{HostBytesBackend, Module};

    use super::*;
    use crate::{CKKSLayout, Scale, SlotsKind};

    fn layout_of(indexes: Vec<i64>) -> LinearTransformationLayout {
        LinearTransformationLayout {
            indexes,
            slots: 32,
            strategy: LinearTransformationStrategy::Bsgs { giant_step: 4 },
        }
    }

    fn infos_of(k: u32) -> CKKSLayout {
        CKKSLayout {
            glwe_layout: GLWELayout {
                n: Degree(16),
                base2k: Base2K(12),
                k: TorusPrecision(k),
                rank: Rank(0),
            },
            meta: GLWEPlaintextMeta {
                scale: Scale::Log(10),
                slots: SlotsKind::Real,
                log_sparsity: 0,
            },
        }
    }

    #[test]
    fn transforms_allocate_the_layout_schedule() {
        let module = Module::<HostBytesBackend>::new(64);
        let layout = layout_of(vec![0, 1, 5, 7]);
        let infos = infos_of(24);
        let lt = module.ckks_linear_transformation_alloc_from_infos(&layout, &infos);
        let plan = layout.index();
        assert_eq!(lt.baby_steps, plan.baby_steps);
        let schedule: Vec<(i64, Vec<i64>)> = lt
            .giant_steps
            .iter()
            .map(|gs| (gs.rot, gs.diagonals.iter().map(|d| d.baby).collect()))
            .collect();
        assert_eq!(
            schedule,
            plan.giant_steps
                .iter()
                .copied()
                .zip(plan.index.iter().cloned())
                .collect::<Vec<_>>()
        );
        assert!(
            lt.giant_steps
                .iter()
                .flat_map(|gs| &gs.diagonals)
                .all(|d| d.plaintext.n() == Degree(16) && d.plaintext.meta() == infos.meta)
        );
    }

    #[test]
    fn built_transform_has_the_layout_allocation_schedule() {
        use crate::layouts::ComplexDiagonals;
        use poulpy_core::layouts::Diagonals;

        let module = Module::<HostBytesBackend>::new(64);
        let mut cd = ComplexDiagonals::new(Diagonals::new(32), Diagonals::new(32));
        cd.re.set(0, vec![1.0, 2.0]);
        cd.im.set(0, vec![0.5]);
        cd.re.set(1, vec![3.0]);
        cd.re.set(5, vec![-1.0, 4.0]);
        cd.im.set(7, vec![1.0, -2.0]);
        let strategy = LinearTransformationStrategy::Bsgs { giant_step: 4 };
        let layout = LinearTransformationLayout {
            indexes: cd.indexes(),
            slots: cd.slots(),
            strategy,
        };
        let mut infos = infos_of(24);
        infos.glwe_layout.n = module.ckks_compact_degree(cd.period());
        let lt = cd.build_transform(strategy, |_, _| module.ckks_plaintext_alloc_from_infos(&infos));
        let alloc = module.ckks_linear_transformation_alloc_from_infos(&layout, &infos);
        assert_eq!(alloc.baby_steps, lt.baby_steps);
        let schedule = |t: &LinearTransformation<_>| -> Vec<(i64, Vec<i64>)> {
            t.giant_steps
                .iter()
                .map(|gs| (gs.rot, gs.diagonals.iter().map(|d| d.baby).collect()))
                .collect()
        };
        assert_eq!(schedule(&alloc), schedule(&lt));
    }

    #[test]
    fn compact_degree_is_the_compact_allocation_degree() {
        let module = Module::<HostBytesBackend>::new(64);
        for slots in [1, 2, 8, 32] {
            assert_eq!(
                module.ckks_compact_degree(slots),
                module.ckks_pt_vec_alloc_compact(slots, Base2K(12), TorusPrecision(24)).n(),
                "slots={slots}"
            );
        }
    }
}
