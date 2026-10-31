use poulpy_core::layouts::{
    Base2K, Degree, GLWEAlloc, GLWEInfos, GLWEPlaintextAlloc, GLWEPlaintextLayout, GetDegree, LinearTransformation,
    LinearTransformationDiagonal, LinearTransformationGiantStep, LinearTransformationLayout, ModuleCoreAlloc, Rank,
    TorusPrecision,
};
use poulpy_hal::layouts::{Backend, Module};

use crate::{CKKSInfos, GLWEPlaintextMeta, SetCKKSInfos, api::CKKSModuleInfos};

use super::{CKKSCiphertext, CKKSCiphertextOwned, CKKSPlaintext, CKKSPlaintextOwned};

/// Allocates [`CKKSCiphertext`]s on a [`Module`].
///
/// Every method is default-bodied over [`GLWEAlloc`]: a ciphertext with explicit rank plus thin conveniences over it.
pub trait CKKSCiphertextAlloc<BE: Backend>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEAlloc {
    /// Allocates a ciphertext with `infos`' layout **and** its CKKS metadata
    /// (`log_delta`, `log_sparsity`), mirroring
    /// [`CKKSPlaintextAlloc::ckks_plaintext_alloc_from_infos`]. Use
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
}

impl<BE: Backend> CKKSCiphertextAlloc<BE> for Module<BE> where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}

/// Allocates [`CKKSPlaintext`]s on a [`Module`].
///
/// Every method is default-bodied over [`GLWEPlaintextAlloc`]: a plaintext with explicit degree plus thin
/// conveniences over it.
pub trait CKKSPlaintextAlloc<BE: Backend>:
    ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord> + GLWEPlaintextAlloc + CKKSModuleInfos
{
    fn ckks_plaintext_alloc_from_infos<A>(&self, infos: &A) -> CKKSPlaintextOwned<BE>
    where
        A: CKKSInfos,
    {
        let mut pt = self.ckks_plaintext_alloc(infos.n(), infos.base2k(), infos.k());
        pt.set_meta(infos.meta());
        pt
    }

    /// Allocates a default-meta plaintext sized to `k` over `base2k`. The semantic
    /// [`GLWEPlaintextMeta`] is not needed to size the buffer: set it afterwards with
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

    /// A plaintext of `slots` slots, at the degree that holds them: `2 * slots` on the standard ring, `slots` on the
    /// invariant ring; [`CKKSModuleInfos::ckks_max_slots`] gives one at the ring degree.
    ///
    /// Panics when `slots` is not a power of two, exceeds `ckks_max_slots`, or needs a degree below the backend's
    /// minimum.
    fn ckks_plaintext_alloc_from_slots(&self, slots: usize, base2k: Base2K, k: TorusPrecision) -> CKKSPlaintextOwned<BE>
    where
        Self: GetDegree,
    {
        let max_slots = self.ckks_max_slots();
        assert!(
            slots.is_power_of_two() && slots <= max_slots,
            "ckks_plaintext_alloc_from_slots: the slot count must be a power of two up to the module's slots"
        );
        let n = slots * (self.ring_degree().as_usize() / max_slots);
        assert!(
            n >= BE::MIN_DEGREE,
            "ckks_plaintext_alloc_from_slots: the slot count needs a degree below the backend's minimum"
        );
        self.ckks_plaintext_alloc(n.into(), base2k, k)
    }
}

impl<BE: Backend> CKKSPlaintextAlloc<BE> for Module<BE> where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
}

/// Allocates plaintext-diagonal [`LinearTransformation`]s on a [`Module`].
pub trait CKKSLinearTransformationAlloc<BE: Backend>: CKKSPlaintextAlloc<BE> {
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
            rotations: plan.rotations,
        }
    }
}

impl<BE: Backend> CKKSLinearTransformationAlloc<BE> for Module<BE> where
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
{
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
        let n = module
            .ckks_plaintext_alloc_from_slots(cd.period(), Base2K(12), TorusPrecision(24))
            .n();
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
