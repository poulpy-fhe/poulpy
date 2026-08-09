//! Phase 8 coverage: the DFT- and big-domain GLWE scratch views.
//!
//! Proves the two new carves are allocatable and, more importantly, that the
//! resulting views are *consumable*: a GLWE is prepared into a DFT-domain
//! scratch view, inverse-transformed into a big-domain scratch view, and
//! normalized back to the coefficient domain, with the round trip checked to be
//! the identity. Every stage is addressed through the semantic wrapper's
//! backend-borrow traits, not through raw HAL payloads.

use poulpy_hal::{
    api::{
        ScratchOwnedAlloc, ScratchOwnedBorrow, VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxDftApply,
        VecZnxDftBytesOf, VecZnxIdftApply, VecZnxIdftApplyTmpBytes,
    },
    layouts::{DigestU64, FillUniform, Module, ScratchOwned},
    source::Source,
    test_suite::TestParams,
};

use crate::{
    layouts::{
        Base2K, GLWEBigToBackendMut, GLWEBigToBackendRef, GLWEInfos, GLWELayout, GLWEPreparedFactory, GLWEPreparedToBackendRef,
        GLWEToBackendMut, LWEInfos, ModuleCoreAlloc, Rank, TorusPrecision,
    },
    scratch::ScratchArenaTakeCore,
};

/// Carves a DFT- and a big-domain GLWE from scratch, then runs
/// `prepare -> idft -> normalize` through them and checks the round trip is the
/// identity.
pub fn test_glwe_domain_scratch_round_trip<BE: crate::test_suite::TestBackend>(params: &TestParams, module: &Module<BE>)
where
    BE::OwnedBuf: poulpy_hal::layouts::HostDataMut,
    for<'a> BE::BufRef<'a>: poulpy_hal::layouts::HostDataRef,
    for<'a> BE::BufMut<'a>: poulpy_hal::layouts::HostDataMut,
    Module<BE>: ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + GLWEPreparedFactory<BE>
        + VecZnxDftBytesOf
        + VecZnxBigBytesOf
        + VecZnxDftApply<BE>
        + VecZnxIdftApply<BE>
        + VecZnxIdftApplyTmpBytes
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
    ScratchOwned<BE>: ScratchOwnedAlloc<BE> + ScratchOwnedBorrow<BE>,
{
    let base2k = Base2K(params.base2k as u32);
    let rank = Rank(2);
    let infos = GLWELayout {
        n: (module.n() as u32).into(),
        base2k,
        k: TorusPrecision(3 * base2k.as_u32()),
        rank,
    };

    let mut source = Source::new([7u8; 32]);

    // A coefficient-domain GLWE with arbitrary (already normalized) content: the
    // round trip below has to reproduce it exactly.
    let mut a = module.glwe_alloc_from_infos(&infos);
    a.fill_uniform(params.base2k, &mut source);
    let a_digest = a.data().digest_u64();

    let mut res = module.glwe_alloc_from_infos(&infos);

    let cols: usize = (infos.rank() + 1).into();
    let scratch_bytes = module.glwe_prepared_bytes_of_from_infos(&infos)
        + BE::bytes_of_vec_znx_big(module.n(), cols, infos.size())
        + module
            .vec_znx_idft_apply_tmp_bytes()
            .max(module.vec_znx_big_normalize_tmp_bytes());
    let mut scratch: ScratchOwned<BE> = ScratchOwned::alloc(scratch_bytes);

    scratch.borrow().scope(|arena| {
        // DFT-domain carve, addressed as a GLWE rather than a bare VecZnxDft.
        let (mut a_dft, arena_1) = arena.take_glwe_prepared_scratch(module, &infos);
        assert_eq!(a_dft.n(), infos.n());
        assert_eq!(a_dft.rank(), infos.rank);
        assert_eq!(a_dft.base2k(), infos.base2k);
        assert_eq!(a_dft.k(), infos.k);
        assert_eq!(a_dft.size(), infos.size());

        module.glwe_prepare(&mut a_dft, &a);

        // Big-domain carve: the deferred-normalization intermediate.
        let (mut a_big, mut arena_2) = arena_1.take_glwe_big_scratch(module, &infos);
        assert_eq!(a_big.n(), infos.n());
        assert_eq!(a_big.rank(), infos.rank);
        assert_eq!(a_big.base2k(), infos.base2k);
        assert_eq!(a_big.k(), infos.k);

        {
            let a_dft_ref = a_dft.to_backend_ref();
            let mut a_big_mut = a_big.to_backend_mut();
            for col in 0..cols {
                module.vec_znx_idft_apply(&mut a_big_mut.data, col, &a_dft_ref.data, col, &mut arena_2.borrow());
            }
        }

        // Back to the coefficient domain.
        let a_big_ref = a_big.to_backend_ref();
        let mut res_mut = GLWEToBackendMut::<BE>::to_backend_mut(&mut res);
        for col in 0..cols {
            module.vec_znx_big_normalize(
                &mut res_mut.data,
                base2k.into(),
                0,
                col,
                &a_big_ref.data,
                base2k.into(),
                col,
                &mut arena_2.borrow(),
            );
        }
    });

    assert_eq!(a.data().digest_u64(), a_digest, "the input must not be mutated");
    assert_eq!(
        res.data().to_host_owned::<BE>(),
        a.data().to_host_owned::<BE>(),
        "prepare -> idft -> normalize through the domain scratch views must be the identity"
    );
}
