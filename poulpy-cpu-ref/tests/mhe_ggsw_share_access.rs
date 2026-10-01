#![cfg(feature = "enable-mhe")]

//! Keep backend access outside `poulpy-mhe`, where crate-private accessors fail.

use poulpy_core::layouts::{
    Base2K, Dnum, Dsize, GGLWEInfos, LWEInfos, Rank, TorusPrecision,
    compressed::{GGLWECompressedSeed, GGLWECompressedSeedMut, GGLWECompressedToBackendMut, GGLWECompressedToBackendRef},
};
use poulpy_cpu_ref::{FFT64Ref, NTT4x30Ref};
use poulpy_hal::{
    AlignedBuf,
    layouts::{Backend, HostDataMut, HostDataRef, Module, WriterTo, ZnxView, ZnxViewMut},
};
use poulpy_mhe::layouts::MHEModuleAlloc;

fn check_share_access<BE>(module: &Module<BE>)
where
    BE: Backend<OwnedBuf = AlignedBuf, ZnxWord = i64>,
    for<'a> BE::BufRef<'a>: HostDataRef,
    for<'a> BE::BufMut<'a>: HostDataMut,
    Module<BE>: MHEModuleAlloc<BE>,
{
    for rank in [1, 2] {
        let mut share = module.ggsw_share_alloc(Base2K(12), Dnum(3), Dsize(1), TorusPrecision(16), Rank(rank));
        let rank = rank as usize;
        assert_eq!(share.parts().count(), 1 + 2 * rank);
        assert_eq!(share.parts_mut().count(), 1 + 2 * rank);

        for (part_index, part) in share.parts_mut().enumerate() {
            let rank_in = if part_index == 0 { 1 } else { rank };
            assert_eq!(part.rank_in().as_usize(), rank_in);
            assert_eq!(part.rank_out().as_usize(), rank);
            assert_eq!(part.seed().len(), 3 * rank_in);
            for (seed_index, seed) in part.seed_mut().iter_mut().enumerate() {
                *seed = [(16 * part_index + seed_index + 1) as u8; 32];
            }

            let mut view = GGLWECompressedToBackendMut::<BE>::to_backend_mut(part);
            for row in 0..3 {
                for col in 0..rank_in {
                    view.at_view_mut(row, col).data_mut().raw_mut().fill((part_index + 1) as i64);
                }
            }
        }

        // Generation needs a mutable circ_u and its circ_s seeds simultaneously.
        for col in 0..rank {
            let mut parts = share.parts_mut();
            let circ_u = parts.nth(1 + col).unwrap();
            let circ_s = parts.nth(rank - 1).unwrap();
            circ_u.seed_mut().copy_from_slice(circ_s.seed());
        }

        let mut parts_bytes = Vec::new();
        for (part_index, part) in share.parts().enumerate() {
            let seed_part = if (1..=rank).contains(&part_index) {
                part_index + rank
            } else {
                part_index
            };
            for (seed_index, seed) in part.seed().iter().enumerate() {
                assert_eq!(*seed, [(16 * seed_part + seed_index + 1) as u8; 32]);
            }

            let view = GGLWECompressedToBackendRef::<BE>::to_backend_ref(part);
            assert_eq!(view.seed(), part.seed());
            for row in 0..part.dnum().as_usize() {
                for col in 0..part.rank_in().as_usize() {
                    let body = view.at_view(row, col);
                    assert_eq!(body.data().raw().len(), part.n().as_usize() * part.max_size());
                    assert!(body.data().raw().iter().all(|&word| word == (part_index + 1) as i64));
                }
            }
            part.write_to(&mut parts_bytes).unwrap();
        }

        let mut share_bytes = Vec::new();
        share.write_to(&mut share_bytes).unwrap();
        assert_eq!(share_bytes, parts_bytes);
    }
}

#[test]
fn ggsw_share_access_fft64_ref() {
    check_share_access(&Module::<FFT64Ref>::new(16));
}

#[test]
fn ggsw_share_access_ntt4x30_ref() {
    check_share_access(&Module::<NTT4x30Ref>::new(16));
}
