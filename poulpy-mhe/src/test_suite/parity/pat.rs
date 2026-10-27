use poulpy_core::{
    GGLWECompressedEncryptSk, GGLWEEncryptSk, GLWECompressedEncryptSk, TransferInto,
    api::{GLWESecretPreparedFactory, GLWESecretSampling},
    layouts::{GGLWELayout, GLWELayout, ModuleCoreAlloc},
    test_suite::with_scratch,
};
use poulpy_hal::{AlignedBuf, api::VecZnxFillUniformSource, layouts::Module, source::Source, test_suite::vec_znx_backend_mut};

use super::{
    ParityBackend, ReferenceBackend, Snapshot, poison_gglwe, poison_glwe, snapshot_gglwe, snapshot_gglwe_compressed,
    snapshot_glwe, snapshot_glwe_compressed,
};
use crate::{
    api::{GGLWEPatCompressedOps, GGLWEPatOps, GLWEPatCompressedOps},
    layouts::{
        GGLWEPat, GGLWEPatCompressed, GGLWEPatCompressedOwned, GGLWEPatOwned, GLWEPatCompressed, GLWEPatCompressedOwned,
        MHEModuleAlloc,
    },
    test_suite::fixtures::{K, SEEDS, gglwe_layout, glwe_layout_at, party_messages, party_secrets},
};

/// Aggregation and finalization of every PAT type, over one secret-key
/// encryption per party built on the comparison backend: every aggregate, the
/// finalized ciphertext and the PAT after finalization.
pub fn test_pat_parity<BR, BT>(module_ref: &Module<BR>, module_test: &Module<BT>)
where
    BR: ReferenceBackend,
    BT: ParityBackend,
    Module<BR>: MHEModuleAlloc<BR>
        + GLWEPatCompressedOps<BR>
        + GGLWEPatCompressedOps<BR>
        + GGLWEPatOps<BR>
        + GLWESecretSampling<BR>
        + GLWESecretPreparedFactory<BR>
        + GLWECompressedEncryptSk<BR>
        + GGLWECompressedEncryptSk<BR>
        + GGLWEEncryptSk<BR>
        + VecZnxFillUniformSource<BR>,
    Module<BT>: MHEModuleAlloc<BT> + GLWEPatCompressedOps<BT> + GGLWEPatCompressedOps<BT> + GGLWEPatOps<BT>,
{
    let secrets = party_secrets(module_ref);
    let messages = party_messages(module_ref);

    let layout = glwe_layout_at(module_ref, K);
    let mut glwe_ref: Vec<GLWEPatCompressed<AlignedBuf, i64>> = Vec::new();
    for (i, (_, sk)) in secrets.iter().enumerate() {
        let mut pt = module_ref.glwe_plaintext_alloc_from_infos(&layout);
        module_ref.vec_znx_fill_uniform_source(
            layout.base2k.as_usize(),
            layout.k.as_usize(),
            &mut vec_znx_backend_mut::<BR>(pt.data_mut()),
            0,
            &mut Source::new([30 + i as u8; 32]),
        );
        let mut pat = module_ref.glwe_pat_compressed_alloc_from_infos(&layout);
        with_scratch::<BR, _>(module_ref.glwe_compressed_encrypt_sk_tmp_bytes(&layout), |scratch| {
            module_ref.glwe_compressed_encrypt_sk(&mut pat, &pt, sk, SEEDS[0], &mut Source::new([10 + i as u8; 32]), scratch)
        });
        glwe_ref.push(pat);
    }
    let mut glwe_test: Vec<GLWEPatCompressedOwned<BT>> = glwe_ref
        .iter()
        .map(|pat| {
            let mut copy = module_test.glwe_pat_compressed_alloc_from_infos(&layout);
            pat.transfer_into(&mut copy);
            copy
        })
        .collect();
    let want = run_glwe_pat_compressed(module_ref, &layout, &mut glwe_ref);
    let have = run_glwe_pat_compressed(module_test, &layout, &mut glwe_test);
    assert_eq!(want, have, "glwe pat compressed");

    let layout = gglwe_layout(module_ref);
    let mut compressed_ref: Vec<GGLWEPatCompressed<AlignedBuf, i64>> = Vec::new();
    let mut full_ref: Vec<GGLWEPat<AlignedBuf, i64>> = Vec::new();
    for (i, ((_, sk), (m, _))) in secrets.iter().zip(&messages).enumerate() {
        let mut pat = module_ref.gglwe_pat_compressed_alloc_from_infos(&layout);
        with_scratch::<BR, _>(module_ref.gglwe_compressed_encrypt_sk_tmp_bytes(&layout), |scratch| {
            module_ref.gglwe_compressed_encrypt_sk(
                &mut pat,
                m.data(),
                sk,
                SEEDS[0],
                &mut Source::new([10 + i as u8; 32]),
                scratch,
            )
        });
        compressed_ref.push(pat);
        let mut pat = module_ref.gglwe_pat_alloc_from_infos(&layout);
        with_scratch::<BR, _>(module_ref.gglwe_encrypt_sk_tmp_bytes(&layout), |scratch| {
            module_ref.gglwe_encrypt_sk(
                &mut pat,
                m.data(),
                sk,
                &mut Source::new([20 + i as u8; 32]),
                &mut Source::new([50 + i as u8; 32]),
                scratch,
            )
        });
        full_ref.push(pat);
    }
    let mut compressed_test: Vec<GGLWEPatCompressedOwned<BT>> = compressed_ref
        .iter()
        .map(|pat| {
            let mut copy = module_test.gglwe_pat_compressed_alloc_from_infos(&layout);
            pat.transfer_into(&mut copy);
            copy
        })
        .collect();
    let mut full_test: Vec<GGLWEPatOwned<BT>> = full_ref
        .iter()
        .map(|pat| {
            let mut copy = module_test.gglwe_pat_alloc_from_infos(&layout);
            pat.transfer_into(&mut copy);
            copy
        })
        .collect();
    let want = run_gglwe_pat_compressed(module_ref, &layout, &mut compressed_ref);
    let have = run_gglwe_pat_compressed(module_test, &layout, &mut compressed_test);
    assert_eq!(want, have, "gglwe pat compressed");
    let want = run_gglwe_pat(module_ref, &layout, &mut full_ref);
    let have = run_gglwe_pat(module_test, &layout, &mut full_test);
    assert_eq!(want, have, "gglwe pat");
}

fn run_glwe_pat_compressed<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GLWELayout,
    pats: &mut [GLWEPatCompressedOwned<BE>],
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GLWEPatCompressedOps<BE>,
{
    let mut out = Vec::new();
    let (acc, rest) = pats.split_first_mut().unwrap();
    for pat in rest.iter() {
        module.glwe_pat_compressed_aggregate_assign(acc, pat);
        out.push(snapshot_glwe_compressed::<BE, _>("aggregate", acc));
    }
    let mut res = module.glwe_alloc_from_infos(layout);
    poison_glwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.glwe_pat_compressed_finalize_tmp_bytes(), |scratch| {
        module.glwe_pat_compressed_finalize(&mut res, acc, scratch)
    });
    out.push(snapshot_glwe::<BE, _>("finalize", &res));
    out.push(snapshot_glwe_compressed::<BE, _>("finalize pat", acc));
    out
}

fn run_gglwe_pat_compressed<BE: ParityBackend>(
    module: &Module<BE>,
    layout: &GGLWELayout,
    pats: &mut [GGLWEPatCompressedOwned<BE>],
) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GGLWEPatCompressedOps<BE>,
{
    let mut out = Vec::new();
    let (acc, rest) = pats.split_first_mut().unwrap();
    for pat in rest.iter() {
        module.gglwe_pat_compressed_aggregate_assign(acc, pat);
        out.push(snapshot_gglwe_compressed::<BE, _>("aggregate", acc));
    }
    let mut res = module.gglwe_alloc_from_infos(layout);
    poison_gglwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.gglwe_pat_compressed_finalize_tmp_bytes(), |scratch| {
        module.gglwe_pat_compressed_finalize(&mut res, acc, scratch)
    });
    out.push(snapshot_gglwe::<BE, _>("finalize", &res));
    out.push(snapshot_gglwe_compressed::<BE, _>("finalize pat", acc));
    out
}

fn run_gglwe_pat<BE: ParityBackend>(module: &Module<BE>, layout: &GGLWELayout, pats: &mut [GGLWEPatOwned<BE>]) -> Vec<Snapshot>
where
    Module<BE>: MHEModuleAlloc<BE> + GGLWEPatOps<BE>,
{
    let mut out = Vec::new();
    let (acc, rest) = pats.split_first_mut().unwrap();
    for pat in rest.iter() {
        module.gglwe_pat_aggregate_assign(acc, pat);
        out.push(snapshot_gglwe::<BE, _>("aggregate", acc));
    }
    let mut res = module.gglwe_alloc_from_infos(layout);
    poison_gglwe::<BE, _>(&mut res);
    with_scratch::<BE, _>(module.gglwe_pat_finalize_tmp_bytes(), |scratch| {
        module.gglwe_pat_finalize(&mut res, acc, scratch)
    });
    out.push(snapshot_gglwe::<BE, _>("finalize", &res));
    out.push(snapshot_gglwe::<BE, _>("finalize pat", acc));
    out
}
