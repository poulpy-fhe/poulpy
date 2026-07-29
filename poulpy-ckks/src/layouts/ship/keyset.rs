//! SHIP evaluation-key storage, generation, and backend preparation.
//!
//! [`ShipKeySet`] is the unprepared form: per support slot the base-B mux
//! blind-rotation keys and the `4*theta` encrypted selector masks, plus the
//! dense -> sparse encapsulation key, the tensor key of the product tree, and
//! the conjugation key. [`ShipKeySet::generate`] derives the whole bundle from
//! the dense secret and a [`ShipSecretSpec`]; [`ShipKeySet::prepare`] returns
//! the backend-ready [`ShipKeysPrepared`] with the masks preprocessed as left
//! convolution operands.

use anyhow::{Result, ensure};
use poulpy_core::{
    EncryptionLayout, GLWEAutomorphismKeyEncryptSk, GLWESwitchingKeyEncryptSk, GLWETensorKeyEncryptSk, ModuleTransfer,
    layouts::{
        Base2K, GGLWEInfos, GGLWEToBackendRef, GLWEAutomorphismKey, GLWEAutomorphismKeyLayout, GLWEAutomorphismKeyPrepared,
        GLWEAutomorphismKeyPreparedFactory, GLWEInfos, GLWELayout, GLWESecret, GLWESecretLayout, GLWESecretPreparedFactory,
        GLWESwitchingKey, GLWESwitchingKeyLayout, GLWESwitchingKeyPrepared, GLWESwitchingKeyPreparedFactory, GLWETensorKey,
        GLWETensorKeyLayout, GLWETensorKeyPrepared, GLWETensorKeyPreparedFactory, GLWEToBackendRef, GetGaloisElement, LWEInfos,
        ModuleCoreAlloc, Rank,
    },
    msb_mask_bottom_limb,
};
use poulpy_hal::{
    api::{CnvPVecAlloc, CnvPVecBytesOf, Convolution},
    layouts::{
        Backend, CnvPVecL, CnvPVecLToBackendMut, Data, GaloisElement, HostBytesBackend, HostDataMut, HostDataRef, Module,
        ScratchArena, TransferFrom, ZnxView, ZnxViewMut,
    },
    source::Source,
};

use crate::{
    CKKSInfos, CKKSMeta,
    api::{CKKSEncodingHostOps, CKKSEncodingOps, CKKSEncryptOps, ShipScalar},
    encoding::ship::masks::ship_mask_slot_vectors,
    layouts::{CKKSCiphertext, CKKSModuleAlloc, CKKSPlaintext},
};

use super::{plan::ShipPlan, secret::ShipSecretSpec};

/// Plan dimensions and radix that determine the SHIP key material's meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShipKeyParameters {
    plan: ShipPlan,
    base2k: usize,
    complex: bool,
}

impl ShipKeyParameters {
    /// Derives the key fingerprint from a validated plan, the generation
    /// radix, and whether the `omega_2` mask set is present.
    pub fn from_plan(plan: &ShipPlan, base2k: Base2K, complex: bool) -> Self {
        Self {
            plan: *plan,
            base2k: base2k.as_usize(),
            complex,
        }
    }

    /// The plan the keys were generated for.
    pub fn plan(&self) -> &ShipPlan {
        &self.plan
    }

    /// Limb radix of the key material.
    pub fn base2k(&self) -> usize {
        self.base2k
    }

    /// Whether the keys carry the `omega_2` masks of the complex bootstrap.
    pub fn complex(&self) -> bool {
        self.complex
    }
}

/// Half-mux-rotate key (SHIP Definition 1, hoisted gadget form of §5.1): a
/// rank-2 -> rank-1 switching key whose input secret is `(beta * s, beta)`
/// and whose output secret is `s(X^{g^-1})` with `g = 5^(-rot)`, so that
/// applying `X -> X^g` to the keyswitch output realizes `beta * Rot_rot`
/// under `s`.
pub struct HMuxRotKey<D: Data> {
    pub(crate) key: GLWESwitchingKey<D>,
    pub(crate) gal_el: i64,
}

impl<D: Data> HMuxRotKey<D> {
    /// The underlying rank-2 -> 1 switching key.
    pub fn key(&self) -> &GLWESwitchingKey<D> {
        &self.key
    }

    /// The output automorphism `X -> X^gal_el` applied after the keyswitch.
    pub fn gal_el(&self) -> i64 {
        self.gal_el
    }
}

/// Prepared form of [`HMuxRotKey`].
pub struct HMuxRotKeyPrepared<D: Data, BE: Backend> {
    pub(crate) key: GLWESwitchingKeyPrepared<D, BE>,
    pub(crate) gal_el: i64,
}

/// Unprepared mux-method key material for one support slot: per digit
/// position the `b` hoisted mux keys, plus the `4*theta` encrypted selector
/// masks in candidate-major order (`masks2` carries the `omega_2` set of the
/// complex bootstrap; the mux keys are shared between both halves).
pub struct ShipIndexKeys<D: Data> {
    pub(crate) mux_keys: Vec<Vec<HMuxRotKey<D>>>,
    pub(crate) masks: Vec<CKKSCiphertext<D>>,
    pub(crate) masks2: Vec<CKKSCiphertext<D>>,
}

/// Prepared form of [`ShipIndexKeys`]: the masks become left convolution
/// operands.
pub struct ShipIndexKeysPrepared<D: Data, BE: Backend> {
    pub(crate) mux_keys: Vec<Vec<HMuxRotKeyPrepared<D, BE>>>,
    pub(crate) masks: Vec<CnvPVecL<D, BE>>,
    pub(crate) masks2: Vec<CnvPVecL<D, BE>>,
}

impl<D: Data, BE: Backend> ShipIndexKeysPrepared<D, BE> {
    /// Mux key groups, one per digit position, low digit first.
    pub fn mux_keys(&self) -> &[Vec<HMuxRotKeyPrepared<D, BE>>] {
        &self.mux_keys
    }

    /// The `4*theta` prepared selector masks (first coefficient half).
    pub fn masks(&self) -> &[CnvPVecL<D, BE>] {
        &self.masks
    }

    /// The `omega_2` mask set (second coefficient half), empty unless the
    /// keys were generated with `complex`.
    pub fn masks2(&self) -> &[CnvPVecL<D, BE>] {
        &self.masks2
    }
}

/// Validated, unprepared SHIP key material.
pub struct ShipKeySet<D: Data> {
    parameters: ShipKeyParameters,
    index_keys: Vec<ShipIndexKeys<D>>,
    dense_to_sparse: GLWESwitchingKey<D>,
    tensor_key: GLWETensorKey<D>,
    conjugation_key: GLWEAutomorphismKey<D>,
}

/// Eagerly prepared SHIP key material ready for a backend pipeline.
pub struct ShipKeysPrepared<D: Data, BE: Backend> {
    parameters: ShipKeyParameters,
    index_keys: Vec<ShipIndexKeysPrepared<D, BE>>,
    dense_to_sparse: GLWESwitchingKeyPrepared<D, BE>,
    tensor_key: GLWETensorKeyPrepared<D, BE>,
    conjugation_key: GLWEAutomorphismKeyPrepared<D, BE>,
}

/// Gadget layouts of the keys produced by [`ShipKeySet::generate`].
///
/// The mux switching keys and the dense -> sparse encapsulation key are fully
/// determined by the plan and radix, so only the gadget digit size of the mux
/// keys is selectable here.
#[derive(Clone, Copy, Debug)]
pub struct ShipKeysLayout {
    /// Gadget digit size of the hoisted mux switching keys.
    pub mux_dsize: usize,
    /// Layout of the product-tree relinearization (tensor) key.
    pub tensor_key: GLWETensorKeyLayout,
    /// Layout of the conjugation (Galois element `-1`) automorphism key.
    pub conjugation_key: GLWEAutomorphismKeyLayout,
    /// Also generate the `omega_2` masks required by the complex bootstrap.
    pub complex: bool,
}

/// Applies the coefficient permutation of `X -> X^gal_el` to host secret
/// coefficients, accumulating into the zeroed `dst`.
fn znx_automorphism_apply(gal_el: i64, src: &[i64], dst: &mut [i64]) {
    let n = src.len();
    let two_n = 2 * n;
    let g = gal_el.rem_euclid(two_n as i64) as usize;
    for (i, &c) in src.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let e = (i * g) % two_n;
        if e >= n {
            dst[e - n] = -c;
        } else {
            dst[e] = c;
        }
    }
}

impl<D: Data> ShipKeySet<D> {
    /// Builds and validates an unprepared SHIP key set.
    pub fn new(
        plan: &ShipPlan,
        base2k: Base2K,
        complex: bool,
        index_keys: Vec<ShipIndexKeys<D>>,
        dense_to_sparse: GLWESwitchingKey<D>,
        tensor_key: GLWETensorKey<D>,
        conjugation_key: GLWEAutomorphismKey<D>,
    ) -> Result<Self> {
        let parameters = ShipKeyParameters::from_plan(plan, base2k, complex);
        validate_material(&parameters, &index_keys, &dense_to_sparse, &tensor_key, &conjugation_key)?;
        Ok(Self {
            parameters,
            index_keys,
            dense_to_sparse,
            tensor_key,
            conjugation_key,
        })
    }

    /// Dimensions that give the key material its SHIP meaning.
    pub fn parameters(&self) -> ShipKeyParameters {
        self.parameters
    }

    /// Prepares every key for `module`: gadget keys are preprocessed and the
    /// selector masks become left convolution operands. `scratch` must hold
    /// the per-key prepare scratch, which is validated up front.
    pub fn prepare<BE: Backend<OwnedBuf = D>>(
        &self,
        module: &Module<BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<ShipKeysPrepared<D, BE>>
    where
        D: HostDataRef,
        CKKSCiphertext<D>: GLWEToBackendRef<BE>,
        GLWESwitchingKey<D>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWETensorKey<D>: GGLWEToBackendRef<BE> + GGLWEInfos,
        GLWEAutomorphismKey<D>: GGLWEToBackendRef<BE> + GetGaloisElement + GGLWEInfos,
        Module<BE>: ModuleCoreAlloc<OwnedBuf = D>
            + GLWESwitchingKeyPreparedFactory<BE>
            + GLWEAutomorphismKeyPreparedFactory<BE>
            + GLWETensorKeyPreparedFactory<BE>
            + Convolution<BE>
            + CnvPVecAlloc<BE>
            + CnvPVecBytesOf,
    {
        let base2k = self.parameters.base2k;
        let kk = self.parameters.plan.raised_k(base2k);
        let mask_size = kk.div_ceil(base2k);
        let mask_msb = msb_mask_bottom_limb(base2k, kk);

        let mut required = module
            .glwe_switching_key_prepare_tmp_bytes(&self.dense_to_sparse)
            .max(module.prepare_tensor_key_tmp_bytes(&self.tensor_key))
            .max(module.glwe_automorphism_key_prepare_tmp_bytes(&self.conjugation_key))
            .max(module.cnv_prepare_left_tmp_bytes(mask_size, mask_size));
        for ik in &self.index_keys {
            for group in &ik.mux_keys {
                for mux in group {
                    required = required.max(module.glwe_switching_key_prepare_tmp_bytes(&mux.key));
                }
            }
        }
        ensure!(
            scratch.available() >= required,
            "SHIP key preparation needs {required} scratch bytes, but only {} are available",
            scratch.available()
        );

        let prepare_masks = |masks: &[CKKSCiphertext<D>], scratch: &mut ScratchArena<'_, BE>| -> Vec<CnvPVecL<D, BE>> {
            masks
                .iter()
                .map(|ct| {
                    let mut prep = module.cnv_pvec_left_alloc(2, mask_size);
                    module.cnv_prepare_left(
                        &mut prep.to_backend_mut(),
                        GLWEToBackendRef::<BE>::to_backend_ref(ct).data(),
                        mask_msb,
                        scratch,
                    );
                    prep
                })
                .collect()
        };

        let mut index_keys = Vec::with_capacity(self.index_keys.len());
        for ik in &self.index_keys {
            let masks = prepare_masks(&ik.masks, scratch);
            let masks2 = prepare_masks(&ik.masks2, scratch);
            let mux_keys = ik
                .mux_keys
                .iter()
                .map(|group| {
                    group
                        .iter()
                        .map(|mux| {
                            let mut key = module.glwe_switching_key_prepared_alloc_from_infos(&mux.key);
                            module.glwe_switching_key_prepare(&mut key, &mux.key, scratch);
                            HMuxRotKeyPrepared { key, gal_el: mux.gal_el }
                        })
                        .collect()
                })
                .collect();
            index_keys.push(ShipIndexKeysPrepared { mux_keys, masks, masks2 });
        }

        let mut dense_to_sparse = module.glwe_switching_key_prepared_alloc_from_infos(&self.dense_to_sparse);
        module.glwe_switching_key_prepare(&mut dense_to_sparse, &self.dense_to_sparse, scratch);
        let mut tensor_key = module.alloc_tensor_key_prepared_from_infos(&self.tensor_key);
        module.prepare_tensor_key(&mut tensor_key, &self.tensor_key, scratch);
        let mut conjugation_key = module.glwe_automorphism_key_prepared_alloc_from_infos(&self.conjugation_key);
        module.glwe_automorphism_key_prepare(&mut conjugation_key, &self.conjugation_key, scratch);

        Ok(ShipKeysPrepared {
            parameters: self.parameters,
            index_keys,
            dense_to_sparse,
            tensor_key,
            conjugation_key,
        })
    }
}

impl<D: Data, BE: Backend> ShipKeysPrepared<D, BE> {
    /// Dimensions that give the key material its SHIP meaning.
    pub fn parameters(&self) -> ShipKeyParameters {
        self.parameters
    }

    /// Per-support-slot key material, in support order.
    pub fn index_keys(&self) -> &[ShipIndexKeysPrepared<D, BE>] {
        &self.index_keys
    }

    /// Dense -> sparse encapsulation key (bottom modulus).
    pub fn dense_to_sparse(&self) -> &GLWESwitchingKeyPrepared<D, BE> {
        &self.dense_to_sparse
    }

    /// Relinearization key of the product tree.
    pub fn tensor_key(&self) -> &GLWETensorKeyPrepared<D, BE> {
        &self.tensor_key
    }

    /// Conjugation (Galois element `-1`) automorphism key.
    pub fn conjugation_key(&self) -> &GLWEAutomorphismKeyPrepared<D, BE> {
        &self.conjugation_key
    }
}

/// Encrypts one [`HMuxRotKey`] for test bit `beta` and rotation amount `rot`
/// at ciphertext precision `k_ct`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hmux_rot_key_encrypt_sk<BE>(
    module: &Module<BE>,
    host_module: &Module<HostBytesBackend>,
    sk_dense_host: &GLWESecret<Vec<u8>>,
    beta: bool,
    rot: usize,
    k_ct: usize,
    base2k: Base2K,
    dsize: usize,
    source_xe: &mut Source,
    source_xa: &mut Source,
    scratch: &mut ScratchArena<'_, BE>,
) -> Result<HMuxRotKey<BE::OwnedBuf>>
where
    BE: Backend + TransferFrom<HostBytesBackend>,
    BE::OwnedBuf: HostDataRef + HostDataMut,
    Module<BE>: GLWESwitchingKeyEncryptSk<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf> + ModuleTransfer<BE> + GaloisElement,
    Module<HostBytesBackend>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
{
    let n = sk_dense_host.n();
    let m = n.as_usize() / 2;
    let gal_el = module.galois_element(((m - (rot % m)) % m) as i64);

    let mut sk_in_host = host_module.glwe_secret_alloc_from_infos(&GLWESecretLayout { n, rank: Rank(2) });
    sk_in_host.fill_zero();
    if beta {
        let src = sk_dense_host.data().at(0, 0).to_vec();
        let data = sk_in_host.data_mut();
        data.at_mut(0, 0).copy_from_slice(&src);
        data.at_mut(1, 0)[0] = 1;
    }
    let sk_in = module.upload_glwe_secret(&sk_in_host);

    let mut sk_out_host = host_module.glwe_secret_alloc_from_infos(&GLWESecretLayout { n, rank: Rank(1) });
    sk_out_host.fill_zero();
    znx_automorphism_apply(
        module.galois_element_inv(gal_el),
        sk_dense_host.data().at(0, 0),
        sk_out_host.data_mut().at_mut(0, 0),
    );
    let sk_out = module.upload_glwe_secret(&sk_out_host);

    let b2k = base2k.as_usize();
    let ksk_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
        n,
        base2k,
        dnum: k_ct.div_ceil(b2k * dsize).into(),
        k_aux: (b2k * dsize).into(),
        rank_in: Rank(2),
        rank_out: Rank(1),
        dsize: dsize.into(),
    })?;
    let mut key = module.glwe_switching_key_alloc_from_infos(&ksk_infos);
    module.glwe_switching_key_encrypt_sk(&mut key, &sk_in, &sk_out, &ksk_infos, source_xe, source_xa, scratch);
    Ok(HMuxRotKey { key, gal_el })
}

impl<D: Data> ShipKeySet<D> {
    /// Generates the full SHIP key set for `sk_dense_host` and the sparse
    /// support of `spec`.
    ///
    /// Key generation is host-side (mux input/output secrets are permuted host
    /// polynomials); the masks are encoded and encrypted through the module's
    /// backend-resident CKKS operations at working scalar precision `F`.
    /// `scratch` must hold the largest of the key-encrypt, mask-encrypt, and
    /// slot-encode scratch requirements.
    #[allow(clippy::too_many_arguments)]
    pub fn generate<BE, F>(
        module: &Module<BE>,
        host_module: &Module<HostBytesBackend>,
        plan: &ShipPlan,
        base2k: Base2K,
        spec: &ShipSecretSpec,
        sk_dense_host: &GLWESecret<Vec<u8>>,
        layout: &ShipKeysLayout,
        source_xe: &mut Source,
        source_xa: &mut Source,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<Self>
    where
        BE: Backend<OwnedBuf = D> + TransferFrom<HostBytesBackend>,
        D: HostDataRef + HostDataMut,
        F: ShipScalar,
        Module<BE>: GLWESwitchingKeyEncryptSk<BE>
            + GLWEAutomorphismKeyEncryptSk<BE>
            + GLWETensorKeyEncryptSk<BE>
            + GLWESecretPreparedFactory<BE>
            + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf>
            + ModuleTransfer<BE>
            + CKKSModuleAlloc<BE>
            + CKKSEncryptOps<BE>
            + CKKSEncodingOps<BE, F>
            + GaloisElement,
        Module<HostBytesBackend>: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
        CKKSCiphertext<BE::OwnedBuf>: GLWEToBackendRef<BE>,
        CKKSPlaintext<BE::OwnedBuf>: GLWEToBackendRef<BE>,
    {
        let n = sk_dense_host.n();
        ensure!(
            n.as_usize() == plan.n(),
            "SHIP dense secret degree {} does not match plan degree {}",
            n,
            plan.n()
        );
        ensure!(
            sk_dense_host.rank().as_usize() == 1,
            "SHIP dense secret must have rank 1, got {}",
            sk_dense_host.rank()
        );
        let b2k = base2k.as_usize();
        ensure!((1..63).contains(&b2k), "SHIP base2k must be in [1, 63), got {b2k}");

        let m = plan.half_n();
        let h = plan.sparse_hamming_weight();
        let theta = plan.theta();
        let bases = plan.mux_bases();
        let kk = plan.raised_k(b2k);
        let ld = plan.log_delta_work();
        let mask_meta = CKKSMeta {
            log_delta: ld,
            log_sparsity: 0,
        };
        let enc_infos = EncryptionLayout::new_from_default_sigma(GLWELayout {
            n,
            base2k,
            k: kk.into(),
            rank: Rank(1),
        })?;

        let sk_dense = module.upload_glwe_secret(sk_dense_host);
        let mut sk_dense_prepared = module.glwe_secret_prepared_alloc_from_infos(&GLWESecretLayout { n, rank: Rank(1) });
        module.glwe_secret_prepare(&mut sk_dense_prepared, &sk_dense);

        let mut index_keys = Vec::with_capacity(h);
        for (slot, &(j, s_j)) in spec.support().iter().enumerate() {
            let u = spec.offset(plan, slot);

            let mut encrypt_masks = |omega2: bool| -> Result<Vec<CKKSCiphertext<BE::OwnedBuf>>> {
                ship_mask_slot_vectors::<F>(plan, slot, j, s_j, u, omega2)
                    .into_iter()
                    .map(|re| {
                        let im = vec![F::zero(); m];
                        let mut pt = module.ckks_pt_vec_alloc(base2k, kk.into());
                        pt.set_meta_checked(mask_meta)?;
                        module.ckks_encode_reim_into(&mut pt, &re, &im, scratch)?;
                        let mut ct = module.ckks_ciphertext_alloc(base2k, kk.into());
                        module.ckks_encrypt_sk(&mut ct, &pt, &sk_dense_prepared, &enc_infos, source_xe, source_xa, scratch)?;
                        Ok(ct)
                    })
                    .collect()
            };
            let masks = encrypt_masks(false)?;
            let masks2 = if layout.complex { encrypt_masks(true)? } else { Vec::new() };

            let mut mux_keys = Vec::with_capacity(bases.len());
            let mut weight = theta;
            for &b in &bases {
                let digit = (u / weight) % b;
                let mut group = Vec::with_capacity(b);
                for d in 0..b {
                    group.push(hmux_rot_key_encrypt_sk(
                        module,
                        host_module,
                        sk_dense_host,
                        d == digit,
                        (d * weight) % m,
                        kk,
                        base2k,
                        layout.mux_dsize,
                        source_xe,
                        source_xa,
                        scratch,
                    )?);
                }
                mux_keys.push(group);
                weight *= b;
            }
            index_keys.push(ShipIndexKeys { mux_keys, masks, masks2 });
        }

        // Dense -> sparse encapsulation key at the bottom modulus: the only
        // object encrypted under the sparse secret.
        let mut sk_sparse_host = host_module.glwe_secret_alloc_from_infos(&GLWESecretLayout { n, rank: Rank(1) });
        spec.fill_glwe_secret(plan, &mut sk_sparse_host)?;
        let sk_sparse = module.upload_glwe_secret(&sk_sparse_host);
        let d2s_infos = EncryptionLayout::new_from_default_sigma(GLWESwitchingKeyLayout {
            n,
            base2k,
            dnum: 1usize.into(),
            k_aux: b2k.into(),
            rank_in: Rank(1),
            rank_out: Rank(1),
            dsize: 1usize.into(),
        })?;
        let mut dense_to_sparse = module.glwe_switching_key_alloc_from_infos(&d2s_infos);
        module.glwe_switching_key_encrypt_sk(
            &mut dense_to_sparse,
            &sk_dense,
            &sk_sparse,
            &d2s_infos,
            source_xe,
            source_xa,
            scratch,
        );

        let tsk_infos = EncryptionLayout::new_from_default_sigma(layout.tensor_key)?;
        let mut tensor_key = module.glwe_tensor_key_alloc_from_infos(&tsk_infos);
        module.glwe_tensor_key_encrypt_sk(&mut tensor_key, &sk_dense, &tsk_infos, source_xe, source_xa, scratch);

        let atk_infos = EncryptionLayout::new_from_default_sigma(layout.conjugation_key)?;
        let mut conjugation_key = module.glwe_automorphism_key_alloc_from_infos(&atk_infos);
        module.glwe_automorphism_key_encrypt_sk(&mut conjugation_key, -1, &sk_dense, &atk_infos, source_xe, source_xa, scratch);

        Self::new(
            plan,
            base2k,
            layout.complex,
            index_keys,
            dense_to_sparse,
            tensor_key,
            conjugation_key,
        )
    }
}

/// Validates the unprepared SHIP key representation against its parameters.
fn validate_material<D: Data>(
    parameters: &ShipKeyParameters,
    index_keys: &[ShipIndexKeys<D>],
    dense_to_sparse: &GLWESwitchingKey<D>,
    tensor_key: &GLWETensorKey<D>,
    conjugation_key: &GLWEAutomorphismKey<D>,
) -> Result<()> {
    let plan = &parameters.plan;
    let n = plan.n();
    let base2k = parameters.base2k;
    let kk = plan.raised_k(base2k);
    let bases = plan.mux_bases();
    let mask_count = 4 * plan.theta();

    ensure!(
        index_keys.len() == plan.sparse_hamming_weight(),
        "SHIP key set has {} index keys, expected the sparse Hamming weight {}",
        index_keys.len(),
        plan.sparse_hamming_weight()
    );
    for (slot, ik) in index_keys.iter().enumerate() {
        ensure!(
            ik.masks.len() == mask_count,
            "SHIP index {slot} has {} masks, expected {mask_count}",
            ik.masks.len()
        );
        let expected2 = if parameters.complex { mask_count } else { 0 };
        ensure!(
            ik.masks2.len() == expected2,
            "SHIP index {slot} has {} omega_2 masks, expected {expected2}",
            ik.masks2.len()
        );
        for ct in ik.masks.iter().chain(&ik.masks2) {
            ensure!(
                ct.n().as_usize() == n
                    && ct.rank().as_usize() == 1
                    && ct.base2k().as_usize() == base2k
                    && ct.k().as_usize() == kk
                    && ct.log_delta() == plan.log_delta_work(),
                "SHIP index {slot} mask layout does not match the plan (degree {}, rank {}, base2k {}, width {}, scale {})",
                ct.n(),
                ct.rank(),
                ct.base2k(),
                ct.k(),
                ct.log_delta()
            );
        }
        ensure!(
            ik.mux_keys.len() == bases.len(),
            "SHIP index {slot} has {} mux digit positions, expected {}",
            ik.mux_keys.len(),
            bases.len()
        );
        for (position, (group, &b)) in ik.mux_keys.iter().zip(&bases).enumerate() {
            ensure!(
                group.len() == b,
                "SHIP index {slot} digit {position} has {} keys, expected base {b}",
                group.len()
            );
            for mux in group {
                ensure!(
                    mux.key.n().as_usize() == n
                        && mux.key.base2k().as_usize() == base2k
                        && mux.key.rank_in().as_usize() == 2
                        && mux.key.rank_out().as_usize() == 1
                        && mux.key.k().as_usize() >= kk,
                    "SHIP index {slot} digit {position} mux key layout does not match the plan"
                );
            }
        }
    }

    ensure!(
        dense_to_sparse.n().as_usize() == n
            && dense_to_sparse.base2k().as_usize() == base2k
            && dense_to_sparse.rank_in().as_usize() == 1
            && dense_to_sparse.rank_out().as_usize() == 1
            && dense_to_sparse.k().as_usize() >= 2 * base2k,
        "SHIP dense -> sparse key layout does not match the plan"
    );
    ensure!(
        tensor_key.n().as_usize() == n && tensor_key.base2k().as_usize() == base2k,
        "SHIP tensor key layout does not match the plan"
    );
    ensure!(
        conjugation_key.n().as_usize() == n && conjugation_key.base2k().as_usize() == base2k,
        "SHIP conjugation key layout does not match the plan"
    );
    ensure!(
        conjugation_key.p() == -1,
        "SHIP conjugation key must use Galois element -1, got {}",
        conjugation_key.p()
    );
    ensure!(
        tensor_key.k().as_usize() >= kk,
        "SHIP tensor key width {} is narrower than the raised precision {kk}",
        tensor_key.k()
    );
    Ok(())
}
