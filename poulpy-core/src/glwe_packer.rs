use poulpy_hal::{
    api::{ModuleLogN, ScratchAvailable},
    layouts::{Backend, GaloisElement, Module, Scratch},
};

pub use crate::api::GLWEPackerOps;
use crate::{
    GLWEAdd, GLWEAutomorphism, GLWECopy, GLWENormalize, GLWERotate, GLWEShift, GLWESub, ScratchTakeCore,
    glwe_trace::GLWETrace,
    layouts::{
        GGLWEInfos, GGLWEPreparedToRef, GLWE, GLWEAutomorphismKeyHelper, GLWEInfos, GLWEToMut, GLWEToRef, GetGaloisElement,
        LWEInfos,
    },
};

/// [GLWEPacker] enables only the fly GLWE packing
/// with constant memory of Log(N) ciphertexts.
/// Main difference with usual GLWE packing is that
/// the output is bit-reversed.
pub struct GLWEPacker {
    pub(crate) accumulators: Vec<Accumulator>,
    log_batch: usize,
    counter: usize,
}

/// [Accumulator] stores intermediate packing result.
/// There are Log(N) such accumulators in a [GLWEPacker].
pub(crate) struct Accumulator {
    data: GLWE<Vec<u8>>,
    value: bool,   // Implicit flag for zero ciphertext
    control: bool, // Can be combined with incoming value
}

impl Accumulator {
    /// Allocates a new [Accumulator].
    ///
    /// #Arguments
    ///
    /// * `module`: static backend FFT tables.
    /// * `base2k`: base 2 logarithm of the GLWE ciphertext in memory digit representation.
    /// * `k`: base 2 precision of the GLWE ciphertext precision over the Torus.
    /// * `rank`: rank of the GLWE ciphertext.
    pub fn alloc<A>(infos: &A) -> Self
    where
        A: GLWEInfos,
    {
        Self {
            data: GLWE::alloc_from_infos(infos),
            value: false,
            control: false,
        }
    }
}

impl GLWEPacker {
    /// Instantiates a new [GLWEPacker].
    ///
    /// # Arguments
    ///
    /// * `log_batch`: packs coefficients which are multiples of X^{N/2^log_batch}.
    ///   i.e. with `log_batch=0` only the constant coefficient is packed
    ///   and N GLWE ciphertext can be packed. With `log_batch=2` all coefficients
    ///   which are multiples of X^{N/4} are packed. Meaning that N/4 ciphertexts
    ///   can be packed.
    pub fn alloc<A>(infos: &A, log_batch: usize) -> Self
    where
        A: GLWEInfos,
    {
        let mut accumulators: Vec<Accumulator> = Vec::<Accumulator>::new();
        let log_n: usize = infos.n().log2();
        (0..log_n - log_batch).for_each(|_| accumulators.push(Accumulator::alloc(infos)));
        GLWEPacker {
            accumulators,
            log_batch,
            counter: 0,
        }
    }

    /// Implicit reset of the internal state (to be called before a new packing procedure).
    fn reset(&mut self) {
        for i in 0..self.accumulators.len() {
            self.accumulators[i].value = false;
            self.accumulators[i].control = false;
        }
        self.counter = 0;
    }

    // module-only API: packing operations are provided as free functions below.
}

/// Number of scratch space bytes required to call [`glwe_packer_add`].
pub fn glwe_packer_tmp_bytes<R, K, M, BE: Backend>(module: &M, res_infos: &R, key_infos: &K) -> usize
where
    R: GLWEInfos,
    K: GGLWEInfos,
    M: GLWEPackerOps<BE>,
{
    GLWE::<Vec<u8>>::bytes_of_from_infos(res_infos)
        + module
            .glwe_shift_tmp_bytes()
            .max(module.glwe_automorphism_tmp_bytes(res_infos, res_infos, key_infos))
}

/// Returns the Galois elements needed by the GLWE packer.
pub fn glwe_packer_galois_elements<M, BE: Backend>(module: &M) -> Vec<i64>
where
    M: GLWETrace<BE>,
{
    module.glwe_trace_galois_elements()
}

/// Adds a GLWE ciphertext to the [`GLWEPacker`].
pub fn glwe_packer_add<A, K, H, M, BE: Backend>(
    module: &M,
    packer: &mut GLWEPacker,
    a: Option<&A>,
    auto_keys: &H,
    scratch: &mut Scratch<BE>,
) where
    A: GLWEToRef + GLWEInfos,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    M: GLWEPackerOps<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert!(
        (packer.counter as u32) < packer.accumulators[0].data.n(),
        "Packing limit of {} reached",
        packer.accumulators[0].data.n().0 as usize >> packer.log_batch
    );
    assert!(
        scratch.available() >= glwe_packer_tmp_bytes(module, &packer.accumulators[0].data, &auto_keys.automorphism_key_infos()),
        "scratch.available(): {} < glwe_packer_tmp_bytes: {}",
        scratch.available(),
        glwe_packer_tmp_bytes(module, &packer.accumulators[0].data, &auto_keys.automorphism_key_infos())
    );

    module.packer_add(packer, a, packer.log_batch, auto_keys, scratch);
    packer.counter += 1 << packer.log_batch;
}

/// Flushes the packed result into `res`.
pub fn glwe_packer_flush<R, M, BE: Backend>(module: &M, packer: &mut GLWEPacker, res: &mut R, scratch: &mut Scratch<BE>)
where
    R: GLWEToMut + GLWEInfos,
    M: GLWEPackerOps<BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    assert!(packer.counter as u32 == packer.accumulators[0].data.n());

    let out: &GLWE<Vec<u8>> = &packer.accumulators[module.log_n() - packer.log_batch - 1].data;

    if out.base2k() == res.base2k() {
        module.glwe_copy(res, out)
    } else {
        module.glwe_normalize(res, out, scratch);
    }

    packer.reset();
}

#[doc(hidden)]
pub trait GLWEPackerOpsDefault<BE: Backend>
where
    Self: Sized
        + ModuleLogN
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy,
{
    fn packer_add_default<A, K, H>(
        &self,
        packer: &mut GLWEPacker,
        a: Option<&A>,
        i: usize,
        auto_keys: &H,
        scratch: &mut Scratch<BE>,
    ) where
        A: GLWEToRef + GLWEInfos,
        K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
        H: GLWEAutomorphismKeyHelper<K, BE>,
        Scratch<BE>: ScratchTakeCore<BE>,
    {
        pack_core(self, a, &mut packer.accumulators, i, auto_keys, scratch)
    }
}

impl<BE: Backend> GLWEPackerOpsDefault<BE> for Module<BE> where
    Self: Sized
        + ModuleLogN
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy
{
}

pub(crate) fn pack_core<A, K, H, M, BE: Backend>(
    module: &M,
    a: Option<&A>,
    accumulators: &mut [Accumulator],
    i: usize,
    auto_keys: &H,
    scratch: &mut Scratch<BE>,
) where
    A: GLWEToRef + GLWEInfos,
    M: ModuleLogN
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let log_n: usize = module.log_n();

    if i == log_n {
        return;
    }

    // Isolate the first accumulator
    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    // Control = true accumlator is free to overide
    if !acc_prev[0].control {
        let acc_mut_ref: &mut Accumulator = &mut acc_prev[0]; // from split_at_mut

        // No previous value -> copies and sets flags accordingly
        if let Some(a_ref) = a {
            if a_ref.base2k() == acc_mut_ref.data.base2k() {
                module.glwe_copy(&mut acc_mut_ref.data, a_ref);
            } else {
                module.glwe_normalize(&mut acc_mut_ref.data, a_ref, scratch);
            }
            acc_mut_ref.value = true
        } else {
            acc_mut_ref.value = false
        }
        acc_mut_ref.control = true; // Able to be combined on next call
    } else {
        // Compresses acc_prev <- combine(acc_prev, a).
        combine(module, &mut acc_prev[0], a, i, auto_keys, scratch);
        acc_prev[0].control = false;

        // Propagates to next accumulator
        if acc_prev[0].value {
            pack_core(module, Some(&acc_prev[0].data), acc_next, i + 1, auto_keys, scratch);
        } else {
            pack_core(module, None::<&GLWE<Vec<u8>>>, acc_next, i + 1, auto_keys, scratch);
        }
    }
}

fn combine<B, K, H, M, BE: Backend>(
    module: &M,
    acc: &mut Accumulator,
    b: Option<&B>,
    i: usize,
    auto_keys: &H,
    scratch: &mut Scratch<BE>,
) where
    B: GLWEToRef + GLWEInfos,
    M: ModuleLogN
        + GLWEAutomorphism<BE>
        + GaloisElement
        + GLWERotate<BE>
        + GLWESub
        + GLWEShift<BE>
        + GLWEAdd
        + GLWENormalize<BE>
        + GLWECopy,
    K: GGLWEPreparedToRef<BE> + GetGaloisElement + GGLWEInfos,
    H: GLWEAutomorphismKeyHelper<K, BE>,
    Scratch<BE>: ScratchTakeCore<BE>,
{
    let log_n: usize = acc.data.n().log2();
    let a: &mut GLWE<Vec<u8>> = &mut acc.data;

    let gal_el: i64 = if i == 0 { -1 } else { module.galois_element(1 << (i - 1)) };

    let t: i64 = 1 << (log_n - i - 1);

    // Goal is to evaluate: a = a + b*X^t + phi(a - b*X^t))
    // We also use the identity: AUTO(a * X^t, g) = -X^t * AUTO(a, g)
    // where t = 2^(log_n - i - 1) and g = 5^{2^(i - 1)}
    // Different cases for wether a and/or b are zero.
    //
    // Implicite RSH without modulus switch, introduces extra I(X) * Q/2 on decryption.
    // Necessary so that the scaling of the plaintext remains constant.
    // It however is ok to do so here because coefficients are eventually
    // either mapped to garbage or twice their value which vanishes I(X)
    // since 2*(I(X) * Q/2) = I(X) * Q = 0 mod Q.
    if acc.value {
        if let Some(b) = b {
            let (mut tmp, scratch_1) = scratch.take_glwe(a);

            // a = a * X^-t
            module.glwe_rotate_assign(-t, a, scratch_1);

            // tmp_b = a * X^-t - b
            module.glwe_sub(&mut tmp, a, b);
            module.glwe_rsh(1, &mut tmp, scratch_1);
            // a = a * X^-t + b
            module.glwe_add_assign(a, b);

            module.glwe_rsh(1, a, scratch_1);
            module.glwe_normalize_assign(&mut tmp, scratch_1);

            // tmp_b = phi(a * X^-t - b)
            if let Some(auto_key) = auto_keys.get_automorphism_key(gal_el) {
                module.glwe_automorphism_assign(&mut tmp, auto_key, scratch_1);
            } else {
                panic!("auto_key[{gal_el}] not found");
            }

            // a = a * X^-t + b - phi(a * X^-t - b)
            module.glwe_sub_assign(a, &tmp);
            module.glwe_normalize_assign(a, scratch_1);

            // a = a + b * X^t - phi(a * X^-t - b) * X^t
            //   = a + b * X^t - phi(a * X^-t - b) * - phi(X^t)
            //   = a + b * X^t + phi(a - b * X^t)
            module.glwe_rotate_assign(t, a, scratch_1);
        } else {
            module.glwe_rsh(1, a, scratch);
            // a = a + phi(a)
            if let Some(auto_key) = auto_keys.get_automorphism_key(gal_el) {
                module.glwe_automorphism_add_assign(a, auto_key, scratch);
            } else {
                panic!("auto_key[{gal_el}] not found");
            }
        }
    } else if let Some(b) = b {
        let (mut tmp_b, scratch_1) = scratch.take_glwe(a);
        module.glwe_rotate(t, &mut tmp_b, b);
        module.glwe_rsh(1, &mut tmp_b, scratch_1);

        // a = (b* X^t - phi(b* X^t))
        if let Some(auto_key) = auto_keys.get_automorphism_key(gal_el) {
            module.glwe_automorphism_sub_negate(a, &tmp_b, auto_key, scratch_1);
        } else {
            panic!("auto_key[{gal_el}] not found");
        }

        acc.value = true;
    }
}
