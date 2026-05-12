use itertools::Itertools;
use poulpy_core::{
    GLWECopy, ScratchArenaTakeCore,
    layouts::{GGSWInfos, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc},
};
use poulpy_hal::layouts::{Backend, HostDataMut, Module, ScratchArena, ZnxZero};

use crate::bdd_arithmetic::{Cmux, Cswap, GetGGSWBit};

/// Stateful accumulator for oblivious retrieval of one GLWE ciphertext from a
/// stream of inputs using an encrypted binary index.
///
/// Implements a binary-carry-save accumulation strategy that processes input
/// ciphertexts one by one via [`add`][GLWEBlindRetriever::add], combining pairs
/// with CMux at successive bit positions.  When all inputs have been added,
/// [`flush`][GLWEBlindRetriever::flush] finalises the result.
///
/// The convenience method [`retrieve`][GLWEBlindRetriever::retrieve] combines
/// `reset`, all `add` calls, and `flush` in a single step.
///
/// ## Capacity
///
/// `alloc(infos, size)` allocates enough internal state to accumulate up to
/// `size` inputs.  Adding more than `size` inputs panics.
///
/// ## Scratch-Space
///
/// All methods that require scratch space accept a mutable `ScratchArena<BE>` arena.
/// The required size is returned by
/// [`retrieve_tmp_bytes`][GLWEBlindRetriever::retrieve_tmp_bytes].
pub struct GLWEBlindRetriever<D: poulpy_hal::layouts::Data = Vec<u8>> {
    accumulators: Vec<Accumulator<D>>,
    counter: usize,
}

impl GLWEBlindRetriever<Vec<u8>> {
    pub fn alloc<A, M>(module: &M, infos: &A, size: usize) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
        A: GLWEInfos,
    {
        let bit_size: usize = (u32::BITS - (size as u32 - 1).leading_zeros()) as usize;
        Self {
            accumulators: (0..bit_size).map(|_| Accumulator::alloc(module, infos)).collect_vec(),
            counter: 0,
        }
    }

    pub fn retrieve_tmp_bytes<M, R, S, BE>(module: &M, res: &R, selector: &S) -> usize
    where
        BE: Backend<OwnedBuf = Vec<u8>>,
        M: Cmux<BE>,
        R: GLWEInfos,
        S: GGSWInfos,
    {
        module.cmux_tmp_bytes(res, res, selector)
    }

    pub fn retrieve<M, R, A, S, BE>(
        &mut self,
        module: &M,
        res: &mut R,
        data: &[A],
        selector: &S,
        offset: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        M: GLWECopy<BE> + Cmux<BE>,
        BE: Backend<OwnedBuf = Vec<u8>> + 'static,
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        S: GetGGSWBit<BE>,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        self.reset();
        for ct in data {
            self.add(module, ct, selector, offset, scratch);
        }
        self.flush(module, res, selector, offset, scratch);
    }

    pub fn add<A, S, M, BE>(&mut self, module: &M, a: &A, selector: &S, offset: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        A: GLWEToBackendRef<BE>,
        S: GetGGSWBit<BE>,
        M: GLWECopy<BE> + Cmux<BE>,
        BE: Backend<OwnedBuf = Vec<u8>> + 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        assert!(
            (self.counter as u32) < 1 << self.accumulators.len(),
            "Accumulating limit of {} reached",
            1 << self.accumulators.len()
        );

        add_core(module, a, &mut self.accumulators, 0, selector, offset, scratch);
        self.counter += 1;
    }

    pub fn flush<R, M, S, BE>(&mut self, module: &M, res: &mut R, selector: &S, offset: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        S: GetGGSWBit<BE>,
        M: GLWECopy<BE> + Cmux<BE>,
        BE: Backend<OwnedBuf = Vec<u8>> + 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        if self.counter == 0 {
            res.to_backend_mut().data_mut().zero();
            self.reset();
            return;
        }
        for i in 0..self.accumulators.len() - 1 {
            let (acc_prev, acc_next) = self.accumulators.split_at_mut(i + 1);
            if acc_prev[i].num != 0 {
                add_core(module, &acc_prev[i].data, acc_next, i + 1, selector, offset, scratch);
                acc_prev[i].num = 0;
            }
        }
        module.glwe_copy(res, &self.accumulators.last().unwrap().data);
        self.reset()
    }

    fn reset(&mut self) {
        for acc in self.accumulators.iter_mut() {
            acc.num = 0;
        }
        self.counter = 0;
    }
}

struct Accumulator<D: poulpy_hal::layouts::Data> {
    data: GLWE<D>,
    num: usize, // Number of accumulated values
}

impl Accumulator<Vec<u8>> {
    pub fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = Vec<u8>>,
        A: GLWEInfos,
    {
        Self {
            data: module.glwe_alloc_from_infos(infos),
            num: 0,
        }
    }
}

fn add_core<A, S, M, BE>(
    module: &M,
    a: &A,
    accumulators: &mut [Accumulator<Vec<u8>>],
    i: usize,
    selector: &S,
    offset: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    A: GLWEToBackendRef<BE>,
    S: GetGGSWBit<BE>,
    M: GLWECopy<BE> + Cmux<BE>,
    BE: Backend<OwnedBuf = Vec<u8>> + 'static,
    for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
    for<'a> BE::BufMut<'a>: HostDataMut,
    for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
{
    // Isolate the first accumulator
    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    match acc_prev[0].num {
        0 => {
            module.glwe_copy(&mut acc_prev[0].data, a);
            acc_prev[0].num = 1;
        }
        1 => {
            let selector_bit = selector.get_bit(i + offset);
            module.cmux_assign_neg(&mut acc_prev[0].data, a, selector_bit, scratch);

            if !acc_next.is_empty() {
                add_core(module, &acc_prev[0].data, acc_next, i + 1, selector, offset, scratch);
            }

            acc_prev[0].num = 0
        }
        _ => {
            panic!("something went wrong")
        }
    }
}

impl<BE: Backend<OwnedBuf = Vec<u8>>> GLWEBlindRetrieval<BE> for Module<BE> where Self: GLWECopy<BE> + Cmux<BE> + Cswap<BE> {}

/// Oblivious in-place sorting / retrieval of a GLWE vector by an encrypted index.
///
/// Where `GLWEBlindSelection` extracts one element from a map given an encrypted
/// key, `GLWEBlindRetrieval` operates on an ordered `Vec<R>` and performs a
/// sorting-network-style rearrangement: after
/// [`glwe_blind_retrieval_statefull`][Self::glwe_blind_retrieval_statefull],
/// element `0` of the vector encrypts the input element whose index equals the
/// encrypted selector.
///
/// The rearrangement uses conditional-swap ([`Cswap`]) operations, one per bit
/// of the selector sub-field.  The `_rev` variant applies the operations in
/// reverse, useful for undoing the permutation.
pub trait GLWEBlindRetrieval<BE: Backend<OwnedBuf = Vec<u8>>>
where
    Self: GLWECopy<BE> + Cmux<BE> + Cswap<BE>,
{
    /// Returns the minimum scratch-space size in bytes required by
    /// [`glwe_blind_retrieval_statefull`][Self::glwe_blind_retrieval_statefull].
    fn glwe_blind_retrieval_tmp_bytes<R, K>(&self, res_infos: &R, k_infos: &K) -> usize
    where
        R: GLWEInfos,
        K: GGSWInfos,
    {
        self.cswap_tmp_bytes(res_infos, res_infos, k_infos)
    }

    /// Rearranges `res` in-place so that `res[0]` encrypts the element at the
    /// encrypted index `(bits >> bit_rsh) % 2^bit_mask`.
    ///
    /// Uses a butterfly network of [`Cswap`] gates, iterating from the
    /// most-significant to the least-significant bit of the selector sub-field.
    fn glwe_blind_retrieval_statefull<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        for i in 0..bit_mask {
            let t: usize = 1 << (bit_mask - i - 1);
            let bit = bits.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal
            for j in 0..t {
                if j + t < res.len() {
                    let (lo, hi) = res.split_at_mut(j + t);
                    self.cswap(&mut lo[j], &mut hi[0], bit, &mut scratch.borrow());
                }
            }
        }
    }

    /// Reverses the permutation applied by
    /// [`glwe_blind_retrieval_statefull`][Self::glwe_blind_retrieval_statefull].
    ///
    /// Applies the same butterfly network in reverse order, restoring the original
    /// element ordering after an oblivious retrieval.
    fn glwe_blind_retrieval_statefull_rev<R, K>(
        &self,
        res: &mut Vec<R>,
        bits: &K,
        bit_rsh: usize,
        bit_mask: usize,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE> + GLWEToBackendRef<BE> + GLWEInfos,
        K: GetGGSWBit<BE>,
        BE: 'static,
        for<'a> ScratchArena<'a, BE>: ScratchArenaTakeCore<'a, BE>,
        for<'a> BE::BufMut<'a>: HostDataMut,
        for<'a> BE: Backend<BufMut<'a> = &'a mut [u8], BufRef<'a> = &'a [u8]>,
    {
        for i in (0..bit_mask).rev() {
            let t: usize = 1 << (bit_mask - i - 1);
            let bit = bits.get_bit(bit_rsh + bit_mask - i - 1); // MSB -> LSB traversal
            for j in 0..t {
                if j < res.len() && j + t < res.len() {
                    let (lo, hi) = res.split_at_mut(j + t);
                    self.cswap(&mut lo[j], &mut hi[0], bit, &mut scratch.borrow());
                }
            }
        }
    }
}
