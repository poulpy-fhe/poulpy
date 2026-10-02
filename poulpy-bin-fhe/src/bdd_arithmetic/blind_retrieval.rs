pub use crate::api::GLWEBlindRetrieval;
use itertools::Itertools;
use poulpy_core::layouts::prepared::GGSWPreparedToBackendRef;
use poulpy_core::{
    GLWECopy, GLWEZero,
    layouts::{GGSWInfos, GLWE, GLWEInfos, GLWEToBackendMut, GLWEToBackendRef, ModuleCoreAlloc},
};
use poulpy_hal::layouts::{Backend, Data, ScratchArena};

use crate::bdd_arithmetic::{Cmux, GetGGSWBit};
use poulpy_core::GLWEBytesOf;

/// Stateful accumulator for oblivious retrieval of one GLWE ciphertext from a
/// stream of inputs using an encrypted binary index.
///
/// Implements a binary-carry-save accumulation strategy that processes input
/// ciphertexts one by one via [`add`][GLWEBlindRetriever::add], combining pairs
/// with CMux at successive bit positions.  When all inputs have been added,
/// [`flush`][GLWEBlindRetriever::flush] finalises the result.
///
/// The convenience method [`retrieve`][GLWEBlindRetriever::retrieve] combines
/// `reset`, all `add` calls, and `flush` in a single step. Inputs are borrowed
/// without mutation. Unlike [`GLWEBlindRetrieval`], this helper does not permute
/// a ciphertext array for later reversal: it owns persistent accumulation state
/// and composes the selected [`Cmux`] and core operations.
///
/// For a nonempty stream, the encrypted index starting at `offset` must refer
/// to an input actually added; missing positions are not padded with zero.
/// Flushing an empty stream writes zero. Every flush resets the accumulation.
///
/// ## Capacity
///
/// `alloc(infos, size)` allocates enough internal state to accumulate up to
/// `size` inputs, including capacities zero and one. Adding more than `size`
/// inputs panics. [`retrieve`][Self::retrieve] checks its input length before
/// resetting the state or writing the result.
///
/// ## Scratch-Space
///
/// Inputs must match the accumulator ring degree, rank and radix. Precision
/// and allocated capacity may differ. For compatible layouts, take the maximum of
/// [`add_tmp_bytes`][Self::add_tmp_bytes] for each input and
/// [`flush_tmp_bytes`][Self::flush_tmp_bytes] for the output. These queries use
/// the actual persistent accumulator layout and remain valid before adding any
/// inputs. The selector metadata must describe every selector bit used.
/// [`retrieve_tmp_bytes`][Self::retrieve_tmp_bytes] is a convenience query for
/// uniform external buffers and compact accumulators.
pub struct GLWEBlindRetriever<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> {
    accumulators: Vec<Accumulator<D, W>>,
    counter: usize,
    capacity: usize,
}

impl<D: Data> GLWEBlindRetriever<D, i64> {
    pub fn alloc<A, M>(module: &M, infos: &A, size: usize) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = i64>,
        A: GLWEInfos,
    {
        let bit_size = if size == 0 {
            0
        } else {
            ((usize::BITS - (size - 1).leading_zeros()) as usize).max(1)
        };
        Self {
            accumulators: (0..bit_size).map(|_| Accumulator::alloc(module, infos)).collect_vec(),
            counter: 0,
            capacity: size,
        }
    }

    /// Scratch required when every input has the same layout and allocated
    /// capacity as `res`, and accumulators have `res.glwe_layout()` (the compact
    /// layout produced by the standard allocator). For other layouts, use
    /// [`add_tmp_bytes`][Self::add_tmp_bytes] and
    /// [`flush_tmp_bytes`][Self::flush_tmp_bytes].
    pub fn retrieve_tmp_bytes<M, R, S, BE>(module: &M, res: &R, selector: &S) -> usize
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: GLWEBytesOf<BE> + GLWECopy<BE> + Cmux<BE>,
        R: GLWEInfos,
        S: GGSWInfos,
    {
        let acc = res.glwe_layout();
        module
            .cmux_tmp_bytes(&acc, res, selector)
            .max(module.cmux_tmp_bytes(&acc, &acc, selector))
            .max(module.glwe_copy_tmp_bytes(&acc, res))
            .max(module.glwe_copy_tmp_bytes(&acc, &acc))
            .max(module.glwe_copy_tmp_bytes(res, &acc))
    }

    /// Scratch required to add an input with the supplied layout and allocated
    /// capacity, including any carries through the persistent accumulators.
    /// The bound does not depend on how many inputs have already been added.
    pub fn add_tmp_bytes<M, A, S, BE>(&self, module: &M, a: &A, selector: &S) -> usize
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: GLWEBytesOf<BE> + GLWECopy<BE> + Cmux<BE>,
        A: GLWEInfos,
        S: GGSWInfos,
    {
        let Some(acc) = self.accumulators.first().map(|acc| &acc.data) else {
            return 0;
        };
        module
            .cmux_tmp_bytes(acc, a, selector)
            .max(module.cmux_tmp_bytes(acc, acc, selector))
            .max(module.glwe_copy_tmp_bytes(acc, a))
            .max(module.glwe_copy_tmp_bytes(acc, acc))
    }

    /// Scratch required to finish accumulation and copy into the supplied
    /// output layout, including its allocated capacity. The bound is valid even
    /// when queried before any inputs have been added.
    pub fn flush_tmp_bytes<M, R, S, BE>(&self, module: &M, res: &R, selector: &S) -> usize
    where
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        M: GLWEBytesOf<BE> + GLWECopy<BE> + Cmux<BE>,
        R: GLWEInfos,
        S: GGSWInfos,
    {
        let Some(acc) = self.accumulators.first().map(|acc| &acc.data) else {
            return 0;
        };
        module
            .cmux_tmp_bytes(acc, acc, selector)
            .max(module.glwe_copy_tmp_bytes(acc, acc))
            .max(module.glwe_copy_tmp_bytes(res, acc))
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
        M: GLWEBytesOf<BE> + GLWECopy<BE> + GLWEZero<BE> + Cmux<BE>,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
        R: GLWEToBackendMut<BE>,
        A: GLWEToBackendRef<BE>,
        S: GetGGSWBit<BE>,
    {
        assert!(data.len() <= self.capacity, "retrieval capacity exceeded");
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
        M: GLWEBytesOf<BE> + GLWECopy<BE> + Cmux<BE>,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
    {
        assert!(self.counter < self.capacity, "retrieval capacity exceeded");

        add_core(module, a, &mut self.accumulators, 0, selector, offset, scratch);
        self.counter += 1;
    }

    pub fn flush<R, M, S, BE>(&mut self, module: &M, res: &mut R, selector: &S, offset: usize, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE>,
        S: GetGGSWBit<BE>,
        M: GLWEBytesOf<BE> + GLWECopy<BE> + GLWEZero<BE> + Cmux<BE>,
        BE: Backend<OwnedBuf = D, ZnxWord = i64>,
    {
        if self.counter == 0 {
            module.glwe_zero(res);
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
        module.glwe_copy(res, &self.accumulators.last().unwrap().data, scratch);
        self.reset()
    }

    fn reset(&mut self) {
        for acc in self.accumulators.iter_mut() {
            acc.num = 0;
        }
        self.counter = 0;
    }
}

struct Accumulator<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> {
    data: GLWE<D, W>,
    num: usize, // Number of accumulated values
}

impl<D: poulpy_hal::layouts::Data, W: poulpy_hal::layouts::ZnxWord> Accumulator<D, W> {
    pub fn alloc<A, M>(module: &M, infos: &A) -> Self
    where
        M: ModuleCoreAlloc<OwnedBuf = D, ZnxWord = W>,
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
    accumulators: &mut [Accumulator<BE::OwnedBuf, BE::ZnxWord>],
    i: usize,
    selector: &S,
    offset: usize,
    scratch: &mut ScratchArena<'_, BE>,
) where
    A: GLWEToBackendRef<BE>,
    S: GetGGSWBit<BE>,
    M: GLWEBytesOf<BE> + GLWECopy<BE> + Cmux<BE>,
    BE: Backend<ZnxWord = i64>,
{
    // Isolate the first accumulator
    let (acc_prev, acc_next) = accumulators.split_at_mut(1);

    match acc_prev[0].num {
        0 => {
            module.glwe_copy(&mut acc_prev[0].data, a, scratch);
            acc_prev[0].num = 1;
        }
        1 => {
            let selector_bit = selector.get_bit(i + offset);
            module.cmux_assign_neg(&mut acc_prev[0].data, a, &selector_bit.to_backend_ref(), scratch);

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
