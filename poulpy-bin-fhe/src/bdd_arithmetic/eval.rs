use core::panic;
use poulpy_core::{
    GLWECopy, GLWENormalize, GLWESub, GLWEZero, ScratchArenaTakeCore,
    api::GLWEExternalProductInternal,
    default::external_product::glwe::glwe_external_product_output_size,
    layouts::{
        GGSWInfos, GLWEInfos, GLWELayout, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, ModuleCoreAlloc,
        prepared::{GGSWPreparedBackendRef, GGSWPreparedToBackendRef},
    },
};
use poulpy_hal::layouts::CoeffNormalized;
use poulpy_hal::{
    api::{
        ModuleN, ScratchArenaTakeBasic, VecZnxAddScalarAssignBackend, VecZnxBigAddSmallAssign, VecZnxBigAddSmallIntoBackend,
        VecZnxBigBytesOf, VecZnxBigNormalize, VecZnxBigNormalizeTmpBytes, VecZnxBigSubSmallABackend, VecZnxDftAddAssign,
        VecZnxDftApply, VecZnxDftBytesOf, VecZnxDftZero, VecZnxIdftApply, VecZnxNormalizeAssignBackend, VecZnxNormalizeTmpBytes,
        VmpApplyDftToDft, VmpApplyDftToDftTmpBytes,
    },
    layouts::{
        Backend, Host, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena, VecZnxBigViewMut, vec_znx_backend_ref_from_mut,
        vec_znx_big_backend_ref_from_mut, vec_znx_dft_backend_ref_from_mut,
    },
};

use crate::bdd_arithmetic::GetGGSWBit;
use poulpy_core::GLWEBytesOf;

/// A single bit-output circuit stored as a flat node array.
///
/// Implementors provide the node sequence and the maximum intermediate state
/// size required during evaluation.
pub trait BitCircuitInfo: Sync {
    /// Returns the node sequence and the maximum intermediate-state count
    /// (`max_inter_state`) for this output bit.
    fn info(&self) -> (&[Node], usize);
}

/// A multi-output BDD circuit that maps encrypted inputs to encrypted output bits.
///
/// Provides the dimensional information and per-bit circuit access needed by
/// [`ExecuteBDDCircuit`].
pub trait GetBitCircuitInfo: Sync {
    /// Number of input bits expected by this circuit (across all input words).
    fn input_size(&self) -> usize;
    /// Number of output bits produced by this circuit.
    fn output_size(&self) -> usize;
    /// Returns the node sequence and intermediate-state count for output bit `bit`.
    fn get_circuit(&self, bit: usize) -> (&[Node], usize);

    /// Maximum `max_inter_state` across all output-bit circuits.
    ///
    /// If [`output_size`][Self::output_size] is zero, returns `0`.
    fn max_state_size(&self) -> usize {
        (0..self.output_size()).map(|i| self.get_circuit(i).1).fold(0, usize::max)
    }
}

/// A statically-sized BDD bit-circuit, produced by the code-generator.
///
/// `N` is the total number of [`Node`] entries in the circuit.
/// `max_inter_state` is the width of the intermediate-state buffer required
/// during evaluation (i.e. the maximum number of live GLWE values at any BDD
/// level).
pub struct BitCircuit<const N: usize> {
    /// The flat node array encoding this circuit's BDD levels.
    pub nodes: [Node; N],
    /// Maximum width of the BDD intermediate state.
    pub max_inter_state: usize,
}

/// Associates compile-time input/output bit counts with a family of [`BitCircuit`]s.
///
/// Implemented by code-generated circuit types.  Used by [`Circuit`] to satisfy
/// the [`GetBitCircuitInfo`] bound.
pub trait BitCircuitFamily {
    /// Total number of input bits across all input words.
    const INPUT_BITS: usize;
    /// Number of output bits produced by circuits in this family.
    const OUTPUT_BITS: usize;
}

/// An array of `N` per-output-bit circuits sharing the same `C` circuit type.
///
/// Implements [`GetBitCircuitInfo`] by delegating each output bit to the
/// corresponding `C` entry.  The circuit type `C` must implement both
/// [`BitCircuitInfo`] and [`BitCircuitFamily`] to supply input/output sizes.
pub struct Circuit<C: BitCircuitInfo, const N: usize>(pub [C; N]);

impl<C, const N: usize> GetBitCircuitInfo for Circuit<C, N>
where
    C: BitCircuitInfo + BitCircuitFamily,
{
    fn input_size(&self) -> usize {
        C::INPUT_BITS
    }
    fn output_size(&self) -> usize {
        C::OUTPUT_BITS
    }
    fn get_circuit(&self, bit: usize) -> (&[Node], usize) {
        self.0[bit].info()
    }
}

/// Backend-level BDD circuit evaluator.
///
/// Evaluates a multi-output BDD circuit on a set of encrypted input bits,
/// producing one GLWE ciphertext per output bit.  The circuit is represented as
/// a sequence of [`Node`] entries arranged in BDD levels; each level is evaluated
/// using [`Cmux`] gates.
pub trait ExecuteBDDCircuit<BE: Backend> {
    /// Returns the minimum scratch-space size in bytes required by a single
    /// thread of BDD circuit evaluation.
    ///
    /// `state_size` is the maximum number of live intermediate GLWE values
    /// (i.e. `max_inter_state` from [`BitCircuit`]).
    fn execute_bdd_circuit_tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos;

    fn execute_bdd_circuit_tmp_bytes_for<R, G, C>(&self, res_infos: &R, circuit: &C, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo,
    {
        self.execute_bdd_circuit_tmp_bytes(res_infos, circuit.max_state_size(), ggsw_infos)
    }

    /// Single-threaded BDD circuit evaluation.
    ///
    /// Evaluates `circuit` on `inputs`, writing one GLWE ciphertext per output
    /// bit into `out[0..circuit.output_size()]`.  Elements beyond
    /// `output_size` are zeroed.
    ///
    /// Delegates to [`execute_bdd_circuit_multi_thread`][Self::execute_bdd_circuit_multi_thread]
    /// with `threads = 1`.
    fn execute_bdd_circuit<C, G, O>(&self, out: &mut [O], inputs: &G, circuit: &C, scratch: &mut ScratchArena<'_, BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + Send,
    {
        self.execute_bdd_circuit_multi_thread(1, out, inputs, circuit, scratch);
    }

    /// Multi-threaded BDD circuit evaluation.
    ///
    /// Partitions the output bits across at most `threads` workers using the
    /// selected backend's task executor. Each worker receives a dedicated slice of the
    /// scratch arena of size
    /// [`execute_bdd_circuit_tmp_bytes`][Self::execute_bdd_circuit_tmp_bytes].
    ///
    /// # Panics
    ///
    /// Panics if the arena cannot provide one scratch slice per active worker.
    fn execute_bdd_circuit_multi_thread<C, G, O>(
        &self,
        threads: usize,
        out: &mut [O],
        inputs: &G,
        circuit: &C,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + Send;
}

pub trait BitSize {
    fn bit_size(&self) -> usize;
}

pub(super) trait BddEvaluator<BE: Backend, L> {
    fn tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos;

    fn tmp_bytes_for<R, G, C>(&self, res_infos: &R, circuit: &C, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo;

    fn execute<C, G, O>(&self, threads: usize, out: &mut [O], inputs: &G, circuit: &C, scratch: &mut ScratchArena<'_, BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + Send;
}

pub(super) trait BddTrivialOne<BE: Backend, L> {
    type Prepared: Sync;

    fn prepare_bdd_trivial_one<R: GLWEInfos>(&self, infos: &R) -> Self::Prepared;

    fn set_bdd_trivial_one<R>(&self, res: &mut R, prepared: &Self::Prepared)
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos;
}

pub(super) fn bdd_parallel_tmp_bytes<BE: Backend>(threads: usize, output_size: usize, worker_bytes: usize) -> usize {
    let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(threads, output_size);
    workers
        .checked_mul(poulpy_hal::execution::worker_scratch_bytes::<BE>(worker_bytes))
        .expect("BDD scratch size overflow")
}

#[allow(private_bounds)]
impl<BE: Backend<ZnxWord = i64>> ExecuteBDDCircuit<BE> for Module<BE>
where
    Self: BddEvaluator<BE, BE::Location>,
{
    fn execute_bdd_circuit_tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
    {
        <Self as BddEvaluator<BE, BE::Location>>::tmp_bytes(self, res_infos, state_size, ggsw_infos)
    }

    fn execute_bdd_circuit_tmp_bytes_for<R, G, C>(&self, res_infos: &R, circuit: &C, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo,
    {
        <Self as BddEvaluator<BE, BE::Location>>::tmp_bytes_for(self, res_infos, circuit, ggsw_infos)
    }

    fn execute_bdd_circuit_multi_thread<C, G, O>(
        &self,
        threads: usize,
        out: &mut [O],
        inputs: &G,
        circuit: &C,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + Send,
    {
        <Self as BddEvaluator<BE, BE::Location>>::execute(self, threads, out, inputs, circuit, scratch);
    }
}

impl<BE: Backend<Location = Host, ZnxWord = i64>> BddEvaluator<BE, Host> for Module<BE>
where
    Self: GLWEBytesOf<BE>
        + Cmux<BE>
        + GLWECopy<BE>
        + GLWEZero<BE>
        + VecZnxAddScalarAssignBackend<BE>
        + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>
        + Sync,
    BE: 'static,
{
    fn tmp_bytes<R, G>(&self, res_infos: &R, state_size: usize, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
    {
        2 * state_size * self.glwe_bytes_of_from_infos(res_infos) + self.cmux_tmp_bytes(res_infos, res_infos, ggsw_infos)
    }

    fn tmp_bytes_for<R, G, C>(&self, res_infos: &R, circuit: &C, ggsw_infos: &G) -> usize
    where
        R: GLWEInfos,
        G: GGSWInfos,
        C: GetBitCircuitInfo,
    {
        <Self as BddEvaluator<BE, Host>>::tmp_bytes(self, res_infos, circuit.max_state_size(), ggsw_infos)
    }

    fn execute<C, G, O>(&self, threads: usize, out: &mut [O], inputs: &G, circuit: &C, scratch: &mut ScratchArena<'_, BE>)
    where
        G: GetGGSWBit<BE> + BitSize,
        C: GetBitCircuitInfo,
        O: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + Send,
    {
        execute_bdd_circuit_default(self, threads, out, inputs, circuit, scratch);
    }
}

impl<BE: Backend<Location = Host, ZnxWord = i64>> BddTrivialOne<BE, Host> for Module<BE>
where
    Self: GLWEZero<BE> + VecZnxAddScalarAssignBackend<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    type Prepared = ScalarZnx<BE::OwnedBuf, i64>;

    fn prepare_bdd_trivial_one<R: GLWEInfos>(&self, infos: &R) -> Self::Prepared {
        trivial_one_scalar::<BE, _>(infos)
    }

    fn set_bdd_trivial_one<R>(&self, res: &mut R, prepared: &Self::Prepared)
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
    {
        glwe_set_trivial_one_with_scalar(self, res, prepared);
    }
}

pub(super) fn execute_bdd_circuit_default<BE, M, C, G, O, L>(
    module: &M,
    threads: usize,
    out: &mut [O],
    inputs: &G,
    circuit: &C,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<ZnxWord = i64> + 'static,
    M: GLWEBytesOf<BE> + Cmux<BE> + GLWECopy<BE> + GLWEZero<BE> + BddTrivialOne<BE, L> + Sync,
    G: GetGGSWBit<BE> + BitSize,
    C: GetBitCircuitInfo,
    O: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos + Send,
{
    assert!(inputs.bit_size() >= circuit.input_size());
    assert!(out.len() >= circuit.output_size());
    let output_size = circuit.output_size();
    for out_i in out.iter_mut().skip(output_size) {
        module.glwe_zero(out_i);
    }
    if output_size == 0 {
        return;
    }

    let one = module.prepare_bdd_trivial_one(&out[0]);
    let scratch_thread_size = poulpy_hal::execution::worker_scratch_bytes::<BE>(
        2 * circuit.max_state_size() * module.glwe_bytes_of_from_infos(&out[0])
            + module.cmux_tmp_bytes(&out[0], &out[0], inputs.get_bit(0)),
    );
    let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(threads, output_size);
    let needed = bdd_parallel_tmp_bytes::<BE>(threads, output_size, scratch_thread_size);
    assert!(scratch.available() >= needed);
    let (worker_scratch, _) = scratch.borrow().split(workers, scratch_thread_size);
    poulpy_hal::execution::for_each_with_scratch::<BE::TaskExecutor, BE, _, _>(
        &mut out[..output_size],
        0,
        worker_scratch,
        &|bit_idx, out_i, scratch| {
            let (nodes, state_size) = circuit.get_circuit(bit_idx);
            if state_size == 0 {
                module.glwe_zero(out_i);
            } else {
                eval_level(module, out_i, inputs, nodes, state_size, &one, scratch);
            }
        },
    );
}

fn eval_level<M, G, R, BE, L>(
    module: &M,
    res: &mut R,
    inputs: &G,
    nodes: &[Node],
    state_size: usize,
    one: &M::Prepared,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: Cmux<BE> + GLWECopy<BE> + GLWEZero<BE> + BddTrivialOne<BE, L>,
    BE: Backend<ZnxWord = i64> + 'static,
    G: GetGGSWBit<BE> + BitSize,
    R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
{
    assert!(nodes.len().is_multiple_of(state_size));

    let (mut level, mut scratch_1) = scratch.borrow().take_glwe_slice_scratch(2 * state_size, res);

    level.iter_mut().for_each(|ct| module.glwe_zero(ct));
    module.set_bdd_trivial_one(&mut level[1], one);

    let (mut prev_level, mut next_level) = level.split_at_mut(state_size);

    let (all_but_last, last) = nodes.split_at(nodes.len() - state_size);

    for nodes_lvl in all_but_last.chunks_exact(state_size) {
        for (j, node) in nodes_lvl.iter().enumerate() {
            match node {
                Node::Cmux(in_idx, hi_idx, lo_idx) => {
                    module.cmux(
                        &mut next_level[j],
                        &prev_level[*hi_idx],
                        &prev_level[*lo_idx],
                        &inputs.get_bit(*in_idx).to_backend_ref(),
                        &mut scratch_1.borrow(),
                    );
                }
                Node::Copy => module.glwe_copy(&mut next_level[j], &prev_level[j]), /* Update BDD circuits to order Cmux -> Copy -> None so that mem swap can be used */
                Node::None => {}
            }
        }

        (prev_level, next_level) = (next_level, prev_level);
    }

    // Last chunck of max_inter_state Nodes is always structured as
    // [CMUX, NONE, NONE, ..., NONE]
    match &last[0] {
        Node::Cmux(in_idx, hi_idx, lo_idx) => {
            module.cmux(
                res,
                &prev_level[*hi_idx],
                &prev_level[*lo_idx],
                &inputs.get_bit(*in_idx).to_backend_ref(),
                &mut scratch_1.borrow(),
            );
        }
        _ => {
            panic!("invalid last node, should be CMUX")
        }
    }
}

fn trivial_one_scalar<BE, R>(infos: &R) -> ScalarZnx<BE::OwnedBuf, i64>
where
    BE: Backend<ZnxWord = i64>,
    R: GLWEInfos,
{
    let base2k = infos.base2k().as_usize();
    let value = if base2k == 1 { -1 } else { 1i64 << (base2k - 2) };
    let mut bytes = vec![0u8; infos.n().as_usize() * size_of::<i64>()];
    bytes[..size_of::<i64>()].copy_from_slice(&value.to_ne_bytes());
    ScalarZnx::from_data(BE::from_host_bytes(&bytes), infos.n().as_usize(), 1)
}

fn glwe_set_trivial_one_with_scalar<BE, R, M>(module: &M, res: &mut R, one: &ScalarZnx<BE::OwnedBuf, i64>)
where
    BE: Backend<ZnxWord = i64>,
    R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
    M: GLWEZero<BE> + VecZnxAddScalarAssignBackend<BE>,
{
    module.glwe_zero(res);
    let limbs = 2usize.div_ceil(res.base2k().as_usize());
    assert!(limbs <= res.size());
    let scalar = <ScalarZnx<BE::OwnedBuf, i64> as ScalarZnxToBackendRef<BE>>::to_backend_ref(one);
    // Writing a small scalar onto zeroed limbs stays within the base2k digit bound, so the
    // CoeffNormalized owner label remains valid after this unnormalized-typed write.
    let mut res = poulpy_core::layouts::glwe_borrowed_carry_view::<BE, _>(res.to_backend_mut());
    for limb in 0..limbs {
        module.vec_znx_add_scalar_assign_backend(res.data_mut(), 0, limb, &scalar, 0);
    }
}

impl<const N: usize> BitCircuit<N> {
    pub const fn new(nodes: [Node; N], max_inter_state: usize) -> Self {
        Self { nodes, max_inter_state }
    }
}
impl<const N: usize> BitCircuitInfo for BitCircuit<N> {
    fn info(&self) -> (&[Node], usize) {
        (self.nodes.as_ref(), self.max_inter_state)
    }
}

/// A single node in a BDD circuit level.
///
/// Nodes are arranged in a flat array divided into chunks of `max_inter_state`
/// entries, one chunk per BDD level.  Each chunk is processed left-to-right
/// during evaluation; the outputs of one level become the inputs of the next.
#[derive(Debug)]
pub enum Node {
    /// `Cmux(selector_bit, hi_index, lo_index)`: evaluates
    /// `res = (hi - lo) * GGSW(selector_bit) + lo`.
    Cmux(usize, usize, usize),
    /// Copy the corresponding entry from the previous level unchanged.
    Copy,
    /// No-op; the corresponding state slot is unused at this level.
    None,
}

impl<BE: Backend<ZnxWord = i64>> Cswap<BE> for Module<BE> where
    Self: GLWEBytesOf<BE>
        + Sized
        + ModuleN
        + GLWEExternalProductInternal<BE>
        + GLWESub<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAddSmallIntoBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSubSmallABackend<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes
{
}

/// Homomorphic conditional swap of two GLWE ciphertexts.
///
/// Given a GGSW ciphertext `s` encrypting a bit `b ∈ {0, 1}`, swaps the
/// contents of `res_a` and `res_b` if `b = 1`, and leaves them unchanged if
/// `b = 0`.  The operation is equivalent to:
///
/// ```text
/// (new_res_a, new_res_b) = if b == 1 { (res_b, res_a) } else { (res_a, res_b) }
/// ```
///
/// but is performed entirely in the ciphertext domain.  Used by
/// `GLWEBlindRetrieval` to implement oblivious array access.
pub trait Cswap<BE: Backend>
where
    Self: GLWEBytesOf<BE>
        + Sized
        + ModuleN
        + GLWEExternalProductInternal<BE>
        + GLWESub<BE>
        + GLWECopy<BE>
        + GLWENormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxBigAddSmallIntoBackend<BE>
        + VecZnxBigBytesOf
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
        + VecZnxBigSubSmallABackend<BE>
        + VecZnxDftAddAssign<BE>
        + VecZnxDftApply<BE>
        + VecZnxDftBytesOf
        + VecZnxDftZero<BE>
        + VecZnxIdftApply<BE>
        + VecZnxNormalizeTmpBytes
        + VmpApplyDftToDft<BE>
        + VmpApplyDftToDftTmpBytes,
{
    /// Returns the minimum scratch-space size in bytes required by [`cswap`][Self::cswap].
    fn cswap_tmp_bytes<R, A, S>(&self, res_a_infos: &R, res_b_infos: &A, s_infos: &S) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        S: GGSWInfos,
    {
        let tmp_c_infos = GLWELayout {
            n: s_infos.n(),
            base2k: s_infos.base2k(),
            k: res_a_infos.k().max(res_b_infos.k()),
            rank: s_infos.rank(),
        };
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(res_a_infos, &tmp_c_infos, s_infos);
        let res_dft: usize = self.bytes_of_vec_znx_dft((s_infos.rank() + 1).into(), output_size);
        let mut tot = res_dft
            + (self.glwe_external_product_internal_tmp_bytes(res_a_infos, &tmp_c_infos, s_infos)
                + self.glwe_bytes_of_from_infos(&tmp_c_infos))
            .max(self.vec_znx_big_normalize_tmp_bytes());

        if res_a_infos.base2k() != s_infos.base2k() {
            tot += self.glwe_bytes_of_from_infos(&GLWELayout {
                n: res_a_infos.n(),
                base2k: s_infos.base2k(),
                k: res_a_infos.k(),
                rank: res_a_infos.rank(),
            });
            tot += self.glwe_bytes_of_from_infos(&GLWELayout {
                n: res_b_infos.n(),
                base2k: s_infos.base2k(),
                k: res_b_infos.k(),
                rank: res_b_infos.rank(),
            });
        } else {
            tot += self.glwe_bytes_of_from_infos(res_a_infos);
            tot += self.glwe_bytes_of_from_infos(res_b_infos);
        }

        tot + self.bytes_of_vec_znx_big(1, output_size)
    }

    fn cswap<'k, A, B>(
        &self,
        res_a: &mut A,
        res_b: &mut B,
        s: &GGSWPreparedBackendRef<'k, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        A: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
        B: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEToBackendRef<BE, State = CoeffNormalized> + GLWEInfos,
        BE: 'k,
    {
        assert_eq!(res_a.base2k(), res_b.base2k());
        assert_eq!(res_a.n(), self.n() as u32);
        assert_eq!(res_b.n(), self.n() as u32);
        assert_eq!(res_a.rank(), s.rank());
        assert_eq!(res_b.rank(), s.rank());

        let scratch = scratch.borrow();
        assert!(
            scratch.available() >= self.cswap_tmp_bytes(res_a, res_b, s),
            "scratch.available(): {} < Cswap::cswap_tmp_bytes: {}",
            scratch.available(),
            self.cswap_tmp_bytes(res_a, res_b, s)
        );

        let res_base2k: usize = res_a.base2k().as_usize();
        let s_base2k: usize = s.base2k().as_usize();
        let cols: usize = (s.rank() + 1).into();
        let tmp_c_infos = GLWELayout {
            n: s.n(),
            base2k: s.base2k(),
            k: res_a.k().max(res_b.k()),
            rank: s.rank(),
        };
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(res_a, &tmp_c_infos, s);

        if res_base2k == s_base2k {
            let (mut a_prev, scratch_1) = scratch.take_glwe_scratch(res_a);
            let (mut b_prev, scratch_2) = scratch_1.take_glwe_scratch(res_b);
            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(self, cols, output_size);
            let (res_big_tmp, scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, output_size);
            self.glwe_copy(&mut a_prev, res_a);
            self.glwe_copy(&mut b_prev, res_b);

            let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
            {
                // (b - a) is carry-producing: normalize it before it enters the DFT domain.
                let (tmp_c, mut scratch_5) = scratch_4.take_glwe_scratch(&tmp_c_infos);
                let mut tmp_c = tmp_c.into_unnormalized();
                self.glwe_sub(&mut tmp_c, res_b, res_a);
                let tmp_c = tmp_c.normalize(self, &mut scratch_5.borrow());
                let (tmp_res_big, mut scratch_6) = scratch_5.take_vec_znx_big_scratch(self, cols, output_size);
                let mut tmp_res_big = tmp_res_big;
                self.glwe_external_product_dft(&mut res_dft, &tmp_c, s, &mut scratch_6.borrow());
                let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
                for col in 0..cols {
                    self.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_6.borrow());
                }
                (res_big, scratch_norm) = (tmp_res_big, scratch_6);
            }

            let mut res_big_tmp = res_big_tmp;
            let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
            let mut res_a_backend = res_a.to_backend_mut();

            for j in 0..cols {
                self.vec_znx_big_add_small_into_backend(
                    &mut res_big_tmp,
                    0,
                    &res_big_ref,
                    j,
                    &vec_znx_backend_ref_from_mut::<BE, _>(a_prev.data()),
                    j,
                );
                let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
                self.vec_znx_big_normalize(
                    res_a_backend.data_mut(),
                    res_base2k,
                    0,
                    j,
                    &res_big_tmp_ref,
                    s_base2k,
                    0,
                    &mut scratch_norm.borrow(),
                );
            }

            let mut res_b_backend = res_b.to_backend_mut();
            for j in 0..cols {
                self.vec_znx_big_sub_small_a_backend(
                    &mut res_big_tmp,
                    0,
                    &vec_znx_backend_ref_from_mut::<BE, _>(b_prev.data()),
                    j,
                    &res_big_ref,
                    j,
                );
                let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
                self.vec_znx_big_normalize(
                    res_b_backend.data_mut(),
                    res_base2k,
                    0,
                    j,
                    &res_big_tmp_ref,
                    s_base2k,
                    0,
                    &mut scratch_norm.borrow(),
                );
            }
        } else {
            let (mut tmp_a, scratch_1) = scratch.take_glwe_scratch(&GLWELayout {
                n: res_a.n(),
                base2k: s.base2k(),
                k: res_a.k(),
                rank: res_a.rank(),
            });
            let (mut tmp_b, mut scratch_2) = scratch_1.take_glwe_scratch(&GLWELayout {
                n: res_b.n(),
                base2k: s.base2k(),
                k: res_b.k(),
                rank: res_b.rank(),
            });

            self.glwe_normalize(&mut tmp_a, res_a, &mut scratch_2);
            self.glwe_normalize(&mut tmp_b, res_b, &mut scratch_2);

            let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(self, cols, output_size);
            let (res_big_tmp, scratch_4) = scratch_3.take_vec_znx_big_scratch(self, 1, output_size);

            let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
            {
                // (b - a) is carry-producing: normalize it before it enters the DFT domain.
                let (tmp_c, mut scratch_5) = scratch_4.take_glwe_scratch(&tmp_c_infos);
                let mut tmp_c = tmp_c.into_unnormalized();
                self.glwe_sub(&mut tmp_c, &tmp_b, &tmp_a);
                let tmp_c = tmp_c.normalize(self, &mut scratch_5.borrow());
                let (tmp_res_big, mut scratch_6) = scratch_5.take_vec_znx_big_scratch(self, cols, output_size);
                let mut tmp_res_big = tmp_res_big;
                self.glwe_external_product_dft(&mut res_dft, &tmp_c, s, &mut scratch_6.borrow());
                let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
                for col in 0..cols {
                    self.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_6.borrow());
                }
                (res_big, scratch_norm) = (tmp_res_big, scratch_6);
            }

            let mut res_big_tmp = res_big_tmp;
            let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
            let mut res_a_backend = res_a.to_backend_mut();

            for j in 0..cols {
                self.vec_znx_big_add_small_into_backend(
                    &mut res_big_tmp,
                    0,
                    &res_big_ref,
                    j,
                    &vec_znx_backend_ref_from_mut::<BE, _>(tmp_a.data()),
                    j,
                );
                let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
                self.vec_znx_big_normalize(
                    res_a_backend.data_mut(),
                    res_base2k,
                    0,
                    j,
                    &res_big_tmp_ref,
                    s_base2k,
                    0,
                    &mut scratch_norm.borrow(),
                );
            }

            let mut res_b_backend = res_b.to_backend_mut();
            for j in 0..cols {
                self.vec_znx_big_sub_small_a_backend(
                    &mut res_big_tmp,
                    0,
                    &vec_znx_backend_ref_from_mut::<BE, _>(tmp_b.data()),
                    j,
                    &res_big_ref,
                    j,
                );
                let res_big_tmp_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big_tmp);
                self.vec_znx_big_normalize(
                    res_b_backend.data_mut(),
                    res_base2k,
                    0,
                    j,
                    &res_big_tmp_ref,
                    s_base2k,
                    0,
                    &mut scratch_norm.borrow(),
                );
            }
        }
    }
}

/// Homomorphic multiplexer (CMux) operation on GLWE ciphertexts.
///
/// Given two GLWE ciphertexts `t` (true branch) and `f` (false branch) and a
/// GGSW ciphertext `s` encrypting a selector bit `b`, computes:
///
/// ```text
/// res = (t - f) · s + f
/// ```
///
/// so that `res` encrypts `t` when `b = 1` and `f` when `b = 0`.  This is the
/// fundamental gate used throughout BDD circuit evaluation.
pub trait Cmux<BE: Backend>
where
    Self: GLWEBytesOf<BE>
        + Sized
        + GLWEExternalProductInternal<BE>
        + GLWECopy<BE>
        + GLWESub<BE>
        + ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + GLWENormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes,
{
    /// Returns the minimum scratch-space size in bytes required by [`cmux`][Self::cmux].
    fn cmux_tmp_bytes<R, A, B>(&self, res_infos: &R, a_infos: &A, selector_infos: &B) -> usize
    where
        Self: GLWEBytesOf<BE>,
        R: GLWEInfos,
        A: GLWEInfos,
        B: GGSWInfos,
    {
        let tmp_infos = GLWELayout {
            n: res_infos.n(),
            base2k: res_infos.base2k(),
            k: res_infos.k().max(a_infos.k()),
            rank: res_infos.rank(),
        };
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(&tmp_infos, &tmp_infos, selector_infos);
        let cols: usize = (selector_infos.rank() + 1).into();
        let res_dft: usize = self.bytes_of_vec_znx_dft(cols, output_size);
        let res_big: usize = self.bytes_of_vec_znx_big(cols, output_size);
        self.glwe_bytes_of_from_infos(res_infos)
            + self
                .glwe_bytes_of_from_infos(a_infos)
                .max(self.glwe_bytes_of_from_infos(&tmp_infos))
            + res_dft
            + res_big
            + self
                .glwe_external_product_internal_tmp_bytes(&tmp_infos, &tmp_infos, selector_infos)
                .max(self.vec_znx_big_normalize_tmp_bytes())
    }

    // res = (t - f) * s + f
    fn cmux<'k, R, T, F>(&self, res: &mut R, t: &T, f: &F, s: &GGSWPreparedBackendRef<'k, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        T: GLWEToBackendRef<BE, State = CoeffNormalized>,
        F: GLWEToBackendRef<BE, State = CoeffNormalized>,
        BE: 'k,
    {
        let f_backend = f.to_backend_ref();

        let scratch = scratch.borrow();
        let res_base2k: usize = res.base2k().into();
        let ggsw_base2k: usize = s.base2k().into();

        let cols: usize = (res.rank() + 1).into();
        // tmp_in = t - f, normalized before it enters the DFT domain.
        let (tmp_in, mut scratch_1) = scratch.take_glwe_scratch(res);
        let mut tmp_in = tmp_in.into_unnormalized();
        self.glwe_sub(&mut tmp_in, t, f);
        let mut tmp_in = tmp_in.normalize(self, &mut scratch_1.borrow());
        let (mut tmp_f, scratch_2) = scratch_1.take_glwe_scratch(&f_backend);
        self.glwe_copy(&mut tmp_f, f);
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(res, res, s);
        let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(self, cols, output_size);
        let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
        {
            let (tmp_res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self, cols, output_size);
            let mut tmp_res_big = tmp_res_big;
            self.glwe_external_product_dft(&mut res_dft, &tmp_in, s, &mut scratch_4.borrow());
            let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
            for col in 0..cols {
                self.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_4.borrow());
            }
            (res_big, scratch_norm) = (tmp_res_big, scratch_4);
        }
        let mut res_big = res_big;
        let tmp_f_ref = vec_znx_backend_ref_from_mut::<BE, _>(tmp_f.data());
        for j in 0..cols {
            self.vec_znx_big_add_small_assign(&mut res_big, j, &tmp_f_ref, j);
            let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
            self.vec_znx_big_normalize(
                tmp_in.data_mut(),
                res_base2k,
                0,
                j,
                &res_big_ref,
                ggsw_base2k,
                j,
                &mut scratch_norm.borrow(),
            );
        }
        self.glwe_copy(res, &tmp_in);
    }

    // res = (a - res) * s + res
    fn cmux_assign_neg<'k, R, A>(
        &self,
        res: &mut R,
        a: &A,
        s: &GGSWPreparedBackendRef<'k, BE>,
        scratch: &mut ScratchArena<'_, BE>,
    ) where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        A: GLWEToBackendRef<BE, State = CoeffNormalized>,
        BE: 'k,
    {
        let a_backend = a.to_backend_ref();

        assert_eq!(res.base2k(), a_backend.base2k());

        let scratch = scratch.borrow();
        let res_base2k: usize = res.base2k().into();
        let ggsw_base2k: usize = s.base2k().into();
        let tmp_infos = GLWELayout {
            n: s.n(),
            base2k: res.base2k(),
            k: res.k().max(a_backend.k()),
            rank: res.rank(),
        };
        // tmp = a - res, normalized before it enters the DFT domain.
        let (tmp, scratch_1) = scratch.take_glwe_scratch(&tmp_infos);
        let mut tmp = tmp.into_unnormalized();
        let (mut res_prev, mut scratch_2) = scratch_1.take_glwe_scratch(res);
        self.glwe_copy(&mut res_prev, res);
        self.glwe_sub(&mut tmp, a, res);
        let mut tmp = tmp.normalize(self, &mut scratch_2.borrow());
        let cols: usize = (res.rank() + 1).into();
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(&tmp_infos, &tmp_infos, s);
        let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(self, cols, output_size);
        let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
        {
            let (tmp_res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self, cols, output_size);
            let mut tmp_res_big = tmp_res_big;
            self.glwe_external_product_dft(&mut res_dft, &tmp, s, &mut scratch_4.borrow());
            let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
            for col in 0..cols {
                self.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_4.borrow());
            }
            (res_big, scratch_norm) = (tmp_res_big, scratch_4);
        }
        let mut res_big = res_big;
        let res_prev_ref = vec_znx_backend_ref_from_mut::<BE, _>(res_prev.data());
        for j in 0..cols {
            self.vec_znx_big_add_small_assign(&mut res_big, j, &res_prev_ref, j);
            let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
            self.vec_znx_big_normalize(
                tmp.data_mut(),
                res_base2k,
                0,
                j,
                &res_big_ref,
                ggsw_base2k,
                j,
                &mut scratch_norm.borrow(),
            );
        }
        self.glwe_copy(res, &tmp);
    }

    // res = (res - a) * s + a
    fn cmux_assign<'k, R, A>(&self, res: &mut R, a: &A, s: &GGSWPreparedBackendRef<'k, BE>, scratch: &mut ScratchArena<'_, BE>)
    where
        R: GLWEToBackendMut<BE, State = CoeffNormalized> + GLWEInfos,
        A: GLWEToBackendRef<BE, State = CoeffNormalized>,
        BE: 'k,
    {
        let a_backend = a.to_backend_ref();
        let scratch = scratch.borrow();
        let res_base2k: usize = res.base2k().into();
        let ggsw_base2k: usize = s.base2k().into();
        let cols: usize = (res.rank() + 1).into();
        // tmp = res - a, normalized before it enters the DFT domain.
        let (tmp, mut scratch_1) = scratch.take_glwe_scratch(res);
        let mut tmp = tmp.into_unnormalized();
        self.glwe_sub(&mut tmp, res, a);
        let mut tmp = tmp.normalize(self, &mut scratch_1.borrow());
        let (mut tmp_a, scratch_2) = scratch_1.take_glwe_scratch(&a_backend);
        self.glwe_copy(&mut tmp_a, a);
        let output_size = glwe_external_product_output_size::<BE, _, _, _>(res, res, s);
        let (mut res_dft, scratch_3) = scratch_2.take_vec_znx_dft_scratch(self, cols, output_size);
        let (res_big, mut scratch_norm): (VecZnxBigViewMut<'_, BE>, _);
        {
            let (tmp_res_big, mut scratch_4) = scratch_3.take_vec_znx_big_scratch(self, cols, output_size);
            let mut tmp_res_big = tmp_res_big;
            self.glwe_external_product_dft(&mut res_dft, &tmp, s, &mut scratch_4.borrow());
            let res_dft_ref = vec_znx_dft_backend_ref_from_mut::<BE>(&res_dft);
            for col in 0..cols {
                self.vec_znx_idft_apply(&mut tmp_res_big, col, &res_dft_ref, col, &mut scratch_4.borrow());
            }
            (res_big, scratch_norm) = (tmp_res_big, scratch_4);
        }
        let mut res_big = res_big;
        let tmp_a_ref = vec_znx_backend_ref_from_mut::<BE, _>(tmp_a.data());
        for j in 0..cols {
            self.vec_znx_big_add_small_assign(&mut res_big, j, &tmp_a_ref, j);
            let res_big_ref = vec_znx_big_backend_ref_from_mut::<BE>(&res_big);
            self.vec_znx_big_normalize(
                tmp.data_mut(),
                res_base2k,
                0,
                j,
                &res_big_ref,
                ggsw_base2k,
                j,
                &mut scratch_norm.borrow(),
            );
        }
        self.glwe_copy(res, &tmp);
    }
}

impl<BE: Backend<ZnxWord = i64>> Cmux<BE> for Module<BE> where
    Self: GLWEBytesOf<BE>
        + Sized
        + GLWEExternalProductInternal<BE>
        + GLWECopy<BE>
        + GLWESub<BE>
        + ModuleN
        + VecZnxBigAddSmallAssign<BE>
        + VecZnxBigBytesOf
        + GLWENormalize<BE>
        + VecZnxNormalizeAssignBackend<BE>
        + VecZnxDftBytesOf
        + VecZnxIdftApply<BE>
        + VecZnxBigNormalize<BE>
        + VecZnxBigNormalizeTmpBytes
{
}
