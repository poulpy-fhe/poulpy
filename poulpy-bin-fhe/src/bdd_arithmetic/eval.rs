pub use crate::api::{Cmux, Cswap, ExecuteBDDCircuit};
use crate::bdd_arithmetic::GetGGSWBit;
use poulpy_core::{
    GLWEBytesOf, GLWECopy, GLWEZero, ScratchArenaTakeCore,
    layouts::{GLWEInfos, GLWEToBackendMut, LWEInfos, ModuleCoreAlloc, prepared::GGSWPreparedToBackendRef},
};
use poulpy_hal::{
    api::VecZnxAddScalarAssign,
    layouts::{Backend, Module, ScalarZnx, ScalarZnxToBackendRef, ScratchArena},
};
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

pub trait BitSize {
    fn bit_size(&self) -> usize;
}

pub(crate) trait BddTrivialOne<BE: Backend> {
    type Prepared: Sync;

    fn prepare_bdd_trivial_one<R: GLWEInfos>(&self, infos: &R) -> Self::Prepared;

    fn set_bdd_trivial_one<R>(&self, res: &mut R, prepared: &Self::Prepared)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos;
}

pub(super) fn bdd_parallel_tmp_bytes<BE: Backend>(threads: usize, output_size: usize, worker_bytes: usize) -> usize {
    let workers = poulpy_hal::execution::worker_count::<BE::TaskExecutor>(threads, output_size);
    workers
        .checked_mul(poulpy_hal::execution::worker_scratch_bytes::<BE>(worker_bytes))
        .expect("BDD scratch size overflow")
}

impl<BE: Backend<ZnxWord = i64>> BddTrivialOne<BE> for Module<BE>
where
    Self: GLWEZero<BE> + VecZnxAddScalarAssign<BE> + ModuleCoreAlloc<OwnedBuf = BE::OwnedBuf, ZnxWord = BE::ZnxWord>,
{
    type Prepared = ScalarZnx<BE::OwnedBuf, i64>;

    fn prepare_bdd_trivial_one<R: GLWEInfos>(&self, infos: &R) -> Self::Prepared {
        trivial_one_scalar::<BE, _>(infos)
    }

    fn set_bdd_trivial_one<R>(&self, res: &mut R, prepared: &Self::Prepared)
    where
        R: GLWEToBackendMut<BE> + GLWEInfos,
    {
        glwe_set_trivial_one_with_scalar(self, res, prepared);
    }
}

pub(crate) fn execute_bdd_circuit_reference<BE, M, C, G, O>(
    module: &M,
    threads: usize,
    out: &mut [O],
    inputs: &G,
    circuit: &C,
    scratch: &mut ScratchArena<'_, BE>,
) where
    BE: Backend<ZnxWord = i64> + 'static,
    M: GLWEBytesOf<BE> + Cmux<BE> + GLWECopy<BE> + GLWEZero<BE> + BddTrivialOne<BE> + Sync,
    G: GetGGSWBit<BE> + BitSize,
    C: GetBitCircuitInfo,
    O: GLWEToBackendMut<BE> + GLWEInfos + Send,
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
            + module
                .cmux_tmp_bytes(&out[0], &out[0], inputs.get_bit(0))
                .max(module.glwe_copy_tmp_bytes(&out[0], &out[0])),
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

fn eval_level<M, G, R, BE>(
    module: &M,
    res: &mut R,
    inputs: &G,
    nodes: &[Node],
    state_size: usize,
    one: &M::Prepared,
    scratch: &mut ScratchArena<'_, BE>,
) where
    M: Cmux<BE> + GLWECopy<BE> + GLWEZero<BE> + BddTrivialOne<BE>,
    BE: Backend<ZnxWord = i64> + 'static,
    G: GetGGSWBit<BE> + BitSize,
    R: GLWEToBackendMut<BE> + GLWEInfos,
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
                Node::Copy => module.glwe_copy(&mut next_level[j], &prev_level[j], &mut scratch_1.borrow()), /* Update BDD circuits to order Cmux -> Copy -> None so that mem swap can be used */
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
    R: GLWEToBackendMut<BE> + GLWEInfos,
    M: GLWEZero<BE> + VecZnxAddScalarAssign<BE>,
{
    module.glwe_zero(res);
    let limbs = 2usize.div_ceil(res.base2k().as_usize());
    assert!(limbs <= res.size());
    let scalar = <ScalarZnx<BE::OwnedBuf, i64> as ScalarZnxToBackendRef<BE>>::to_backend_ref(one);
    let mut res = res.to_backend_mut();
    for limb in 0..limbs {
        module.vec_znx_add_scalar_assign(res.data_mut(), 0, limb, &scalar, 0);
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
