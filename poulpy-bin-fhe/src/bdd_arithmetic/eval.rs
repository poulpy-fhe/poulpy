pub use crate::api::{Cmux, Cswap, ExecuteBDDCircuit};

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
