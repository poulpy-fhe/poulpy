use crate::layouts::CKKSCiphertextOwned;
use anyhow::{Result, ensure};
use poulpy_core::GLWENormalize;
use poulpy_core::layouts::GetTensorKey;
use poulpy_core::layouts::{
    GGLWEInfos, GLWEInfos, GLWELayout, GLWEPlaintextMeta, GLWEToBackendMut, GLWEToBackendRef, LWEInfos, Rank, split_degree,
};
use poulpy_hal::layouts::{Backend, Data, Module, Ring, ScratchArena, Standard, ZnxWord};

use crate::{
    CKKSCtBounds, CKKSInfos, SetCKKSInfos,
    api::{CKKSMulOps, CKKSPow2Ops, CKKSSubOps},
    checked_mul_ct_log_budget,
    layouts::{CKKSCiphertext, CKKSCiphertextAlloc, CKKSCiphertextViewMut, ScratchArenaTakeCKKSCiphertext},
};

pub use crate::api::{Basis, Parity};
pub use poulpy_core::layouts::{PowerBasis, PowerBasisHelper};

/// CKKS computation of the power basis entries used by BSGS evaluation.
pub trait CKKSPowerBasisInsert<D: Data, W: ZnxWord, Rg: Ring> {
    /// Inserts a caller-provided pre-computed ciphertext power.
    fn insert(&mut self, n: usize, value: CKKSCiphertext<D, W, Rg>) -> Result<()>;
}

/// CKKS computation of the power basis entries used by BSGS evaluation.
pub trait CKKSPowerBasisGen<BE: Backend, Rg: Ring = Standard> {
    /// Recursively computes and stores X^`n` (monomial basis).
    fn gen_power<H>(&mut self, n: usize, module: &Module<BE>, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Module<BE>: CKKSMulOps<BE, Rg> + CKKSCiphertextAlloc<BE>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>;

    /// Recursively computes and stores `T_n(X)` (Chebyshev basis).
    fn gen_power_chebyshev<H>(
        &mut self,
        n: usize,
        module: &Module<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>:
            CKKSPow2Ops<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSSubOps<BE, Rg> + CKKSCiphertextAlloc<BE> + GLWENormalize<BE, Rg>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>;

    /// Pre-computes all powers required to evaluate a polynomial of the given
    /// `degree` using BSGS, for the basis stored in `self`.
    fn populate<H>(
        &mut self,
        degree: usize,
        log_split: usize,
        parity: Parity,
        module: &Module<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSPow2Ops<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSSubOps<BE, Rg> + GLWENormalize<BE, Rg>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>;
}

impl<D: Data, W: ZnxWord, Rg: Ring> CKKSPowerBasisInsert<D, W, Rg> for PowerBasis<CKKSCiphertext<D, W, Rg>> {
    fn insert(&mut self, n: usize, value: CKKSCiphertext<D, W, Rg>) -> Result<()> {
        ensure!(
            n >= 2,
            "PowerBasis::insert: power must be at least 2; power 1 is set at construction"
        );
        ensure!(!self.contains_power(n), "PowerBasis::insert: power {n} is already present");

        let (expected_n, expected_base2k, expected_rank) = {
            let x = self
                .get_stored(1)
                .expect("PowerBasis::new always stores the degree-one power");
            (x.n(), x.base2k(), x.rank())
        };
        ensure!(
            value.n() == expected_n,
            "PowerBasis::insert: power {n} ring degree {} does not match degree-one ring degree {}",
            value.n(),
            expected_n
        );
        ensure!(
            value.base2k() == expected_base2k,
            "PowerBasis::insert: power {n} base2k {} does not match degree-one base2k {}",
            value.base2k(),
            expected_base2k
        );
        ensure!(
            value.rank() == expected_rank,
            "PowerBasis::insert: power {n} rank {} does not match degree-one rank {}",
            value.rank(),
            expected_rank
        );

        self.set_power(n, value);
        Ok(())
    }
}

/// Where a basis keeps the powers it computes: the heap for one a caller keeps, the arena for one that lives through a
/// single evaluation.
pub(crate) trait PowerStorage<'a, BE: Backend, Rg: Ring = Standard>:
    GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos
{
    fn carve(module: &Module<BE>, layout: &GLWELayout, scratch: ScratchArena<'a, BE>) -> (Self, ScratchArena<'a, BE>);
}

impl<'a, BE: Backend, Rg: Ring> PowerStorage<'a, BE, Rg> for CKKSCiphertextOwned<BE, Rg>
where
    Module<BE>: CKKSCiphertextAlloc<BE>,
    CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
{
    fn carve(module: &Module<BE>, layout: &GLWELayout, scratch: ScratchArena<'a, BE>) -> (Self, ScratchArena<'a, BE>) {
        (module.ckks_ciphertext_alloc_from_glwe_infos(layout), scratch)
    }
}

impl<'a, BE: Backend + 'a, Rg: Ring> PowerStorage<'a, BE, Rg> for CKKSCiphertextViewMut<'a, BE, Rg> {
    fn carve(_module: &Module<BE>, layout: &GLWELayout, scratch: ScratchArena<'a, BE>) -> (Self, ScratchArena<'a, BE>) {
        scratch.take_ckks_ciphertext_scratch(layout, GLWEPlaintextMeta::default())
    }
}

/// The layout of a rank-1 power, `k` bits wide, at the degree and radix of `a`.
pub(crate) fn power_layout<A: GLWEInfos>(a: &A, k: usize) -> GLWELayout {
    GLWELayout {
        n: a.n(),
        base2k: a.base2k(),
        k: k.into(),
        rank: Rank(1),
    }
}

fn gen_monomial<'a, BE, A, H, Rg: Ring>(
    basis: &mut PowerBasis<A>,
    n: usize,
    module: &Module<BE>,
    tsk: &H,
    scratch: ScratchArena<'a, BE>,
) -> Result<ScratchArena<'a, BE>>
where
    BE: Backend,
    A: PowerStorage<'a, BE, Rg>,
    Module<BE>: CKKSMulOps<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
{
    ensure!(
        basis.basis() == Basis::Monomial,
        "PowerBasis::gen_power only supports the monomial basis; use gen_power_chebyshev for Chebyshev"
    );

    if basis.contains_power(n) {
        return Ok(scratch);
    }

    ensure!(n >= 2, "gen_power: n={n} < 2; X^1 must be provided at construction");

    let (a, b) = split_degree(n);
    let scratch = gen_monomial(basis, a, module, tsk, scratch)?;
    let scratch = gen_monomial(basis, b, module, tsk, scratch)?;

    let a_val = basis.get_stored(a).expect("gen_power(a) just succeeded");
    let b_val = basis.get_stored(b).expect("gen_power(b) just succeeded");
    let k = mul_ct_k("ckks_mul_into", a_val, b_val)?;
    let (mut result, mut scratch) = A::carve(module, &power_layout(a_val, k), scratch);
    module.ckks_mul_into(&mut result, a_val, b_val, tsk, &mut scratch)?;
    basis.set_power(n, result);
    Ok(scratch)
}

fn gen_chebyshev<'a, BE, A, H, Rg: Ring>(
    basis: &mut PowerBasis<A>,
    n: usize,
    module: &Module<BE>,
    tsk: &H,
    scratch: ScratchArena<'a, BE>,
) -> Result<ScratchArena<'a, BE>>
where
    BE: Backend,
    A: PowerStorage<'a, BE, Rg>,
    Module<BE>: CKKSPow2Ops<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSSubOps<BE, Rg> + GLWENormalize<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
{
    ensure!(
        basis.basis() == Basis::Chebyshev,
        "gen_power_chebyshev requires a Chebyshev PowerBasis"
    );

    if basis.contains_power(n) {
        return Ok(scratch);
    }

    ensure!(n >= 2, "gen_power_chebyshev: n={n} < 2; T_1 must be provided at construction");

    let (a, b) = split_degree(n);
    let scratch = gen_chebyshev(basis, a, module, tsk, scratch)?;
    let scratch = gen_chebyshev(basis, b, module, tsk, scratch)?;

    let c = a.abs_diff(b);
    let scratch = if c != 0 {
        gen_chebyshev(basis, c, module, tsk, scratch)?
    } else {
        scratch
    };

    let a_val = basis.get_stored(a).expect("gen_power_chebyshev(a) just succeeded");
    let b_val = basis.get_stored(b).expect("gen_power_chebyshev(b) just succeeded");
    let k = mul_ct_k(if a == b { "ckks_square_into" } else { "ckks_mul_into" }, a_val, b_val)?;
    let layout = power_layout(a_val, k);
    let (mut result, mut scratch) = A::carve(module, &layout, scratch);

    // `2·T_a·T_b − T_c`, normalized once at the end.
    scratch.scope(|scratch| -> Result<()> {
        let (mut product, mut scratch) = scratch.take_ckks_ciphertext_scratch(&layout, GLWEPlaintextMeta::default());
        if a == b {
            module.ckks_square_into(&mut product, a_val, tsk, &mut scratch)?;
        } else {
            module.ckks_mul_into(&mut product, a_val, b_val, tsk, &mut scratch)?;
        }
        module.ckks_double_into(&mut result, &product, &mut scratch)?;

        if c == 0 {
            module.ckks_sub_one_assign(&mut result, &mut scratch)?;
        } else {
            let c_val = basis.get_stored(c).expect("gen_power_chebyshev(c) just succeeded");
            module.ckks_sub_assign(&mut result, c_val, &mut scratch)?;
        }
        module.glwe_normalize_assign(&mut result, &mut scratch);
        Ok(())
    })?;

    basis.set_power(n, result);
    Ok(scratch)
}

/// The powers a basis computes directly for a polynomial of `degree`: its largest power of two, then its baby steps.
fn populate_targets(degree: usize, log_split: usize, parity: Parity) -> impl Iterator<Item = usize> {
    let log_degree = (usize::BITS - degree.leading_zeros()) as usize;
    let largest_pow2 = 1usize << (log_degree - 1);
    let baby_limit = (1usize << log_split).min(degree + 1);
    let (first, step) = match parity {
        Parity::Even => (4, 2),
        Parity::Odd => (3, 2),
        Parity::Full => (3, 1),
    };
    (largest_pow2 >= 2)
        .then_some(largest_pow2)
        .into_iter()
        .chain((first..baby_limit).step_by(step))
}

/// Largest baby-step range [`populated_powers`] replays: `log_split` 12, a degree around `2^24`.
const MAX_BABY_STEPS: usize = 1 << 12;

/// Powers [`populate_in`] adds to a `basis` of a polynomial of `degree`, replaying [`split_degree`] on the stack: a query
/// reads this, and a step runs its query at entry. Past [`MAX_BABY_STEPS`] it bounds the count by the whole schedule.
pub(crate) fn populated_powers(degree: usize, log_split: usize, parity: Parity, basis: Basis) -> usize {
    struct Seen {
        pow2: u64,
        rest: [u64; MAX_BABY_STEPS / 64],
    }
    fn add(seen: &mut Seen, n: usize, basis: Basis) -> usize {
        if n < 2 {
            return 0;
        }
        let (word, bit) = if n.is_power_of_two() {
            (&mut seen.pow2, n.trailing_zeros())
        } else {
            (&mut seen.rest[n / 64], (n % 64) as u32)
        };
        if *word >> bit & 1 == 1 {
            return 0;
        }
        *word |= 1 << bit;
        let (a, b) = split_degree(n);
        let c = if basis == Basis::Chebyshev { a.abs_diff(b) } else { 0 };
        1 + add(seen, a, basis) + add(seen, b, basis) + add(seen, c, basis)
    }
    let baby_limit = (1usize << log_split).min(degree + 1);
    if baby_limit > MAX_BABY_STEPS {
        return baby_limit + usize::BITS as usize;
    }
    let mut seen = Seen {
        pow2: 0,
        rest: [0; MAX_BABY_STEPS / 64],
    };
    populate_targets(degree, log_split, parity)
        .map(|n| add(&mut seen, n, basis))
        .sum()
}

/// Computes every power of a polynomial of `degree` into `basis`, in the storage of its entries, and returns what is left
/// of `scratch`.
pub(crate) fn populate_in<'a, BE, A, H, Rg: Ring>(
    basis: &mut PowerBasis<A>,
    degree: usize,
    log_split: usize,
    parity: Parity,
    module: &Module<BE>,
    tsk: &H,
    mut scratch: ScratchArena<'a, BE>,
) -> Result<ScratchArena<'a, BE>>
where
    BE: Backend,
    A: PowerStorage<'a, BE, Rg>,
    Module<BE>: CKKSPow2Ops<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSSubOps<BE, Rg> + GLWENormalize<BE, Rg>,
    H: GetTensorKey<BE, Rg>,
{
    ensure!(degree >= 1, "populate: degree must be ≥ 1");

    // Several products read every entry: normalize each once, where it is made.
    if let Some(mut x) = basis.take_power(1) {
        if !x.is_canonical() {
            module.glwe_normalize_assign(&mut x, &mut scratch);
        }
        basis.set_power(1, x);
    }

    for n in populate_targets(degree, log_split, parity) {
        scratch = match basis.basis() {
            Basis::Monomial => gen_monomial(basis, n, module, tsk, scratch)?,
            Basis::Chebyshev => gen_chebyshev(basis, n, module, tsk, scratch)?,
        };
    }
    Ok(scratch)
}

impl<BE: Backend, Rg: Ring> CKKSPowerBasisGen<BE, Rg> for PowerBasis<CKKSCiphertextOwned<BE, Rg>> {
    fn gen_power<H>(&mut self, n: usize, module: &Module<BE>, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Module<BE>: CKKSMulOps<BE, Rg> + CKKSCiphertextAlloc<BE>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>,
    {
        gen_monomial(self, n, module, tsk, scratch.borrow()).map(|_| ())
    }

    fn gen_power_chebyshev<H>(&mut self, n: usize, module: &Module<BE>, tsk: &H, scratch: &mut ScratchArena<'_, BE>) -> Result<()>
    where
        Module<BE>:
            CKKSPow2Ops<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSSubOps<BE, Rg> + CKKSCiphertextAlloc<BE> + GLWENormalize<BE, Rg>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>,
    {
        gen_chebyshev(self, n, module, tsk, scratch.borrow()).map(|_| ())
    }

    fn populate<H>(
        &mut self,
        degree: usize,
        log_split: usize,
        parity: Parity,
        module: &Module<BE>,
        tsk: &H,
        scratch: &mut ScratchArena<'_, BE>,
    ) -> Result<()>
    where
        Module<BE>: CKKSPow2Ops<BE, Rg> + CKKSMulOps<BE, Rg> + CKKSSubOps<BE, Rg> + GLWENormalize<BE, Rg>,
        CKKSCiphertextOwned<BE, Rg>: GLWEToBackendMut<BE, Rg> + GLWEToBackendRef<BE, Rg> + CKKSCtBounds + SetCKKSInfos,
        H: GetTensorKey<BE, Rg>,
    {
        populate_in(self, degree, log_split, parity, module, tsk, scratch.borrow()).map(|_| ())
    }
}

/// Scratch of one product a basis takes for ciphertexts of the layout of `ct`: the product carved beside its
/// multiplication, then the doubling and the subtraction of the Chebyshev recurrence.
pub(crate) fn ckks_power_product_tmp_bytes<BE, C, T, Rg: Ring>(module: &Module<BE>, ct: &C, tsk: &T) -> usize
where
    BE: Backend,
    Module<BE>: CKKSMulOps<BE, Rg> + CKKSPow2Ops<BE, Rg> + CKKSSubOps<BE, Rg> + GLWENormalize<BE, Rg>,
    C: CKKSCtBounds,
    T: GGLWEInfos,
{
    BE::scratch_aligned(BE::vec_znx_bytes_of(ct.n().into(), (ct.rank() + 1).into(), ct.max_size()))
        + module
            .ckks_mul_tmp_bytes(ct, ct, ct, tsk)
            .max(module.ckks_square_tmp_bytes(ct, ct, tsk))
            .max(module.ckks_double_tmp_bytes(ct.max_size()))
            .max(module.ckks_sub_one_tmp_bytes(ct.max_size()))
            .max(module.ckks_sub_tmp_bytes(ct.max_size()))
            .max(module.glwe_normalize_tmp_bytes())
}

/// `k` of a ciphertext square, shared with the input-transform path.
pub(crate) fn square_ct_k<A: GLWEInfos + CKKSInfos>(a: &A) -> Result<usize> {
    mul_ct_k("ckks_square_into", a, a)
}

fn mul_ct_k<A, B>(op: &'static str, a: &A, b: &B) -> Result<usize>
where
    A: GLWEInfos + CKKSInfos,
    B: GLWEInfos + CKKSInfos,
{
    let log_budget = checked_mul_ct_log_budget(op, a.log_budget(), b.log_budget(), a.log_delta(), b.log_delta())?;
    Ok(log_budget + a.log_delta().min(b.log_delta()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    fn add(powers: &mut BTreeSet<usize>, n: usize, basis: Basis) {
        if n < 2 || !powers.insert(n) {
            return;
        }
        let (a, b) = split_degree(n);
        add(powers, a, basis);
        add(powers, b, basis);
        if basis == Basis::Chebyshev {
            add(powers, a.abs_diff(b), basis);
        }
    }

    #[test]
    fn populated_powers_counts_the_replayed_schedule() {
        for degree in 1..2048 {
            for log_split in 1..9 {
                for parity in [Parity::Even, Parity::Odd, Parity::Full] {
                    for basis in [Basis::Monomial, Basis::Chebyshev] {
                        let mut powers = BTreeSet::new();
                        for n in populate_targets(degree, log_split, parity) {
                            add(&mut powers, n, basis);
                        }
                        assert_eq!(
                            populated_powers(degree, log_split, parity, basis),
                            powers.len(),
                            "degree {degree}, log_split {log_split}, {parity:?}, {basis:?}"
                        );
                    }
                }
            }
        }
    }
}
