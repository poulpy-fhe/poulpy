//! Backend-resident operands of the negacyclic NTT over `Z/q`.
//!
//! A residue modulo `q` is stored as `limbs` little-endian 64-bit words,
//! `limbs = ceil(bits(q) / 64)`, so `q` may have any size. Both layouts carry only
//! shape metadata and opaque backend storage: a device backend can keep the
//! residues and the twiddle table in device memory.

use anyhow::{Result, ensure};
use dashu_int::{UBig, fast_div::ConstDivisor, ops::BitTest};
use poulpy_hal::layouts::{Backend, Data};

/// Bytes of one 64-bit word of a residue.
const WORD_BYTES: usize = 8;

/// Shape shared by NTT buffers and plans: the transform degree and the residue width.
pub trait GBFVNttInfos {
    /// Base-2 logarithm of the transform degree.
    fn log_n(&self) -> usize;

    /// Number of 64-bit words per residue.
    fn limbs(&self) -> usize;

    /// Transform degree `n`.
    fn n(&self) -> usize {
        1 << self.log_n()
    }
}

/// `n` residues modulo `q` in backend storage, `limbs` little-endian 64-bit words each.
#[repr(C)]
pub struct GBFVNttBuffer<D: Data> {
    /// Opaque host- or device-resident bytes.
    pub data: D,
    log_n: usize,
    limbs: usize,
}

impl<D: Data> GBFVNttBuffer<D> {
    /// Wraps storage of at least [`Self::bytes_of`] bytes.
    pub fn from_data(data: D, log_n: usize, limbs: usize) -> Self {
        Self { data, log_n, limbs }
    }

    /// Bytes of a buffer of `2^log_n` residues of `limbs` words.
    pub fn bytes_of(log_n: usize, limbs: usize) -> usize {
        entries_bytes(
            1usize.checked_shl(log_n as u32).expect("gbfv ntt: degree overflows usize"),
            limbs,
        )
    }

    /// Allocates a zero buffer of `2^log_n` residues of `limbs` words in backend storage.
    pub fn alloc<BE: Backend<OwnedBuf = D>>(log_n: usize, limbs: usize) -> Self {
        Self::from_data(BE::alloc_zeroed_bytes(Self::bytes_of(log_n, limbs)), log_n, limbs)
    }

    /// Uploads host residues into backend storage. Errors when the number of values is
    /// not a power of two or a value needs more than `limbs` words.
    pub fn from_host<BE: Backend<OwnedBuf = D>>(values: &[UBig], limbs: usize) -> Result<Self> {
        ensure!(
            values.len().is_power_of_two(),
            "gbfv ntt buffer: length must be a power of two"
        );
        let log_n: usize = values.len().trailing_zeros() as usize;
        let mut bytes: Vec<u8> = vec![0; Self::bytes_of(log_n, limbs)];
        for (index, value) in values.iter().enumerate() {
            ensure!(
                value.bit_len() <= limbs * 64,
                "gbfv ntt buffer: value exceeds the residue width"
            );
            write_residue(&mut bytes, index, limbs, value);
        }
        Ok(Self::from_data(BE::from_host_bytes(&bytes), log_n, limbs))
    }

    /// Downloads the residues of backend storage to the host.
    pub fn to_host<BE: Backend<OwnedBuf = D>>(&self) -> Vec<UBig> {
        let mut bytes: Vec<u8> = vec![0; Self::bytes_of(self.log_n, self.limbs)];
        BE::copy_to_host(&self.data, &mut bytes);
        (0..self.n()).map(|index| read_residue(&bytes, index, self.limbs)).collect()
    }
}

impl<D: Data> GBFVNttInfos for GBFVNttBuffer<D> {
    fn log_n(&self) -> usize {
        self.log_n
    }

    fn limbs(&self) -> usize {
        self.limbs
    }
}

pub type GBFVNttBufferBackendRef<'a, BE> = GBFVNttBuffer<<BE as Backend>::BufRef<'a>>;
pub type GBFVNttBufferBackendMut<'a, BE> = GBFVNttBuffer<<BE as Backend>::BufMut<'a>>;

/// Converts owned or borrowed NTT buffer storage to a shared backend-native view.
pub trait GBFVNttBufferToBackendRef<BE: Backend>: GBFVNttInfos {
    fn to_backend_ref(&self) -> GBFVNttBufferBackendRef<'_, BE>;
}

/// Converts owned or borrowed NTT buffer storage to a mutable backend-native view.
pub trait GBFVNttBufferToBackendMut<BE: Backend>: GBFVNttBufferToBackendRef<BE> {
    fn to_backend_mut(&mut self) -> GBFVNttBufferBackendMut<'_, BE>;
}

impl<BE: Backend> GBFVNttBufferToBackendRef<BE> for GBFVNttBuffer<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GBFVNttBufferBackendRef<'_, BE> {
        GBFVNttBuffer::from_data(BE::view(&self.data), self.log_n, self.limbs)
    }
}

impl<BE: Backend> GBFVNttBufferToBackendMut<BE> for GBFVNttBuffer<BE::OwnedBuf> {
    fn to_backend_mut(&mut self) -> GBFVNttBufferBackendMut<'_, BE> {
        GBFVNttBuffer::from_data(BE::view_mut(&mut self.data), self.log_n, self.limbs)
    }
}

/// Precomputed table of the negacyclic NTT of degree `n = 2^log_n` modulo an odd `q > 1`,
/// for a root `psi` with `psi^n = -1 mod q`.
///
/// The table holds `2n + 2` residues of `limbs = ceil(bits(q) / 64)` little-endian
/// 64-bit words, in backend storage:
///
/// | entry | residue |
/// |---|---|
/// | `0` | `q` |
/// | `1` | `n^(-1) mod q` |
/// | `2 + i`, `i < n` | `psi^(brv(i)) mod q` |
/// | `2 + n + i`, `i < n` | `psi^(-brv(i)) mod q` |
///
/// `brv` reverses the `log_n` low bits of its argument.
pub struct GBFVNttPlan<D: Data> {
    data: D,
    log_n: usize,
    limbs: usize,
}

impl<D: Data> GBFVNttPlan<D> {
    /// Builds the table of degree `2^log_n` modulo `modulus` for the root `root` on the
    /// host and uploads it to backend storage.
    ///
    /// Errors when `modulus` is not odd and greater than one, or when
    /// `root^n != -1 mod modulus`. For a power-of-two `n` and an odd `modulus`, that
    /// condition makes `psi^2` a principal `n`-th root of unity, so the transform is
    /// invertible.
    pub fn new<BE: Backend<OwnedBuf = D>>(log_n: usize, modulus: &UBig, root: &UBig) -> Result<Self> {
        ensure!(log_n < usize::BITS as usize - 1, "gbfv ntt plan: degree overflows usize");
        ensure!(
            *modulus > UBig::ONE && modulus.bit(0),
            "gbfv ntt plan: modulus must be odd and greater than one"
        );
        let n: usize = 1 << log_n;
        let limbs: usize = modulus.bit_len().div_ceil(64);
        let ring: ConstDivisor = ConstDivisor::new(modulus.clone());
        let psi = ring.reduce(root.clone());
        ensure!(
            psi.pow(&UBig::from(n)) == ring.reduce(modulus - UBig::ONE),
            "gbfv ntt plan: root must satisfy root^n = -1"
        );
        let psi_inv = psi.pow(&UBig::from(2 * n - 1));
        let n_inv = ring
            .reduce(UBig::from(n))
            .inv()
            .expect("an odd modulus makes the degree invertible");

        let mut bytes: Vec<u8> = vec![0; entries_bytes(2 * n + 2, limbs)];
        write_residue(&mut bytes, 0, limbs, modulus);
        write_residue(&mut bytes, 1, limbs, &n_inv.residue());
        let (mut forward, mut inverse) = (ring.reduce(1u8), ring.reduce(1u8));
        for k in 0..n {
            let index: usize = bit_reverse(k, log_n);
            write_residue(&mut bytes, 2 + index, limbs, &forward.residue());
            write_residue(&mut bytes, 2 + n + index, limbs, &inverse.residue());
            forward *= &psi;
            inverse *= &psi_inv;
        }
        Ok(Self {
            data: BE::from_host_bytes(&bytes),
            log_n,
            limbs,
        })
    }

    /// Opaque backend storage of the table.
    pub fn data(&self) -> &D {
        &self.data
    }
}

impl<D: Data> GBFVNttInfos for GBFVNttPlan<D> {
    fn log_n(&self) -> usize {
        self.log_n
    }

    fn limbs(&self) -> usize {
        self.limbs
    }
}

pub type GBFVNttPlanBackendRef<'a, BE> = GBFVNttPlan<<BE as Backend>::BufRef<'a>>;

/// Converts an owned NTT plan to a shared backend-native view.
pub trait GBFVNttPlanToBackendRef<BE: Backend>: GBFVNttInfos {
    fn to_backend_ref(&self) -> GBFVNttPlanBackendRef<'_, BE>;
}

impl<BE: Backend> GBFVNttPlanToBackendRef<BE> for GBFVNttPlan<BE::OwnedBuf> {
    fn to_backend_ref(&self) -> GBFVNttPlanBackendRef<'_, BE> {
        GBFVNttPlan {
            data: BE::view(&self.data),
            log_n: self.log_n,
            limbs: self.limbs,
        }
    }
}

/// `k` with its `log_n` low bits reversed.
pub(crate) fn bit_reverse(k: usize, log_n: usize) -> usize {
    if log_n == 0 {
        0
    } else {
        k.reverse_bits() >> (usize::BITS as usize - log_n)
    }
}

/// Bytes of `count` residues of `limbs` words.
fn entries_bytes(count: usize, limbs: usize) -> usize {
    count
        .checked_mul(limbs)
        .and_then(|words| words.checked_mul(WORD_BYTES))
        .expect("gbfv ntt: size overflows usize")
}

/// Residue `index` of a table of `limbs`-word residues.
pub(crate) fn read_residue(bytes: &[u8], index: usize, limbs: usize) -> UBig {
    let width: usize = limbs * WORD_BYTES;
    UBig::from_le_bytes(&bytes[index * width..(index + 1) * width])
}

/// Writes `value`, of at most `limbs` words, as residue `index` of a table.
pub(crate) fn write_residue(bytes: &mut [u8], index: usize, limbs: usize, value: &UBig) {
    let width: usize = limbs * WORD_BYTES;
    let slot: &mut [u8] = &mut bytes[index * width..(index + 1) * width];
    let le: Box<[u8]> = value.to_le_bytes();
    slot[..le.len()].copy_from_slice(&le);
    slot[le.len()..].fill(0);
}
