# SHIP bootstrapping

`poulpy-ckks` implements SHIP (Cheon, Hanrot, Kim and Stehlé, [ePrint 2025/784](https://eprint.iacr.org/2025/784)) as a native CKKS operation.
SHIP is a half bootstrap: a one-limb bottom ciphertext, holding real cleartexts with an explicit gap `gamma` in its coefficients, is refreshed into a slots-domain ciphertext at the raised precision — without ModUp or EvalMod.
The bottom ciphertext is switched to a regularly-spaced sparse secret; each support slot then contributes one exponential factor through encrypted selector masks and hoisted base-B mux blind rotations, and a binary product tree over the `h + 1` factors assembles the result.
Its multiplicative depth is `log2(h + 1) + 1` products, independent of the bottom modulus.

## Public API

Invariant-bearing data is exported from `poulpy_ckks::layouts`:

- `ShipPlan` carries the validated instance dimensions: the encoding gap `log_gamma`, the working precision `log_delta_work` of the omega ciphertexts, the residual output budget `log_budget_out`, the sparse Hamming weight `h`, the offset window half-width `w`, the mux base `B`, and the low-digit base `theta` absorbed into the masking.
  The raised precision is `((tree_depth + 2) * log_delta_work + log_budget_out)` rounded up to a limb boundary (`ShipPlan::raised_k`).
- `ShipSecretSpec` samples and validates the regularly-spaced sparse support: the `k`-th nonzero coefficient sits at `k*N/h + delta_k` with `delta_k` in `[-w, w]` and a uniform sign, all indices distinct.
- `ShipKeySet` is validated, unprepared key material; `ShipKeySet::generate` derives the whole bundle from the dense secret and a `ShipSecretSpec`; `ShipKeySet::prepare` returns the backend-ready `ShipKeysPrepared`.
  `ShipKeyParameters` fingerprints the key-defining plan, radix, and complex flag; `ShipKeysLayout` selects the mux gadget digit size and the tensor and conjugation key layouts.

Evaluation is exposed through `poulpy_ckks::api::CKKSShipOps` on `Module<BE>`.
All outputs are caller-allocated:

- `ckks_ship_bootstrap_into`: real cleartexts in the first `N/2` coefficients of the bottom ciphertext.
- `ckks_ship_bootstrap_complex_into`: `Re(mu)` in the first `N/2` coefficients and `Im(mu)` in the last `N/2` (the slots-to-coeffs layout); requires keys generated with `complex`, which doubles the mask material (the mux keys are shared between the halves).
- `ckks_ship_bootstrap_tmp_bytes` reports the caller-arena bound, validating the ciphertext and key layouts along the way.

The only SHIP-specific backend hook is the coefficient encoding: the input-dependent conversion of the bottom ciphertext's public residues into `pt0` and the rotated `pi` plaintexts.
A backend opts in by implementing `poulpy_ckks::oep::CKKSShipCoeffEncodingImpl`; the scheme definition is exported as `poulpy_ckks::encoding::ship_coeff_encodings_host`, and a CPU backend adopts it wholesale with `poulpy-cpu-ref`'s `impl_ckks_ship_coeff_encoding!` macro.
The rest of SHIP composes existing CKKS multiplication, keyswitching, convolution, and DFT APIs.

## Construction outline

```rust,ignore
use poulpy_ckks::{
    api::CKKSShipOps,
    layouts::{ShipKeySet, ShipKeysLayout, ShipPlan, ShipSecretSpec},
};

let plan = ShipPlan::new(log_n, log_gamma, log_delta_work, log_budget_out,
                         sparse_hamming_weight, window, mux_base, theta)?;
let spec = ShipSecretSpec::sample(&plan, &mut source_xs);

let key_set = ShipKeySet::generate::<MyBackend, f64>(
    &module, &host_module, &plan, base2k, &spec, &sk_dense_host,
    &keys_layout, &mut source_xe, &mut source_xa, &mut scratch,
)?;
let keys = key_set.prepare(&module, &mut scratch)?;

let mut output = module.ckks_ciphertext_alloc(base2k, plan.raised_k(base2k.as_usize()).into());
let bytes = CKKSShipOps::<MyBackend, f64>::ckks_ship_bootstrap_tmp_bytes(&module, &output, &input, &keys)?;
// ... allocate scratch of `bytes` ...
CKKSShipOps::<MyBackend, f64>::ckks_ship_bootstrap_into(&module, &mut output, &input, &keys, &mut scratch)?;
```

## Key and ciphertext contract

The bottom input has ring degree `N`, rank one, the key radix, and torus width exactly one limb (`base2k` bits).
Its torus content is `sum_i round(q0 * mu_i / gamma) X^i` — value `mu_i / gamma` at `log_delta = base2k` with no budget.
It decrypts under the application's dense secret; the pipeline switches it to the sparse secret internally through the bundle's encapsulation key.
The output is under the dense secret, at scale `log_delta_work`, with at least `log_budget_out` budget.

The key bundle carries, per support slot, the `4*theta` encrypted selector masks (prepared as left convolution operands; a second `omega_2` set when generated with `complex`) and one hoisted mux key group per mixed-radix digit position.
A mux key is a rank-2 -> rank-1 switching key whose input secret is `(beta * s, beta)` and whose output secret is `s(X^{g^-1})`, so the output automorphism realizes `beta * Rot_rot` under `s`; the input gadget decomposition is shared across a digit position, so the input DFT is computed once per position.
The bundle also carries the dense -> sparse encapsulation key — the only object encrypted under the sparse secret, kept at the bottom modulus — plus the product-tree tensor key and the conjugation key, all under the dense secret.

`ShipKeySet::new` and operation preflight validate layouts, mask and mux counts, gadget dimensions, and the `ShipKeyParameters` fingerprint.
They cannot inspect the secrets: `ShipKeySet::generate` enforces provenance by construction, and custom key material must satisfy the same relationships.

## Security and parameter selection

SHIP relies on a non-standard sparse secret distribution: ternary, weight exactly `h`, with the `k`-th nonzero confined to a `2w + 1` window around `k*N/h`.
The paper estimates its security through the ternary-key attack of May (Crypto'21), whose cost grows as `(4w)^(0.4*h)`, and uses `w = max(175, N/(2h))` in its experiments; treat that estimate as an explicit deployment assumption.
The long-lived application key stays dense — the sparse secret only ever encrypts the bottom modulus `2^(2*base2k)` through the encapsulation key.

The phase embedding imposes the same precision contract as PaCo: residues modulo `q0 = 2^base2k` must be exact in the working scalar, so the `f64` path accepts at most 52 residue bits.
For a cleartext of magnitude `|mu|`, the leading gap-model error is `(2*pi)^2 * |mu|^3 / (6 * gamma^2)` before homomorphic noise; `log_delta_work` should sit far enough above the accumulated CKKS noise (roughly 14 bits over the tree) that the gap model dominates.
The mux chain must cover the window: the mixed-radix bases of `ShipPlan::mux_bases` multiply to at least `(2w + 1) / theta` candidates, and raising `theta` trades mux keyswitches for masking convolutions.
Validate precision and security with application-scale parameters rather than the small test instances.
