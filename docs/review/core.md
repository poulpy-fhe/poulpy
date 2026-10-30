# poulpy-core review rules

## 1. Layering, OEP and the reference macros

- **Direct Module impls.** Only `GLWEBytesOf`, the host-only noise diagnostics and allocation and byte counts (`ModuleCoreAlloc`, `ModuleCoreCompressedAlloc`, the allocating and sizing methods `<X>PreparedFactory` provides as default bodies) have no OEP method, and `TransferInto` on the layout types.
- **Host-only diagnostics.** Only `GLWENoise`, `GGLWENoise` and `GGSWNoise` read host coefficients, bounding `HostBackend<ZnxWord = i64>` and `HostData*` on the method with no OEP family; other host helpers (`at`, `Display`, codecs) bound `HostData*`, metadata accessors any `D: Data`, and a backend-owned value reaches the host only through `to_host_owned::<BE>()` or `display_host::<BE>()`.
- **Coefficient-word fence.** Code working only for `ZnxWord = i64` says so in a comment at the impl or module naming the forcing HAL item.
- **Reference placement.** A circuit over HAL operations is a `pub trait <ApiTrait>Reference<BE: Backend>` (or its grouped family's, `ConversionReference`) in `poulpy-core/src/reference/<family>`, its `<op>_reference` and `<op>_tmp_bytes_reference` written there or forwarded to same-named `pub fn`s.
- **Derived defaults.** A composition of core operations is an OEP default body calling one `<op>_derived` (argument adaptation aside), with a `<op>_tmp_bytes_derived` query default.
- **Internal hooks.** An OEP method without an API twin is only an internal hook, called by the reference or derived body, recording no metadata, re-deriving no noise-model or precision decision, and documenting that the caller records its metadata.
- **DFT limb layout.** A body forming a partial DFT view or calling `VecZnxDft::zero_at` states `const { assert!(BE::DFT_LIMBS_CONTIGUOUS, ..) }` and documents it, its hook exposed without dispatch where its reference would recurse. Runtime narrowing helpers keep their runtime assert.
- **What may call reference.** The only operation bodies without API or OEP form that a core reference calls directly are `gglwe_product_dft_reference`, `gadget_dft_apply`, `glwe_keyswitch_internal`, `glwe_encrypt_sk_internal` and `glwe_encrypt_pk_internal`, each sized by its own query, a new such helper dispatching its own constituents, gadget sizing and placement helpers living in `poulpy_core::gadget`.
- **Opt-in macros.** A family's opt-in macro, with one `<method>_reference` call per required method, is listed in `oep/mod.rs`'s `pub use crate::{impl_...}` block.
- **Supertraits and bounds.** Only an allocation or sizing dependency (`ModuleCoreAlloc`, `GLWEBytesOf`) is a `Module<Self>: Trait<Self>` bound on the method needing it.
- **Registration lists.** A new core family is registered in the same diff:
  - ring-independent: in `impl_core_reference_full!` (`poulpy-core/src/oep/mod.rs`, directly or through `impl_operations_reference_full!`), `impl_oracle_core!` (`backends/poulpy-cpu-oracle/src/core_impl.rs`) and `impl_cpu_core_defaults!` (`backends/poulpy-cpu-portable/src/backend_defaults.rs`), per type instead of the last where a CPU backend overrides it;
  - standard-ring: in none of these, but on each standard backend type it serves, in or beside `impl_oracle_core_standard!`, and among the omissions the `impl_core_reference_full!` doc lists;
  - `SamplingImpl`: only by each backend's own sampler (`impl_sampling_host!`, the oracle's independent and controlled samplers).
- **API and OEP docs.** A new core API method's doc states the operands read canonical, the output's canonical state and width and any bit-exact identity with an existing operation, and its OEP method's doc names the body an override reproduces bit for bit, for prepared operands under any hint.
- **Public reference surface.** A core reference item is `pub` only as a forwarding target or a circuit helper another crate's reference calls.

## 2. Naming and signatures

- **Trait and family names.** A core API trait is `<Layout><Family>` with upper-case layout acronyms (`GLWE`, `GGLWE`, `GGSW`, `LWE`, `CI`), holding each method's `_assign`, `_add` and `_sub` siblings, and a new operation joins its closest neighbour's grouped OEP family (`GLWETensoringImpl` for products and relinearizations) when its circuit belongs there.
- **Method grammar.** A method's object is the written layout in snake case, then any storage tier and `ci` (`glwe_switching_key_compressed_encrypt_sk`, `glwe_ci_embed`):
  - qualifiers `_add`, `_sub`, `_sub_negate` (into or from `old(res)`), `_prepared` (prepared operands), `_sum` (term slices), `_dft`, `_dft_add` (DFT-domain result), `_at_col`, `_with_mask`, `_smudged`;
  - encryption `<layout>[_compressed]_encrypt[_zero]_{sk|pk}[_<qualifier>]` in `<Layout>[Compressed]Encrypt{Sk|Pk}[<Qualifier>]`, generation from a secret alone `<layout>[_compressed]_generate`;
  - conversions `<target>_from_<source>`, gadget products `_keyswitch`, `_automorphism`, `_external_product`.
- **Destination suffixes.** Only new variants of existing `_into` operations (`glwe_add_into`) keep `_into`.
- **Layout plumbing names.** Core's owned alias sits in `layouts/mod.rs`, entry views are `<X>AtViewRef`/`<X>AtViewMut`, decompression `decompress_<x>(res, other)` on `<X>Decompress`, and carving `take_<x>_scratch(infos)`, `(n, rank)` for secrets and `(module, infos)` for prepared GGLWE and GGSW.
- **Key and secret parameters.** A single switching or automorphism key is `key`, a per-Galois-element source `keys: &H` (`H: GetAutomorphismKey<BE>`), a tensor key at a computed width `tsk: &H` (`H: GetTensorKey<BE>`), a public key `pk`, and secrets `sk`, `sk_in`/`sk_out`, `sk_lwe`/`sk_glwe` or `sk_prepared`; one evaluation key is a concrete prepared backend reference (`&GGLWEPreparedBackendRef<'_, BE>`, `&GGSWPreparedBackendRef<'_, BE>`), while ciphertexts, plaintexts, secrets and public keys stay generic over their view traits.
- **Argument order.** After each operand with its qualifiers (`a, a_size, a_k`) come qualifiers shared by all terms or by infos-less prepared operands (a `galois` slice, their `base2k`), then keys or key sources.
- **Encryption and decryption signatures.** Encryption takes `(res, [pt,] <key parameters>, <sources>, scratch)`, key encryptions' key parameters being `p, sk`, `sk_in, sk_out` or `sk_lwe, sk_glwe`, decryption takes `(res: &mut P, a: &A, sk, [sk_tensor,] scratch)`, and the host noise diagnostics `(res, [res_row, res_col,] pt_want, sk_prepared, scratch)`.
- **Samplers and fills.** A `SamplingImpl` method `<layout>_add_noise` (adds) or `<layout>_fill_<what>` (overwrites) takes `base2k, k`, `res, res_col` (`res` alone for a whole-layout fill), the law (`dist`, `noise`), then `source: &mut Source`, or `seed_xa: [u8; 32]` on `_from_seed` fills.
- **No optional operands.** Each variant of a public core API operation is its own method (`glwe_square_relinearize`), with its own `_assign` form where in place makes sense, never merged through an `Option` operand, which only internal OEP hooks take, or a caller-named type parameter.
- **DFT-domain variants.** `<op>_dft` takes `res: &mut VecZnxDftBackendMut<'_, BE>`, `<op>_dft_add` (product into scratch, then `vec_znx_dft_add_assign`) is a required method with its own query, and their docs state the domain, input radix and how to finish the result.
- **Reference functions.** A free reference function takes `module: &M` first with generics `<BE: Backend, M, ..>` (operand types after, in argument order), `M` bounded by the API traits its body calls, as the `*Reference` impl restates.

## 3. Errors and validation

- **The CoreError variant.** Core returns only `CoreError::GGLWEKeyUse { op, detail }`, `op` naming the public entry point, from the key-source getters, `GGLWELayout::at_k_digit` and `GGLWEPrepared::with_k_digit`, and an operation receiving one panics with the error's text.
- **Shared refusal helpers.** A multi-comparison refusal an override must reproduce is a `pub fn assert_*` (`assert_key_covers`), called first by the reference and by every override, fused path and scheme helper on the same inputs, and a changed refusal changes the helper.

## 4. Scratch

- **Core byte counts.** A carvable layout's `GLWEBytesOf::<x>_bytes_of_from_infos` (prepared: `<x>_prepared_bytes_of[_from_infos]`) pads every separately carved buffer but the last to `SCRATCH_ALIGN` and equals what `take_<x>_scratch` consumes at `infos.size()`.
- **Query arguments.** A core query takes `<param>_infos` (or its file neighbours' bare names, `infos` for one layout), `res_size` for a raw DFT accumulator and a limb count (`a_size`) for an infos-less prepared operand or a single-count body.
- **Shared queries.** A form without its own query (`_assign`, `_add`, `_sub`, `_sub_negate`) is sized by its base query with `res` as input, a shared query is the maximum over its forms, each asserting it on its own operands (`(res, res, key)` for an assign form), an `_add`, `_sub` or `_sub_negate` form needing more getting its own, and a method sized by a query named otherwise says so.
- **Prepare hints.** Single-product temporaries use `PrepareHint::OneShot`, and a byte count or query takes both hints' maximum when its infos do not fix the hint of what it sizes.
- **Scratch asserts.** A core entry check is `assert!(scratch.available() >= q, ..)`, its message static where the query counts a secret or a secret-derived temporary.
- **Per-call collections.** A per-call collection of views or terms (`Vec::with_capacity`) is refilled per column (`clear`, `extend`).

## 5. Degree and ring

- **Operand degrees.** A core body and its query check same-degree operands with `let n = operand_degree(module.n(), &[res.n(), a.n(), ..])` and use that `n`, never `module.n()`, and only degree-mapping operations assert their own relation, the larger degree at most `module.n()`: embed `res.n() == 2 * a.n()`, trace `a.n() == 2 * res.n()`, `sk.n() <= res.n()`, `lwe.n() <= key.n()`, embedded operands.
- **Smaller-degree plaintexts.** A product's plaintext or diagonal operand of power-of-two degree `n_b >= BE::MIN_DEGREE` dividing the ciphertext degree is prepared at its own degree and read through the embedding slot (`take_cnv_pvec_right_scratch(b.n(), ..)`), never resized.
- **Galois elements.** A rotation's element on a degree-`n` operand is `operand_galois_element(module, n, rot)`, keys are looked up under it, and an element stored or used as a key identity is reduced into the operand's ring.
- **Allocation degree.** A factory's parameter form (`glwe_alloc(base2k, k, rank)`) builds its `<X>Layout` at `n: self.ring_degree()` for `<x>_alloc_from_infos`; among allocators only the LWE family takes a degree (`lwe_alloc(n, base2k, k)`); other parameters follow `base2k, dnum, k_digit | k, k_aux, rank | rank_in, rank_out`; prepared allocators, counts and carves check `operand_degree(module.n(), &[infos.n()])` first.
- **Ring scope.** A family on the HAL monomial family or another standard-only family bounds `Backend<Ring = Standard>` on its OEP trait, directly or through that supertrait, as the cross-ring `GLWECIConversionImpl` does.

## 6. Layouts and allocation

- **Module factories.** Outside its module a core layout is allocated only through `ModuleCoreAlloc`, `ModuleCoreCompressedAlloc` and `<X>PreparedFactory`, where a new layout adds its factory methods.
- **Fresh state.** Every core allocation starts with `noise: None` and `plaintext_meta: None`.
- **Layout shape.** A new core layout is one struct `X` (its prepared tier over `<D: Data, BE: Backend>`) with `pub(crate)` fields of the domain newtypes (`Degree`, `Base2K`, `TorusPrecision`, `Rank`, `Dnum`, never `Dsize`), a public-field `<X>Layout` deriving `PartialEq, Eq, Copy, Clone, Debug` with the same Infos traits, an Infos trait providing `<x>_layout()`, and allocators and queries taking `&impl <X>Infos`.
- **Threading a new field.** A field added to a core layout is also threaded in the same diff through:
  - `<x>_layout()` and every forwarding impl (`&T`, `&mut T`, key wrappers, backend newtypes, scratch views, `impl_gglwe_infos_for_inner!`);
  - module, host and compressed allocators, static and backend-routed byte counts, carves;
  - `to_backend_ref`, `to_backend_mut`, row and entry views, `reinterpret`;
  - hand-written `PartialEq` and `Display`, decompress;
  - `From<GGLWELayout>` conversions, asserting its neutral value where the target lacks it.
- **Distinct types for distinct meanings.** A new key or secret variant sharing a storage shape but not a meaning gets its own types or traits, never an existing one's: the tensor secret has its own `GLWESecretTensor*` traits and none of the base secret's, and `GLWECIKey` carries a `CIEmbed` or `CITrace` marker, preparing only into the same marker.
- **Backend-reachable storage.** Every layout's storage and metadata are reachable through public traits on its owned and borrowed forms (`<X>ToBackendRef`/`Mut`, `<X>AtViewRef`/`Mut`, Infos, `GetDistribution`/`Mut`, seed traits) and `data()`/`data_mut()`; an overlapping borrowed form is a newtype (`from_inner`, `into_inner`, `Deref`, `DerefMut` when mutable) implementing each trait itself, pinned by a witness test or doctest.
- **A new key type.** A new key type over `GGLWE` or `GLWESwitchingKey` ships in one diff with standard, compressed and prepared types, a `<X>Layout` whose `From<GGLWELayout>` asserts `stride == 1` (and `k_offset == 0` without offsets), module factories, Infos and views via `impl_gglwe_to_backend_for_field!`, `impl_gglwe_at_view_for_field!` and `impl_glwe_host_at_for_field!`, standard and compressed serialization (wrapper header, then the inner key) with a `test_serialization` case, and decompress, prepare and `TransferInto` carrying its wrapper fields.
- **Shape constraints everywhere.** A key family's structural constraints (public key `rank >= 1`, `k_digit == base2k` and fixed ranks of the LWE keys, automorphism `rank_in == rank_out`) are release-asserted in every host, module, compressed and prepared allocator and size query of the family.
- **Preparation checks and records.** A preparation checks its destination against independent source quantities (degree, `base2k`, limbs against `ceil(k / base2k)`, digit width, row offset, rank) before writing, normalizes a flag-clear GLWE source into counted scratch, then records the source's noise, distribution and wrapper fields on the owner.

## 7. Canonical flag and normalization

- **Flag per operation class.** A GLWE-writing operation sets the owner's flag after its writes (after the nested call when it hands one an output view): true after a final normalization at the destination's `k` or on canonical-by-construction data, false after raw limb arithmetic (`mul_xp_minus_one`, the conjugate-invariant embed, `glwe_encrypt_zero_sk`), and the source's for digit permutations and copies only under `keeps_canonical_digits`.
- **DFT readers normalize their input.** Every operation reading GLWE digits through a DFT, convolution or gadget decomposition (decryption, noise measurement and preparations included) normalizes a flag-clear operand with the public `GLWENormalize`, in place when `&mut` (`glwe_normalize_assign`, unless its radix conversion normalizes) and into an always-carved temporary when `&A`, unless its API asserts a documented canonical precondition.
- **Flagless layouts stay canonical.** GGLWE and GGSW row views, public-key entries, `GLWEPlaintext`, `GLWETensor` and their views report canonical and ignore `set_canonical(false)`, and public-key preparation is one `vmp_prepare`.
- **Canonical at its own k.** A result with `k` below its allocation is normalized at `res.k()`, never `size * base2k`, before its flag is set, after a full-width kernel (`vec_znx_rsh_assign`) or into a temporary with the shift as offset, copied back.
- **Narrowing clears metadata.** `SetK::set_k` clears the canonical flag and noise estimate exactly when it lowers `k`, in every flag- or noise-carrying implementor, wrappers (the CKKS ciphertext) forwarding it to their inner GLWE.

## 8. Operation bodies and products

### 8.1. Operands

- **Live limbs only.** A product, DFT, gadget digit layout or key-switch body addend reads, and its query counts, only live limbs (`size()`, `precision_limbs(data, base2k, k)`, `None` at precision 0), never up to `max_size()` unless the body must touch the whole allocation, which a comment explains.
- **Plaintext width.** A convolution's plaintext is bounded on `IntPolyInfos` and read at `encoded_k()`, never `k()`, `max_size()` or `max_k()`, and every new plaintext-like operand implements `IntPolyInfos`.
- **Lower-rank operands.** A column-wise operation on a lower-rank operand (rank 0: a plaintext on the body) reads only its `rank + 1` columns and writes every other `res` column explicitly.
- **Assign forms.** An operation reading `res` as an operand documents it as read at its current width.
- **Summed operations.** A summed core operation panics on an empty first slice or a later term's layout differing from the first's or `res`'s, naming the index, equals the single operation bit for bit on one term, and documents a term-count-independent query.
- **Prepared operands.** An operation over `glwe_prepare_left`/`glwe_prepare_right` operands takes each prepared limb count (and width where needed) explicitly, checking before writing each prepared layout's degree, columns and limbs against `res` and that count, `ceil(k / base2k)`, and preparing into a caller-owned slot panics unless it holds exactly that many limbs, taken from the source's top limbs.

### 8.2. Gadget products

- **Gadget operands.** Every gadget-product operand, in references and overrides, is built with `gadget_dft_apply` and sized with `gadget_input_size` and `gadget_dft_apply_tmp_bytes`, and a body transforming input limbs straight into the product is gated on `digits_aligned` for the width in use.
- **Digit loop widths.** An interleaved-digit product, override included, writes digit 0 over the full accumulator (`vmp_apply_dft_to_dft` at limb offset 0) and adds later digits with `vmp_apply_dft_to_dft_add` at `gglwe_product_digit_output_size(res.size(), key.size(), dsize, di, product_limbs)`, `product_limbs` coming from `gglwe_product_limbs` or `gadget_product_limbs`.
- **Product windows.** A gadget product, override included, materializes and normalizes the window `gadget_product_output_size` returns, never a hard-coded or worst-case guard count.
- **One-pass gate.** A one-pass product relinearizes `d2` as the convolution leaves it only when `one_pass_keeps_two_step_noise` holds, which past the gadget at a sub-limb offset requires the key's noise metadata and a known secret law, else falling back to `glwe_tensor_relinearize`, bit-exact with the two-step product; an override makes the same decision, and a new metadata-driven gate falls back to the exact path without metadata.

### 8.3. Transforms and normalizations

- **One final normalization.** Products, key-switch contributions, bodies and cross-term sums accumulate in DFT or BIG, each output column normalized once (besides key-switch mask columns, radix conversions and tensor stages), and `cnv_offset_to_limb_offset` gives `cnv_offset_hi` to the convolutions and `cnv_offset_lo` to that normalization's offset.
- **Transform once.** The key-switch body is added untransformed, and a streamed diagonal is prepared once per evaluation into one reused slot.
- **Tensor stages.** Tensor products round at exactly these points, never merged, moved or dropped: each diagonal and pairwise product at `res.size() * base2k` (sub-limb offset applied) before the pairwise subtractions, then each tensor column at `res.k()`; a two-pass sum of tensor products rounds each term at the tensor width.

### 8.4. Linear transformations

- **One schedule.** Baby and giant steps, baby caches, `galois_elements` and the plaintext mirror `Diagonals::evaluate` derive only from `LinearTransformationLayout::index()`/`plan()` and `LinearTransformation::index()`, keys covering only rotations a non-empty giant step uses.
- **Engine paths.** `glwe_eval_giant_steps` (`poulpy-core/src/reference/linear_transformation/giant_steps.rs`) keeps both paths: the lazy one, only when destination, product output and every giant key share one base2k or no giant rotation exists, keeping products, body add, giant automorphisms and the cross-giant sum in DFT until one IDFT and one normalization, and the fallback, taken only where the lazy one cannot run, rotating each normalized giant contribution with the public `glwe_automorphism`.
- **Diagonal periods.** A `Diagonals` entry stores one period (`set` takes any length dividing `slots`), readers index modulo the stored length, slot-wise combinations use `period_lcm`, never `max`, and rotations use `rotate_slots_into`.

## 9. Noise and plaintext metadata

Model: `poulpy-core/docs/core-contracts.md` (Component noise).

- **Who records noise.** Secret-key encryption and key generation record `ComponentNoise::from_secret_at(dist, res.k(), rank)` (scalar LWE: its dimension as `rank`) and public-key encryption `public_key_encryption_noise`; equal-or-wider copies, stored-layout preparation (`*_prepare`), compression, decompression and transfers keep the source's estimate; multiparty fresh collective objects use `ComponentNoise::aggregate`; every other output gets `set_noise(None)`; a public key takes the secret's distribution.
- **Mutable access clears noise.** Every mutable accessor of a noise-carrying layout (`to_backend_mut`, `data_mut`, row and entry views), on owned, `&mut` borrowed and newtype forms alike, clears the owner's `ComponentNoise` before handing out storage and documents it, shared views copying the estimate.
- **Borrow order.** A core body runs:
  1. checks and key lookups through infos and `to_backend_ref()`, scratch last;
  2. metadata reads the result derives from;
  3. the mutable borrow and its writes, in their own block;
  4. owner metadata writes: `res.set_noise(..)`, `res.set_canonical(..)`, `*res.dist_mut() = ..`, seeds through `seed_mut()` and wrapper fields.
- **Component counts.** An estimate has exactly `rank + 1` components (`cols_out` for keys, `n + 1` for scalar LWE), ordered `[body, mask_0, ..]`, set through `checked_noise`, widened only by `with_rank` and never truncated, and a noiseless layout's `set_noise` is a no-op.
- **Composite keys.** A multi-key layout reports an estimate only when every key carries the same one (`common_noise`) and its `set_noise` writes every key, and a whole GGSW records per component the largest estimate across its columns.
- **Model honesty.** An estimate the fresh-noise model cannot compute (law `NONE`, an invalid parameter, a noncentered law with inherited mask error, an overflow) is positive infinity, a ties-to-even rounding of `d` uniform bits has zero mean and second moment `(1 + 2 * 4^-d) / 12`, and a precision change uses `variance_at`/`std_dev_at`.
- **Scheme-owned plaintext metadata.** Core operations take scheme information as explicit integers (`cnv_offset`, `a_k`, `res_k`) and never touch `Option<GLWEPlaintextMeta>`, which every conversion carries, equality compares and key rows, public-key entries and carves leave `None`, an unset field reading as `Scale::Log(0)`, `SlotsKind::Complex`, sparsity 0; a new `SlotsKind` variant keeps the claim order (`join` coarser, `meet` finer, `Complex` always holds) with its byte in `GLWEPlaintextMeta::write_to`/`read_from`, and only the scheme sets quantities on core prepared operands (`PreparedDiagonal::log_scale`, `real_slots`), core checking that summed transformations share them.

## 10. Keys, key sources and gadget decomposition

- **Coverage is max_input_k.** Coverage is `GGLWELayout::max_input_k` (`min(dnum * k_digit, k - k_offset - k_digit)`) alone, asserted by `assert_key_covers("<op>", k, &key)` at the decomposed width (`a.k()`, `res.k()` for `_assign`) before the key is read, source lookups using the width read (baby rotations the source's `k`, giant rotations `res.k()`, relinearizations the tensor width), and a key switching another key's rows covers `k_offset + dnum * k_digit + k_aux`.
- **Key sources.** A key source implements only `lookup_automorphism_key`/`lookup_tensor_key`, never overriding the provided `get_automorphism_key`/`get_tensor_key`.
- **Own-layout switches.** A coarsened key view (`at_k_digit`, `with_k_digit`, `stride > 1`) is read-only, the `From<GGLWELayout>` conversions asserting `stride == 1`, and never allocated, serialized or prepared from, and an own-layout switch is a rounding copy into a smaller key followed by the `_assign` operation, never a row-subset view.
- **Row offsets.** `k_offset` exists only on layouts all of whose consumers support it (others report `k_offset() == 0` and panic on conversion from an offset layout); `glwe_keyswitch`, `glwe_keyswitch_dft` and `glwe_keyswitch_dft_add` honour it, another GGLWE consumer honouring it as `glwe_keyswitch` does, and preparation, decompression and key-producing operations assert equal offsets.
- **Key-level operations.** A key-level operation runs the GLWE operation on each `at_view_mut(row, col)`, leaving rows canonical, asserts the result keeps the source's `k_digit`, `k_offset` and `base2k` with no more rows, checks its own scratch, and writes wrapper metadata after the rows.
- **Digit geometry.** Gadget layouts store `k_digit: TorusPrecision`, never taking `dsize`; widths come from `k_digit_of(base2k, dsize)` or `TorusPrecision::from(base2k)`, whole-limb digits are tested with `digits_aligned` on the width in use, `k_aux >= k_digit`, and precision and limbs come only from `key_k`/`offset_key_k` and `key_size`/`offset_key_size`.

## 11. Encryption, sampling and secrets

- **Sampling calls.** Core draws errors with `vec_znx_add_noise` or `vec_znx_big_add_noise` at `Noise::ENCRYPTION`, ephemerals and secrets with `scalar_znx_fill_distribution` and masks with `fill_glwe_mask_from_source`/`fill_lwe_mask_from_source` (or `_from_seed`), never through sampler kernels.
- **A new sampled quantity.** A new sampled quantity is a required `SamplingImpl` method behind a one-method API trait, landing in the same diff in the `@impl` arm of `impl_sampling_host!`, the oracle's `impl_independent_sampling!`, the hand-written test samplers and the `DifferentSamplingFFT64Portable` probe, and the controlled-sampling adapters (`impl_controlled_sampling!` in `backends/poulpy-cpu-portable/src/test_suite/controlled_sampling.rs`, the oracle's `ControlledSampling` backends), which replay the tested backend's draws through a new `SampleProvider` method and a public draw-copy function beside `scalar_samples` in `test_suite::parity::controlled_sampling`.
- **SamplingImpl contract.** A `SamplingImpl` method takes backend views and the API's `source`, no scratch, validates law (`Noise::validate`), radix, precision and placement before drawing, then draws one seed `source.new_seed()` and values set by that seed, degree, law, precision and radix alone, and writes only its column (scalar fills panic on `NONE` and `ENCAPSULATED`; noise additions write balanced unnormalized digits over `ceil(k / base2k)` limbs, padding below `k` left zero), sampling every law exactly but a Gaussian with `floor(6 sigma) <= 64`, which may use a 128-bit cumulative table.
- **Noise law and floods.** Errors and floods take a `Noise` (dyadic `sigma >= 1` cut at `CUTOFF_FACTOR = 6` sigma, or uniform `bits`), with no precision field or smudging-law type, and secrets a `Distribution`; noise constants (`DEFAULT_SIGMA_XE`) derive from `Noise::ENCRYPTION`, and `sigma`-derived bounds use `Noise::gaussian_parts`, never `f64` products.
- **Distribution tags.** A secret's `Distribution` is set only after the fill, by the samplers or a documented law-changing transform (the conjugate-invariant embedding tags `ENCAPSULATED`), and copied by every other secret operation, tensor secrets included; an `ENCAPSULATED` tag is never serialized, and `ComponentNoise::from_secret_at` records it as `NONE`.
- **A new law variant.** A new `Distribution` variant lands in one diff with its lossless wire tag and payload (`write_to`/`read_from`, `validate_wire`), bitwise `PartialEq`, `fresh_noise_model::base_moments`, `*_secret_fill_*` on `GLWESecretSampling` and `LWESecretSampling`, every `SamplingImpl`'s `scalar_znx_fill_distribution` arm, a `test_suite::sampling` check, every scheme `match` and the docs; a new `Noise` variant updates `variance`, `validate`, `assert_valid_for` and every backend's noise kernels.
- **Refuse unusable secrets first.** Before any draw or write, encryption refuses a `NONE` encrypting secret (a key's output secret, or the source of a secret the call derives, such as a conjugate-invariant embedding), public-key generation a `NONE` or `ENCAPSULATED` secret, and public-key encryption a `NONE` or `ENCAPSULATED` key, a zero ephemeral law (`ZERO`, weight 0, probability `<= 0` or NaN), a key less precise than the output, an entry count other than its rank and a `k_sample` outside the output and key precisions.
- **Public-key encryption.** Public-key encryption draws one independent ephemeral per key entry from the key's law (`source_xu`), computes `Sum_l u_l pk_l` as one `vmp_apply_dft_to_dft` of `vec(u)` over the key's leading `ceil(k_sample / base2k)` limbs, adds in BIG each column's error at `k_sample` (the body's unless the form omits or floods it) and the plaintext, and normalizes each column once at the output's `k`, the derived layer choosing `k_sample` (`public_key_sample_precision`) for the hook `glwe_encrypt_pk_sampled_at`.
- **Draw orders.** Core draws:
  - GLWE secret-key encryption: masks `1..=rank` from `source_xa`, then the body error from `source_xe`;
  - GGLWE entry by entry, input column outer and row inner; GGSW row outer and column inner;
  - public-key encryption: all `rank` ephemerals from `source_xu`, then per-column errors from `source_xe`.
- **Compressed forms.** A compressed encryption mirrors its standard form entry for entry, regenerating each mask with `fill_glwe_mask_from_seed` into a full scratch ciphertext whose body alone is copied out, its query being the standard query plus that ciphertext, normalizing a zero encryption's body, and storing one seed per entry row-major (`seed[row * cols_in + col]`) through `*CompressedSeedMut` after the borrow.
- **Wiping secret budgets.** A core reference reading a secret (encryption, key generation and encryption, decryption, mask inner products, secret preparation and tensoring) wipes its budget, and a derived default of such an operation wipes `<op>_tmp_bytes_derived(..)` once its carves and nested calls end, unless its only use of the arena is one nested secret operation, whose wipe it inherits.

## 12. Serialization, transfer and decompression

- **Serialized core types.** Core serializes only standard and compressed GLWE, LWE, GGLWE, GGSW, public keys and key wrappers, opening with the `PNM3` noise block, a noiseless layout writing `None` and rejecting a tagged stream.
- **Writers check first.** `write_to` validates before its first byte, wrapper headers included, the noise component count (`validate_noise_components`) and every distribution (`validate_wire`).
- **Reader order.** A core `read_from` clears the noise metadata first, checks the HAL shape header (`read_vec_znx_with_shape`/`read_mat_znx_with_shape`) and the component and seed counts against the destination before any allocation they size, and validates degree-dependent metadata against the stream's degree before committing; NaN or negative variance, sparsity past the degree and a zero degree or `base2k` are `InvalidData`, and a failed read may leave the payload partly overwritten but never changes the component count.
- **Format changes.** Core keeps no legacy reader; a new noise wire layout bumps the `PNM` marker, readers rejecting every other, and `test_serialization` covers it.
- **Transfers between backends.** A layout moves only by `TransferInto::transfer_into(&self, dst)` into a caller-allocated destination (or `reinterpret::<To>()` for a shared `OwnedBuf` and word), each impl asserting layout parameters and container shapes (the `move_*` helpers) before moving buffers with `transfer_buf_into`, and copying the metadata fields (flag, noise, plaintext metadata, distribution, seeds, Galois element, degrees, wrapper fields).
- **Decompression restores everything.** Decompression checks degree, digit width, row offset and rank before writing, copies bodies, regenerates masks from the seeds, then sets on the owner the source's noise and plaintext metadata, `set_canonical(true)` for a GLWE, a public key's distribution and the wrapper fields, which a key wrapper's prepare, transfer and serialization also carry.

## 13. Tests and registration

### 13.1. Placement and registration

- **Shared test helpers.** Backend-registered `test_suite` functions reuse the shared helpers (`parity/mod.rs`, `noise/mod.rs`, `test_suite/mod.rs`, `noise::glwe_tensor::assert_canonical`).
- **Suite registration lists.** A new core suite function is registered in the same diff:
  - noise tests in the `tests = { .. }` of `core_backend_test_suite!` (`poulpy-core/src/test_suite/mod.rs`);
  - `test_<op>_parity` or `test_<x>_contract` in every `core_parity_test_suite!` of every backend serving the family, sub-degree and Rayon ones included, and in the oracle's conjugate-invariant suite when that ring has the operation;
  - encryption parity in the body of `core_encryption_parity_test_suite!` (`parity/controlled_sampling.rs`), inside `with_backend_samples`;
  - other signatures as one-line `#[test]`s per backend crate.

  Registration modules start with `core_parity_` or `core_encryption_`.

  Entries drop `test_`, `_parity` and `_contract`; registration modules start with `core_parity_` or `core_encryption_`.

  Entries drop `test_`, `_parity` and `_contract`; registration modules start with `core_parity_` or `core_encryption_`.
- **Suite signatures and bundles.** A noise-suite function is `pub fn test_<op><BE: crate::test_suite::noise::TestBackend>(params: &TestParams, module: &Module<BE>)` with API bounds as `Module<BE>: ..` (a refusal test: the refused call's only), a parity function takes `(params: &TestParams, shapes: &ParityShapes, module_ref: &Module<BR>, module_test: &Module<BT>)` with `BR, BT: ParityBackend` or an extension (`EncryptionParityBackend`), and `TestBackend`, whose supertrait list and blanket-impl `where` clause change together, holds only ring-independent families, a standard-only family being bounded on the function.
- **Suite docs and visibility.** A new suite function's doc states the property it checks.

### 13.2. Degrees, rings and radices

- **Test degrees.** Noise tests compute at `module.n()` (registrations pass `n == size`, `TestParams::size` being the module degree, not a limb count), and parity tests allocate compared operands at `params.n` after `assert!(params.n <= module_ref.n() && params.n <= module_test.n())`, never from a module's degree, except per-backend contract checks comparing nothing across modules.
- **Rings and radices.** A suite function is generic over `BE::Ring`, takes ring expectations from it (`Ring::product_moments(n)`), is left out of conjugate-invariant lists instead of branching when it cannot hold there, and derives radices from `params.base2k` (clamps such as `params.base2k.min(12)` allowed), checking a needed lower bound with a message, holding at every registered radix, a literal radix carrying a why comment.

### 13.3. Fixtures and comparisons

- **Parity fixtures.** Core parity inputs are built once on the reference module (`ref_glwe`, `ref_gglwe`, `fill_by_digit`) and moved by `transfer_into` (raw HAL layouts: `upload_*`/`download_*`), DFT-domain results being read through `vec_znx_idft_normalize`.
- **Comparison macros.** Evaluation outputs are compared with `assert_glwe_eq!` (GLWE) or `assert_eval_eq!`, plain `assert_eq!` serving copies, tagless layouts and tag-cleared host oracles, and a tag-carrying output asserts the expected `noise()`.
- **Each side's own selection.** A pairwise parity helper (`<BR, BT>`) runs both backends through their public API or `BE::` dispatch; only a single-module check of an override against the reference body calls `*_reference` directly.
- **No accidental passes.** Core parity outputs start from an uploaded `ref_glwe`/`ref_gglwe` fixture or poison (distinct 0x55 and 0x77 fills), exact tests from a 0x55 fill or zero where the expected value is non-zero.
- **Checked decrypt helpers.** A core test decrypts or measures the GLWE under test with `glwe_decrypt_checked` or `glwe_noise_checked`, plain `glwe_decrypt` or `glwe_noise` serving only reference phases.

### 13.4. Coverage

- **Lazy and wide twins.** A DFT reader is also tested on `unnormalized_twin` of its operand (as destination for an assign form) and, when it sizes reads from `k`, on `wide_twin` at the narrow query, both matching the canonical output exactly, twins built from the reference fixture on the tested side, or on each side where each backend runs alone (encryption parity).
- **Uncovered input refusal.** A gadget-product test with a GGLWE, automorphism or tensor key asserts with `assert_refuses_uncovered_input` that an input one bit past `gglwe_layout().max_input_k()` panics with "wider than the key's max_input_k".
- **Noise bounds.** A noise test bounds measured `log2(std)` by the analytic model on its own layouts (`GGLWENoiseModel`, `GGSWNoiseModel`, `log2_std_noise_glwe_tensor*`; fresh: `DEFAULT_SIGMA_XE.log2() - k + 0.5`) plus 0.5 or 1 bit, a wider slack being a documented named constant (`TENSOR_NOISE_MARGIN`) and a bound that cannot fail a finding; a test of the model itself is two-sided (pooled second moment within a stated ratio, or `|have - model| <= 1`).
- **Noise sweeps.** A gadget-product noise test sweeps ranks 1 and 2 (3 for tensoring), every whole-limb `dsize` up to `ceil(k_in / key_base2k)`, distinct input, key and output radices unless equal ones are required, and off-limb digit widths, in the sweep or a companion test from 2 bits to three limbs.
- **Parity shapes.** A parity test with a rank or digit axis iterates `shapes.ranks` and `shapes.k_digits(k, base2k, rank)` (whole-limb: `shapes.dsizes(k, base2k)`), and `ParityShapes::default()` stays the full sweep.
- **Boundaries and paths.** Exact and parity sweeps cover precisions 1, `base2k - 1`, `base2k`, `base2k + 1` and a partial last limb for operand and result separately, precision 0 for gadget products at every swept width, shifts by 0, 1, `base2k - 1`, `base2k` and past the operand, rotations and monomial products by negative values, 0, `n` and past `2n`, and extremal digits, carries and ties.
- **Variants against their definition.** Core also checks resident against streamed transforms and a `_zero` encryption's final source states against a zero-plaintext encryption's, and tests every mutation variant from the same uploaded accumulator on both sides, exact expectations coming from an independent host model or tested operations.
- **Summed operation tests.** A summed-operation test checks that one term beside a zero term equals the single operation bit for bit and distinct terms the sum of single operations.
- **Prepare hint tests.** A caller-chosen `PrepareHint` (`glwe_prepare_left`/`glwe_prepare_right`) is tested under both hints (`prepare_hints(i)`) and, in parity, swapped across the backends.
- **Fixes extend the sweeps.** A fix or new layout capability adds its case to the existing noise and parity sweeps of every operation it reaches, its CHANGELOG entry naming those suites.

### 13.5. Refusals

- **Refusal tests.** A refusal test runs the call in `catch_unwind`, matching `test_suite::decryption::panic_message(..)` against the full message, and a refusal placed before the scratch check and any key read runs on a zero-byte arena without keys.

### 13.6. Encryption, sampling, preparation and serialization

- **Encryption parity snapshots.** An encryption or key-generation entry in encryption parity poisons its output with 0xA5, pushes a labeled snapshot of it (bytes, layout and noise metadata; an `encrypt` label asserting fresh-noise metadata) and checks that preparation and decompression keep noise metadata and seeds; host-sampler backends pair directly, an own-sampler backend replaying draws inside one non-nested `with_backend_samples` on the module its closure receives.
- **Sampling contract tests.** A new sampled quantity gets `test_sampling_contract` checks on each backend alone: support and moments within 4 to 7 standard errors, sentinels on untouched columns and limbs, seed determinism and source advance, independence from column, column count and allocation, and full-width noise reconstructed past 128 bits with its low bits and partial-limb padding checked.
- **Preparation contract tests.** A new prepared-layout factory joins `PreparationBounds` and `test_preparation_contract` (a `check!` case), pinning that its alloc and byte-count forms match the allocated length, that preparing on exactly the poisoned `*_prepare_tmp_bytes` keeps the source's noise, and that a destination of another rank panics with noise `None`.
- **Serialization tests.** A new serializable layout or header field joins `test_serialization` with distinct non-`None` noise fixtures, a `test_reader_writer_interface` round trip and, for a header field, a malformed-header case through `test_reused_reader_shape_bound`.
- **Public-key digests.** The `glwe_public_key_rank1_digests` pinned by `test_glwe_public_key_rank1_golden` change only with a stated change of the Gaussian sampler, the public-key generation or encryption circuit (draw order included) or a serialized field.
