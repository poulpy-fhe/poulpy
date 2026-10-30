# poulpy-mhe review rules

## 1. Protocol grammar and share types

- **Protocol traits.** Every protocol is one API trait and one OEP trait holding exactly its share generation, aggregation and finalization and the generation and finalization queries.
- **One share type per protocol.** Each protocol has its own share type; share types hold only published data, a party's private share (`secret` in the sharing conversions) staying a caller-owned core layout.
- **PAT base traits.** Each PAT type has one trait with only an in-place aggregation `(res, a)`, a finalization query `()` and a finalization `(res, pat, scratch)`, operands generic over core's backend-borrow, seed and infos traits.
- **Scheme-specific protocols.** A scheme-specific protocol lives under `poulpy-mhe/src/<scheme>/` mirroring the crate layers; scheme-agnostic protocols stay at the crate root.
- **Owned shares, generic results.** Protocol methods take the share as the owned alias (`res: &mut MHE<Thing>ShareOwned<BE>`, `a` and `share` as `&MHE<Thing>ShareOwned<BE>`), other operands and the output generic over core traits.
- **Argument order.** Signatures follow core's encryption order:
  - share generation: `res` (encryption-to-shares: `public`, then `secret`), key parameters (`p`), the data operand (`pt`, `mask` or input share `secret`), `sk` (or `sk_in`, `sk_out`), then `u`, `pk`, `pk_out`, `flood: Noise`, `seed: [u8; 32]`, `source_xu`, `source_xe`, `source_xm`, `source_smudge`, `scratch`;
  - finalization: `(res, [ct], share, [key], scratch)`.
- **Mask-only share generation.** A generation reading only a ciphertext's mask takes `mask: &C` with `C: GLWEMaskToBackendRef<BE> + GLWEInfos`, its trait doc requiring a canonical mask and its body asserting the mask's canonical flag, the body entering only at finalization.
- **Allocators mirror core.** Every PAT and share type has its own allocation trait of default bodies over core's allocation traits, its blanket `impl` for `Module<BE>` empty, the `_alloc_from_infos(&self, infos)` form asserting the infos degree is the module's and the `_alloc(&self, ..)` form at the module degree, taking the wrapped core allocator's parameters in core's order minus those the protocol fixes.

## 2. Layers and extension points

- **One stem per family.** A family's stem also names `test_suite/<stem>.rs`; a new protocol joins its family (switching and automorphism keys share `evaluation_key`) or adds its stem to every layer.
- **Lists that enumerate protocols.** A new protocol or PAT type also updates:
  - under `poulpy-mhe/src/`: `api/mod.rs` (module doc list, private `mod`, `pub use`), `delegates/mod.rs`, `oep/mod.rs` and `impl_mhe_reference_full!` (scheme families included), `reference/mod.rs`, `layouts/mod.rs` and the type's `MHE<Type>Alloc` trait, `test_suite/mod.rs` and `mhe_backend_test_suite!`;
  - `poulpy-mhe/docs/mhe-contracts.md`: "Public API organization", the "Operation map" row, the "Replacing an operation" macro list;
  - the lists of `poulpy-mhe/README.md` and the poulpy-mhe line of the root `README.md`.
- **Module exports.** `api` and `layouts` are glob re-exported at the crate root, `oep` and `reference` namespaced, in scheme modules too.
- **OEP method docs.** Each OEP method's doc names the seeds, degrees, Galois element and checks it reproduces.
- **Compositions are default bodies.** An aggregation or finalization composing only mhe operations (PAT and protocol methods) is a default body, its checks or metadata steps in `<method>_derived<BE: <families it calls>>(module: &Module<BE>, ..)`, and its query the dispatched family's query.
- **Reference traits.** A family's reference has one `<method>_reference` per required OEP method, queries included, and none for a defaulted one.
- **Opt-in macros.** Each opt-in documents the reference it selects with the methods left to default bodies and forwards to `<::poulpy_hal::layouts::Module<$be> as $crate::reference::<X>Reference<$be>>::<method>_reference`.
- **Crate-private shared helpers.** Shared helpers live in a crate-private module outside `reference`.

## 3. Gen, aggregate and finalize

### 3.1. Constructions

- **Public-key encryption through core.** A protocol encrypts under a public key only through core's public-key encryption API (`glwe_encrypt_pk_at_col` for a chosen column, `glwe_encrypt_pk_smudged` when a flood replaces the body error).
- **Gadget rows and row offsets.** Gadget-row paths cover `vec_znx_normalize_tmp_bytes` in their query; a path without row-offset support refuses one with `assert_no_row_offset` or builds `k_offset = 0` parts.
- **Collective public key.** The public key share holds one seeded body per entry, entry seeds derived from the protocol seed, and records the secret's distribution, which generation and finalization require samplable (not `NONE` or `ENCAPSULATED`), finalization also distinct entry seeds.
- **Collective GGSW.** Each published GGSW share body is a rank-`r` encryption under the whole `s_i` or ephemeral `u_i` over common `r x r` mask matrices per gadget row, `u_i` caller-owned at the GGSW's rank, fresh, independent of `s_i` and reused per key set, the ephemeral key's seed distinct from GGSW seeds.

### 3.2. Aggregation

- **Assign-only raw sum.** Aggregation is only `res += a` from the first share, raw limb addition through the share layout's family with no normalization, scratch or query.
- **Agreement before the first write.** Before adding, aggregation asserts for every part equal layouts (`*_layout()`), seeds and result metadata (degrees, Galois element, secret distribution, ranks), combining `ComponentNoise` first through the shared asserting helpers.

### 3.3. Normalization

- **Where mhe normalizes.** No mhe trait exposes normalize: generation yields a canonical share (normalizing after a flag-clearing last step), aggregation an unnormalized sum and finalization the canonical result of normalizing that sum.
- **Normalize only non-canonical parts.** A seeded finalization normalizes only aggregated bodies, regenerating masks from seeds (`fill_glwe_mask_from_seed`, `decompress_gglwe`); an unseeded sum (`GGLWEPat`, GLWE-backed shares) normalizes every column.

### 3.4. Metadata

- **Provenance tags at generation.** A generation not inheriting a core tag records `ComponentNoise::from_secret_at(<decrypting secret's distribution>, share k, share rank)` per "Share metadata" in `poulpy-mhe/docs/mhe-contracts.md`, that secret being `sk_out` (private key switch), `sk` (encryption-to-shares) or `u` (GGSW message halves).
- **Combining aggregate metadata.** Aggregation combines noise with `aggregate_metadata` for independent errors or `aggregate_common_key_metadata(left, right, n)` for public-key encryptions under the one collective key, untagged if a share is.
- **Finalization copies the share.** Finalization copies the share's metadata (degrees, Galois element, distribution, noise) into the result and requires its exact layout ("invalid finalization: layouts differ"), except protocols converting by design (collective key switching, encryption-to-shares), which compare fields one by one.
- **Ciphertext and key finalizations.** Adding the aggregate to an input ciphertext leaves noise `None`; consuming a collective key derives the estimate from share and key, first checking that `share.k()` is within the key's `max_input_k` and, when both are tagged, `same_secret`.

### 3.5. Queries

- **Query shapes.** A gen query takes the share's infos first, documenting other layouts, a finalize query nothing when only normalizing and `(res_infos, key_infos)` when key-switching.

## 4. Validation

- **Static messages.** Every mhe path is secret-carrying: checks are `assert!` with a literal and no format argument.
- **Checks at the protocol boundary.** A reference or default body checks every precondition (ranks, layouts, precision order `pk.k() >= share.k()` and `secret.k() <= res.k()`, samplable distributions, flood, row offsets, key coverage) with its own mhe message, flood, row offsets and key coverage excepted (core's helpers), before its first core call.
- **The module degree only.** Generation, its query and finalization first assert that the share, mask or ciphertext has degree `module.n()` ("invalid share: [mask ]degree differs from the module's", "invalid layout: degree differs from the module's", "invalid finalization: [ciphertext ]degree differs from the module's"); other operands have the share's degree, switching key share secrets any power of two up to the key's.
- **Public-key checks.** Generation under a collective public key first checks its degree, radix and rank against the share, `pk.k() >= share.k()` ("invalid share: public key less precise than the share"), a samplable distribution and `assert_public_key_distribution`, sizes with `glwe_encrypt_pk_tmp_bytes(res_infos, pk_infos)` or its smudged variant, and is tested with a key one limb wider than the share.

## 5. Seeds, floods and the threat model

- **Semi-honest threat model.** Each protocol trait documents what a dishonest input reveals instead of checking it.
- **Common seed.** Common randomness comes only from `seed: [u8; 32]` (no `source_xa`), a seed copied where two parts share masks (each `circ_u[j]` takes `circ_s[j]`'s seeds).
- **Private streams.** Floods draw from `source_smudge`, never `source_xe`, and library code never seeds a private stream from the common seed.
- **Caller-owned randomness obligations.** Caller obligations, documented and never enforced: a fresh seed per result, seeds separated by protocol, session and key, private streams independently seeded per party and purpose and never replayed, the secret distribution. Structural checks within one object stay, such as distinct public key entry seeds.
- **Floods on the share's grid.** A partial-decryption share generation takes `flood: Noise`, adding it to the share or calling `glwe_encrypt_pk_smudged` at the share's `k`, and records the flood variance as body noise, plus the modeled conversion term when the share is narrower than the mask.
- **Torus masks.** A scheme-agnostic protocol's additive mask is torus-uniform at the share's precision (`vec_znx_fill_uniform_source(base2k, k, ..)` from `source_xm`), without `log_bound` or a shift at shares-to-encryption; bounded masks need a scheme-specific protocol.
- **Scratch wipes.** A finalization writing a party's private share (encryption-to-shares) ends with `scratch.wipe(tmp_bytes)` of its own `_tmp_bytes_reference(..)`.

## 6. Serialization

- **Core wire format.** Every PAT and share type implements `ReaderFrom` and `WriterTo` exactly as its wrapped core layout, as its type doc states, wrapper metadata first, in core's field order (switching key share: input, output degree `u32`; automorphism key share: `p`, 8 bytes); a share without core counterpart documents its part order (GGSW share: `col0`, `circ_u`, `circ_s`).
- **Share readers.** A share reader returns `InvalidData` unless the bytes form one share of the receiver's allocation (rank, part count, part layouts).

## 7. Layouts and the MatZnx family

- **Matrices through the MatZnx family.** MatZnx-backed share data goes whole: one `mat_znx_add_assign` per aggregation, one `mat_znx_normalize_assign` or `mat_znx_normalize` per normalized column, sized by `mat_znx_normalize_tmp_bytes`; per-entry GLWE views only for per-entry work (GGSW cells, per-entry seeds, differing column counts).
- **Share type shape.** A PAT or share type is `pub struct X<D: Data, W: ZnxWord>` with `pub type XOwned<BE> = X<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>`, forwarding to its core layout:
  - `LWEInfos` and `GLWEInfos`, `GGLWEInfos` or `GGSWInfos`;
  - `*ToBackendRef<BE>`, `*ToBackendMut<BE>`, `set_noise`, and `set_canonical` when GLWE-backed;
  - the seed traits when seeded and its metadata traits (`GLWESwitchingKeyDegrees(Mut)`, `GetGaloisElement`/`SetGaloisElement`, `GetDistribution(Mut)`);
  - `Clone`, and `parts()`/`parts_mut()` in serialization order when multi-part.
- **GLWE-backed shares.** A share that is one unseeded core `GLWE` is declared with the crate-private `glwe_share!(/// doc, Name, NameOwned)` macro, never hand-written.
- **No canonical flag on PATs.** PAT types carry no canonical flag.
- **Seeds on the owner.** Seeds are written through `seed_mut()` on the part, and `parts_mut()` edits keep part layouts, seed counts and equal `circ_u[j]`/`circ_s[j]` seeds.
- **Host access in layouts only.** Host data is touched only in the layouts' `Debug`, `Display`, `ReaderFrom` and `WriterTo` impls and the test suite.

## 8. Documentation and CHANGELOG

- **Protocol docs on the trait.** Each `MHE*Protocol` trait and its methods document the protocol's specifics (seed rules on `_share_gen`): what parties compute and the shares encrypt, what the common seed derives, what seed reuse or a dishonest input reveals, its private streams and their seeding rules, flood grid and required input properties, what shares must agree on and what finalization copies.
- **Contracts doc is cross-protocol.** `poulpy-mhe/docs/mhe-contracts.md` holds only cross-protocol rules, which a protocol diff touches only for such a rule or its rows and lists.

## 9. Tests and registration

- **Where tests live.** Backend-dependent tests are generic `pub fn test_*<BE>(module: &Module<BE>)` in `poulpy-mhe/src/test_suite/<stem>.rs`, bounded by `BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>` plus their API traits, constants and helpers in the `pub(crate)` `fixtures.rs`; `#[cfg(test)]` holds only backend-free unit tests of pure helpers.
- **Registration by hand.** Every new test enters `mhe_backend_test_suite!` as `#[test] fn <name without test_>() { $crate::test_suite::<stem>::test_<name>(&Module::<$backend>::new(N)); }`, `N = 256` for correctness and `N = 64` for guards, panics and small checks.
- **Rejection tests.** A rejection test pins each full message once, then compares each private stream with a fresh `Source` on its seed.
- **Protocol correctness tests.** A correctness test runs `PARTIES` parties with distinct fixed seeds, covers a wider key, `dsize > 1` or off-limb digits, a narrower share and binary and ternary secrets, and checks decryption under `fixtures::ideal_secret` within a stated bound, metadata of shares, aggregate and result (`assert_collective_metadata`, `assert_fresh_noise`, `assert_noise_components`) and measured noise against the recorded estimate.
- **Parity on every type.** Every backend type registering mhe runs the mhe parity suite (`mhe_parity_test_suite!`) against its attested link (`docs/review/backends.md` section 3.5).
