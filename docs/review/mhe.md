# poulpy-mhe review rules

Read with `docs/review/REVIEW.md`; this file adds the rules specific to `poulpy-mhe`.

## 1. Protocol grammar and share types

Generic rules: `docs/review/workspace.md` section 12.

- **Protocol traits.** Every protocol is one API trait `<Thing>MHEProtocol<BE>` (OEP `<Thing>MHEProtocolImpl`, reference `<Thing>MHEProtocolReference<BE>`), `<Thing>` naming the finalized core object (`GGSW`) or the ciphertext-transforming protocol (`GLWEEncToShare`, share `GLWEEncToShareShare`).
- **Protocol methods.** A protocol's API and OEP traits hold exactly, in this order, `mhe_<thing>_share_gen_tmp_bytes`, `mhe_<thing>_share_gen`, `mhe_<thing>_share_aggregate`, `mhe_<thing>_share_finalize_tmp_bytes`, `mhe_<thing>_share_finalize`, `<thing>` being `<Thing>` in snake case, and its reference trait keeps that order.
- **One share type per protocol.** Each protocol has its own `<Thing>Share<D, W>` (alias `<Thing>ShareOwned<BE>`) wrapping PAT or core layouts plus its result's metadata, taken instead of a bare PAT.
- **PAT base traits.** Each PAT type has one trait `<PatType>Ops` (OEP `<PatType>Impl`, reference `<PatType>Reference`) with only `<pat>_aggregate_assign(res, a)`, `<pat>_finalize_tmp_bytes()` and `<pat>_finalize(res, pat, scratch)`, operands generic over core's backend-borrow, seed and infos traits.
- **Non-protocol operations.** An operation outside gen, aggregate and finalize has a trait without `MHEProtocol` and `mhe_`-prefixed methods (`GLWEWideSecretPrepare::mhe_glwe_wide_secret_prepare`).
- **Scheme-specific protocols.** A scheme-specific protocol lives under `poulpy-mhe/src/<scheme>/` mirroring the crate layers, with the scheme in every name (`<Scheme><Thing>MHEProtocol`, `<Scheme><Thing>Share`, `mhe_<scheme>_<thing>_share_*`, `impl_mhe_<scheme>_<thing>_reference!`) and share allocators in `MHE<Scheme>ModuleAlloc: MHEModuleAlloc`; scheme-agnostic protocols stay at the crate root.
- **Owned shares, generic results.** Protocol methods take the share as the owned alias (`res: &mut <Thing>ShareOwned<BE>`, `a` and `share` as `&<Thing>ShareOwned<BE>`), other operands and the output generic over core traits.
- **Argument order.** Signatures follow core's encryption order:
  - share generation: `res` (encryption-to-shares: `public`, then `secret`), scalars (`p`), the data operand (`pt`, `mask` or input share `secret`), `sk` (or `sk_in`, `sk_out`), then `u`, `pk`, `pk_out`, `flood: Noise`, `seed: [u8; 32]`, `source_xu`, `source_xe`, `source_xm`, `source_smudge`, `scratch`;
  - aggregation: `(res, a)`;
  - finalization: `(res, [ct], share, [key], scratch)`.
- **Mask-only share generation.** A generation reading only a ciphertext's mask takes `mask: &C` with `C: GLWEMaskToBackendRef<BE> + GLWEInfos`, its trait doc requiring a canonical mask, the body entering only at finalization.
- **Allocators mirror core.** Every PAT and share type has `MHEModuleAlloc` default bodies over core's allocation supertraits, leaving the blanket `impl MHEModuleAlloc<BE> for Module<BE>` empty: `<type_snake>_alloc_from_infos(&self, infos)` at the infos degree and `<type_snake>_alloc(&self, ..)` at the module degree, taking the wrapped core allocator's parameters in core's order minus those the protocol fixes.

## 2. Layers and extension points

Generic rules: `docs/review/workspace.md` section 3.

- **One stem per family.** A family's stem also names `test_suite/<stem>.rs` and `impl_mhe_<stem>_reference!`; a new protocol joins its family (switching and automorphism keys share `evaluation_key`) or adds its stem to every layer, with its PAT and share types in a `layouts/` file named after the type or family.
- **Lists that enumerate protocols.** A new protocol or PAT type also updates:
  - under `poulpy-mhe/src/`: `api/mod.rs` (module doc list, `pub mod`, `pub use`), `delegates/mod.rs`, `oep/mod.rs` and `impl_mhe_reference_full!` (scheme families included), `reference/mod.rs`, `layouts/mod.rs` and the `MHEModuleAlloc` pair, `test_suite/mod.rs` and `mhe_backend_test_suite!`;
  - `poulpy-mhe/docs/mhe-contracts.md`: "Public API organization", the "Operation map" row, the "Replacing an operation" macro list;
  - the lists of `poulpy-mhe/README.md` and the poulpy-mhe line of the root `README.md`.
- **Module exports.** `api` and `layouts` are glob re-exported at the crate root, `oep` and `reference` namespaced, `delegates` private, in scheme modules too.
- **Delegates pass seeds through.** An mhe delegate passes `seed` and every `&mut Source` unchanged, with no child-seed adaptation (`docs/review/workspace.md` section 3).
- **OEP trait supertraits.** An mhe OEP trait's supertraits are exactly the OEP families its default bodies call, else `Backend`; its `# Safety` names the seeds, degrees, Galois element and checks reproduced.
- **Compositions are default bodies.** An aggregation or finalization composing only PAT, hal `MatZnx` or core seeded-mask-fill operations is a default body, its checks or metadata steps in `pub(crate) fn <method>_derived<BE: <families it calls>>(module: &Module<BE>, ..)` of `oep/derived/<stem>.rs` (a pure one-call forward may stay inline) and its query the dispatched family's query.
- **Reference traits.** A family's reference is a public `trait <X>Reference<BE: Backend>` blanket-implemented on `Module<BE>`, with one `<method>_reference` per required OEP method, queries included, and none for a defaulted one, calling the public API on `self` or public core helpers without an API form (`gadget_row_plaintext`); moving a method between default body and reference updates the opt-in macro, its "Operation map" row and the CHANGELOG.
- **Opt-in macros.** Each `oep/<stem>.rs` defines one `#[macro_export]` `impl_mhe_<stem>_reference!($be:ty)`, documented "Selects the reference ..." with the methods left to default bodies, implementing exactly the required methods of the file's OEP traits by forwarding to `<::poulpy_hal::layouts::Module<$be> as $crate::reference::<X>Reference<$be>>::<method>_reference`; `impl_mhe_reference_full!` invokes every family macro.
- **No reference outside the macros.** `layouts`, `api`, `delegates` and `oep` (`oep::derived` included) never import `crate::reference`, without exemption (`docs/review/workspace.md` section 3): shared helpers live in a crate-private module outside `reference`.

## 3. Gen, aggregate and finalize

### 3.1. Constructions

- **Public-key encryption through core.** A protocol encrypts under a public key only through core's public-key encryption API (`glwe_encrypt_pk_at_col` for a chosen column, `glwe_encrypt_pk_smudged` when a flood replaces the body error).
- **Gadget rows and row offsets.** Gadget-row plaintexts use core's `gadget_row_plaintext`, the query covering `vec_znx_normalize_tmp_bytes`; a path without row-offset support refuses one with `assert_no_row_offset` or builds `k_offset = 0` parts.
- **Collective public key.** The public key share holds one seeded body per entry, entry seeds derived from the protocol seed, and records the secret's distribution, which generation and finalization require samplable (not `NONE` or `ENCAPSULATED`), finalization also distinct entry seeds.
- **Collective GGSW.** Each published GGSW share body is a rank-`r` encryption under the whole `s_i` or ephemeral `u_i` over common `r x r` mask matrices per gadget row, `u_i` caller-owned at the GGSW's rank, fresh, independent of `s_i` and reused per key set, the ephemeral key's seed distinct from GGSW seeds.

### 3.2. Aggregation

- **Assign-only raw sum.** Aggregation is only `res += a` from the first share, raw limb addition through the share layout's family (section 7) with no normalization, scratch, query or headroom check.
- **Agreement before the first write.** Before adding, aggregation asserts for every part equal layouts (`*_layout()`), seeds and result metadata (degrees, Galois element, secret distribution, ranks), combining `ComponentNoise` first through the shared asserting helpers (`docs/review/workspace.md` section 4).

### 3.3. Normalization

- **Where mhe normalizes.** No mhe trait exposes normalize; finalization yields a canonical result and generation a canonical share, normalizing only after a flag-clearing last step and to round before a flood (section 5).
- **Normalize only non-canonical parts.** A seeded finalization normalizes only aggregated bodies, regenerating masks from seeds (`fill_glwe_mask_from_seed`, `decompress_gglwe`) under a one-line comment that they are canonical by construction; an unseeded sum (`GGLWEPat`, GLWE-backed shares) normalizes every column.

### 3.4. Metadata

Generic rule: `docs/review/workspace.md` section 9.

- **Provenance tags at generation.** A generation not inheriting a core tag records `ComponentNoise::from_secret_at(<decrypting secret's distribution>, share k, share rank)` per "Share metadata" in `poulpy-mhe/docs/mhe-contracts.md`, that secret being `sk_out` (private key switch), `sk` (encryption-to-shares) or `u` (GGSW message halves).
- **Combining aggregate metadata.** Aggregation combines noise with `aggregate_metadata` for independent errors or `aggregate_common_key_metadata(left, right, n)` for public-key encryptions under the one collective key, untagged if a share is, set on the owner after the backend borrow (`drop(res_be); res.set_noise(md)`).
- **Finalization copies the share.** Finalization copies the share's metadata (degrees, Galois element, distribution, noise) into the result after its last write, keeps the share unchanged and requires its exact layout ("invalid finalization: layouts differ"), except protocols converting by design (collective key switching, encryption-to-shares), which compare fields one by one.
- **Ciphertext and key finalizations.** Adding the aggregate to an input ciphertext leaves noise `None`; consuming a collective key derives the estimate from share and key, first checking `key.dnum() * key.k_digit() >= share.k()` ("invalid finalization: key does not cover the GGSW precision"; `docs/review/workspace.md` section 2) and, when both are tagged, `same_secret`.

### 3.5. Queries

- **Query shapes.** A protocol or PAT query sits right before its operation (`docs/review/workspace.md` section 5); a gen query takes the share's infos first, documenting other layouts, a finalize query nothing when only normalizing and `(res_infos, key_infos)` when key-switching.

## 4. Validation

- **Message grammar.** Every mhe path is secret-carrying (`docs/review/workspace.md` section 4): checks are `assert!(<cond>, "invalid <subject>: <reason>")` with a literal and no format argument, like reader `InvalidData` text, `<subject>` the stage (`share` in generation, `aggregation`, `finalization`, `layout` in a query) or a shared check's operand (`secret`, `public key`, `distribution`; `noise` comes from core's flood check).
- **Checks at the protocol boundary.** A reference or default body checks every precondition (ranks, layouts, precision order `pk.k() >= share.k()` and `secret.k() <= res.k()`, samplable distributions, flood, row offsets, key coverage) with its own mhe message, row offsets excepted (section 3.1), before its first core call, write or draw, even where core rechecks (`docs/review/workspace.md` section 4).
- **The module degree only.** Narrowing `docs/review/workspace.md` section 6, generation, its query and finalization first assert that the share, mask or ciphertext has degree `module.n()` ("invalid share: [mask ]degree differs from the module's", "invalid layout: degree differs from the module's", "invalid finalization: [ciphertext ]degree differs from the module's"); other operands have the share's degree, switching key share secrets any power of two up to the key's.
- **Public-key checks.** Generation under a collective public key first checks its degree, radix and rank against the share, `pk.k() >= share.k()` ("invalid share: public key less precise than the share"), a samplable distribution and `assert_public_key_distribution`, sizes with `glwe_encrypt_pk_tmp_bytes(res_infos, pk_infos)` or its smudged variant, and is tested with a key one limb wider than the share.

## 5. Seeds, floods and the threat model

- **Semi-honest threat model.** Each protocol trait documents what a dishonest input reveals instead of checking it (section 10).
- **Common seed.** Common randomness comes only from `seed: [u8; 32]` (no `source_xa`), further seeds derived by `Source::new(seed)` and `new_seed()` in a trait-documented order (`docs/review/workspace.md` section 10), a seed copied where two parts share masks (each `circ_u[j]` takes `circ_s[j]`'s seeds).
- **Private streams.** Floods draw from `source_smudge`, never `source_xe`, and library code never seeds a private stream from the common seed.
- **Caller-owned randomness obligations.** Caller obligations (`docs/review/workspace.md` section 10), documented (section 8) and never enforced: a fresh seed per result, seeds separated by protocol, session and key, private streams independently seeded per party and purpose and never replayed, the secret distribution. Structural checks within one object stay, such as distinct public key entry seeds (section 3.1).
- **Floods on the share's grid.** A partial-decryption share generation takes `flood: Noise`, checked by `flood.assert_valid_for(share_base2k, share_k)`, then rounds at the share's `k`, adds the flood on that grid (`vec_znx_add_noise(base2k, k, ..)`) and normalizes, or calls `glwe_encrypt_pk_smudged` at the share's `k`, recording the flood variance as body noise, plus the modeled conversion term when the share is narrower than the mask.
- **Torus masks.** A scheme-agnostic protocol's additive mask is torus-uniform at the share's precision (`vec_znx_fill_uniform_source(base2k, k, ..)` from `source_xm`), without `log_bound` or a shift at shares-to-encryption; bounded masks need a scheme-specific protocol (section 1).
- **Scratch wipes.** Share generations carving secret-derived temporaries or composing several core secret operations, and a finalization writing a party's private share (encryption-to-shares), end with `scratch.wipe(tmp_bytes)` of their own `_tmp_bytes_reference(..)` (`docs/review/workspace.md` section 10); a single core secret operation relies on its own wipe.

## 6. Serialization

Generic rules: `docs/review/workspace.md` section 11.

- **Core wire format.** Every PAT and share type implements `ReaderFrom` and `WriterTo` exactly as its wrapped core layout, as its type doc states, wrapper metadata first, little-endian, in core's field order (switching key share: input, output degree `u32`; automorphism key share: `p`, 8 bytes); a share without core counterpart documents its part order (GGSW share: `col0`, `circ_u`, `circ_s`).
- **Share readers.** A share reader returns `InvalidData` ("invalid share: parts do not form one GGSW") unless the bytes form one share of the receiver's allocation (rank, part count, part layouts).

## 7. Layouts and the MatZnx family

- **Matrices through the MatZnx family.** MatZnx-backed share data goes whole through `HalMatZnxImpl` (`docs/review/workspace.md` section 3): one `mat_znx_add_assign` per aggregation, one `mat_znx_normalize_assign` or `mat_znx_normalize` per normalized column, sized by `mat_znx_normalize_tmp_bytes`; per-entry GLWE views only for per-entry work (GGSW cells, per-entry seeds, differing column counts).
- **Share type shape.** A PAT or share type is `pub struct X<D: Data, W: ZnxWord>` with only `pub(crate)` fields and `pub type XOwned<BE> = X<<BE as Backend>::OwnedBuf, <BE as Backend>::ZnxWord>`, forwarding to its core layout:
  - `LWEInfos` and `GLWEInfos`, `GGLWEInfos` or `GGSWInfos`;
  - `*ToBackendRef<BE>`, `*ToBackendMut<BE>`, `set_noise`, and `set_canonical` when GLWE-backed;
  - the seed traits when seeded and its metadata traits (`GLWESwitchingKeyDegrees(Mut)`, `GetGaloisElement`/`SetGaloisElement`, `GetDistribution(Mut)`);
  - `Clone`, `PartialEq`/`Eq`, `Debug`, and `parts()`/`parts_mut()` in serialization order when multi-part.
- **GLWE-backed shares.** A share that is one unseeded core `GLWE` is declared with the crate-private `glwe_share!(/// doc, Name, NameOwned)` macro, never hand-written.
- **No canonical flag on PATs.** PAT types carry no canonical flag (`docs/review/workspace.md` section 8).
- **Seeds on the owner.** Seeds are written on the owner (`seed_mut()` on the part), not a backend view, and `parts_mut()` edits keep part layouts, seed counts and equal `circ_u[j]`/`circ_s[j]` seeds.
- **Host access in layouts only.** Host data is touched only in the layouts' `Debug`, `Display`, `ReaderFrom` and `WriterTo` impls and the test suite (`docs/review/workspace.md` section 2).

## 8. Documentation and CHANGELOG

- **Protocol docs on the trait.** Each `*MHEProtocol` trait and its methods document the protocol's specifics (seed rules on `_share_gen`): what parties compute and the shares encrypt, what the common seed derives, what seed reuse or a dishonest input reveals, its private streams and their seeding rules (section 5), flood grid and required input properties, what shares must agree on and what finalization copies.
- **Contracts doc is cross-protocol.** `poulpy-mhe/docs/mhe-contracts.md` holds only cross-protocol rules, which a protocol diff touches only for such a rule or its rows and lists (section 2).

## 9. Tests and registration

- **Where tests live.** Backend-dependent tests are generic `pub fn test_*<BE>(module: &Module<BE>)` in `poulpy-mhe/src/test_suite/<stem>.rs`, bounded by `BE: HostBackend<OwnedBuf = AlignedBuf, ZnxWord = i64>` plus their API traits (the host-only exception of `docs/review/workspace.md` section 15), constants and helpers in the `pub(crate)` `fixtures.rs`; `#[cfg(test)]` holds only backend-free unit tests of pure helpers, next to the helper.
- **Registration by hand.** Every new test enters `mhe_backend_test_suite!` as `#[test] fn <name without test_>() { $crate::test_suite::<stem>::test_<name>(&Module::<$backend>::new(N)); }`, `N = 256` for correctness and `N = 64` for guards, panics and small checks.
- **Rejection tests.** A rejection test calls `fixtures::assert_panics_with("<full message>", || ..)` once per message, then compares the output with a pre-call clone and each private stream with a fresh `Source` on its seed (`docs/review/workspace.md` section 15).
- **Protocol correctness tests.** A correctness test runs `PARTIES` parties with distinct fixed seeds (`Source::new([base + i; 32])`), covers a wider key, `dsize > 1` or off-limb digits, a narrower share and binary and ternary secrets, and checks decryption under `fixtures::ideal_secret` within a stated bound, metadata of shares, aggregate and result (`assert_collective_metadata`, `assert_fresh_noise`, `assert_noise_components`) and measured noise against the recorded estimate.
- **Overrides bring the parity suite.** An mhe OEP override registers in the mhe parity suite (`mhe_parity_test_suite!`) against an attested backend (`docs/review/workspace.md` section 2), its diff creating the suite if missing.

## 10. Do not report

In addition to `docs/review/workspace.md` section 18:

- Active-adversary findings: missing commitments, proofs, authentication or provenance checks on shares and inputs, and malicious-party attacks.
- A missing headroom check or normalization in aggregation, or a missing normalize operation on a PAT or protocol trait.
- Missing layout or serialization unit tests of a PAT or share wrapper that only forwards to a tested core layout.
- A point repeated across protocol trait docs, an exception to `docs/review/workspace.md` section 13.
