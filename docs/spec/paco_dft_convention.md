# PaCo DFT convention: where and why the library diverges from the paper

The PaCo bootstrapping (`poulpy-ckks`, split across `layouts/paco/`, `encoding/paco/`, and `default/paco/`) does **not** use the DFT conventions of the paper and its reference implementation ([se-tim/PaCo-Implementation](https://github.com/se-tim/PaCo-Implementation)). It builds the same pipeline on the crate's official DFT factorization instead. This note explains the divergence: what each convention is, why the library's one was chosen, and what the trade-offs cost.

## Why diverge

The library convention is that special-purpose re-implementations exist only where the homomorphic evaluation requires them (a factorization to extract encodable diagonals from). Following the paper's conventions requires a private DFT stack — the `D_{n,2^l}`/`E^{(B)}_{n,2^l}` factor generators and an extended-bit-reversal permutation `Π^{(C/2)}` — living next to, and subtly different from, the official generator (`gen_dft_matrices`) that the standard bootstrapping already uses. The packing of the bootstrapping keys and of the public coefficient encodings is "just a small FFT per block"; there is no reason for it to come from anywhere but the official pipeline. Nothing in PaCo's math depends on the paper's specific transform: any truncated DFT with consistent per-block points works, so the choice of convention is free — and the library exercises that freedom.

## The hybrid encoding, in both conventions

PaCo packs data so that one slot-wise ciphertext multiplication performs `h·k` independent small-ring products (the blind rotation). Both conventions realize the same idea — a **truncated DFT**, stopping after `log 2C` butterfly levels, so each `2C`-wide slot block is an isomorphic image of a small polynomial ring — and differ only in *which* truncated DFT:

| | Paper (se-tim reference) | This library (official generator) |
|---|---|---|
| Block ring | `ℂ[Z]/(Z^{2C} − α_v)`, `α_v` **varies** per block (twiddles `2^l·5^{bitrev(i)} mod 4n`) | `ℂ[Z]/(Z^{2C} − i)`, **uniform** for every block and chunk |
| Slot order | extended-bit-reversed by `C/2` (`Π^{(C/2)}` applied after packing) | natural coefficient order (bit-reversed *point* order, invisible to the pipeline) |
| Factor generator | dedicated `D`/`E` matrices (`E = Π·D·Π` to stay tridiagonal in the permuted layout) | `gen_dft_matrices_blockwise` (same butterflies as `gen_dft_matrices`, diagonals canonicalized modulo the `n`-tile) |
| Packing (cleartext) | `D`-chain layers + extended bit-reversal | the encoder's own FFT per `2C` block (the test suite's `ReferenceEncoder::unpack_reim_coeffs`) + bit-reversal gather (`P_br·U_enc`) |
| Partial CoeffToSlot | inverse `E^{(C/2)}` chain | blockwise `2C`-point Encode chain (`bit_reversed = true`) |
| PartialC2s output | coefficient `b'_{ext_bit_rev(i, C/2)}` at slot `i` | coefficient `b'_i` at slot `i` (**natural**) |
| PackedPairs layout | `m̃_{br(j)} + i·m̃_{br(j)+C/2}` | `m̃_j + i·m̃_{j+C/2}` (**natural**) |
| SlotToCoeff′ | forward `E` chain (consumes bit-reversed input for free) | `C/2`-point Decode chain (`bit_reversed = false`, encoder convention) with the input-side bit-reversal **folded into the first factor** |

### Conservation of the bit-reversal

An FFT butterfly network cannot have natural ordering on both sides without a permutation somewhere; no convention avoids the bit-reversal, it only decides where to pay it. The paper distributes it across the whole pipeline (permuted packing, `Π`-conjugated matrices, permuted masks and read-off); the library concentrates it at the single boundary where PaCo meets the encoder's coefficient convention — the first SlotToCoeff′ factor, the smallest matrix in the system (`C/2 × C/2`). Everything between the blind rotation and the SlotToCoeff′ input is then in natural order, and every stage formula (μ mask `i mod 2C < C`, ψ-pairing `acc += conj(rot_C(acc))`, the η routing `i mod C < C/2`, the folds, the read-off) is the plain, unpermuted one — including the paper's Eq. 11 packing relation, which becomes literally `slot v·2C + i = b'_{λ_v, i}`.

A bit-reversal can end up in exactly three places: cancelled against a cleartext boundary (free — the `br` readout in `pack_chunk` cancels the input permutation of the bit-reversed-input Encode factorization, which is why the partial-C2S chain carries no permutation diagonals at all), left in the data ordering (free until an index-structured operation touches it), or merged into a factor matrix (diagonals). The library's default convention (`PaCoSlotOrder::Natural`) does the first at the input and the third at the output; the opt-in `PaCoSlotOrder::BitRevLow` convention (see the shipped-alternative section below) does the first at **both** boundaries by conjugating the C2S chain, leaving no permutation diagonals anywhere.

### The rejected convention: full-width bit-reversed mid-pipeline

Dropping the `br` readout from the packing pushes the permutation into the data ordering instead: the partial C2S switches to the natural-input flavor (identical butterfly counts, measured `[3,3,3,3]`/`[7,7]` in both flavors), the middle of the pipeline runs in a within-block bit-reversed layout, and SlotToCoeff′ becomes the bit-reversed-input factorization with no merged permutation. Whether this wins is decided by which operations commute with the relabeling. The invariant: the relabeling permutes the low `log 2C` bits of the slot index, so everything indexed by strides that are multiples of the block size — the chunk fold `Tr_{N/2→n}`, the class product `Pr_{n→2C}` — commutes exactly, and every point-wise operation (blind-rotation products, `−conj`, diagonal masks) commutes trivially. What does not commute is a **top-bit fold**: an operation pairing index `i` with `i + half-domain`, which bit-reversal maps to "flip the bottom bit" — a 2-diagonal XOR pattern where a 1-diagonal rotation used to be. The pipeline has exactly two such folds: the ψ-pairing `i ↔ i+C` (the deferred `Y^C = −1` reduction) and the re/im pair-packing `j ↔ j+C/2`.

Measured on the generated factors: the pair-packing lives in the first StC′ factor, which the composed pack already holds at or near the dense ceiling (`16/16` diagonals at `C = 8`, `24/32` at `C = 16`), so both the merged bit-reversal *and* its XOR-pattern alternative are absorbed for ≈ nothing there. The ψ-pairing lives in the sparse conjugation-augmented `B` factor, where the XOR pattern doubles the diagonals (`3 → 6` at `g0 = 1`, `7 → 14` at `g0 = 2`, every C). Net: at any parameter set one would run (small C), the bit-reversed-middle convention pays `+3` to `+7` diagonals at the fused C2S level to save ≈ 0 on the StC′ side. The natural-order middle is kept as the default: top-bit folds are free next to dense factors and cost 2× next to sparse ones, so the layout convention is chosen to park them where the dense factors are.

The sharp variable in this analysis is the **reversal width**. This rejected convention leaves the packing's full `log 2C`-bit readout reversal in the data, and `log 2C` *contains* bit `log C` — which is why the ψ-pairing fold `i ↔ i+C` degenerates into the XOR pattern. Reversing only the low `log(C/2)` bits — the paper's `Π^{(C/2)}`, and exactly the reversal the StC′ input actually needs — leaves bits `log(C/2)` and `log C` fixed, so *every* fold and mask in the pipeline commutes with it exactly. That narrower relabeling is not a rejected convention but a shipped one:

### The shipped alternative: `PaCoSlotOrder::BitRevLow`

`PaCoPlan::with_slot_order(PaCoSlotOrder::BitRevLow)` relabels the mid-pipeline (partial-C2S output → StC′ input) by the tiled permutation `P` that reverses the low `log(C/2)` bits of the slot index (`ext_bitrev_low`). Three code sites change, all behind the flag:

- `paco_c2s_factors` conjugates each factor by `P` (`conjugate_by_low_bitrev`): the chain telescopes to `Π·chain·Π`, whose output is the relabeled mid-pipeline. Butterfly support structure keeps the conjugated factors as sparse as the originals (a level-`ℓ` layer's offsets `{0, ±2^ℓ}` map to `{0, ±2^{log(C/2)−1−ℓ}}` below the boundary and are fixed points above it — the paper's Prop. 1, pinned by `conjugate_by_low_bitrev_prop1_offsets`).
- The trailing input-side `Π` is absorbed into the cleartext packing gather in `pack_chunk_into_with` (`br(P(k))` instead of `br(k)`), free, and automatically consistent between `bskGen` and the coefficient encodings.
- `paco_stc_factors` uses the raw bit-reversed-input Decode factorization with **no** `bit_reverse_columns` fold — the `P`-ordered data *is* its input order (`P` restricted to the `C/2`-periodic packed layout is the full input bit-reversal).

Everything between the relabeling points is untouched code, because it commutes with `P` exactly: the fold strides (multiples of `2C`), the ψ-pairing offset `C`, the μ/η mask strata (bits `log C` and `log(C/2)`), the pack factor's `{0, C/2, C, 3C/2}` diagonals (all fixed points of the conjugation, pinned by `conjugate_by_low_bitrev_invariants`), and every pointwise op. The oracle (`seq_paco_reference`) stays convention-agnostic; gates comparing production mid-pipeline data relabel its `z_7` by `P`. Final output, budget, levels, and noise are identical across conventions — `BitRevLow` is purely a layout relabel.

Measured, pinned costs (`bitrevlow_stc_chain_unfolded`, `conjugate_by_low_bitrev_counts`, `galois_element_counts_by_convention`, `stc_fold_cost_pins`):

| Configuration | Natural | BitRevLow |
|---|---|---|
| StC′ diagonals, `C = 32`, schedule `[1,1,1,1,1]` | `[4, 14, 3, 3, 2]` = 26 | `[4, 3, 3, 3, 2]` = 15 |
| StC′ first factor, `C = 32`, schedule `[2,2,1]` | 56 | 12 |
| StC′ fully merged (`[5]`) | 64 | 64 |
| Automorphism keys, `C = 32, g1 = 1` plan | 22 | 20 |
| C2S factor counts | baseline | equal, except sparse merges spanning the `log(C/2)` boundary (`+2` at `C = 16`, schedule `[2,2,1]`) |

#### Convention selection across C (measured sweep)

The pinned table above fixes the `C = 32` values in tests; the broader sweep below was measured with the same generators (`stc_fold_cost()` / `galois_elements()` / `conjugate_by_low_bitrev` on the c2s factors) across the range of `C`. StC′ totals count chain diagonals ≈ plaintext multiplies in the last linear stage:

| C | StC′ schedule | Natural | BitRevLow | saving | automorphism keys Nat → BRL |
|---|---|---|---|---|---|
| 8 | `[1,1,1]` | 10 | 9 | 1 | 15 → 15 |
| 8 | `[2,1]` | 18 | 14 | 4 | 18 → 18 |
| 16 | `[1,1,1,1]` | 15 | 12 | 3 | 18 → 18 |
| 16 | `[2,2]` | 28 | 16 | 12 | 27 → 21 |
| 32 | `[1,1,1,1,1]` | 26 | 15 | 11 | 22 → 20 |
| 32 | `[2,2,1]` | 65 | 21 | 44 | 43 → 25 |
| 64 | `[1×6]` | 33 | 18 | 15 | 29 → 23 |
| 64 | `[2,2,2]` | 83 | 23 | 60 | 62 → 28 |
| 128 | `[1×7]` | 60 | 21 | 39 | 44 → 25 |
| 128 | `[2,2,2,1]` | 184 | 28 | 156 | 127 → 32 |
| any | fully merged | tie | tie | 0 | tie |

The C2S-side penalty (the only place `BitRevLow` loses) fires only when a merge group straddles the `log(C/2)` boundary — a pure alignment property of the schedule, zero at `g0 = 1` for every `C`:

| c2s grouping | penalty by `C = 8, 16, 32, 64, 128` |
|---|---|
| `g0 = 1` | 0, 0, 0, 0, 0 |
| `g0 = 2` | 0, +2, 0, +2, 0 |
| `g0 = 3` | 0, 0, +6, +6, 0 |

Net extremes: the worst case for `BitRevLow` is small `C` with a fully merged StC′ and a boundary-straddling C2S schedule — net **+2 diagonals** (~2 extra plaintext multiplies out of a ~50-multiply pipeline), avoidable by aligning the C2S groups with the boundary. The best case is large `C` with two-level StC′ grouping — at `C = 128`, `[2,2,2,1]`: **156 fewer plaintext multiplies** (184 → 28, a 6.5× reduction of the stage) and **95 fewer automorphism keys** (127 → 32, ~4× less of that key material).

The regime where this matters is not exotic: `C` is capped at `B = N/(4h)`, so large `C` is precisely the full-width, throughput-oriented configurations — at the paper's own parameters (`N = 2^15, h = 64`) the cap is `C = 128`, and sparse-secret encapsulation (which relaxes the security constraint on `h`) pushes `B` higher still. The `C = 8` shape of the paper's benchmarks, where the fold costs 0–1 diagonals and `Natural` is the right call, is the small end of the range, not the typical operating point.

Guidance: keep `Natural` (the default) for `C ≤ 8`, fully merged StC′ schedules, or wherever a persisted key bundle must stay valid. Choose `BitRevLow` for `C ≥ 16` with light StC′ grouping (`g1 ∈ {1, 2}`); `PaCoPlan::stc_fold_cost()` returns both StC′ totals and `galois_elements().len()` the key delta for the decision. The convention changes the BSGS diagonal offsets and hence the Galois key *values* — it must stay stable per persisted key bundle. For `C ≤ 4` the permutation is the identity and the conventions coincide. If a chosen C2S schedule merges across the `log(C/2)` boundary and pays the penalty, realign the groups (splitting at the boundary changes the diagonal totals, so compare — at `C = 16` the aligned `[3,2]` costs 22 c2s diagonals versus `[2,2,1]`'s 17 + 2, and eating the +2 is the cheaper move).

## Cost model

Let `n = 2hC`, `k = B/C = N/(4hC)` chunks, and let `g0`, `g1`
be the maximum unit-group widths passed to the balanced uniform schedulers.

**Cleartext (keygen σ_t, per-bootstrap β_t).** Both conventions: `log 2C` sparse tridiagonal layers per chunk, `O(N log C)` flops per packed vector — identical asymptotics and constants (the extended-bit-reversal pass simply disappears).

**Homomorphic levels.** Identical by construction between the two conventions.
For uniform schedules the exact count is `1 (blind rotation) +
⌈(log C + 2)/g0⌉ (partial C2S) + log h (product) + ⌈log C/g1⌉
(StC′)`. The C2S unit list contains `log(2C) = log C + 1` butterfly
layers plus the schedulable ψ/μ unit; the StC′ list contains the schedulable
pack/η unit plus `log C − 1` butterfly layers. A caller-supplied schedule
uses its actual factor counts instead. The masks add no level beyond those
explicit units: ψ/μ becomes the last conjugation-augmented C2S factor, while
pack/η becomes the first StC′ factor (and may be merged with adjacent
butterflies).

**Factor diagonals / keyswitches.** A factor merging `m` butterfly layers has at most `3^m` diagonals in either convention (both generators merge the same tridiagonal layers). The one structural difference: the first StC′ factor absorbs the bit-reversal permutation, growing from ≤ 3 to at most `C/2` diagonals. Per extra diagonal the BSGS evaluation costs one plaintext multiply and at most one baby-step rotation; the automorphism-key set grows only if the new rotation indexes are not already present.

**Key material.** Four bsk ciphertexts, one tensor key, one automorphism key per Galois element — structurally identical; only the element *values* differ (everything derives from `plan.galois_elements()`, so key generation is oblivious to the convention).

**Numerics.** The two factorizations are different, equally valid floating-point paths through the same transform; their noise is the same up to fractions of a bit and far below any parameter budget.

## At real parameters

Paper-scale instance `N = 2^15, h = 64, C = 8, q = 2^52, g0 = 2, g1 = 1`
(13 levels, `n = 1024`, `k = 16` chunks). The balanced schedules are
`c2s = [2, 2, 1]` and `stc = [1, 1, 1]`; the singleton units are the
standalone ψ/μ and pack/η factors:

| Metric | Paper convention | Library convention | Δ |
|---|---|---|---|
| Partial C2S schedule | 2 seven-diagonal butterfly factors + 1 ψ/μ pair | same | none |
| StC′ schedule (dim 4) | pack + 2 Decode factors | same | none |
| DFT-dependent StC′ diagonals | 4 (pack), 3, 2 | 4 (pack), **4**, 2 | **+1 diagonal** |
| Galois elements (automorphism keys) | plan-derived set | same measured set | none |
| Levels / consumed bits | 13 | 13 | none |

The *entire* structural overhead of the library convention at these parameters is one extra plaintext diagonal on a `4 × 4` factor — one additional plaintext multiply in the last linear stage, zero additional keys, zero additional levels.

The folded permutation's cost depends on how the StC′ units are grouped (the
schedule trades levels for diagonals, paper §3.5). Merging `g` butterfly
layers gives a factor with at most `min(3^g, C/2)` diagonals; composing the
permutation into a factor can only push it up to the dense ceiling `C/2`. The
closer the first Decode factor already is to dense, the less the permutation
costs, reaching zero when the full StC′ chain is merged. For `C = 8` the
current schedulable-unit convention gives:

| StC′ unit schedule | folded vs plain diagonals | extra keys |
|---|---|---|
| `[3]` (fully merged) | [4] vs [4] | 0 |
| `[2, 1]` (pack merged with one butterfly) | [4, 2] vs [3, 2] | 0 |
| `[1, 1, 1]` (standalone pack) | [4, 4, 2] vs [4, 3, 2] | 0 |

For a `C/2`-point transform the fully merged or two-level schedules are normally the useful choices; at `C = 8` the permutation costs zero or one diagonal and no additional key, and `Natural` is the right convention. The alternative that removes even the residual diagonal — conjugating the factor chain by `Π^{(C/2)}` so the permutation telescopes into the cleartext packing — is shipped as `PaCoSlotOrder::BitRevLow` (see the shipped-alternative section above) and pays off at large `C` with light StC′ grouping.

## Benefits

- **One DFT convention in the crate**: PaCo's factors come from the same generator, diagonal type, encoder, BSGS engine and key plumbing as the standard bootstrapping; the packing and its homomorphic inverse are structurally the same object, so they cannot drift apart.
- **No private DFT machinery**: the paper's `D`/`E` generators do not exist in the codebase. The `BitRevLow` convention realizes the paper's `Π^{(C/2)}` conjugation as a generic post-transform (`conjugate_by_low_bitrev`) on the official generator's factors, not as a parallel factor stack.
- **Natural-order semantics** everywhere in the pipeline: every mask, rotation, packing relation and read-off is the unpermuted formula, which simplifies the documentation, the annotated example, and debugging.
- **Block-uniform `α = i`**: the paper's per-block `α_v` was never load-bearing (products stay below degree `2C`), and uniformity removes a whole class of "which block am I in" reasoning.
- **Reusable facility**: `gen_dft_matrices_blockwise` is general — any future scheme that needs a small transform replicated across a tile (CinS-style hybrid encodings, batched small convolutions à la NeuJeans) uses the same entry point.

## Costs

- **A wider first StC′ factor** when the StC′ chain is not fully merged: +1 plaintext diagonal at two-level configurations at `C = 8`, zero when merged to one level (measured table above), but growing toward the dense ceiling `C/2` at large `C` with light grouping (26 vs 15 total StC′ diagonals at `C = 32, g1 = 1`) — which is what `PaCoSlotOrder::BitRevLow` removes.
- **Divergence from the reference implementation**: packed vectors are not bit-for-bit comparable with se-tim/PaCo-Implementation, so validation cannot lean on it. Correctness gates instead include internal structural identities (round-trip, Vandermonde pins, and the Eq. 9/Eq. 11 relations), direct decryption identities, cleartext-model checks, and homomorphic recovery against the original message coefficients.
