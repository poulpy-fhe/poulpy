# Poulpy API request: finish resolved prepared-key execution

## Goal

Complete the level-aware GGLWE key API so every key product is resolved for
its exact input precision before sizing or execution. A regular key resolves
with its stored `dsize`; a thin `GGLWEKeyUse` wrapper may request another
compatible `dsize` without copying or projecting the key.

The ordinary scalar and fused OEPs must consume the resulting key bound. Do
not introduce a separate selected-product OEP, public row view, or projected
key object.

This request is generic. It is not specific to linear transformations and does
not introduce a configurable modulus cap.

## 1. Add a common resolved key bound

Add one resolved bound that keeps the complete physical prepared key together
with the logical product geometry for one exact input precision. Its semantics
must include:

```text
exact input precision
complete physical GGLWEPreparedBackendRef / VmpPMatBackendRef
physical rows and physical limb pitch
logical dnum, dsize, k_aux, and work size
first physical row
physical row step
```

For a positive-precision use, logical row `i` addresses:

```text
physical_row(i) = first_physical_row + i * physical_row_step
```

The physical limb size remains the storage pitch. The logical work size is the
readable prefix within each selected physical polynomial. Changing only the
logical `size` on a dense `VmpPMatBackendRef` is incorrect because it changes
the inferred row and column pitches.

Provide one binding operation conceptually equivalent to:

```rust
key.bind_for(input_k)
key.with_dsize(effective_dsize).bind_for(input_k)
```

The default implementation resolves `self.effective_dsize()`. Consequently a
raw key uses `self.dsize()`, while the existing thin wrapper selects another
decomposition. Resolve before converting the wrapper into the current concrete
prepared backend reference, because that conversion currently erases the
effective `dsize`.

Binding must distinguish:

- a valid positive-precision bound;
- a valid zero-precision use, handled in core with no key read or OEP launch;
- an invalid or unrealizable use, returned as an error.

Exact naming and whether the zero result is represented by an enum or another
explicit result are implementation choices. Do not represent zero precision
as an invalid key.

The bound must retain `input_k` and validate that the product input has
`ceil(input_k / base2k)` limbs. The same resolver must support layout-only
sizing and prepared-key execution. This may be one generic resolved-bound type or layout and prepared forms that
share the same geometry. Scratch queries and execution must not independently
recompute different bounds.

## 2. Make the ordinary OEPs consume the bound

Change the regular product OEP contract to consume the resolved prepared-key
bound instead of an unresolved dense `GGLWEPreparedBackendRef`. Remove the need
for `gglwe_product_dft_selected` as a separate execution path.

The generic implementation may materialize a non-dense bound for a backend
that cannot address it directly. CUDA must be able to consume the physical
prepared buffer directly using the bound row origin, row step, physical
pitch, and logical limb prefix.

The ordinary scalar OEP receives the bound. Existing public composites and
backend-private pair, quad, or cross-rotation batching must propagate one
already-resolved bound per lane; this request does not add new fused OEP entry
points. Compatible lanes may batch. Incompatible lanes must split or use the
ordinary scalar OEP. A backend must not duplicate the Poulpy resolver
arithmetic or infer selection from physical metadata.

The ordinary path must also use a bound when effective and physical `dsize`
are equal. Native-`dsize` use can still have fewer active rows and a shorter
logical prefix than the stored key.

The current `dsize == 1` specialization drops directly to dense
`VmpApplyDftToDft`. Use that specialization only for a full dense bound, make
it bound-aware, or fall through to the unified product path. It must not ignore
native low-precision row or prefix trimming.

## 3. Size from the resolved logical layout

Every product, accumulated-product, scratch, and linear-transformation sizing
helper must consume the same resolved bound and use its logical `dnum`,
`dsize`, `k_aux`, and work size. Derive `product_terms` from that logical
bound plus the operation-local `term_count`; term count is not key-bound
state. Forwarding physical metadata while changing only `effective_dsize` is
insufficient.

A concrete regression is a physical key:

```text
dsize = 8, dnum = 3, k_aux = 8K + g
```

used at `D = 16` and input precision `16K`. The physical calculation is
approximately 25 limbs, while the resolved logical key has `k_aux = 16K + g`
and needs approximately 33 limbs. Immediate exact-backend live-window caps can
hide this, but accumulated products and giant-step tails can expose the
underestimate.

Fix `gglwe_product_output_size`, its accumulated/tail variant, and every caller
that currently passes `key.with_dsize(D)` expecting logical sizing. Directly
assert that the resolved bound has a roughly 33-limb logical work size. For
each output/scratch helper, assert that its result equals the existing formula
evaluated from the resolved layout, never from the roughly 25-limb physical
layout. An exact backend may legitimately cap the returned output below 33
limbs; use an approximate backend or sufficient terms/extra live limbs when a
test needs the output helper itself to reach the full work size.

## 4. Bind keys in every operation variant

Every operation must bind its key at the exact input precision of the operation
and then call a shared bound-aware body:

- GLWE keyswitch and keyswitch-assign;
- automorphism plain, assign, add, add-assign, sub, sub-assign,
  sub-negate, and sub-negate-assign;
- tensor relinearization and its scalar/batched composite forms;
- trace, packing, rotations, and linear-transformation baby/giant products.

Do not maintain native and selected arithmetic bodies. No variant may call
`glwe_keyswitch_internal` or a product OEP with unresolved physical metadata.
Scratch queries and execution must bind identically.

## 5. Generalize lockstep EvalMod to RLK helpers

The paired EvalMod API accepts `GLWERelinearizationKeyHelper`, but the lockstep
driver and its internal frontier functions still require one concrete
`GLWETensorKeyPrepared`. Generalize:

- `ckks_eval_mod_pair_lockstep_default` to an execution helper;
- `ckks_eval_mod_pair_lockstep_tmp_bytes_default` to the layout helper;
- every internal square/multiply/prepared frontier to the corresponding
  helper.

At every multiply, ask the helper for the physical key and requested `dsize`,
then bind that thin key use at the exact input precision of that
multiplication. A frontier may batch entries only when their resolved bounds
are compatible; otherwise split the frontier without changing its dependency
order. Preserve the existing CUDA pair/quad batching for a single native key.

## 6. Validate before ordinary OEP dispatch

Before constructing a prepared bound or calling an OEP, validate the complete
physical prepared object against its physical metadata:

- degree, rows, input columns, and output columns;
- `data.size() == key.max_size()` for a complete prepared key;
- backing span covers `bytes_of_vmp_pmat` for the physical shape;
- checked or widened row, pitch, byte-offset, and product-term arithmetic;
- every logical selected row and prefix lies within the physical object.

Malformed restored objects and unrealizable bounds must fail before a backend
receives them.

## Acceptance

1. Dense-oracle product cases cover native and wrapped keys, row steps 1, 2,
   3, and 4, truncated limb prefixes, and two active selected rows. Random or
   poisoned skipped rows must not affect the output.
2. Address/launch instrumentation proves the kernel generates no addresses or
   memory transactions for inactive logical rows or suffixes of a
   native-`dsize` low-precision bound. Output parity alone is not evidence of
   this; hardware cache-line overfetch is outside the contract.
3. Zero precision performs no key read and no OEP/materialization launch,
   verified with launch instrumentation.
4. The physical-8/effective-16 regression resolves to an approximately
   33-limb logical work size. Immediate and accumulated/tail helpers equal
   their formulas evaluated from that resolved layout, including any valid
   exact-backend live-window cap.
5. Every keyswitch and automorphism variant matches the dense logical-key
   oracle, including assign and add/sub forms.
6. Bound-aware tensor relinearization, trace, packing, and one multi-factor
   linear transformation match CPU/reference results.
7. EvalMod query logging proves helper lookup and binding occur before every
   multiply and may select different physical keys or `dsize` values by level.
8. Native paired EvalMod retains its existing fused batch dispatch counts.
   Wrapped pair/quad/cross-rotation lanes either batch with their resolved
   bounds or fall back to the same ordinary scalar bound-aware OEP.
