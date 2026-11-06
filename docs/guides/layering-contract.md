# Layering contract

Every library crate (poulpy-hal, poulpy-core, poulpy-ckks, poulpy-bin-fhe,
poulpy-mhe, poulpy-blas) dispatches its operations through the same layers.
This file states them once. Each crate's contracts doc lists its families and
its own rules, and links here.

## Layers

| Layer | What it is |
|---|---|
| API trait | The public operation, implemented on `Module<BE>` (`GLWERotate`). |
| Delegate | The blanket `impl Api<BE> for Module<BE>`. It forwards every call, unchanged, to the backend's OEP method and checks nothing. |
| OEP family | An `unsafe` trait the backend implements (`GLWERotateImpl`); OEP means open extension point. Its methods take `module: &Module<Self>`, then the API method's parameters. |
| Reference | A public `<ApiTrait>Reference<BE>` trait, blanket-implemented on `Module<BE>`, whose `<op>_reference` methods define the result by composing lower-layer operations through their public APIs. |
| Derived default | The default body of an OEP method that composes operations of the same layer. It calls one crate-private `<op>_derived` function, which reaches its constituents through the backend's selected implementations. |

A call runs `api -> delegate -> oep`. Only the reference and derived bodies
define a result: an OEP method forwards to the reference, inherits its derived
default, or is an override that reproduces one of them.

The API, delegate, OEP and layout modules name the `reference` module only in
opt-in macro bodies and rustdoc links. A helper that code outside `reference`
also calls (a check, a size, a placement) lives outside it, refusal checks in
`layouts/validation.rs` (`layouts/validate.rs` in poulpy-blas), and
`reference` imports it.

poulpy-hal has no reference layer. The contract blocks of `poulpy_hal::api`
define its basis operations, poulpy-cpu-portable implements them and the
independent poulpy-cpu-oracle checks them. Its derived bodies are
`#[doc(hidden)] pub` functions of `poulpy_hal::oep::derived`, so that a
backend override can fall back to them.

## Selecting an implementation

A backend implements every OEP family it serves, choosing exactly one of these
per family:

- **The family's opt-in macro.** It implements every required method by
  forwarding it to the reference, and inherits every derived default:

  ```rust,ignore
  poulpy_core::impl_glwe_rotate_reference_full!(MyBackend);
  ```

  A family whose methods all have default bodies has
  `impl_<family>_derived_full!`, an empty `unsafe impl`. Each crate has one
  umbrella, `impl_<crate>_reference_full!`, that expands to its family macros;
  poulpy-ckks's covers the families serving both rings, a standard backend
  adding the standard-only ones.
- **A hand-written `unsafe impl` of the family.** It implements the methods it
  replaces, forwards the other required methods and their `_tmp_bytes` queries
  to the reference, and keeps the derived defaults.

A macro and a hand-written implementation of the same family exclude each
other, so a backend that replaces one family lists the macros of the other
families instead of the umbrella. Trait bounds and imports never select an
implementation.

Derived defaults call their constituents through the selected
implementations. An override of a constituent therefore reaches every
composition built on it, and an override of a derived method replaces that
method alone.

## What an override reproduces

The reference or derived body defines the result: values and rounding,
metadata and the canonical flag, input mutation, the refusals and their order,
and the order of random draws. An override reproduces it bit for bit. It
chooses only its kernels, its prepared representation and its scratch size.
An override that needs a different amount of scratch also overrides the
matching `_tmp_bytes` query, which reports at least what the body takes, and
its tests run it on exactly that budget.

Every override has a registered bit-exact parity test against the oracle or
against a backend already attested for those operations. Prepared and
transformed storage stays opaque: tests prepare each backend's objects
independently and compare the outputs of the operations that read them.

## Names in each crate

poulpy-core is the model above poulpy-hal: OEP methods share the API method's
name, references are traits, derived bodies are crate-private, and each opt-in
macro is named after its OEP family (`GLWEAddImpl`:
`impl_glwe_add_reference_full!`). The crates name their layers as follows.

| Crate | OEP methods | Reference | Derived bodies | Opt-in macros |
|---|---|---|---|---|
| poulpy-hal | the API name | none | `#[doc(hidden)] pub fn <op>_derived` in `oep::derived` | `hal_impl_*!` in poulpy-cpu-portable |
| poulpy-core | the API name | `<Trait>Reference<BE>` | `pub(crate) fn <op>_derived` | `impl_<family>_reference_full!`, `impl_<family>_derived_full!`, `impl_core_reference_full!` |
| poulpy-ckks | the API name; hooks: keyed `ckks_{rotate,conjugate}_with_key_*`, encoding `ckks_{quantize,dequantize}_coeffs_into`, `ckks_encoding_buffer_{slots_to_coeffs,coeffs_to_slots}_assign_with_plan`, `ckks_encoding_plan_cache` and `ckks_encoding_plans_create`, ModUp `ckks_encapsulated_mod_up` and `ckks_mod_up_first_factor`, PaCo `ckks_slot_{trace,product}_assign`, SHIP `ckks_ship_{masking_accumulate,mux_rotate}` | `CKKS<Family>Reference<BE>` | `pub(crate) fn <op>_derived` | `impl_<family>_reference_full!`, `impl_<family>_derived_full!`, `impl_ckks_reference_full!` (both-ring families) |
| poulpy-bin-fhe | the API name | `<Trait>Reference<BE>` | `pub(crate) fn <op>_derived` | `impl_<family>_reference_full!`, `impl_<family>_derived_full!`, group umbrellas `impl_<group>_reference_full!`, `impl_bin_fhe_reference_full!` |
| poulpy-mhe | the API name | `MHE<Thing>ProtocolReference<BE>`, `MHE<Pat>Reference<BE>` | `pub(crate) fn <op>_derived` | `impl_<family>_reference_full!`, `impl_mhe_reference_full!` |
| poulpy-blas | the API name; hooks `blas_<hook>` | `BlasMat<Family>Reference<BE>`, `BlasJklsReference<BE>`, `BlasEncodingReference<BE>` | `pub(crate) fn blas_<op>_derived` | `impl_<family>_reference_full!`, `impl_blas_reference_full!` |
