# Getting performance out of Poulpy

Many things affect how fast an application runs; this page covers five that are chosen at build or parameter-selection time and are easy to get wrong: the arithmetic family, the backend, the limb size `base2k`, the gadget granularity `dsize`, and the number of threads.
It describes what each one trades and how to measure the last one on your own machine.
Use the [diagnostic](#measuring-your-own-thread-count) for numbers from your own hardware.

## The short version

1. Use `FFT64` for bin-FHE and gate-level work, and `NTT3x42` for CKKS and leveled work where IFMA is available.
   Without IFMA the choice between `FFT64` and `NTT4x30` for CKKS depends on your operation mix; see below.
2. Within a family, take the widest ISA your CPU supports — run the [capability report](#3-compilation-options) to see which ones this machine has.
   In our measurements the ordering between backends did not change with the thread count.
3. Use the largest `base2k` the backend allows — 19 bits for `FFT64`, 52 for `NTT4x30` and `NTT3x42`.
4. Tune `dsize`: it sets key size and the key's auxiliary precision together, and has a smaller, machine-dependent effect on speed. `dsize = 4` is a reasonable starting point.
5. Give the Rayon backends a handful of threads, not all of them, and measure where your own knee is.

## Key material is usually the bottleneck

Most of the cost of an FHE circuit sits in two operations: the external product in gate-level work, and the key-switch in leveled work.
Both stream a large, prepared key through the CPU and use each byte of it once.
At realistic parameters those keys are hundreds of megabytes, far beyond any cache, so the loop runs at the speed the memory system can deliver them and the arithmetic hides underneath.

The size of a prepared key is therefore the first thing to look at, and the backend decides it.
A prepared key holds one DFT-domain word per coefficient per limb, so what matters is the number of bytes it spends per bit of torus precision:

| family | bytes per coefficient | max `base2k` | bytes per torus bit |
| --- | --- | --- | --- |
| `NTT3x42` (IFMA) | 16 | 52 | 0.31 |
| `FFT64` | 8 | 19 | 0.42 |
| `NTT4x30` | 32 | 52 | 0.62 |

A wider limb is only worth what it costs to store.
`NTT4x30` halves the limb count relative to `FFT64` and more, but spends four times the bytes on each one, so its keys end up the largest of the three.
`NTT3x42` gets the wide limbs at half the storage and has the smallest keys.

This ranking applies to leveled work, where a parameter set fixes the torus precision and the limb counts follow from it.
In our measurements the CKKS key-switch came out in that order: `NTT3x42` fastest, `FFT64` next, `NTT4x30` last.

Gate-level parameters fix a handful of limbs rather than a precision, and the ranking changes accordingly.
`FFT64` needs three narrow limbs where the NTT families need two wide ones, which gives it the smallest key of the three; it was also the fastest, ahead of `NTT3x42` and far ahead of `NTT4x30`.
Compare the size of the prepared key at your own parameters rather than the per-bit figure alone.

Key size governs only half of the cost.
Everything that works in the coefficient domain instead — addition, encoding and decoding, plaintext operations, normalization — costs one pass per limb, so there the limb count is what matters and a small `base2k` is a straight penalty.
The two halves pull in opposite directions for `FFT64`, whose limbs are the cheapest to store and the most numerous, so its ranking depends on which operations a circuit spends its time in.

## 1. Arithmetic family

`FFT64` stores a limb as one `f64` polynomial; the NTT families store it as several word-sized residues, `4 × 30` bits or `3 × 42` bits with IFMA.
The NTT limbs are wider, so a given torus precision needs fewer of them, but each limb costs several transforms instead of one and, per the table above, more bytes.

**Gate-level and bin-FHE work: use `FFT64`.**
Blind rotation runs at a few limbs on a small ring, where the NTT's wider limbs cannot pay for themselves.
`FFT64` was several times faster than any NTT backend on this workload, by a margin wide enough that the portable `FFT64Ref` also beat every accelerated NTT backend.

**CKKS and leveled work: use `NTT3x42` where IFMA is available.**
It was roughly twice as fast as either alternative on a full bootstrap at equal output precision, and fastest on every individual operation from moderate ring degrees upward, with its lead growing as the ring grows.
At small ring degrees `FFT64` can win an isolated key-switch, though not the surrounding pipeline.

**Without IFMA, the choice depends on your operation mix.**
`FFT64` and `NTT4x30` each win a different half of the workload:

- Key-switch-dominated work — rotations, conjugation, relinearized multiplication — favors `FFT64` by a factor of two or more, since its keys are the smallest and the loop streams keys.
  The margin shrinks as the ring degree grows.
- Coefficient-domain work — addition, encoding and decoding, plaintext multiplication — favors `NTT4x30` by a similar factor, because `FFT64` needs roughly three times the limbs to reach the same precision and every one of them costs a pass.

On a full bootstrapping, which mixes both, the two landed within about fifteen percent of each other in our measurements.
Profile your own circuit, or start from whichever half dominates it.

## 2. Backend

Within a family, prefer the most accelerated backend your CPU and build flags allow, and pick it before choosing a thread count: in our measurements the ordering between backends was the same on one thread as on sixteen.
Threads multiply what the serial backend gives you; they do not reorder the choice.

How much a wider ISA buys depends on how far the workload is from the memory roof.
On leveled work with many limbs there is arithmetic to accelerate, and each ISA step is worth a solid fraction.
On gate-level work there is less: the blind rotation already runs close to what a single core can stream, so `FFT64Avx512` and `FFT64Avx` landed within a couple of percent of each other and the scalar `FFT64Ref` only modestly behind.
The same workload on `NTT4x30`, which does four times the transforms and sits further from the memory roof, still gained a large factor from SIMD.

Take the widest ISA available, but do not expect it to compensate for the wrong family or limb size.

A `*Rayon` backend in a single-threaded pool costs nothing measurable over its serial counterpart, so the parallel variant can be compiled in and the pool width decided later.

## 3. Compilation options

The accelerated backends need both a Cargo feature and the matching target features.
`Module::new` checks the CPU at runtime and panics if they are missing.

| backend | crate | feature | `RUSTFLAGS` |
| --- | --- | --- | --- |
| `FFT64Avx`, `NTT4x30Avx` | `poulpy-cpu-avx` | `enable-avx` | `-C target-feature=+avx2,+fma` |
| `FFT64Avx512`, `NTT4x30Avx512` | `poulpy-cpu-avx512` | `enable-avx512f` | `-C target-feature=+avx512f` |
| `NTT3x42Ifma` | `poulpy-cpu-avx512` | `enable-ifma` | `-C target-feature=+avx512f,+avx512ifma,+avx512vl` |
| `FFT64Neon`, `NTT4x30Neon` | `poulpy-cpu-arm` | `enable-neon` | none (AArch64) |

Build in release mode: an unoptimized build is slower by orders of magnitude and says nothing about any of the choices on this page.
`-C target-cpu=native` is the simplest way to enable everything the build machine supports.
Add `enable-rayon` to expose the `*Rayon` types, and `enable-core` / `enable-ckks` to wire the scheme layers in.
The workspace also ships a `profiling` profile — release plus debug symbols — for use with a sampling profiler.

To list the instruction sets a machine has, run, with no features or flags:

```sh
cargo test -p poulpy-cpu-ref capabilities -- --ignored --nocapture
```

```text
instruction set   present  build with
avx512f           yes      RUSTFLAGS="-C target-feature=+avx512f"
avx512ifma+vl     yes      RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl"
neon              no
```

The table above maps each set to the backends it unlocks.

To check which backends a given build enabled, run the report of the backend crate itself:

```sh
cargo test -p poulpy-cpu-avx512 capabilities -- --ignored --nocapture
```

```text
backend              crate               cpu  built  build with
NTT3x42Ifma          poulpy-cpu-avx512   yes  no     --features enable-ifma   RUSTFLAGS="-C target-feature=+avx512f,+avx512ifma,+avx512vl"
```

`cpu` describes the machine; `built` describes the command that produced the report, and reads `no` until the feature and flags in the last column are added.

## 4. `base2k`

The backend caps the limb size: 19 bits for `FFT64`, 52 for `NTT4x30` and `NTT3x42`.
Within that cap, larger is better, and by a wide margin.

A given torus precision needs `⌈k / base2k⌉` limbs, and cost grows with the limb count — linearly for the transforms and the key traffic, quadratically for the tensor product.
Halving `base2k` therefore roughly doubles the work, and the whole span from a small limb to the backend's maximum is worth several times the runtime on every backend we measured.

Use the largest `base2k` the backend allows, unless the noise budget of your circuit forces a smaller one.
The key-size table above assumes this: a family is only as good as the `base2k` it is run at.

## 5. `dsize`

`dsize` is the number of base-`2^K` limbs grouped into one gadget digit.
It moves key size, the key's auxiliary precision, and — to a smaller extent — evaluation speed.

**Key size.**
Raising `dsize` cuts the digit count `dnum` roughly as `1/dsize` and amortizes the key's fixed precision overhead over a wider digit, so evaluation keys shrink substantially.
Key generation time follows them down, by a factor of several across the useful range.
When RAM is the binding constraint, this is the main lever: a bootstrapping key set runs to hundreds of megabytes, and resident keys often decide whether a workload fits at all.

**Auxiliary precision.**
A wider digit needs a more precise key: `k_aux`, the key's own torus precision, grows with `dsize`.
That growth is the part to account for when choosing parameters — it enters the key-switching noise, and so the budget the circuit has left to work with — which makes `dsize` a parameter-selection knob rather than a free dial.

**Speed.**
The effect on evaluation time is small next to the two above, and it depends on the machine and the circuit.
`dsize = 4` is a reasonable starting point; measure if it matters to you.

Pick `dsize` from the key size and the precision budget your parameters need, and check the evaluation cost of the value you land on.

## 6. Threads

More threads is not faster, and past a point it is slower.

The kernels the Rayon backends parallelize are the same key-streaming loops described above.
Once the memory system is saturated — which happens at a handful of cores, not at the core count — extra workers add scheduling and cache pressure without adding throughput.
In our measurements the best speed-up over a single thread was around a factor of two, reached at a single-digit number of threads; beyond that the curve turns back up, and at full core count a blind rotation lands close to its single-threaded time.
Giving the pool every core is therefore slower than giving it a few.

Two further points:

- The knee moves with the ring degree, the limb count and the backend, so it has to be measured per workload.
- Choosing a `*Rayon` backend enlarges `*_tmp_bytes`, since the parallel kernels reserve one scratch slice per worker up to a fixed cap.
  Widening the pool beyond that cap changes neither the reservation nor the work done.

### Inside one operation, or across many?

The `*Rayon` backends parallelize inside a single operation, which suits refreshing one ciphertext as fast as possible.
For many independent ciphertexts, run the serial backend on threads of your own so each thread keeps its own working set; that scales better, since the threads are not competing for the same key stream.
Use the Rayon backends for latency and the serial ones for throughput.

### Setting the pool

The Rayon backends use the ambient pool, so either set `RAYON_NUM_THREADS` or build a pool explicitly and run inside it:

```rust
let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build()?;
pool.install(|| {
    // Poulpy calls made here use this pool.
});
```

Nesting a Rayon backend inside your own Rayon tasks is safe — the executor serializes the inner level rather than oversubscribing — but it gains nothing.
Parallelize at one level only.

## Measuring your own thread count

`poulpy-cpu-rayon` ships a diagnostic that sweeps pool widths at a given ring degree and limb count, timing one primitive per kernel family: the vector-matrix product used by key-switching and blind rotation, the convolution used by plaintext products, the inverse transform, and a coefficient-domain addition.
The families do not share a knee — the coefficient-domain one regresses at widths where the transform is still gaining — so a single probe would not describe the machine.

Each backend crate instantiates it as an ignored test:

```sh
RUSTFLAGS="-C target-cpu=native" cargo test --release -p poulpy-cpu-avx512 \
  --features enable-avx512f,enable-ifma,enable-rayon tuning -- --ignored --nocapture
```

The report gives, per probe, the time and speed-up at each pool width, the agreement between the fastest repetitions, and the smallest width within a few percent of the best, together with the other widths the data cannot separate from it:

```text
NTT3x42IfmaRayon
vmp (worker slices capped at 4)
  threads=1        7.893 ms    1.00x   spread 1.01
  threads=4        5.705 ms    1.38x   spread 1.02
  threads=16       5.854 ms    1.35x   spread 1.00
  recommended: 4 threads (within noise of 4, 32)
idft (worker slices capped at 8)
  threads=1        1.684 ms    1.00x   spread 1.04
  threads=4        0.417 ms    4.04x   spread 1.01
  threads=16       0.264 ms    6.37x   spread 1.09
  recommended: 16 threads
coefficient
  threads=1        0.056 ms    1.00x   spread 1.00
  threads=4        0.028 ms    1.98x   spread 1.03
  threads=16       0.070 ms    0.80x   spread 1.25
  recommended: 4 threads
  => 4 threads: the width no probe is badly hurt by
```

Use the last line.
One pool serves every kernel, and since their knees disagree the figure is not the widest per-probe recommendation but the width minimizing the worst relative loss across them; probes whose repetitions do not agree are excluded from it.

Widths are measured round-robin across several rounds rather than one at a time, so clock drift and cache warming are not attributed to the thread count.
`Mode::Fast` is the default and takes well under a second per backend.
`Mode::Precise` spends about ten times that on more rounds and repetitions, for a machine that is not idle or a band that comes out wide.

The sweep times the production path, whose worker slices are capped per kernel; the report prints each probe's cap beside its name.
Pool widths above a cap run the same number of slices as the cap, so those curves flatten there, and the uncapped coefficient-domain probe is the one that shows what a wider pool does on its own.
The sweep answers how many threads to give the pool, not whether a kernel would benefit from more slices than the library allows — that requires measuring a kernel against a scratch arena sized for more.

Two caveats:

- Run it on an idle machine.
  A busy one produces a curve that is mostly noise; the report prints the repetition spread and warns above ten percent, and `Mode::Precise` copes with a moderately busy machine.
- Give it your parameters.
  The constants at the top of the test are the ring degree, limb count and rank to tune for; the knee at `2^12` is not the knee at `2^16`.

Point the pool at the width it reports, and measure again when the parameters change.

The result is an estimate.
The sweep covers four primitives at one shape, so it rules out the obviously wrong settings — one thread, or every core — but the last of the performance comes from timing the circuit itself across pool widths, and across backends and `base2k` values when the choice is close.
