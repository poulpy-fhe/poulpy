# Encrypted ResNet-20 on CIFAR-10

ResNet-20 inference over the real-slot S2C-first bootstrap of `poulpy-ckks`: every convolution is folded into the last factor of SlotsToCoeffs, which it runs in the level of that factor. Stages 2 and 3 fit half the slots: their bootstraps run on `Z[X^2]`, with transforms of one layer less.

The weights and the first ten CIFAR-10 test images are built into the crate.

```sh
# Encrypted, on the GPU, the circuit recorded as one graph.
cargo run --release -p poulpy-resnet20 --features cuda --bin encrypted_inference -- 10 --replay
# In the clear, with the exact and the polynomial ReLU.
cargo run --release -p poulpy-resnet20 --bin plain_accuracy -- 10
```

Without `cuda`, `encrypted_inference` runs on the CPU. `--trace` compares each layer with the cleartext on the first image.

## The full test set

The dataset is not in this repository. Download the CIFAR-10 binary version, 170 MB, and point `CIFAR10_TEST_BATCH` at its test batch:

```sh
curl -LO https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzf cifar-10-binary.tar.gz
export CIFAR10_TEST_BATCH=$PWD/cifar-10-batches-bin/test_batch.bin
cargo run --release -p poulpy-resnet20 --features cuda --bin encrypted_inference -- 10000 --replay
```

CIFAR-10 is from A. Krizhevsky, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), 2009.
