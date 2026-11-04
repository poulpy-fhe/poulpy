//! Accuracy of the cleartext forward pass, with the exact and the polynomial ReLU, on the first images of the CIFAR-10
//! test batch: `plain_accuracy [images]`. Ten are built in; the crate's README gets the others.

use anyhow::Result;
use poulpy_resnet20::model::{Relu, ResNet20, argmax, test_images};

fn main() -> Result<()> {
    let limit = std::env::args().nth(1).map_or(Ok(10), |a| a.parse())?;
    let model = ResNet20::load()?;
    let images = test_images(limit)?;
    for relu in [Relu::Exact, Relu::Polynomial] {
        let correct = images
            .iter()
            .filter(|(image, label)| model.forward(image, relu).is_ok_and(|logits| argmax(&logits) == *label))
            .count();
        println!(
            "{} ReLU: {correct} / {} correct",
            if relu == Relu::Exact { "exact" } else { "polynomial" },
            images.len()
        );
    }
    Ok(())
}
