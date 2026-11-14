//! Accuracy of the cleartext forward pass on the first images of the CIFAR-10 test batch, `plain_accuracy [images]`:
//! the network as trained, on the rows of the encrypted network, and with its polynomial ReLU as well. Ten images are
//! built in; the crate's README gets the others.

use anyhow::Result;
use poulpy_resnet20::model::{ResNet20, Variant, argmax, test_images};

fn main() -> Result<()> {
    let limit = std::env::args().nth(1).map_or(Ok(10), |a| a.parse())?;
    let model = ResNet20::load()?;
    let images = test_images(limit)?;
    for (variant, name) in [
        (Variant::Trained, "zero padding, exact ReLU"),
        (Variant::Wrapped, "wrapped rows, exact ReLU"),
        (Variant::Polynomial, "wrapped rows, polynomial ReLU"),
    ] {
        let correct = images
            .iter()
            .filter(|(image, label)| model.forward(image, variant).is_ok_and(|logits| argmax(&logits) == *label))
            .count();
        println!("{name}: {correct} / {} correct", images.len());
    }
    Ok(())
}
