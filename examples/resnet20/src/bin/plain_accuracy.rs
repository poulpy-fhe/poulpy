//! Accuracy of the cleartext forward pass on the first images of the CIFAR-10 test batch, `plain_accuracy [images]`:
//! the network as trained, on the rows of the encrypted network, and with its polynomial ReLU as well. Ten images are
//! built in; the crate's README gets the others.

use std::thread;

use anyhow::Result;
use poulpy_resnet20::model::{ResNet20, Variant, argmax, test_images};

const VARIANTS: [(Variant, &str); 3] = [
    (Variant::Trained, "zero padding, exact ReLU"),
    (Variant::Wrapped, "wrapped rows, exact ReLU"),
    (Variant::Polynomial, "wrapped rows, polynomial ReLU"),
];

fn main() -> Result<()> {
    let limit = std::env::args().nth(1).map_or(Ok(10), |a| a.parse())?;
    let model = ResNet20::load()?;
    let images = test_images(limit)?;
    // Correct images per variant, the images shared out among the cores.
    let correct = |images: &[(Vec<f64>, usize)]| {
        VARIANTS.map(|(variant, _)| {
            images
                .iter()
                .filter(|(image, label)| argmax(&model.forward(image, variant)) == *label)
                .count()
        })
    };
    let share = images.len().div_ceil(thread::available_parallelism()?.get()).max(1);
    let counts = thread::scope(|scope| {
        let workers: Vec<_> = images
            .chunks(share)
            .map(|images| scope.spawn(move || correct(images)))
            .collect();
        workers.into_iter().fold([0; 3], |sum, worker| {
            let counts = worker.join().expect("a forward pass does not panic");
            std::array::from_fn(|v| sum[v] + counts[v])
        })
    });
    for ((_, name), correct) in VARIANTS.iter().zip(counts) {
        println!("{name}: {correct} / {} correct", images.len());
    }
    Ok(())
}
