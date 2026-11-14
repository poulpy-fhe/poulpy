//! Encrypted ResNet-20 inference on the first images of the CIFAR-10 test batch, `encrypted_inference [images]`. Ten
//! are built in; the crate's README gets the others. The server runs on the CPU, or on the GPU with the `enable-cuda`
//! feature, where `--replay` records the whole circuit once as a graph. Each image is compared with the cleartext
//! forward pass that uses the same polynomial ReLU; `--trace` decrypts every layer of the first image on the client and
//! compares it with the cleartext; `--no-stage-sync` drops the time per phase and the waits it needs from the wall
//! clock.

use std::time::{Duration, Instant};

use anyhow::{Result, ensure};
use poulpy_ckks::{CKKSInfos, layouts::CKKSCiphertextOwned};
use poulpy_core::layouts::LWEInfos;
use poulpy_resnet20::{
    Client, CoeffLayout, Host, Params, ProbePoint, Server, Timings,
    model::{ResNet20, SPAN, Variant, argmax, divisor, max_error, relu, test_images},
};

const FLAGS: [&str; 3] = ["--replay", "--trace", "--no-stage-sync"];

/// The encrypted logits against the labels and the cleartext polynomial network, image by image.
struct Score<'a> {
    model: &'a ResNet20,
    images: usize,
    correct: usize,
    agree: usize,
    /// Wall clock of every image but the first.
    warm: Duration,
}

impl<'a> Score<'a> {
    fn new(model: &'a ResNet20) -> Self {
        Self {
            model,
            images: 0,
            correct: 0,
            agree: 0,
            warm: Duration::ZERO,
        }
    }

    /// Counts and reports the next image, whose inference took `wall`, reported as `time`, and gave `logits`.
    fn image(&mut self, image: &[f64], label: usize, logits: &[f64], wall: Duration, time: &str) {
        let mirror = self.model.forward(image, Variant::Polynomial);
        let (class, want) = (argmax(logits), argmax(&mirror));
        println!(
            "image {}: label {label}, encrypted {class}, cleartext {want}, logit error 2^{:.1}, {time}",
            self.images,
            max_error(logits, &mirror).log2()
        );
        if self.images > 0 {
            self.warm += wall;
        }
        self.images += 1;
        self.correct += usize::from(class == label);
        self.agree += usize::from(class == want);
    }

    fn print(&self) {
        if self.images > 1 {
            println!(
                "warm wall clock: {:.1} ms per image",
                1e3 * self.warm.as_secs_f64() / (self.images - 1) as f64
            );
        }
        println!(
            "{} / {} correct, {} agree with the cleartext polynomial network",
            self.correct, self.images, self.agree
        );
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    ensure!(
        args.iter().all(|a| !a.starts_with("--") || FLAGS.contains(&a.as_str())),
        "the flags are {FLAGS:?}"
    );
    ensure!(
        cfg!(feature = "enable-cuda") || !flag("--replay"),
        "--replay takes the enable-cuda feature"
    );
    let count = args.iter().find(|a| !a.starts_with("--")).map_or(Ok(1), |a| a.parse())?;
    let model = ResNet20::load()?;
    let images = test_images(count)?;
    ensure!(!images.is_empty(), "an inference takes at least one image");
    let params = Params::new()?;
    println!(
        "widths: input {} output {} raised {}",
        params.preset().input_k(),
        params.preset().output_k(),
        params.preset().bootstrap_k()
    );

    let start = Instant::now();
    let mut client = Client::new(&params, rand::random());
    let key_set = client.keys()?;
    println!("client keys: {:.1} s", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let server = Server::new(&params, &model, key_set)?;
    println!("server setup: {:.1} s", start.elapsed().as_secs_f64());

    #[cfg(feature = "enable-cuda")]
    if flag("--replay") {
        return replayed(&model, &images, &mut client, &server);
    }
    let mut score = Score::new(&model);
    for (i, (image, label)) in images.iter().enumerate() {
        let ct = client.encrypt_image(image)?;
        let start = Instant::now();
        let (out, t) = if flag("--trace") && i == 0 {
            let (out, t) = traced(&model, image, &mut client, &server, &ct)?;
            (out, Some(t))
        } else if flag("--no-stage-sync") {
            (server.infer(&ct)?, None)
        } else {
            let (out, t) = server.infer_timed(&ct, None)?;
            (out, Some(t))
        };
        let wall = start.elapsed();
        let phases = t.map_or(String::new(), |t: Timings| {
            format!(
                " (s2c {:.2}, conv {:.2}, raise {:.2}, relu {:.2})",
                t.s2c.as_secs_f64(),
                t.conv.as_secs_f64(),
                t.raise.as_secs_f64(),
                t.relu.as_secs_f64()
            )
        });
        score.image(
            image,
            *label,
            &client.decrypt_logits(&out)?,
            wall,
            &format!("{:.2} s{phases}", wall.as_secs_f64()),
        );
    }
    score.print();
    Ok(())
}

/// An inference of `ct`, the encryption of `image`, reporting each layer against the cleartext one before and after
/// its ReLU, normalized by its divisor.
fn traced(
    model: &ResNet20,
    image: &[f64],
    client: &mut Client,
    server: &Server,
    ct: &CKKSCiphertextOwned<Host>,
) -> Result<(CKKSCiphertextOwned<Host>, Timings)> {
    let mut inputs = Vec::new();
    model.forward_traced(image, Variant::Polynomial, &mut |_, x| inputs.push(x.to_vec()));
    let mut probe = |layer: usize, point: ProbePoint, ct: &CKKSCiphertextOwned<Host>, layout: &CoeffLayout| {
        let normalized = inputs[layer].iter().map(|x| x / divisor(layer));
        let (want, have): (Vec<f64>, _) = match point {
            ProbePoint::Input => (normalized.map(|u| SPAN * u).collect(), client.decrypt_coeffs(ct, layout)),
            ProbePoint::Activation => (normalized.map(relu).collect(), client.decrypt_slots(ct, layout)),
        };
        let have = have.expect("the server hands the probe ciphertexts the client decrypts");
        println!(
            "  layer {layer} {point:?}: max error 2^{:.1}, max value {:.3}, k {}, scale 2^{}",
            max_error(&have, &want).log2(),
            want.iter().fold(0.0f64, |m, w| m.max(w.abs())),
            ct.k().as_usize(),
            ct.log_delta()
        );
    };
    server.infer_timed(ct, Some(&mut probe))
}

/// The images through the circuit recorded once as a graph.
#[cfg(feature = "enable-cuda")]
fn replayed(model: &ResNet20, images: &[(Vec<f64>, usize)], client: &mut Client, server: &Server) -> Result<()> {
    let mut encrypted = client.encrypt_image(&images[0].0)?;
    let start = Instant::now();
    let mut recorded = server.record(&encrypted)?;
    let host = recorded.replay.host_events();
    println!(
        "recorded in {:.1} s: {} launches, graph {}, {:.2} GiB retained, host reads {}, writes {}, syncs {}, allocations {}",
        start.elapsed().as_secs_f64(),
        recorded.replay.launches(),
        recorded.replay.is_graph(),
        recorded.replay.retained_bytes() as f64 / (1u64 << 30) as f64,
        host.reads,
        host.writes,
        host.syncs,
        host.allocations,
    );
    recorded.replay.check();

    let mut score = Score::new(model);
    for (i, (image, label)) in images.iter().enumerate() {
        if i > 0 {
            encrypted = client.encrypt_image(image)?;
        }
        let start = Instant::now();
        let out = recorded.infer(&encrypted)?;
        let wall = start.elapsed();
        score.image(
            image,
            *label,
            &client.decrypt_logits(&out)?,
            wall,
            &format!("{:.1} ms", 1e3 * wall.as_secs_f64()),
        );
    }
    score.print();
    Ok(())
}
