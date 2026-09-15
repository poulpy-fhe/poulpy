pub mod convolution;
pub mod module;
pub mod reim;
pub mod reim4;
pub mod svp;
pub mod vec_znx_big;
pub mod vec_znx_dft;
pub mod vmp;

#[allow(dead_code)]
pub(crate) fn assert_approx_eq_slice(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len(), "Slices have different lengths");

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff: f64 = (x - y).abs();
        let scale: f64 = x.abs().max(y.abs()).max(1.0);
        assert!(
            diff <= tol * scale,
            "Difference at index {}: left={} right={} rel_diff={} > tol={}",
            i,
            x,
            y,
            diff / scale,
            tol
        );
    }
}
