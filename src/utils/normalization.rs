//! Normalization utilities

use ndarray::{Array1, Array2, Axis};

/// Normalize a probability vector to sum to 1
pub fn normalize_vector(mut vec: Array1<f64>) -> Array1<f64> {
    let sum: f64 = vec.sum();
    if sum > 0.0 {
        vec /= sum;
    }
    vec
}

/// Normalize each row of a matrix to sum to 1
pub fn normalize_matrix_rows(mut matrix: Array2<f64>) -> Array2<f64> {
    for mut row in matrix.axis_iter_mut(Axis(0)) {
        let sum: f64 = row.sum();
        if sum > 0.0 {
            row /= sum;
        }
    }
    matrix
}

/// Convert probabilities to log space safely
pub fn log_normalize(probs: &Array1<f64>) -> Array1<f64> {
    probs.mapv(|x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
}

/// Convert log probabilities back to probability space
pub fn exp_normalize(log_probs: &Array1<f64>) -> Array1<f64> {
    let max_log = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let shifted = log_probs.mapv(|x| (x - max_log).exp());
    normalize_vector(shifted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_normalize_vector() {
        let vec = array![1.0, 2.0, 3.0];
        let normalized = normalize_vector(vec);
        assert_relative_eq!(normalized.sum(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[0], 1.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[1], 2.0 / 6.0, epsilon = 1e-10);
        assert_relative_eq!(normalized[2], 3.0 / 6.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_vector_zero_sum() {
        let vec = array![0.0, 0.0, 0.0];
        let normalized = normalize_vector(vec);
        assert_eq!(normalized, array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_matrix_rows() {
        let matrix = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let normalized = normalize_matrix_rows(matrix);

        for i in 0..normalized.nrows() {
            let row_sum: f64 = normalized.row(i).sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_log_normalize() {
        let probs = array![0.5, 0.3, 0.2];
        let log_probs = log_normalize(&probs);

        assert_relative_eq!(log_probs[0], 0.5_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(log_probs[1], 0.3_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(log_probs[2], 0.2_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_log_normalize_zero() {
        let probs = array![0.5, 0.0, 0.2];
        let log_probs = log_normalize(&probs);

        assert_relative_eq!(log_probs[0], 0.5_f64.ln(), epsilon = 1e-10);
        assert_eq!(log_probs[1], f64::NEG_INFINITY);
        assert_relative_eq!(log_probs[2], 0.2_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_exp_normalize() {
        let log_probs = array![-0.693147, -1.203973, -1.609438];
        let probs = exp_normalize(&log_probs);

        assert_relative_eq!(probs.sum(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(probs[0], 0.5, epsilon = 1e-6);
        assert_relative_eq!(probs[1], 0.3, epsilon = 1e-6);
        assert_relative_eq!(probs[2], 0.2, epsilon = 1e-6);
    }
}
