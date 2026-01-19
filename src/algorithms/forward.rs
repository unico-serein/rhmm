//! Forward algorithm implementation

use crate::errors::Result;
use ndarray::{Array1, Array2};

/// Compute forward probabilities (alpha values)
///
/// The forward algorithm computes the probability of observing a partial sequence
/// and being in a particular state at each time step.
///
/// # Arguments
///
/// * `start_prob` - Initial state probabilities
/// * `transition_matrix` - State transition matrix
/// * `emission_probs` - Emission probabilities for each observation
///
/// # Returns
///
/// Forward probabilities matrix of shape (n_samples, n_states)
pub fn forward_algorithm(
    start_prob: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    emission_probs: &Array2<f64>,
) -> Result<Array2<f64>> {
    let n_samples = emission_probs.nrows();
    let n_states = start_prob.len();

    let mut alpha = Array2::zeros((n_samples, n_states));

    // Initialize first time step
    for i in 0..n_states {
        alpha[[0, i]] = start_prob[i] * emission_probs[[0, i]];
    }

    // Forward pass
    for t in 1..n_samples {
        for j in 0..n_states {
            let mut sum = 0.0;
            for i in 0..n_states {
                sum += alpha[[t - 1, i]] * transition_matrix[[i, j]];
            }
            alpha[[t, j]] = sum * emission_probs[[t, j]];
        }
    }

    Ok(alpha)
}

/// Compute the log probability of an observation sequence
///
/// # Arguments
///
/// * `start_prob` - Initial state probabilities
/// * `transition_matrix` - State transition matrix
/// * `emission_probs` - Emission probabilities for each observation
///
/// # Returns
///
/// Log probability of the observation sequence
pub fn forward_log_probability(
    start_prob: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    emission_probs: &Array2<f64>,
) -> Result<f64> {
    let alpha = forward_algorithm(start_prob, transition_matrix, emission_probs)?;
    let last_row = alpha.row(alpha.nrows() - 1);
    let prob: f64 = last_row.sum();
    Ok(prob.ln())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_forward_algorithm_simple() {
        let start_prob = array![0.6, 0.4];
        let transition_matrix = array![[0.7, 0.3], [0.4, 0.6]];
        let emission_probs = array![[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]];

        let alpha = forward_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();

        assert_eq!(alpha.shape(), &[3, 2]);

        // Check first time step
        assert_relative_eq!(alpha[[0, 0]], 0.6 * 0.9, epsilon = 1e-10);
        assert_relative_eq!(alpha[[0, 1]], 0.4 * 0.1, epsilon = 1e-10);
    }

    #[test]
    fn test_forward_algorithm_dimensions() {
        let start_prob = array![0.5, 0.5];
        let transition_matrix = array![[0.7, 0.3], [0.4, 0.6]];
        let emission_probs = array![[0.9, 0.1], [0.8, 0.2]];

        let alpha = forward_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();
        assert_eq!(alpha.nrows(), 2);
        assert_eq!(alpha.ncols(), 2);
    }

    #[test]
    fn test_forward_log_probability() {
        let start_prob = array![0.6, 0.4];
        let transition_matrix = array![[0.7, 0.3], [0.4, 0.6]];
        let emission_probs = array![[0.9, 0.1], [0.8, 0.2]];

        let log_prob =
            forward_log_probability(&start_prob, &transition_matrix, &emission_probs).unwrap();

        // Log probability should be negative
        assert!(log_prob < 0.0);
    }

    #[test]
    fn test_forward_algorithm_single_observation() {
        let start_prob = array![1.0, 0.0];
        let transition_matrix = array![[1.0, 0.0], [0.0, 1.0]];
        let emission_probs = array![[0.5, 0.5]];

        let alpha = forward_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();

        assert_eq!(alpha.shape(), &[1, 2]);
        assert_relative_eq!(alpha[[0, 0]], 0.5, epsilon = 1e-10);
        assert_relative_eq!(alpha[[0, 1]], 0.0, epsilon = 1e-10);
    }
}
