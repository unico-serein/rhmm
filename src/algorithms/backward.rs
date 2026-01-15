//! Backward algorithm implementation

use ndarray::Array2;
use crate::errors::Result;

/// Compute backward probabilities (beta values)
///
/// The backward algorithm computes the probability of observing the remaining
/// sequence given a particular state at each time step.
///
/// # Arguments
///
/// * `transition_matrix` - State transition matrix
/// * `emission_probs` - Emission probabilities for each observation
///
/// # Returns
///
/// Backward probabilities matrix of shape (n_samples, n_states)
pub fn backward_algorithm(
    transition_matrix: &Array2<f64>,
    emission_probs: &Array2<f64>,
) -> Result<Array2<f64>> {
    let n_samples = emission_probs.nrows();
    let n_states = transition_matrix.nrows();

    let mut beta = Array2::zeros((n_samples, n_states));

    // Initialize last time step
    for i in 0..n_states {
        beta[[n_samples - 1, i]] = 1.0;
    }

    // Backward pass
    for t in (0..n_samples - 1).rev() {
        for i in 0..n_states {
            let mut sum = 0.0;
            for j in 0..n_states {
                sum += transition_matrix[[i, j]] * emission_probs[[t + 1, j]] * beta[[t + 1, j]];
            }
            beta[[t, i]] = sum;
        }
    }

    Ok(beta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_relative_eq;

    #[test]
    fn test_backward_algorithm_simple() {
        let transition_matrix = array![
            [0.7, 0.3],
            [0.4, 0.6]
        ];
        let emission_probs = array![
            [0.9, 0.1],
            [0.8, 0.2]
        ];

        let beta = backward_algorithm(&transition_matrix, &emission_probs).unwrap();
        
        assert_eq!(beta.shape(), &[2, 2]);
        
        // Last time step should be all 1.0
        assert_relative_eq!(beta[[1, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta[[1, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_backward_algorithm_dimensions() {
        let transition_matrix = array![
            [0.5, 0.5],
            [0.5, 0.5]
        ];
        let emission_probs = array![
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3]
        ];

        let beta = backward_algorithm(&transition_matrix, &emission_probs).unwrap();
        
        assert_eq!(beta.nrows(), 3);
        assert_eq!(beta.ncols(), 2);
    }

    #[test]
    fn test_backward_algorithm_single_observation() {
        let transition_matrix = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let emission_probs = array![
            [0.5, 0.5]
        ];

        let beta = backward_algorithm(&transition_matrix, &emission_probs).unwrap();
        
        assert_eq!(beta.shape(), &[1, 2]);
        assert_relative_eq!(beta[[0, 0]], 1.0, epsilon = 1e-10);
        assert_relative_eq!(beta[[0, 1]], 1.0, epsilon = 1e-10);
    }
}
