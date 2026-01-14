//! Viterbi algorithm implementation

use ndarray::{Array1, Array2};
use crate::errors::Result;

/// Find the most likely state sequence using the Viterbi algorithm
///
/// The Viterbi algorithm finds the most probable sequence of hidden states
/// given the observations.
///
/// # Arguments
///
/// * `start_prob` - Initial state probabilities
/// * `transition_matrix` - State transition matrix
/// * `emission_probs` - Emission probabilities for each observation
///
/// # Returns
///
/// Tuple of (log probability, most likely state sequence)
pub fn viterbi_algorithm(
    start_prob: &Array1<f64>,
    transition_matrix: &Array2<f64>,
    emission_probs: &Array2<f64>,
) -> Result<(f64, Array1<usize>)> {
    let n_samples = emission_probs.nrows();
    let n_states = start_prob.len();

    let mut viterbi = Array2::zeros((n_samples, n_states));
    let mut backpointer = Array2::zeros((n_samples, n_states));

    // Initialize first time step
    for i in 0..n_states {
        viterbi[[0, i]] = (start_prob[i] * emission_probs[[0, i]]).ln();
    }

    // Forward pass
    for t in 1..n_samples {
        for j in 0..n_states {
            let mut max_prob = f64::NEG_INFINITY;
            let mut max_state = 0;

            for i in 0..n_states {
                let prob = viterbi[[t - 1, i]] + transition_matrix[[i, j]].ln();
                if prob > max_prob {
                    max_prob = prob;
                    max_state = i;
                }
            }

            viterbi[[t, j]] = max_prob + emission_probs[[t, j]].ln();
            backpointer[[t, j]] = max_state as f64;
        }
    }

    // Find the most likely final state
    let mut max_prob = f64::NEG_INFINITY;
    let mut last_state = 0;
    for i in 0..n_states {
        if viterbi[[n_samples - 1, i]] > max_prob {
            max_prob = viterbi[[n_samples - 1, i]];
            last_state = i;
        }
    }

    // Backtrack to find the most likely path
    let mut path = Array1::zeros(n_samples);
    path[n_samples - 1] = last_state;

    for t in (0..n_samples - 1).rev() {
        path[t] = backpointer[[t + 1, path[t + 1]]] as usize;
    }

    Ok((max_prob, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_viterbi_algorithm_simple() {
        let start_prob = array![0.6, 0.4];
        let transition_matrix = array![
            [0.7, 0.3],
            [0.4, 0.6]
        ];
        let emission_probs = array![
            [0.9, 0.1],
            [0.8, 0.2]
        ];

        let (log_prob, path) = viterbi_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();
        
        assert_eq!(path.len(), 2);
        assert!(log_prob < 0.0); // Log probability should be negative
    }

    #[test]
    fn test_viterbi_algorithm_deterministic() {
        let start_prob = array![1.0, 0.0];
        let transition_matrix = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let emission_probs = array![
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0]
        ];

        let (_log_prob, path) = viterbi_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();
        
        assert_eq!(path, array![0, 0, 0]);
    }

    #[test]
    fn test_viterbi_algorithm_path_length() {
        let start_prob = array![0.5, 0.5];
        let transition_matrix = array![
            [0.7, 0.3],
            [0.4, 0.6]
        ];
        let emission_probs = array![
            [0.9, 0.1],
            [0.8, 0.2],
            [0.7, 0.3],
            [0.6, 0.4]
        ];

        let (_log_prob, path) = viterbi_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();
        
        assert_eq!(path.len(), 4);
    }
}
