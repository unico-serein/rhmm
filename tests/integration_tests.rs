//! Integration tests for rhmm library

use rhmm::models::GaussianHMM;
use rhmm::base::HiddenMarkovModel;
use rhmm::algorithms::{forward_algorithm, backward_algorithm, viterbi_algorithm};
use rhmm::utils::{normalize_vector, validate_probability_vector};
use ndarray::array;
use approx::assert_relative_eq;

#[test]
fn test_gaussian_hmm_workflow() {
    // Create and fit a Gaussian HMM
    let mut hmm = GaussianHMM::new(2);
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0]
    ];
    
    assert!(hmm.fit(&observations, None).is_ok());
    assert!(hmm.is_fitted());
    
    // Test prediction
    let test_obs = array![
        [1.5, 2.5],
        [3.5, 4.5]
    ];
    let states = hmm.predict(&test_obs);
    assert!(states.is_ok());
    
    // Test scoring
    let score = hmm.score(&test_obs);
    assert!(score.is_ok());
}

#[test]
fn test_forward_backward_consistency() {
    let start_prob = array![0.6, 0.4];
    let transition_matrix = array![
        [0.7, 0.3],
        [0.4, 0.6]
    ];
    let emission_probs = array![
        [0.9, 0.1],
        [0.8, 0.2],
        [0.7, 0.3]
    ];
    
    // Run forward algorithm
    let alpha = forward_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();
    
    // Run backward algorithm
    let beta = backward_algorithm(&transition_matrix, &emission_probs).unwrap();
    
    // Check dimensions
    assert_eq!(alpha.shape(), beta.shape());
    assert_eq!(alpha.nrows(), 3);
    assert_eq!(alpha.ncols(), 2);
}

#[test]
fn test_viterbi_with_known_path() {
    // Create a scenario where the path is deterministic
    let start_prob = array![1.0, 0.0];
    let transition_matrix = array![
        [0.9, 0.1],
        [0.1, 0.9]
    ];
    let emission_probs = array![
        [0.9, 0.1],
        [0.9, 0.1],
        [0.1, 0.9]
    ];
    
    let (_log_prob, path) = viterbi_algorithm(&start_prob, &transition_matrix, &emission_probs).unwrap();
    
    // First two observations should prefer state 0, last one state 1
    assert_eq!(path[0], 0);
    assert_eq!(path[1], 0);
}

#[test]
fn test_normalization_and_validation() {
    let vec = array![1.0, 2.0, 3.0];
    let normalized = normalize_vector(vec);
    
    // Check that it sums to 1
    assert_relative_eq!(normalized.sum(), 1.0, epsilon = 1e-10);
    
    // Validate the normalized vector
    assert!(validate_probability_vector(&normalized, "test").is_ok());
}

#[test]
fn test_multiple_sequences() {
    let mut hmm = GaussianHMM::new(2);
    
    // Concatenated sequences
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [5.0, 6.0],
        [6.0, 7.0]
    ];
    
    // Lengths of individual sequences
    let lengths = vec![2, 2];
    
    assert!(hmm.fit(&observations, Some(&lengths)).is_ok());
}

#[test]
fn test_error_handling() {
    let mut hmm = GaussianHMM::new(2);
    
    // Test with empty observations
    let empty_obs = array![[]];
    assert!(hmm.fit(&empty_obs, None).is_err());
    
    // Test prediction before fitting
    let obs = array![[1.0, 2.0]];
    assert!(hmm.predict(&obs).is_err());
}
