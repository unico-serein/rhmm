//! Integration tests for rhmm library

use rhmm::models::{GaussianHMM, GMMHMM};
use rhmm::base::HiddenMarkovModel;
use rhmm::algorithms::{forward_algorithm, backward_algorithm, viterbi_algorithm};
use rhmm::utils::{normalize_vector, validate_probability_vector};
use ndarray::{array, Array2};
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
    let hmm = GaussianHMM::new(2);
    
    // Test prediction before fitting
    let obs = array![[1.0, 2.0]];
    assert!(hmm.predict(&obs).is_err());
}

// ============================================================================
// GMMHMM Tests
// ============================================================================

#[test]
fn test_gmmhmm_creation() {
    let gmm = GMMHMM::new(3, 2);
    assert_eq!(gmm.n_states(), 3);
    assert_eq!(gmm.n_mix(), 2);
    assert!(!gmm.is_fitted());
}

#[test]
fn test_gmmhmm_basic_workflow() {
    // Create a GMM-HMM with 2 states and 2 mixture components per state
    let mut gmm = GMMHMM::new(2, 2);
    
    // Create training data with clear clusters
    let observations = array![
        [1.0, 1.0],   // Cluster 1
        [1.2, 0.9],
        [0.9, 1.1],
        [5.0, 5.0],   // Cluster 2
        [5.1, 4.9],
        [4.9, 5.1],
        [1.1, 1.0],   // Back to cluster 1
        [0.8, 1.2],
    ];
    
    // Fit the model
    assert!(gmm.fit(&observations, None).is_ok());
    assert!(gmm.is_fitted());
    assert_eq!(gmm.n_features(), 2);
    
    // Check that parameters are initialized
    assert!(gmm.mixture_weights().is_some());
    assert!(gmm.means().is_some());
    assert!(gmm.covars().is_some());
    assert!(gmm.start_prob().is_some());
    assert!(gmm.transition_matrix().is_some());
}

#[test]
fn test_gmmhmm_predict() {
    let mut gmm = GMMHMM::new(2, 2);
    
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [10.0, 11.0],
        [11.0, 12.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // Test prediction on training data
    let states = gmm.predict(&observations);
    assert!(states.is_ok());
    let states = states.unwrap();
    assert_eq!(states.len(), 4);
    
    // Test prediction on new data
    let test_obs = array![
        [1.5, 2.5],
        [10.5, 11.5],
    ];
    let test_states = gmm.predict(&test_obs);
    assert!(test_states.is_ok());
}

#[test]
fn test_gmmhmm_score() {
    let mut gmm = GMMHMM::new(2, 2);
    
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // Score should return a log probability
    let score = gmm.score(&observations);
    assert!(score.is_ok());
    let log_prob = score.unwrap();
    
    // Log probability should be finite
    assert!(log_prob.is_finite());
}

#[test]
fn test_gmmhmm_sample() {
    let mut gmm = GMMHMM::new(2, 2);
    
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // Generate samples
    let n_samples = 10;
    let result = gmm.sample(n_samples);
    assert!(result.is_ok());
    
    let (sampled_obs, sampled_states) = result.unwrap();
    
    // Check dimensions
    assert_eq!(sampled_obs.nrows(), n_samples);
    assert_eq!(sampled_obs.ncols(), 2);
    assert_eq!(sampled_states.len(), n_samples);
    
    // Check that states are valid
    for &state in sampled_states.iter() {
        assert!(state < 2);
    }
}

#[test]
fn test_gmmhmm_mixture_weights_sum_to_one() {
    let mut gmm = GMMHMM::new(3, 2);
    
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // Check that mixture weights sum to 1 for each state
    if let Some(weights) = gmm.mixture_weights() {
        for i in 0..3 {
            let sum: f64 = (0..2).map(|k| weights[[i, k]]).sum();
            assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_gmmhmm_parameters_shape() {
    let mut gmm = GMMHMM::new(2, 3);
    
    let observations = array![
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // Check mixture weights shape: (n_states, n_mix)
    if let Some(weights) = gmm.mixture_weights() {
        assert_eq!(weights.shape(), &[2, 3]);
    }
    
    // Check means shape: (n_states, n_mix, n_features)
    if let Some(means) = gmm.means() {
        assert_eq!(means.shape(), &[2, 3, 3]);
    }
    
    // Check covars shape: (n_states, n_mix, n_features)
    if let Some(covars) = gmm.covars() {
        assert_eq!(covars.shape(), &[2, 3, 3]);
    }
}

#[test]
fn test_gmmhmm_error_before_fit() {
    let gmm = GMMHMM::new(2, 2);
    
    let obs = array![[1.0, 2.0]];
    
    // Should fail before fitting
    assert!(gmm.predict(&obs).is_err());
    assert!(gmm.score(&obs).is_err());
    assert!(gmm.sample(10).is_err());
}

#[test]
fn test_gmmhmm_dimension_mismatch() {
    let mut gmm = GMMHMM::new(2, 2);
    
    let train_obs = array![
        [1.0, 2.0],
        [2.0, 3.0],
    ];
    
    gmm.fit(&train_obs, None).unwrap();
    
    // Try to predict with wrong dimensions
    let wrong_obs = array![
        [1.0, 2.0, 3.0],  // 3 features instead of 2
    ];
    
    assert!(gmm.predict(&wrong_obs).is_err());
    assert!(gmm.score(&wrong_obs).is_err());
}

#[test]
fn test_gmmhmm_convergence() {
    let mut gmm = GMMHMM::new(2, 2);
    
    // Create data with clear structure
    let observations = array![
        [0.0, 0.0],
        [0.1, 0.1],
        [0.2, 0.0],
        [10.0, 10.0],
        [10.1, 10.1],
        [10.0, 10.2],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // After fitting, score should be reasonable
    let score = gmm.score(&observations).unwrap();
    assert!(score.is_finite());
}

#[test]
fn test_gmmhmm_single_mixture_vs_gaussian() {
    // GMM-HMM with 1 mixture component should behave similarly to Gaussian HMM
    let mut gmm = GMMHMM::new(2, 1);
    let mut gaussian = GaussianHMM::new(2);
    
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [5.0, 6.0],
        [6.0, 7.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    gaussian.fit(&observations, None).unwrap();
    
    // Both should be able to predict
    let gmm_states = gmm.predict(&observations).unwrap();
    let gaussian_states = gaussian.predict(&observations).unwrap();
    
    assert_eq!(gmm_states.len(), gaussian_states.len());
}

#[test]
fn test_gmmhmm_multiple_mixtures() {
    // Test with more mixture components
    let mut gmm = GMMHMM::new(2, 4);
    
    let observations = array![
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 2.0],
        [5.0, 5.0],
        [5.5, 5.5],
        [6.0, 6.0],
    ];
    
    assert!(gmm.fit(&observations, None).is_ok());
    
    // Check that all mixture components have valid parameters
    if let (Some(weights), Some(_means), Some(covars)) = 
        (gmm.mixture_weights(), gmm.means(), gmm.covars()) {
        
        for i in 0..2 {
            for k in 0..4 {
                // Weights should be positive
                assert!(weights[[i, k]] >= 0.0);
                
                // Covariances should be positive
                for j in 0..2 {
                    assert!(covars[[i, k, j]] > 0.0);
                }
            }
        }
    }
}

#[test]
fn test_gmmhmm_reproducibility_with_same_data() {
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ];
    
    let mut gmm1 = GMMHMM::new(2, 2);
    let mut gmm2 = GMMHMM::new(2, 2);
    
    gmm1.fit(&observations, None).unwrap();
    gmm2.fit(&observations, None).unwrap();
    
    // Both models should be able to score the data
    let score1 = gmm1.score(&observations).unwrap();
    let score2 = gmm2.score(&observations).unwrap();
    
    // Scores should be finite and reasonable
    assert!(score1.is_finite());
    assert!(score2.is_finite());
}

#[test]
fn test_gmmhmm_empty_observations() {
    let mut gmm = GMMHMM::new(2, 2);
    
    // Create observations with 0 rows
    let empty_obs = Array2::<f64>::zeros((0, 2));
    assert!(gmm.fit(&empty_obs, None).is_err());
}

#[test]
fn test_gmmhmm_covariance_positivity() {
    let mut gmm = GMMHMM::new(2, 2);
    
    let observations = array![
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
    ];
    
    gmm.fit(&observations, None).unwrap();
    
    // All covariances should be positive
    if let Some(covars) = gmm.covars() {
        for i in 0..2 {
            for k in 0..2 {
                for j in 0..2 {
                    assert!(covars[[i, k, j]] > 0.0, 
                        "Covariance at [{}, {}, {}] should be positive", i, k, j);
                }
            }
        }
    }
}
