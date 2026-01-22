//! Multinomial Hidden Markov Model
//!
//! This module provides a Hidden Markov Model with discrete/multinomial emission distributions.
//! It is suitable for modeling sequences of discrete observations, such as:
//!
//! - **Natural Language Processing**: Part-of-speech tagging, named entity recognition
//! - **Bioinformatics**: DNA/RNA sequence analysis
//! - **User Behavior**: Clickstream analysis, session modeling
//! - **Weather Modeling**: Discrete weather state sequences
//!
//! # Example
//!
//! ```rust
//! use rhmm::models::MultinomialHMM;
//! use rhmm::base::HiddenMarkovModel;
//! use ndarray::array;
//!
//! // Create a model with 2 hidden states and 3 possible symbols (0, 1, 2)
//! let mut model = MultinomialHMM::new(2, 3);
//!
//! // Training data: discrete observations as integers in [0, n_symbols)
//! // Each row is one observation, stored as f64 for API consistency
//! let observations = array![
//!     [0.0], [1.0], [2.0], [0.0], [1.0],
//!     [2.0], [2.0], [1.0], [0.0], [0.0]
//! ];
//!
//! // Fit the model
//! model.fit(&observations, None).unwrap();
//!
//! // Predict hidden states
//! let states = model.predict(&observations).unwrap();
//!
//! // Calculate log-likelihood
//! let score = model.score(&observations).unwrap();
//! ```
//!
//! # Multiple Sequences
//!
//! You can train on multiple sequences by concatenating them and providing lengths:
//!
//! ```rust
//! use rhmm::models::MultinomialHMM;
//! use rhmm::base::HiddenMarkovModel;
//! use ndarray::array;
//!
//! let mut model = MultinomialHMM::new(2, 3);
//!
//! // Two sequences concatenated: [0,1,2] and [2,1,0]
//! let observations = array![[0.0], [1.0], [2.0], [2.0], [1.0], [0.0]];
//! let lengths = vec![3, 3];
//!
//! model.fit(&observations, Some(&lengths)).unwrap();
//! ```

use crate::algorithms::{backward_algorithm, compute_gamma, forward_algorithm, viterbi_algorithm};
use crate::base::{HiddenMarkovModel, InitialProbs, TransitionMatrix};
use crate::errors::{HmmError, Result};
use crate::utils::{
    default_lengths, split_sequences, validate_probability_vector, validate_transition_matrix,
};
use ndarray::{Array1, Array2};
use rand::Rng;

/// Multinomial Hidden Markov Model
///
/// A Hidden Markov Model with discrete/multinomial emission distributions.
/// Suitable for discrete observation sequences where each observation is an integer
/// in the range `[0, n_symbols)`.
///
/// # Type Parameters
///
/// - `n_states`: Number of hidden states in the model
/// - `n_symbols`: Number of possible discrete observation values (vocabulary size)
///
/// # Observation Format
///
/// Observations should be provided as `Array2<f64>` with shape `(n_samples, 1)`,
/// where each value is an integer in `[0, n_symbols)` stored as `f64`.
///
/// # Example
///
/// ```rust
/// use rhmm::models::MultinomialHMM;
/// use rhmm::base::HiddenMarkovModel;
/// use ndarray::array;
///
/// let mut model = MultinomialHMM::new(2, 4);  // 2 states, 4 symbols
/// let obs = array![[0.0], [1.0], [2.0], [3.0], [0.0]];
/// model.fit(&obs, None).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultinomialHMM {
    /// Number of hidden states
    n_states: usize,
    /// Number of possible discrete observations (vocabulary/symbol size)
    /// Note: This is called n_symbols internally but exposed as n_features for trait compatibility
    n_symbols: usize,
    /// Initial state probabilities π[i] = P(state_0 = i)
    start_prob: Option<InitialProbs>,
    /// State transition matrix A[i,j] = P(state_t = j | state_{t-1} = i)
    transition_matrix: Option<TransitionMatrix>,
    /// Emission probabilities B[i,k] = P(obs_t = k | state_t = i)
    emission_prob: Option<Array2<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
}

impl MultinomialHMM {
    /// Create a new Multinomial HMM
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    /// * `n_symbols` - Number of possible discrete observation values (vocabulary size)
    ///
    /// # Example
    ///
    /// ```rust
    /// use rhmm::models::MultinomialHMM;
    ///
    /// // Create a model for POS tagging with 5 states and 1000 word vocabulary
    /// let model = MultinomialHMM::new(5, 1000);
    /// ```
    pub fn new(n_states: usize, n_symbols: usize) -> Self {
        Self {
            n_states,
            n_symbols,
            start_prob: None,
            transition_matrix: None,
            emission_prob: None,
            is_fitted: false,
        }
    }

    /// Get the number of possible symbols (vocabulary size)
    ///
    /// This is an alias for `n_features()` with clearer semantics for discrete models.
    pub fn n_symbols(&self) -> usize {
        self.n_symbols
    }

    /// Get the initial state probabilities π
    ///
    /// Returns `None` if the model has not been fitted.
    pub fn start_prob(&self) -> Option<&InitialProbs> {
        self.start_prob.as_ref()
    }

    /// Get the state transition matrix A
    ///
    /// Returns `None` if the model has not been fitted.
    pub fn transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.transition_matrix.as_ref()
    }

    /// Get the emission probability matrix B
    ///
    /// Shape: `(n_states, n_symbols)` where `B[i,k] = P(obs = k | state = i)`
    ///
    /// Returns `None` if the model has not been fitted.
    pub fn emission_prob(&self) -> Option<&Array2<f64>> {
        self.emission_prob.as_ref()
    }

    /// Check if the model has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Compute emission probabilities for all observations and states
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence of shape (n_samples, 1)
    ///
    /// # Returns
    ///
    /// Emission probabilities of shape (n_samples, n_states)
    fn compute_emission_probs(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = observations.nrows();
        let mut emission_probs = Array2::zeros((n_samples, self.n_states));

        let ep = self.emission_prob.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Emission probabilities not initialized".to_string())
        })?;

        for t in 0..n_samples {
            let obs_val = observations[[t, 0]] as usize;
            if obs_val >= self.n_symbols {
                return Err(HmmError::InvalidParameter(format!(
                    "Observation index {} exceeds n_symbols {}",
                    obs_val, self.n_symbols
                )));
            }

            for i in 0..self.n_states {
                emission_probs[[t, i]] = ep[[i, obs_val]];
            }
        }

        Ok(emission_probs)
    }

    /// Initialize parameters with uniform distributions if not already set
    fn initialize_parameters(&mut self) -> Result<()> {
        if self.start_prob.is_none() {
            self.start_prob = Some(Array1::from_elem(self.n_states, 1.0 / self.n_states as f64));
        }

        if self.transition_matrix.is_none() {
            self.transition_matrix = Some(Array2::from_elem(
                (self.n_states, self.n_states),
                1.0 / self.n_states as f64,
            ));
        }

        if self.emission_prob.is_none() {
            self.emission_prob = Some(Array2::from_elem(
                (self.n_states, self.n_symbols),
                1.0 / self.n_symbols as f64,
            ));
        }

        Ok(())
    }

    /// Compute xi (state transition probabilities) for a sequence
    fn compute_xi(
        alpha: &Array2<f64>,
        beta: &Array2<f64>,
        transition_matrix: &Array2<f64>,
        emission_probs: &Array2<f64>,
    ) -> Result<Vec<Array2<f64>>> {
        let n_samples = alpha.nrows();
        let n_states = alpha.ncols();
        let mut xi = Vec::with_capacity(n_samples - 1);

        for t in 0..n_samples - 1 {
            let mut xi_t = Array2::zeros((n_states, n_states));
            let mut sum = 0.0;

            for i in 0..n_states {
                for j in 0..n_states {
                    xi_t[[i, j]] = alpha[[t, i]]
                        * transition_matrix[[i, j]]
                        * emission_probs[[t + 1, j]]
                        * beta[[t + 1, j]];
                    sum += xi_t[[i, j]];
                }
            }

            if sum > 0.0 {
                xi_t /= sum;
            }
            xi.push(xi_t);
        }

        Ok(xi)
    }
    /// Update emission probabilities based on gamma (M-step)
    ///
    /// Uses the accumulated gamma values to re-estimate emission probabilities
    /// using maximum likelihood estimation.
    fn update_emission_parameters(
        &mut self,
        observations: &Array2<f64>,
        gamma: &Array2<f64>,
    ) -> Result<()> {
        let n_samples = observations.nrows();
        let mut emission_prob = Array2::zeros((self.n_states, self.n_symbols));

        for i in 0..self.n_states {
            let mut denom = 0.0;
            for t in 0..n_samples {
                let obs_val = observations[[t, 0]] as usize;
                emission_prob[[i, obs_val]] += gamma[[t, i]];
                denom += gamma[[t, i]];
            }

            if denom > 0.0 {
                for j in 0..self.n_symbols {
                    emission_prob[[i, j]] /= denom;
                }
            } else {
                // If a state is never visited, set uniform emission probabilities
                for j in 0..self.n_symbols {
                    emission_prob[[i, j]] = 1.0 / self.n_symbols as f64;
                }
            }
        }

        self.emission_prob = Some(emission_prob);
        Ok(())
    }
}

impl HiddenMarkovModel for MultinomialHMM {
    fn n_states(&self) -> usize {
        self.n_states
    }

    /// Returns the number of symbols (vocabulary size)
    ///
    /// Note: For MultinomialHMM, this returns `n_symbols` (the vocabulary size),
    /// not the number of observation dimensions.
    fn n_features(&self) -> usize {
        self.n_symbols
    }

    fn fit(&mut self, observations: &Array2<f64>, lengths: Option<&[usize]>) -> Result<()> {
        if observations.nrows() == 0 || observations.ncols() == 0 {
            return Err(HmmError::InvalidParameter(
                "Observations cannot be empty".to_string(),
            ));
        }

        // Validate observations: must be integers in [0, n_symbols)
        for t in 0..observations.nrows() {
            let obs_val = observations[[t, 0]];
            if obs_val < 0.0 || obs_val >= self.n_symbols as f64 || obs_val.fract() != 0.0 {
                return Err(HmmError::InvalidParameter(format!(
                    "Invalid observation at row {}: {}. Must be integer in [0, {})",
                    t, obs_val, self.n_symbols
                )));
            }
        }

        // Get sequence lengths (default to single sequence if not provided)
        let lengths_vec = lengths
            .map(|l| l.to_vec())
            .unwrap_or_else(|| default_lengths(observations.nrows()));

        // Split observations into sequences
        let sequences = split_sequences(observations, &lengths_vec)?;

        // Initialize parameters if not set
        self.initialize_parameters()?;

        // Validate parameters
        if let Some(ref sp) = self.start_prob {
            validate_probability_vector(sp, "Initial state probabilities")?;
        }
        if let Some(ref tm) = self.transition_matrix {
            validate_transition_matrix(tm)?;
        }
        if let Some(ref ep) = self.emission_prob {
            for i in 0..self.n_states {
                validate_probability_vector(&ep.row(i).to_owned(), "Emission probabilities")?;
            }
        }

        // Baum-Welch algorithm for parameter estimation
        let max_iter = 100;
        let tol = 1e-4;
        let mut prev_log_prob = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            let mut total_log_prob = 0.0;

            // Accumulators for statistics across all sequences
            let mut start_prob_acc = Array1::zeros(self.n_states);
            let mut trans_acc = Array2::zeros((self.n_states, self.n_states));
            let mut gamma_acc = Array2::zeros((observations.nrows(), self.n_states));

            let start_prob = self.start_prob.as_ref().unwrap();
            let trans_mat = self.transition_matrix.as_ref().unwrap();

            let mut row_offset = 0;

            // E-step: Process each sequence independently
            for seq in &sequences {
                let seq_len = seq.nrows();
                let seq_owned = seq.to_owned();

                // Compute emission probabilities for this sequence
                let emission_probs = self.compute_emission_probs(&seq_owned)?;

                // Compute forward and backward probabilities
                let alpha = forward_algorithm(start_prob, trans_mat, &emission_probs)?;
                let beta = backward_algorithm(trans_mat, &emission_probs)?;

                // Accumulate log probability
                let seq_log_prob = alpha.row(alpha.nrows() - 1).sum().ln();
                total_log_prob += seq_log_prob;

                // Compute gamma (state occupation probabilities)
                let gamma = compute_gamma(&alpha, &beta)?;

                // Copy gamma to accumulator
                for t in 0..seq_len {
                    for i in 0..self.n_states {
                        gamma_acc[[row_offset + t, i]] = gamma[[t, i]];
                    }
                }

                // Accumulate initial state probabilities
                for i in 0..self.n_states {
                    start_prob_acc[i] += gamma[[0, i]];
                }

                // Compute and accumulate xi (state transition probabilities)
                let xi = Self::compute_xi(&alpha, &beta, trans_mat, &emission_probs)?;
                for xi_t in xi.iter().take(seq_len - 1) {
                    for i in 0..self.n_states {
                        for j in 0..self.n_states {
                            trans_acc[[i, j]] += xi_t[[i, j]];
                        }
                    }
                }

                row_offset += seq_len;
            }

            // Check convergence
            if (total_log_prob - prev_log_prob).abs() < tol {
                break;
            }
            prev_log_prob = total_log_prob;

            // M-step: Update parameters

            // Update initial state probabilities
            if let Some(ref mut sp) = self.start_prob {
                let sum: f64 = start_prob_acc.sum();
                if sum > 0.0 {
                    *sp = start_prob_acc.clone() / sum;
                }
            }

            // Update transition matrix
            if let Some(ref mut tm) = self.transition_matrix {
                for i in 0..self.n_states {
                    let row_sum: f64 = trans_acc.row(i).sum();
                    if row_sum > 0.0 {
                        for j in 0..self.n_states {
                            tm[[i, j]] = trans_acc[[i, j]] / row_sum;
                        }
                    }
                }
            }

            // Update emission probabilities
            self.update_emission_parameters(observations, &gamma_acc)?;
        }

        self.is_fitted = true;
        Ok(())
    }

    fn predict(&self, observations: &Array2<f64>) -> Result<Array1<usize>> {
        if !self.is_fitted {
            return Err(HmmError::ModelNotFitted(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        let emission_probs = self.compute_emission_probs(observations)?;
        let start_prob = self.start_prob.as_ref().unwrap();
        let trans_mat = self.transition_matrix.as_ref().unwrap();

        let (_log_prob, path) = viterbi_algorithm(start_prob, trans_mat, &emission_probs)?;
        Ok(path)
    }

    fn score(&self, observations: &Array2<f64>) -> Result<f64> {
        if !self.is_fitted {
            return Err(HmmError::ModelNotFitted(
                "Model must be fitted before scoring".to_string(),
            ));
        }

        let emission_probs = self.compute_emission_probs(observations)?;
        let start_prob = self.start_prob.as_ref().unwrap();
        let trans_mat = self.transition_matrix.as_ref().unwrap();

        let alpha = forward_algorithm(start_prob, trans_mat, &emission_probs)?;
        let prob = alpha.row(alpha.nrows() - 1).sum();
        Ok(prob.ln())
    }

    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)> {
        if !self.is_fitted {
            return Err(HmmError::ModelNotFitted(
                "Model must be fitted before sampling".to_string(),
            ));
        }

        let mut rng = rand::rng();
        let mut observations = Array2::zeros((n_samples, 1));
        let mut states = Array1::zeros(n_samples);

        let start_prob = self.start_prob.as_ref().unwrap();
        let trans_mat = self.transition_matrix.as_ref().unwrap();
        let emission_prob = self.emission_prob.as_ref().unwrap();

        // Sample initial state from π
        let mut current_state = 0;
        let mut cumsum = 0.0;
        let r: f64 = rng.random();
        for i in 0..self.n_states {
            cumsum += start_prob[i];
            if r < cumsum {
                current_state = i;
                break;
            }
        }
        states[0] = current_state;

        // Sample initial observation from B[current_state, :]
        let r_obs_init: f64 = rng.random();
        let mut cumsum_obs = 0.0;
        for j in 0..self.n_symbols {
            cumsum_obs += emission_prob[[current_state, j]];
            if r_obs_init < cumsum_obs {
                observations[[0, 0]] = j as f64;
                break;
            }
        }

        // Sample remaining states and observations
        for t in 1..n_samples {
            // Sample next state from A[current_state, :]
            let mut next_state = 0;
            let mut cumsum_state = 0.0;
            let r_state: f64 = rng.random();
            for i in 0..self.n_states {
                cumsum_state += trans_mat[[current_state, i]];
                if r_state < cumsum_state {
                    next_state = i;
                    break;
                }
            }
            current_state = next_state;
            states[t] = current_state;

            // Sample observation from B[current_state, :]
            let r_obs: f64 = rng.random();
            let mut cumsum_obs = 0.0;
            for j in 0..self.n_symbols {
                cumsum_obs += emission_prob[[current_state, j]];
                if r_obs < cumsum_obs {
                    observations[[t, 0]] = j as f64;
                    break;
                }
            }
        }

        Ok((observations, states))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_multinomial_hmm_new() {
        let hmm = MultinomialHMM::new(3, 5);
        assert_eq!(hmm.n_states(), 3);
        assert_eq!(hmm.n_symbols(), 5);
        assert_eq!(hmm.n_features(), 5); // n_features returns n_symbols for trait compatibility
        assert!(!hmm.is_fitted());
    }

    #[test]
    fn test_multinomial_hmm_fit() {
        let mut hmm = MultinomialHMM::new(2, 3);
        let observations = array![[0.0], [1.0], [2.0], [0.0], [1.0]];

        assert!(hmm.fit(&observations, None).is_ok());
        assert!(hmm.is_fitted());
    }

    #[test]
    fn test_multinomial_hmm_fit_empty_observations() {
        let mut hmm = MultinomialHMM::new(2, 3);
        let observations = ndarray::Array2::<f64>::zeros((0, 1));
        assert!(hmm.fit(&observations, None).is_err());
    }

    #[test]
    fn test_multinomial_hmm_fit_invalid_observation() {
        let mut hmm = MultinomialHMM::new(2, 3);
        // Observation 5.0 is out of range [0, 3)
        let observations = array![[0.0], [1.0], [5.0]];
        assert!(hmm.fit(&observations, None).is_err());
    }

    #[test]
    fn test_multinomial_hmm_fit_non_integer_observation() {
        let mut hmm = MultinomialHMM::new(2, 3);
        // Observation 1.5 is not an integer
        let observations = array![[0.0], [1.5], [2.0]];
        assert!(hmm.fit(&observations, None).is_err());
    }

    #[test]
    fn test_multinomial_hmm_fit_negative_observation() {
        let mut hmm = MultinomialHMM::new(2, 3);
        // Negative observation
        let observations = array![[0.0], [-1.0], [2.0]];
        assert!(hmm.fit(&observations, None).is_err());
    }

    #[test]
    fn test_multinomial_hmm_predict() {
        let mut hmm = MultinomialHMM::new(2, 2);
        let observations = array![[0.0], [1.0], [0.0]];
        hmm.fit(&observations, None).unwrap();

        let predictions = hmm.predict(&observations).unwrap();
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    fn test_multinomial_hmm_predict_not_fitted() {
        let hmm = MultinomialHMM::new(2, 2);
        let observations = array![[0.0], [1.0]];
        assert!(hmm.predict(&observations).is_err());
    }

    #[test]
    fn test_multinomial_hmm_score() {
        let mut hmm = MultinomialHMM::new(2, 2);
        let observations = array![[0.0], [1.0], [0.0]];
        hmm.fit(&observations, None).unwrap();

        let score = hmm.score(&observations).unwrap();
        assert!(score < 0.0); // Log probability should be negative
    }

    #[test]
    fn test_multinomial_hmm_score_not_fitted() {
        let hmm = MultinomialHMM::new(2, 2);
        let observations = array![[0.0], [1.0]];
        assert!(hmm.score(&observations).is_err());
    }

    #[test]
    fn test_multinomial_hmm_sample() {
        let mut hmm = MultinomialHMM::new(2, 3);
        let observations = array![[0.0], [1.0], [2.0]];
        hmm.fit(&observations, None).unwrap();

        let (sampled_obs, sampled_states) = hmm.sample(10).unwrap();
        assert_eq!(sampled_obs.nrows(), 10);
        assert_eq!(sampled_states.len(), 10);

        // All sampled observations should be valid symbols
        for i in 0..10 {
            let obs = sampled_obs[[i, 0]];
            assert!(obs >= 0.0 && obs < 3.0);
            assert_eq!(obs.fract(), 0.0); // Should be integer
        }

        // All sampled states should be valid
        for &state in sampled_states.iter() {
            assert!(state < 2);
        }
    }

    #[test]
    fn test_multinomial_hmm_sample_not_fitted() {
        let hmm = MultinomialHMM::new(2, 3);
        assert!(hmm.sample(10).is_err());
    }

    #[test]
    fn test_multinomial_hmm_multiple_sequences() {
        let mut hmm = MultinomialHMM::new(2, 3);
        
        // Two sequences concatenated
        let observations = array![
            [0.0], [1.0], [2.0],  // Sequence 1
            [2.0], [1.0], [0.0]   // Sequence 2
        ];
        let lengths = vec![3, 3];

        assert!(hmm.fit(&observations, Some(&lengths)).is_ok());
        assert!(hmm.is_fitted());

        // Verify we can predict and score
        let predictions = hmm.predict(&observations).unwrap();
        assert_eq!(predictions.len(), 6);

        let score = hmm.score(&observations).unwrap();
        assert!(score < 0.0);
    }

    #[test]
    fn test_multinomial_hmm_multiple_sequences_different_lengths() {
        let mut hmm = MultinomialHMM::new(2, 4);
        
        // Three sequences with different lengths
        let observations = array![
            [0.0], [1.0],              // Sequence 1 (length 2)
            [2.0], [3.0], [0.0],       // Sequence 2 (length 3)
            [1.0], [2.0], [3.0], [0.0] // Sequence 3 (length 4)
        ];
        let lengths = vec![2, 3, 4];

        assert!(hmm.fit(&observations, Some(&lengths)).is_ok());
        assert!(hmm.is_fitted());
    }

    #[test]
    fn test_multinomial_hmm_decode() {
        let mut hmm = MultinomialHMM::new(2, 3);
        let observations = array![[0.0], [1.0], [2.0], [0.0]];
        hmm.fit(&observations, None).unwrap();

        let (log_prob, states) = hmm.decode(&observations).unwrap();
        assert!(log_prob < 0.0);
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_multinomial_hmm_getters() {
        let mut hmm = MultinomialHMM::new(2, 3);
        let observations = array![[0.0], [1.0], [2.0]];
        hmm.fit(&observations, None).unwrap();

        // Check that all parameters are available after fitting
        assert!(hmm.start_prob().is_some());
        assert!(hmm.transition_matrix().is_some());
        assert!(hmm.emission_prob().is_some());

        // Check dimensions
        let start_prob = hmm.start_prob().unwrap();
        assert_eq!(start_prob.len(), 2);

        let trans_mat = hmm.transition_matrix().unwrap();
        assert_eq!(trans_mat.shape(), &[2, 2]);

        let emission_prob = hmm.emission_prob().unwrap();
        assert_eq!(emission_prob.shape(), &[2, 3]);
    }

    #[test]
    fn test_multinomial_hmm_emission_probabilities_sum_to_one() {
        let mut hmm = MultinomialHMM::new(2, 4);
        let observations = array![[0.0], [1.0], [2.0], [3.0], [0.0], [1.0]];
        hmm.fit(&observations, None).unwrap();

        let emission_prob = hmm.emission_prob().unwrap();
        
        // Each row should sum to approximately 1
        for i in 0..2 {
            let row_sum: f64 = emission_prob.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multinomial_hmm_transition_probabilities_sum_to_one() {
        let mut hmm = MultinomialHMM::new(3, 4);
        let observations = array![[0.0], [1.0], [2.0], [3.0], [0.0], [1.0]];
        hmm.fit(&observations, None).unwrap();

        let trans_mat = hmm.transition_matrix().unwrap();
        
        // Each row should sum to approximately 1
        for i in 0..3 {
            let row_sum: f64 = trans_mat.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }
}
