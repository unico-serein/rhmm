//! Multinomial Hidden Markov Model

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
/// Suitable for discrete observation sequences.
#[derive(Debug, Clone)]
pub struct MultinomialHMM {
    /// Number of hidden states
    n_states: usize,
    /// Number of possible discrete observations (vocabulary size)
    n_features: usize,
    /// Initial state probabilities
    start_prob: Option<InitialProbs>,
    /// State transition matrix
    transition_matrix: Option<TransitionMatrix>,
    /// Emission probabilities (n_states, n_features)
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
    /// * `n_features` - Number of possible discrete observations
    pub fn new(n_states: usize, n_features: usize) -> Self {
        Self {
            n_states,
            n_features,
            start_prob: None,
            transition_matrix: None,
            emission_prob: None,
            is_fitted: false,
        }
    }

    /// Get the initial state probabilities
    pub fn start_prob(&self) -> Option<&InitialProbs> {
        self.start_prob.as_ref()
    }

    /// Get the transition matrix
    pub fn transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.transition_matrix.as_ref()
    }

    /// Get the emission probabilities
    pub fn emission_prob(&self) -> Option<&Array2<f64>> {
        self.emission_prob.as_ref()
    }

    /// Check if the model has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Compute emission probabilities for all observations and states
    fn compute_emission_probs(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = observations.nrows();
        let mut emission_probs = Array2::zeros((n_samples, self.n_states));

        let ep = self.emission_prob.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Emission probabilities not initialized".to_string())
        })?;

        for t in 0..n_samples {
            // Observations are expected to be single discrete values per sample
            // but the trait uses Array2<f64> (n_samples, n_features_per_sample)
            // For Multinomial, we expect n_features_per_sample to be 1, or we take the first column
            let obs_val = observations[[t, 0]] as usize;
            if obs_val >= self.n_features {
                return Err(HmmError::InvalidParameter(format!(
                    "Observation index {} exceeds n_features {}",
                    obs_val, self.n_features
                )));
            }

            for i in 0..self.n_states {
                emission_probs[[t, i]] = ep[[i, obs_val]];
            }
        }

        Ok(emission_probs)
    }

    /// Initialize parameters if they are not set
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
                (self.n_states, self.n_features),
                1.0 / self.n_features as f64,
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
    fn update_emission_parameters(
        &mut self,
        observations: &Array2<f64>,
        gamma: &Array2<f64>,
    ) -> Result<()> {
        let n_samples = observations.nrows();
        let mut emission_prob = Array2::zeros((self.n_states, self.n_features));

        for i in 0..self.n_states {
            let mut denom = 0.0;
            for t in 0..n_samples {
                let obs_val = observations[[t, 0]] as usize;
                emission_prob[[i, obs_val]] += gamma[[t, i]];
                denom += gamma[[t, i]];
            }

            if denom > 0.0 {
                for j in 0..self.n_features {
                    emission_prob[[i, j]] /= denom;
                }
            } else {
                // If a state is never visited, set uniform emission probabilities
                for j in 0..self.n_features {
                    emission_prob[[i, j]] = 1.0 / self.n_features as f64;
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

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn fit(&mut self, observations: &Array2<f64>, lengths: Option<&[usize]>) -> Result<()> {
        if observations.nrows() == 0 || observations.ncols() == 0 {
            return Err(HmmError::InvalidParameter(
                "Observations cannot be empty".to_string(),
            ));
        }

        // Validate observations (for Multinomial, we check if they are within [0, n_features))
        for t in 0..observations.nrows() {
            let obs_val = observations[[t, 0]];
            if obs_val < 0.0 || obs_val >= self.n_features as f64 || obs_val.fract() != 0.0 {
                return Err(HmmError::InvalidParameter(format!(
                    "Invalid observation at row {}: {}. Must be integer in [0, {})",
                    t, obs_val, self.n_features
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

        // Sample initial state
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

        // Sample initial observation
        let r_obs_init: f64 = rng.random();
        let mut cumsum_obs = 0.0;
        for j in 0..self.n_features {
            cumsum_obs += emission_prob[[current_state, j]];
            if r_obs_init < cumsum_obs {
                observations[[0, 0]] = j as f64;
                break;
            }
        }

        // Sample remaining states and observations
        for t in 1..n_samples {
            // Sample next state
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

            // Sample observation
            let r_obs: f64 = rng.random();
            let mut cumsum_obs = 0.0;
            for j in 0..self.n_features {
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
        assert_eq!(hmm.n_features(), 5);
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
    fn test_multinomial_hmm_predict() {
        let mut hmm = MultinomialHMM::new(2, 2);
        let observations = array![[0.0], [1.0], [0.0]];
        hmm.fit(&observations, None).unwrap();

        let predictions = hmm.predict(&observations).unwrap();
        assert_eq!(predictions.len(), 3);
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
    fn test_multinomial_hmm_sample() {
        let mut hmm = MultinomialHMM::new(2, 3);
        let observations = array![[0.0], [1.0], [2.0]];
        hmm.fit(&observations, None).unwrap();

        let (sampled_obs, sampled_states) = hmm.sample(10).unwrap();
        assert_eq!(sampled_obs.nrows(), 10);
        assert_eq!(sampled_states.len(), 10);

        for i in 0..10 {
            assert!(sampled_obs[[i, 0]] >= 0.0 && sampled_obs[[i, 0]] < 3.0);
        }
    }
}
