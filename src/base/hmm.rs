//! Core HMM trait definition

use crate::errors::Result;
use ndarray::{Array1, Array2};

/// Core trait that all HMM models must implement
pub trait HiddenMarkovModel {
    /// Get the number of hidden states
    fn n_states(&self) -> usize;

    /// Get the number of features/dimensions
    fn n_features(&self) -> usize;

    /// Fit the model to observed data
    ///
    /// # Arguments
    ///
    /// * `observations` - Training data of shape (n_samples, n_features)
    /// * `lengths` - Optional lengths of sequences if multiple sequences are concatenated
    fn fit(&mut self, observations: &Array2<f64>, lengths: Option<&[usize]>) -> Result<()>;

    /// Predict the most likely state sequence using Viterbi algorithm
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Array of predicted states
    fn predict(&self, observations: &Array2<f64>) -> Result<Array1<usize>>;

    /// Compute the log probability of observations
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Log probability of the observation sequence
    fn score(&self, observations: &Array2<f64>) -> Result<f64>;

    /// Sample from the model
    ///
    /// # Arguments
    ///
    /// * `n_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Tuple of (observations, states)
    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)>;

    /// Decode the most likely state sequence (alias for predict)
    fn decode(&self, observations: &Array2<f64>) -> Result<(f64, Array1<usize>)> {
        let states = self.predict(observations)?;
        let log_prob = self.score(observations)?;
        Ok((log_prob, states))
    }
}
