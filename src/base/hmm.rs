//! Core trait and types for Hidden Markov Models

use ndarray::{Array1, Array2};
use crate::errors::Result;

#[cfg(feature = "polars")]
use crate::utils::polars::{series_to_array2, dataframe_to_array2, array1_to_series, array2_to_dataframe};
#[cfg(feature = "polars")]
use polars::prelude::*;

/// Trait for Hidden Markov Model implementations
pub trait HiddenMarkovModel {
    /// Get the number of hidden states
    fn n_states(&self) -> usize;
    
    /// Get the number of features
    fn n_features(&self) -> usize;
    
    /// Train the model on observation data
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence of shape (n_samples, n_features)
    /// * `lengths` - Optional sequence lengths for multiple sequences
    fn fit(&mut self, observations: &Array2<f64>, lengths: Option<&[usize]>) -> Result<()>;
    
    /// Predict the most likely state sequence (Viterbi algorithm)
    fn predict(&self, observations: &Array2<f64>) -> Result<Array1<usize>>;
    
    /// Compute log probability of observations (Forward algorithm)
    fn score(&self, observations: &Array2<f64>) -> Result<f64>;
    
    /// Generate synthetic samples from the model
    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)>;
    
    /// Decode state sequence with probability (Viterbi with probability)
    fn decode(&self, observations: &Array2<f64>) -> Result<(f64, Array1<usize>)> {
        let score = self.score(observations)?;
        let path = self.predict(observations)?;
        Ok((score, path))
    }
    
    /// Train the model using Polars Series (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn fit_from_series(&mut self, series_list: &[Series], lengths: Option<&[usize]>) -> Result<()> {
        let observations = series_to_array2(series_list)?;
        self.fit(&observations, lengths)
    }
    
    /// Train the model using Polars DataFrame (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn fit_from_dataframe(&mut self, df: &DataFrame, columns: Option<&[&str]>, lengths: Option<&[usize]>) -> Result<()> {
        let observations = dataframe_to_array2(df, columns)?;
        self.fit(&observations, lengths)
    }
    
    /// Predict states using Polars Series (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn predict_from_series(&self, series_list: &[Series]) -> Result<Series> {
        let observations = series_to_array2(series_list)?;
        let states = self.predict(&observations)?;
        Ok(array1_to_series(&states, "predicted_states"))
    }
    
    /// Predict states using Polars DataFrame (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn predict_from_dataframe(&self, df: &DataFrame, columns: Option<&[&str]>) -> Result<Series> {
        let observations = dataframe_to_array2(df, columns)?;
        let states = self.predict(&observations)?;
        Ok(array1_to_series(&states, "predicted_states"))
    }
    
    /// Compute log probability using Polars Series (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn score_from_series(&self, series_list: &[Series]) -> Result<f64> {
        let observations = series_to_array2(series_list)?;
        self.score(&observations)
    }
    
    /// Compute log probability using Polars DataFrame (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn score_from_dataframe(&self, df: &DataFrame, columns: Option<&[&str]>) -> Result<f64> {
        let observations = dataframe_to_array2(df, columns)?;
        self.score(&observations)
    }
    
    /// Generate synthetic samples and return as Polars DataFrame (requires "polars" feature)
    #[cfg(feature = "polars")]
    fn sample_to_dataframe(&self, n_samples: usize, column_prefix: &str) -> Result<DataFrame> {
        let (observations, states) = self.sample(n_samples)?;
        let obs_df = array2_to_dataframe(&observations, column_prefix)?;
        let states_series = array1_to_series(&states, "states");
        
        // Combine observations and states
        let mut df = obs_df.clone();
        df.with_column(states_series)
            .map_err(|e| HmmError::InvalidParameter(format!("Failed to add states column: {}", e)))
    }
}