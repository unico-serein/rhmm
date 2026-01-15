//! Gaussian Hidden Markov Model

use ndarray::{Array1, Array2};
use crate::base::{HiddenMarkovModel, CovarianceType, TransitionMatrix, InitialProbs};
use crate::errors::{Result, HmmError};
use crate::algorithms::{forward_algorithm, backward_algorithm, viterbi_algorithm, compute_gamma};
use crate::utils::{validate_observations, validate_probability_vector, validate_transition_matrix};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Gaussian Hidden Markov Model
///
/// A Hidden Markov Model with Gaussian emission distributions.
#[derive(Debug, Clone)]
pub struct GaussianHMM {
    /// Number of hidden states
    n_states: usize,
    /// Number of features
    n_features: usize,
    /// Type of covariance matrix
    covariance_type: CovarianceType,
    /// Initial state probabilities
    start_prob: Option<InitialProbs>,
    /// State transition matrix
    transition_matrix: Option<TransitionMatrix>,
    /// Means for each state
    means: Option<Array2<f64>>,
    /// Covariances for each state
    covars: Option<Array2<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
}

impl GaussianHMM {
    /// Create a new Gaussian HMM
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            n_features: 0,
            covariance_type: CovarianceType::default(),
            start_prob: None,
            transition_matrix: None,
            means: None,
            covars: None,
            is_fitted: false,
        }
    }

    /// Create a new Gaussian HMM with specified covariance type
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    /// * `covariance_type` - Type of covariance matrix
    pub fn with_covariance_type(n_states: usize, covariance_type: CovarianceType) -> Self {
        Self {
            n_states,
            n_features: 0,
            covariance_type,
            start_prob: None,
            transition_matrix: None,
            means: None,
            covars: None,
            is_fitted: false,
        }
    }

    /// Get the covariance type
    pub fn covariance_type(&self) -> CovarianceType {
        self.covariance_type
    }

    /// Get the means
    pub fn means(&self) -> Option<&Array2<f64>> {
        self.means.as_ref()
    }

    /// Get the covariances
    pub fn covars(&self) -> Option<&Array2<f64>> {
        self.covars.as_ref()
    }

    /// Get the transition matrix
    pub fn transition_matrix(&self) -> Option<&TransitionMatrix> {
        self.transition_matrix.as_ref()
    }

    /// Get the initial state probabilities
    pub fn start_prob(&self) -> Option<&InitialProbs> {
        self.start_prob.as_ref()
    }

    /// Check if the model has been fitted
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Compute emission probabilities for all observations and states
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence of shape (n_samples, n_features)
    ///
    /// # Returns
    ///
    /// Emission probabilities of shape (n_samples, n_states)
    fn compute_emission_probs(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = observations.nrows();
        let mut emission_probs = Array2::zeros((n_samples, self.n_states));

        let means = self.means.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Means not initialized".to_string())
        })?;
        let covars = self.covars.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Covariances not initialized".to_string())
        })?;

        for t in 0..n_samples {
            let obs = observations.row(t);
            for i in 0..self.n_states {
                let mean = means.row(i);
                let covar = covars.row(i);
                emission_probs[[t, i]] = self.gaussian_pdf(&obs, &mean, &covar)?;
            }
        }

        Ok(emission_probs)
    }

    /// Compute Gaussian probability density function
    ///
    /// # Arguments
    ///
    /// * `x` - Observation vector
    /// * `mean` - Mean vector
    /// * `covar` - Covariance (diagonal elements only for diagonal/spherical)
    ///
    /// # Returns
    ///
    /// Probability density value
    fn gaussian_pdf(
        &self,
        x: &ndarray::ArrayView1<f64>,
        mean: &ndarray::ArrayView1<f64>,
        covar: &ndarray::ArrayView1<f64>,
    ) -> Result<f64> {
        let n_features = x.len();
        
        match self.covariance_type {
            CovarianceType::Diagonal => {
                // For diagonal covariance
                let mut log_prob = -0.5 * n_features as f64 * (2.0 * PI).ln();
                let mut sum_log_det = 0.0;
                let mut mahalanobis = 0.0;

                for i in 0..n_features {
                    let var = covar[i].max(1e-10); // Prevent division by zero
                    sum_log_det += var.ln();
                    let diff = x[i] - mean[i];
                    mahalanobis += diff * diff / var;
                }

                log_prob -= 0.5 * sum_log_det;
                log_prob -= 0.5 * mahalanobis;

                Ok(log_prob.exp())
            }
            CovarianceType::Spherical => {
                // For spherical covariance (single variance value)
                let var = covar[0].max(1e-10);
                let log_prob = -0.5 * n_features as f64 * (2.0 * PI * var).ln();
                
                let mut mahalanobis = 0.0;
                for i in 0..n_features {
                    let diff = x[i] - mean[i];
                    mahalanobis += diff * diff;
                }
                mahalanobis /= var;

                Ok((log_prob - 0.5 * mahalanobis).exp())
            }
            CovarianceType::Full | CovarianceType::Tied => {
                // Simplified implementation - treat as diagonal for now
                // Full implementation would require matrix inversion
                self.gaussian_pdf(x, mean, covar)
            }
        }
    }

    /// Initialize parameters using k-means-like approach
    fn initialize_parameters(&mut self, observations: &Array2<f64>) -> Result<()> {
        let n_samples = observations.nrows();
        
        // Initialize means by randomly selecting observations
        let mut rng = rand::thread_rng();
        let mut means = Array2::zeros((self.n_states, self.n_features));
        for i in 0..self.n_states {
            let idx = rng.gen_range(0..n_samples);
            means.row_mut(i).assign(&observations.row(idx));
        }
        self.means = Some(means);

        // Initialize covariances as variance of the data
        let mut covars = Array2::zeros((self.n_states, self.n_features));
        for j in 0..self.n_features {
            let col = observations.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            for i in 0..self.n_states {
                covars[[i, j]] = var.max(1e-3); // Ensure minimum variance
            }
        }
        self.covars = Some(covars);

        Ok(())
    }

    /// Compute xi values (state transition probabilities)
    ///
    /// # Arguments
    ///
    /// * `alpha` - Forward probabilities
    /// * `beta` - Backward probabilities
    /// * `transition_matrix` - State transition matrix
    /// * `emission_probs` - Emission probabilities
    ///
    /// # Returns
    ///
    /// Xi values of shape (n_samples-1, n_states, n_states)
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
                    let val = alpha[[t, i]]
                        * transition_matrix[[i, j]]
                        * emission_probs[[t + 1, j]]
                        * beta[[t + 1, j]];
                    xi_t[[i, j]] = val;
                    sum += val;
                }
            }

            // Normalize
            if sum > 0.0 {
                xi_t /= sum;
            }

            xi.push(xi_t);
        }

        Ok(xi)
    }

    /// Update model parameters based on gamma and xi
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence
    /// * `gamma` - State occupation probabilities
    /// * `xi` - State transition probabilities
    fn update_parameters(
        &mut self,
        observations: &Array2<f64>,
        gamma: &Array2<f64>,
        xi: &[Array2<f64>],
    ) -> Result<()> {
        let n_samples = observations.nrows();
        let n_states = self.n_states;
        let n_features = self.n_features;

        // Update initial state probabilities
        if let Some(ref mut start_prob) = self.start_prob {
            for i in 0..n_states {
                start_prob[i] = gamma[[0, i]];
            }
        }

        // Update transition matrix
        if let Some(ref mut trans_mat) = self.transition_matrix {
            for i in 0..n_states {
                let mut row_sum = 0.0;
                for j in 0..n_states {
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;

                    for t in 0..n_samples - 1 {
                        numerator += xi[t][[i, j]];
                        denominator += gamma[[t, i]];
                    }

                    trans_mat[[i, j]] = if denominator > 0.0 {
                        numerator / denominator
                    } else {
                        1.0 / n_states as f64
                    };
                    row_sum += trans_mat[[i, j]];
                }

                // Normalize 
                if row_sum > 0.0 {
                    for j in 0..n_states {
                        trans_mat[[i, j]] /= row_sum;
                    }
                }
            }
        }

        // Update means
        if let Some(ref mut means) = self.means {
            for i in 0..n_states {
                let gamma_sum: f64 = gamma.column(i).sum();
                
                if gamma_sum > 0.0 {
                    for j in 0..n_features {
                        let mut weighted_sum = 0.0;
                        for t in 0..n_samples {
                            weighted_sum += gamma[[t, i]] * observations[[t, j]];
                        }
                        means[[i, j]] = weighted_sum / gamma_sum;
                    }
                }
            }
        }

        // Update covariances
        if let Some(ref mut covars) = self.covars {
            let means = self.means.as_ref().unwrap();
            
            for i in 0..n_states {
                let gamma_sum: f64 = gamma.column(i).sum();
                
                if gamma_sum > 0.0 {
                    for j in 0..n_features {
                        let mut weighted_var = 0.0;
                        for t in 0..n_samples {
                            let diff = observations[[t, j]] - means[[i, j]];
                            weighted_var += gamma[[t, i]] * diff * diff;
                        }
                        covars[[i, j]] = (weighted_var / gamma_sum).max(1e-6); // Ensure minimum variance
                    }
                }
            }
        }

        Ok(())
    }
}

impl HiddenMarkovModel for GaussianHMM {
    fn n_states(&self) -> usize {
        self.n_states
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn fit(&mut self, observations: &Array2<f64>, _lengths: Option<&[usize]>) -> Result<()> {
        if observations.nrows() == 0 {
            return Err(HmmError::InvalidParameter(
                "Observations cannot be empty".to_string(),
            ));
        }

        self.n_features = observations.ncols();
        
        // Validate observations if n_features was already set
        if self.n_features > 0 {
            validate_observations(observations, self.n_features)?;
        }

        // Initialize parameters if not set
        if self.start_prob.is_none() {
            self.start_prob = Some(Array1::from_elem(self.n_states, 1.0 / self.n_states as f64));
        }
        
        // Validate initial probabilities
        if let Some(ref start_prob) = self.start_prob {
            validate_probability_vector(start_prob, "Initial state probabilities")?;
        }

        if self.transition_matrix.is_none() {
            self.transition_matrix = Some(Array2::from_elem(
                (self.n_states, self.n_states),
                1.0 / self.n_states as f64,
            ));
        }
        
        // Validate transition matrix
        if let Some(ref trans_mat) = self.transition_matrix {
            validate_transition_matrix(trans_mat)?;
        }

        if self.means.is_none() || self.covars.is_none() {
            self.initialize_parameters(observations)?;
        }

        // Baum-Welch algorithm for parameter estimation
        let max_iter = 100;
        let tol = 1e-4;
        let mut prev_log_prob = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // E-step: Compute emission probabilities
            let emission_probs = self.compute_emission_probs(observations)?;

            // Compute forward and backward probabilities
            let start_prob = self.start_prob.as_ref().unwrap();
            let trans_mat = self.transition_matrix.as_ref().unwrap();
            
            let alpha = forward_algorithm(start_prob, trans_mat, &emission_probs)?;
            let beta = backward_algorithm(trans_mat, &emission_probs)?;

            // Compute current log probability
            let log_prob = alpha.row(alpha.nrows() - 1).sum().ln();

            // Check convergence
            if (log_prob - prev_log_prob).abs() < tol {
                break;
            }
            prev_log_prob = log_prob;

            // Compute gamma (state occupation probabilities)
            let gamma = compute_gamma(&alpha, &beta)?;

            // Compute xi (state transition probabilities)
            let xi = Self::compute_xi(&alpha, &beta, trans_mat, &emission_probs)?;

            // M-step: Update parameters
            self.update_parameters(observations, &gamma, &xi)?;
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

        if observations.ncols() != self.n_features {
            return Err(HmmError::DimensionMismatch {
                expected: self.n_features,
                actual: observations.ncols(),
            });
        }

        // Compute emission probabilities
        let emission_probs = self.compute_emission_probs(observations)?;

        // Use Viterbi algorithm to find most likely state sequence
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

        if observations.ncols() != self.n_features {
            return Err(HmmError::DimensionMismatch {
                expected: self.n_features,
                actual: observations.ncols(),
            });
        }

        // Compute emission probabilities
        let emission_probs = self.compute_emission_probs(observations)?;

        // Use forward algorithm to compute log probability
        let start_prob = self.start_prob.as_ref().unwrap();
        let trans_mat = self.transition_matrix.as_ref().unwrap();
        
        let alpha = forward_algorithm(start_prob, trans_mat, &emission_probs)?;
        let prob: f64 = alpha.row(alpha.nrows() - 1).sum();
        
        Ok(prob.ln())
    }

    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)> {
        if !self.is_fitted {
            return Err(HmmError::ModelNotFitted(
                "Model must be fitted before sampling".to_string(),
            ));
        }

        let mut rng = rand::thread_rng();
        let mut observations = Array2::zeros((n_samples, self.n_features));
        let mut states = Array1::zeros(n_samples);

        let start_prob = self.start_prob.as_ref().unwrap();
        let trans_mat = self.transition_matrix.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let covars = self.covars.as_ref().unwrap();

        // Sample initial state
        let mut cumsum = 0.0;
        let r: f64 = rng.gen();
        let mut current_state = 0;
        for i in 0..self.n_states {
            cumsum += start_prob[i];
            if r < cumsum {
                current_state = i;
                break;
            }
        }
        states[0] = current_state;

        // Sample initial observation
        for j in 0..self.n_features {
            let mean = means[[current_state, j]];
            let std = covars[[current_state, j]].sqrt();
            let normal = Normal::new(mean, std).map_err(|e| {
                HmmError::NumericalError(format!("Failed to create normal distribution: {}", e))
            })?;
            observations[[0, j]] = normal.sample(&mut rng);
        }

        // Sample remaining states and observations
        for t in 1..n_samples {
            // Sample next state
            let mut cumsum = 0.0;
            let r: f64 = rng.gen();
            for i in 0..self.n_states {
                cumsum += trans_mat[[current_state, i]];
                if r < cumsum {
                    current_state = i;
                    break;
                }
            }
            states[t] = current_state;

            // Sample observation
            for j in 0..self.n_features {
                let mean = means[[current_state, j]];
                let std = covars[[current_state, j]].sqrt();
                let normal = Normal::new(mean, std).map_err(|e| {
                    HmmError::NumericalError(format!("Failed to create normal distribution: {}", e))
                })?;
                observations[[t, j]] = normal.sample(&mut rng);
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
    fn test_gaussian_hmm_new() {
        let hmm = GaussianHMM::new(3);
        assert_eq!(hmm.n_states(), 3);
        assert_eq!(hmm.n_features(), 0);
        assert!(!hmm.is_fitted());
    }

    #[test]
    fn test_gaussian_hmm_with_covariance_type() {
        let hmm = GaussianHMM::with_covariance_type(3, CovarianceType::Full);
        assert_eq!(hmm.covariance_type(), CovarianceType::Full);
    }

    #[test]
    fn test_gaussian_hmm_fit() {
        let mut hmm = GaussianHMM::new(2);
        let observations = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0]
        ];
        
        assert!(hmm.fit(&observations, None).is_ok());
        assert!(hmm.is_fitted());
        assert_eq!(hmm.n_features(), 2);
    }

    #[test]
    fn test_gaussian_hmm_fit_empty_observations() {
        let mut hmm = GaussianHMM::new(2);
        let observations = Array2::<f64>::zeros((0, 2));
        
        assert!(hmm.fit(&observations, None).is_err());
    }

    #[test]
    fn test_gaussian_hmm_predict_not_fitted() {
        let hmm = GaussianHMM::new(2);
        let observations = array![
            [1.0, 2.0],
            [2.0, 3.0]
        ];
        
        assert!(hmm.predict(&observations).is_err());
    }

    #[test]
    fn test_gaussian_hmm_predict_dimension_mismatch() {
        let mut hmm = GaussianHMM::new(2);
        let train_obs = array![
            [1.0, 2.0],
            [2.0, 3.0]
        ];
        hmm.fit(&train_obs, None).unwrap();
        
        let test_obs = array![
            [1.0, 2.0, 3.0]
        ];
        assert!(hmm.predict(&test_obs).is_err());
    }

    #[test]
    fn test_gaussian_hmm_getters() {
        let mut hmm = GaussianHMM::new(2);
        let observations = array![
            [1.0, 2.0],
            [2.0, 3.0]
        ];
        hmm.fit(&observations, None).unwrap();
        
        assert!(hmm.start_prob().is_some());
        assert!(hmm.transition_matrix().is_some());
        assert!(hmm.means().is_some());
        assert!(hmm.covars().is_some());
    }
}
