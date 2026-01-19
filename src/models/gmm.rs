//! Gaussian Mixture Model Hidden Markov Model

use ndarray::{Array1, Array2, Array3, s};
use crate::base::{HiddenMarkovModel, CovarianceType, TransitionMatrix, InitialProbs};
use crate::errors::{Result, HmmError};
use crate::algorithms::{forward_algorithm, backward_algorithm, viterbi_algorithm, compute_gamma};
use crate::utils::{validate_observations, validate_probability_vector, validate_transition_matrix};
use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Gaussian Mixture Model Hidden Markov Model
///
/// A Hidden Markov Model with Gaussian Mixture Model emission distributions.
/// Each state emits observations from a mixture of Gaussian distributions.
#[derive(Debug, Clone)]
pub struct GMMHMM {
    /// Number of hidden states
    n_states: usize,
    /// Number of features
    n_features: usize,
    /// Number of mixture components per state
    n_mix: usize,
    /// Type of covariance matrix
    covariance_type: CovarianceType,
    /// Initial state probabilities
    start_prob: Option<InitialProbs>,
    /// State transition matrix
    transition_matrix: Option<TransitionMatrix>,
    /// Mixture weights for each state: shape (n_states, n_mix)
    mixture_weights: Option<Array2<f64>>,
    /// Means for each mixture component: shape (n_states, n_mix, n_features)
    means: Option<Array3<f64>>,
    /// Covariances for each mixture component: shape (n_states, n_mix, n_features)
    covars: Option<Array3<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
}

impl GMMHMM {
    /// Create a new GMM HMM
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    /// * `n_mix` - Number of mixture components per state
    pub fn new(n_states: usize, n_mix: usize) -> Self {
        Self {
            n_states,
            n_features: 0,
            n_mix,
            covariance_type: CovarianceType::default(),
            start_prob: None,
            transition_matrix: None,
            mixture_weights: None,
            means: None,
            covars: None,
            is_fitted: false,
        }
    }

    /// Create a new GMM HMM with specified covariance type
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    /// * `n_mix` - Number of mixture components per state
    /// * `covariance_type` - Type of covariance matrix
    pub fn with_covariance_type(n_states: usize, n_mix: usize, covariance_type: CovarianceType) -> Self {
        Self {
            n_states,
            n_features: 0,
            n_mix,
            covariance_type,
            start_prob: None,
            transition_matrix: None,
            mixture_weights: None,
            means: None,
            covars: None,
            is_fitted: false,
        }
    }

    /// Get the number of mixture components
    pub fn n_mix(&self) -> usize {
        self.n_mix
    }

    /// Get the covariance type
    pub fn covariance_type(&self) -> CovarianceType {
        self.covariance_type
    }

    /// Get the mixture weights
    pub fn mixture_weights(&self) -> Option<&Array2<f64>> {
        self.mixture_weights.as_ref()
    }

    /// Get the means
    pub fn means(&self) -> Option<&Array3<f64>> {
        self.means.as_ref()
    }

    /// Get the covariances
    pub fn covars(&self) -> Option<&Array3<f64>> {
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
                let mut log_prob = -0.5 * n_features as f64 * (2.0 * PI).ln();
                let mut sum_log_det = 0.0;
                let mut mahalanobis = 0.0;

                for i in 0..n_features {
                    let var = covar[i].max(1e-10);
                    sum_log_det += var.ln();
                    let diff = x[i] - mean[i];
                    mahalanobis += diff * diff / var;
                }

                log_prob -= 0.5 * sum_log_det;
                log_prob -= 0.5 * mahalanobis;

                Ok(log_prob.exp())
            }
            CovarianceType::Spherical => {
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
                // Treat as diagonal for now
                self.gaussian_pdf(x, mean, covar)
            }
        }
    }

    /// Compute emission probability for a single observation and state
    ///
    /// For GMM: p(x|state) = sum_k(weight_k * N(x|mean_k, covar_k))
    fn compute_emission_prob_single(
        &self,
        obs: &ndarray::ArrayView1<f64>,
        state: usize,
    ) -> Result<f64> {
        let weights = self.mixture_weights.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Mixture weights not initialized".to_string())
        })?;
        let means = self.means.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Means not initialized".to_string())
        })?;
        let covars = self.covars.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Covariances not initialized".to_string())
        })?;

        let mut prob = 0.0;
        for k in 0..self.n_mix {
            let weight = weights[[state, k]];
            let mean = means.slice(s![state, k, ..]);
            let covar = covars.slice(s![state, k, ..]);
            let gaussian_prob = self.gaussian_pdf(&obs, &mean, &covar)?;
            prob += weight * gaussian_prob;
        }

        Ok(prob.max(1e-300)) // Prevent underflow
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

        for t in 0..n_samples {
            let obs = observations.row(t);
            for i in 0..self.n_states {
                emission_probs[[t, i]] = self.compute_emission_prob_single(&obs, i)?;
            }
        }

        Ok(emission_probs)
    }

    /// Initialize parameters using k-means-like approach
    fn initialize_parameters(&mut self, observations: &Array2<f64>) -> Result<()> {
        let n_samples = observations.nrows();
        let mut rng = rand::thread_rng();
        
        // Initialize mixture weights uniformly
        let mixture_weights = Array2::from_elem((self.n_states, self.n_mix), 1.0 / self.n_mix as f64);
        
        // Initialize means by randomly selecting observations
        let mut means = Array3::zeros((self.n_states, self.n_mix, self.n_features));
        for i in 0..self.n_states {
            for k in 0..self.n_mix {
                let idx = rng.gen_range(0..n_samples);
                for j in 0..self.n_features {
                    means[[i, k, j]] = observations[[idx, j]];
                }
            }
        }
        
        // Initialize covariances as variance of the data
        let mut covars = Array3::zeros((self.n_states, self.n_mix, self.n_features));
        for j in 0..self.n_features {
            let col = observations.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            let var = var.max(1e-3); // Ensure minimum variance
            
            for i in 0..self.n_states {
                for k in 0..self.n_mix {
                    covars[[i, k, j]] = var;
                }
            }
        }
        
        self.mixture_weights = Some(mixture_weights);
        self.means = Some(means);
        self.covars = Some(covars);
        
        Ok(())
    }

    /// Compute component responsibilities (posterior probabilities)
    ///
    /// gamma_ik = weight_k * N(x|mean_k, covar_k) / sum_j(weight_j * N(x|mean_j, covar_j))
    fn compute_component_responsibilities(
        &self,
        observations: &Array2<f64>,
        state_posteriors: &Array2<f64>,
    ) -> Result<Array3<f64>> {
        let n_samples = observations.nrows();
        let mut responsibilities = Array3::zeros((n_samples, self.n_states, self.n_mix));

        let weights = self.mixture_weights.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let covars = self.covars.as_ref().unwrap();

        for t in 0..n_samples {
            let obs = observations.row(t);
            for i in 0..self.n_states {
                let mut component_probs = Vec::with_capacity(self.n_mix);
                let mut sum = 0.0;

                // Compute weighted probabilities for each component
                for k in 0..self.n_mix {
                    let weight = weights[[i, k]];
                    let mean = means.slice(s![i, k, ..]);
                    let covar = covars.slice(s![i, k, ..]);
                    let prob = self.gaussian_pdf(&obs, &mean, &covar)?;
                    let weighted_prob = weight * prob;
                    component_probs.push(weighted_prob);
                    sum += weighted_prob;
                }

                // Normalize to get responsibilities
                if sum > 1e-10 {
                    for k in 0..self.n_mix {
                        responsibilities[[t, i, k]] = state_posteriors[[t, i]] * component_probs[k] / sum;
                    }
                } else {
                    // Uniform distribution if sum is too small
                    for k in 0..self.n_mix {
                        responsibilities[[t, i, k]] = state_posteriors[[t, i]] / self.n_mix as f64;
                    }
                }
            }
        }

        Ok(responsibilities)
    }

    /// Update GMM parameters based on responsibilities
    fn update_gmm_parameters(
        &mut self,
        observations: &Array2<f64>,
        responsibilities: &Array3<f64>,
    ) -> Result<()> {
        let n_samples = observations.nrows();
        
        let weights = self.mixture_weights.as_mut().unwrap();
        let means = self.means.as_mut().unwrap();
        let covars = self.covars.as_mut().unwrap();

        for i in 0..self.n_states {
            for k in 0..self.n_mix {
                // Compute sum of responsibilities for this component
                let mut resp_sum = 0.0;
                for t in 0..n_samples {
                    resp_sum += responsibilities[[t, i, k]];
                }

                if resp_sum > 1e-10 {
                    // Update mixture weight
                    let state_resp_sum: f64 = (0..self.n_mix)
                        .map(|kk| (0..n_samples).map(|t| responsibilities[[t, i, kk]]).sum::<f64>())
                        .sum();
                    weights[[i, k]] = resp_sum / state_resp_sum.max(1e-10);

                    // Update mean
                    for j in 0..self.n_features {
                        let mut weighted_sum = 0.0;
                        for t in 0..n_samples {
                            weighted_sum += responsibilities[[t, i, k]] * observations[[t, j]];
                        }
                        means[[i, k, j]] = weighted_sum / resp_sum;
                    }

                    // Update covariance
                    for j in 0..self.n_features {
                        let mut weighted_var = 0.0;
                        for t in 0..n_samples {
                            let diff = observations[[t, j]] - means[[i, k, j]];
                            weighted_var += responsibilities[[t, i, k]] * diff * diff;
                        }
                        covars[[i, k, j]] = (weighted_var / resp_sum).max(1e-6);
                    }
                }
            }

            // Normalize mixture weights for this state
            let weight_sum: f64 = (0..self.n_mix).map(|k| weights[[i, k]]).sum();
            if weight_sum > 1e-10 {
                for k in 0..self.n_mix {
                    weights[[i, k]] /= weight_sum;
                }
            }
        }

        Ok(())
    }

    /// Update HMM parameters (start_prob and transition_matrix)
    fn update_hmm_parameters(
        &mut self,
        gamma: &Array2<f64>,
        xi: &[Array2<f64>],
    ) -> Result<()> {
        let n_samples = gamma.nrows();
        let n_states = self.n_states;

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

                    trans_mat[[i, j]] = if denominator > 1e-10 {
                        numerator / denominator
                    } else {
                        1.0 / n_states as f64
                    };
                    row_sum += trans_mat[[i, j]];
                }

                // Normalize row
                if row_sum > 1e-10 {
                    for j in 0..n_states {
                        trans_mat[[i, j]] /= row_sum;
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute xi values (state transition probabilities)
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
            if sum > 1e-10 {
                xi_t /= sum;
            } else {
                // Uniform distribution if sum is too small
                xi_t.fill(1.0 / (n_states * n_states) as f64);
            }

            xi.push(xi_t);
        }

        Ok(xi)
    }
}

impl HiddenMarkovModel for GMMHMM {
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
        
        // Validate observations
        if self.n_features > 0 {
            validate_observations(observations, self.n_features)?;
        }

        // Initialize HMM parameters if not set
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

        // Initialize GMM parameters if not set
        if self.mixture_weights.is_none() || self.means.is_none() || self.covars.is_none() {
            self.initialize_parameters(observations)?;
        }

        // Baum-Welch algorithm for GMM-HMM
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

            // M-step: Update HMM parameters
            self.update_hmm_parameters(&gamma, &xi)?;

            // Compute component responsibilities
            let responsibilities = self.compute_component_responsibilities(observations, &gamma)?;

            // Update GMM parameters
            self.update_gmm_parameters(observations, &responsibilities)?;
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

        // Use Viterbi algorithm
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

        // Use forward algorithm
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
        let weights = self.mixture_weights.as_ref().unwrap();
        let means = self.means.as_ref().unwrap();
        let covars = self.covars.as_ref().unwrap();

        // Sample initial state
        let mut cumsum = 0.0;
        let r: f64 = rng.gen();
        let mut current_state = 0;
        for (i, &p) in start_prob.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                current_state = i;
                break;
            }
        }
        states[0] = current_state;

        // Sample initial observation from GMM
        self.sample_from_gmm(current_state, &mut observations.row_mut(0), &mut rng, weights, means, covars)?;

        // Sample remaining states and observations
        for t in 1..n_samples {
            // Sample next state
            let mut cumsum = 0.0;
            let r: f64 = rng.gen();
            for (j, &p) in trans_mat.row(current_state).iter().enumerate() {
                cumsum += p;
                if r < cumsum {
                    current_state = j;
                    break;
                }
            }
            states[t] = current_state;

            // Sample observation from GMM
            self.sample_from_gmm(current_state, &mut observations.row_mut(t), &mut rng, weights, means, covars)?;
        }

        Ok((observations, states))
    }
}

impl GMMHMM {
    /// Sample an observation from the GMM for a given state
    fn sample_from_gmm(
        &self,
        state: usize,
        obs: &mut ndarray::ArrayViewMut1<f64>,
        rng: &mut impl Rng,
        weights: &Array2<f64>,
        means: &Array3<f64>,
        covars: &Array3<f64>,
    ) -> Result<()> {
        // Sample mixture component
        let mut cumsum = 0.0;
        let r: f64 = rng.gen();
        let mut component = 0;
        for k in 0..self.n_mix {
            cumsum += weights[[state, k]];
            if r < cumsum {
                component = k;
                break;
            }
        }

        // Sample from the selected Gaussian component
        for j in 0..self.n_features {
            let mean = means[[state, component, j]];
            let std = covars[[state, component, j]].sqrt();
            let normal = Normal::new(mean, std).map_err(|e| {
                HmmError::InvalidParameter(format!("Failed to create normal distribution: {}", e))
            })?;
            obs[j] = normal.sample(rng);
        }

        Ok(())
    }
}
