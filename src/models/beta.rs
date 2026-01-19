//! Beta Hidden Markov Model

use ndarray::{Array1, Array2, Axis};
use crate::base::{HiddenMarkovModel, TransitionMatrix, InitialProbs};
use crate::errors::{Result, HmmError};
use crate::algorithms::{forward_algorithm, backward_algorithm, viterbi_algorithm, compute_gamma};
use crate::utils::{validate_observations, validate_probability_vector, validate_transition_matrix};
use rand::Rng;
use std::f64::consts::PI;

/// Beta Hidden Markov Model
///
/// A Hidden Markov Model with Beta emission distributions.
/// Beta distribution is suitable for modeling data in the range [0, 1],
/// such as proportions, probabilities, or rates.
///
/// The Beta distribution is parameterized by two shape parameters α (alpha) and β (beta):
/// - Mean = α / (α + β)
/// - Variance = (α * β) / ((α + β)² * (α + β + 1))
#[derive(Debug, Clone)]
pub struct BetaHMM {
    /// Number of hidden states
    n_states: usize,
    /// Number of features (dimensions)
    n_features: usize,
    /// Initial state probabilities
    start_prob: Option<InitialProbs>,
    /// State transition matrix
    transition_matrix: Option<TransitionMatrix>,
    /// Alpha parameters for each state and feature (shape parameter 1)
    alphas: Option<Array2<f64>>,
    /// Beta parameters for each state and feature (shape parameter 2)
    betas: Option<Array2<f64>>,
    /// Whether the model has been fitted
    is_fitted: bool,
}

impl BetaHMM {
    /// Create a new Beta HMM
    ///
    /// # Arguments
    ///
    /// * `n_states` - Number of hidden states
    ///
    /// # Example
    ///
    /// ```
    /// use rhmm::models::BetaHMM;
    /// let model = BetaHMM::new(3);
    /// ```
    pub fn new(n_states: usize) -> Self {
        Self {
            n_states,
            n_features: 0,
            start_prob: None,
            transition_matrix: None,
            alphas: None,
            betas: None,
            is_fitted: false,
        }
    }

    /// Get the alpha parameters
    pub fn alphas(&self) -> Option<&Array2<f64>> {
        self.alphas.as_ref()
    }

    /// Get the beta parameters
    pub fn betas(&self) -> Option<&Array2<f64>> {
        self.betas.as_ref()
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

    /// Compute the mean of Beta distribution
    ///
    /// Mean = α / (α + β)
    pub fn compute_means(&self) -> Option<Array2<f64>> {
        if let (Some(alphas), Some(betas)) = (&self.alphas, &self.betas) {
            Some(alphas / (alphas + betas))
        } else {
            None
        }
    }

    /// Compute the variance of Beta distribution
    ///
    /// Variance = (α * β) / ((α + β)² * (α + β + 1))
    pub fn compute_variances(&self) -> Option<Array2<f64>> {
        if let (Some(alphas), Some(betas)) = (&self.alphas, &self.betas) {
            let sum = alphas + betas;
            let numerator = alphas * betas;
            let denominator = &sum * &sum * &(&sum + 1.0);
            Some(numerator / denominator)
        } else {
            None
        }
    }

    /// Compute emission probabilities for all observations and states
    ///
    /// # Arguments
    ///
    /// * `observations` - Observation sequence of shape (n_samples, n_features)
    ///   Values must be in the range (0, 1)
    ///
    /// # Returns
    ///
    /// Emission probabilities of shape (n_samples, n_states)
    fn compute_emission_probs(&self, observations: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = observations.nrows();
        let mut emission_probs = Array2::zeros((n_samples, self.n_states));

        let alphas = self.alphas.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Alpha parameters not initialized".to_string())
        })?;
        let betas = self.betas.as_ref().ok_or_else(|| {
            HmmError::ModelNotFitted("Beta parameters not initialized".to_string())
        })?;

        // Validate observations are in (0, 1)
        for val in observations.iter() {
            if *val <= 0.0 || *val >= 1.0 {
                return Err(HmmError::InvalidParameter(
                    format!("Beta HMM requires observations in range (0, 1), got {}", val)
                ));
            }
        }

        for t in 0..n_samples {
            let obs = observations.row(t);
            for i in 0..self.n_states {
                let alpha_params = alphas.row(i);
                let beta_params = betas.row(i);
                emission_probs[[t, i]] = self.beta_pdf(&obs, &alpha_params, &beta_params)?;
            }
        }

        Ok(emission_probs)
    }

    /// Compute Beta probability density function
    ///
    /// PDF(x; α, β) = [x^(α-1) * (1-x)^(β-1)] / B(α, β)
    /// where B(α, β) is the Beta function
    ///
    /// # Arguments
    ///
    /// * `x` - Observation vector (values in (0, 1))
    /// * `alpha` - Alpha parameters
    /// * `beta` - Beta parameters
    ///
    /// # Returns
    ///
    /// Probability density value (product over all features)
    fn beta_pdf(
        &self,
        x: &ndarray::ArrayView1<f64>,
        alpha: &ndarray::ArrayView1<f64>,
        beta: &ndarray::ArrayView1<f64>,
    ) -> Result<f64> {
        let n_features = x.len();
        let mut log_prob = 0.0;

        for i in 0..n_features {
            let a = alpha[i].max(1e-6); // Ensure positive parameters
            let b = beta[i].max(1e-6);
            let xi = x[i].max(1e-10).min(1.0 - 1e-10); // Clamp to (0, 1)

            // Compute log PDF to avoid numerical issues
            // log PDF = (α-1)*log(x) + (β-1)*log(1-x) - log B(α, β)
            // log B(α, β) = log Γ(α) + log Γ(β) - log Γ(α+β)
            let log_beta_func = Self::log_gamma(a) + Self::log_gamma(b) - Self::log_gamma(a + b);
            let log_pdf = (a - 1.0) * xi.ln() + (b - 1.0) * (1.0 - xi).ln() - log_beta_func;
            
            log_prob += log_pdf;
        }

        Ok(log_prob.exp())
    }

    /// Compute log of Gamma function using Stirling's approximation
    ///
    /// For x > 0: log Γ(x) ≈ (x - 0.5) * log(x) - x + 0.5 * log(2π)
    fn log_gamma(x: f64) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        // Use Stirling's approximation for large x
        if x > 10.0 {
            (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI).ln()
        } else {
            // For small x, use a more accurate approximation
            // This is a simplified version; for production, use a proper gamma function library
            let mut result = 0.0;
            let mut z = x;
            
            // Shift to larger value using Γ(x+1) = x * Γ(x)
            while z < 10.0 {
                result -= z.ln();
                z += 1.0;
            }
            
            result + (z - 0.5) * z.ln() - z + 0.5 * (2.0 * PI).ln()
        }
    }

    /// Initialize parameters using method of moments
    fn initialize_parameters(&mut self, observations: &Array2<f64>) -> Result<()> {
        let n_samples = observations.nrows();
        let mut rng = rand::rng();
        
        // Initialize alpha and beta parameters
        let mut alphas = Array2::zeros((self.n_states, self.n_features));
        let mut betas = Array2::zeros((self.n_states, self.n_features));
        
        for i in 0..self.n_states {
            for j in 0..self.n_features {
                // Randomly assign observations to states for initialization
                let mut state_obs = Vec::new();
                for t in 0..n_samples {
                    if rng.random::<f64>() < 1.0 / self.n_states as f64 {
                        state_obs.push(observations[[t, j]]);
                    }
                }
                
                if state_obs.is_empty() {
                    // Fallback: use overall statistics
                    let col = observations.column(j);
                    let mean = col.mean().unwrap_or(0.5);
                    let var = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
                    let (a, b) = Self::moments_to_params(mean, var);
                    alphas[[i, j]] = a;
                    betas[[i, j]] = b;
                } else {
                    // Use method of moments for this state
                    let mean = state_obs.iter().sum::<f64>() / state_obs.len() as f64;
                    let var = state_obs.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() 
                        / state_obs.len() as f64;
                    let (a, b) = Self::moments_to_params(mean, var);
                    alphas[[i, j]] = a;
                    betas[[i, j]] = b;
                }
            }
        }
        
        self.alphas = Some(alphas);
        self.betas = Some(betas);
        
        Ok(())
    }

    /// Convert mean and variance to Beta distribution parameters
    ///
    /// Given mean μ and variance σ²:
    /// α = μ * ((μ * (1 - μ) / σ²) - 1)
    /// β = (1 - μ) * ((μ * (1 - μ) / σ²) - 1)
    fn moments_to_params(mean: f64, var: f64) -> (f64, f64) {
        let mean = mean.max(0.01).min(0.99); // Clamp to valid range
        let var = var.max(1e-6).min(mean * (1.0 - mean) * 0.99); // Ensure valid variance
        
        let common = (mean * (1.0 - mean) / var) - 1.0;
        let alpha = (mean * common).max(0.5); // Ensure minimum value
        let beta = ((1.0 - mean) * common).max(0.5);
        
        (alpha, beta)
    }

    /// Compute xi values (state transition probabilities) - optimized version
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
            // Vectorized computation using broadcasting
            let alpha_t = alpha.row(t).insert_axis(Axis(1)); // (n_states, 1)
            let beta_emission = &beta.row(t + 1) * &emission_probs.row(t + 1); // (n_states,)
            let beta_emission = beta_emission.insert_axis(Axis(0)); // (1, n_states)
            
            // xi_t[i,j] = alpha[t,i] * trans[i,j] * emission[t+1,j] * beta[t+1,j]
            let mut xi_t = &alpha_t * transition_matrix * &beta_emission;
            
            // Numerical stable normalization
            let sum = xi_t.sum();
            if sum > 1e-10 {
                xi_t /= sum;
            } else {
                // Handle numerical underflow: use uniform distribution
                xi_t.fill(1.0 / (n_states * n_states) as f64);
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

        // Update Beta parameters using weighted method of moments
        if let (Some(ref mut alphas), Some(ref mut betas)) = (&mut self.alphas, &mut self.betas) {
            for i in 0..n_states {
                let gamma_sum: f64 = gamma.column(i).sum();
                
                if gamma_sum > 1e-10 {
                    for j in 0..n_features {
                        // Compute weighted mean
                        let mut weighted_mean = 0.0;
                        for t in 0..n_samples {
                            weighted_mean += gamma[[t, i]] * observations[[t, j]];
                        }
                        weighted_mean /= gamma_sum;
                        
                        // Compute weighted variance
                        let mut weighted_var = 0.0;
                        for t in 0..n_samples {
                            let diff = observations[[t, j]] - weighted_mean;
                            weighted_var += gamma[[t, i]] * diff * diff;
                        }
                        weighted_var /= gamma_sum;
                        
                        // Convert to Beta parameters
                        let (alpha, beta) = Self::moments_to_params(weighted_mean, weighted_var);
                        alphas[[i, j]] = alpha;
                        betas[[i, j]] = beta;
                    }
                }
            }
        }

        Ok(())
    }
}

impl HiddenMarkovModel for BetaHMM {
    fn n_states(&self) -> usize {
        self.n_states
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn fit(&mut self, observations: &Array2<f64>, _lengths: Option<&[usize]>) -> Result<()> {
        if observations.nrows() == 0 || observations.ncols() == 0 {
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

        if self.alphas.is_none() || self.betas.is_none() {
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
        
        // Log probability is log of sum of final alpha values
        Ok(alpha.row(alpha.nrows() - 1).sum().ln())
    }

    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)> {
        if !self.is_fitted {
            return Err(HmmError::ModelNotFitted(
                "Model must be fitted before sampling".to_string(),
            ));
        }

        use rand_distr::{Distribution, Beta as BetaDist};
        let mut rng = rand::rng();
        
        let mut observations = Array2::zeros((n_samples, self.n_features));
        let mut states = Array1::zeros(n_samples);

        let start_prob = self.start_prob.as_ref().unwrap();
        let trans_mat = self.transition_matrix.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        let betas = self.betas.as_ref().unwrap();

        // Sample initial state
        let mut cumsum = 0.0;
        let r: f64 = rng.random();
        let mut current_state = 0;
        for i in 0..self.n_states {
            cumsum += start_prob[i];
            if r < cumsum {
                current_state = i;
                break;
            }
        }
        states[0] = current_state;

        // Sample initial observation from Beta distribution
        for j in 0..self.n_features {
            let alpha = alphas[[current_state, j]];
            let beta = betas[[current_state, j]];
            let beta_dist = BetaDist::new(alpha, beta).map_err(|e| {
                HmmError::NumericalError(format!("Failed to create Beta distribution: {}", e))
            })?;
            observations[[0, j]] = beta_dist.sample(&mut rng);
        }

        // Sample remaining states and observations
        for t in 1..n_samples {
            // Sample next state based on transition probabilities
            let mut cumsum = 0.0;
            let r: f64 = rng.random();
            for i in 0..self.n_states {
                cumsum += trans_mat[[current_state, i]];
                if r < cumsum {
                    current_state = i;
                    break;
                }
            }
            states[t] = current_state;

            // Sample observation from Beta distribution for current state
            for j in 0..self.n_features {
                let alpha = alphas[[current_state, j]];
                let beta = betas[[current_state, j]];
                let beta_dist = BetaDist::new(alpha, beta).map_err(|e| {
                    HmmError::NumericalError(format!("Failed to create Beta distribution: {}", e))
                })?;
                observations[[t, j]] = beta_dist.sample(&mut rng);
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
    fn test_beta_hmm_creation() {
        let model = BetaHMM::new(3);
        assert_eq!(model.n_states(), 3);
        assert_eq!(model.n_features(), 0);
        assert!(!model.is_fitted());
    }

    #[test]
    fn test_moments_to_params() {
        let (alpha, beta) = BetaHMM::moments_to_params(0.5, 0.05);
        assert!(alpha > 0.0);
        assert!(beta > 0.0);
        
        // Check that mean is approximately correct
        let mean = alpha / (alpha + beta);
        assert!((mean - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_log_gamma() {
        // Test some known values
        let log_gamma_1 = BetaHMM::log_gamma(1.0);
        assert!((log_gamma_1 - 0.0).abs() < 0.1); // Γ(1) = 1, log(1) = 0
        
        let log_gamma_2 = BetaHMM::log_gamma(2.0);
        assert!((log_gamma_2 - 0.0).abs() < 0.1); // Γ(2) = 1, log(1) = 0
    }

    #[test]
    fn test_beta_pdf() {
        let model = BetaHMM::new(2);
        let x = array![0.3, 0.7];
        let alpha = array![2.0, 3.0];
        let beta = array![5.0, 2.0];
        
        let pdf = model.beta_pdf(&x.view(), &alpha.view(), &beta.view());
        assert!(pdf.is_ok());
        let pdf_val = pdf.unwrap();
        assert!(pdf_val > 0.0);
        assert!(pdf_val.is_finite());
    }

    #[test]
    fn test_compute_means_and_variances() {
        let mut model = BetaHMM::new(2);
        model.alphas = Some(array![[2.0, 3.0], [5.0, 4.0]]);
        model.betas = Some(array![[3.0, 2.0], [5.0, 6.0]]);
        
        let means = model.compute_means().unwrap();
        assert_eq!(means.shape(), &[2, 2]);
        assert!((means[[0, 0]] - 0.4).abs() < 0.01); // 2/(2+3) = 0.4
        
        let vars = model.compute_variances().unwrap();
        assert_eq!(vars.shape(), &[2, 2]);
        assert!(vars[[0, 0]] > 0.0);
    }
}
