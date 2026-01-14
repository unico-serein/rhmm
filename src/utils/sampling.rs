//! Sampling utilities

use ndarray::Array1;
use rand::Rng;
use crate::errors::{Result, HmmError};

/// Sample from a discrete distribution
///
/// # Arguments
///
/// * `probs` - Probability distribution
/// * `rng` - Random number generator
///
/// # Returns
///
/// Sampled index
pub fn sample_discrete<R: Rng>(probs: &Array1<f64>, rng: &mut R) -> Result<usize> {
    let sum: f64 = probs.sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(HmmError::InvalidProbability(format!(
            "Probabilities must sum to 1.0, got {}",
            sum
        )));
    }

    let mut cumsum = 0.0;
    let rand_val: f64 = rng.gen();

    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if rand_val <= cumsum {
            return Ok(i);
        }
    }

    // Fallback to last index (handles floating point errors)
    Ok(probs.len() - 1)
}

/// Sample from a multivariate normal distribution
///
/// # Arguments
///
/// * `mean` - Mean vector
/// * `covar` - Covariance (diagonal elements only for now)
/// * `rng` - Random number generator
///
/// # Returns
///
/// Sampled vector
pub fn sample_gaussian<R: Rng>(
    mean: &Array1<f64>,
    covar: &Array1<f64>,
    rng: &mut R,
) -> Result<Array1<f64>> {
    use rand_distr::{Distribution, Normal};

    let n_features = mean.len();
    let mut sample = Array1::zeros(n_features);

    for i in 0..n_features {
        let std_dev = covar[i].sqrt();
        let normal = Normal::new(mean[i], std_dev).map_err(|e| {
            HmmError::InvalidParameter(format!("Invalid normal distribution parameters: {}", e))
        })?;
        sample[i] = normal.sample(rng);
    }

    Ok(sample)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_sample_discrete() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = array![0.5, 0.3, 0.2];
        
        let mut counts = vec![0; 3];
        for _ in 0..1000 {
            let idx = sample_discrete(&probs, &mut rng).unwrap();
            counts[idx] += 1;
        }
        
        // Check that sampling roughly follows the distribution
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
    }

    #[test]
    fn test_sample_discrete_invalid_sum() {
        let mut rng = StdRng::seed_from_u64(42);
        let probs = array![0.5, 0.3, 0.3];
        
        assert!(sample_discrete(&probs, &mut rng).is_err());
    }

    #[test]
    fn test_sample_gaussian() {
        let mut rng = StdRng::seed_from_u64(42);
        let mean = array![0.0, 1.0];
        let covar = array![1.0, 0.5];
        
        let sample = sample_gaussian(&mean, &covar, &mut rng).unwrap();
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_sample_gaussian_multiple() {
        let mut rng = StdRng::seed_from_u64(42);
        let mean = array![5.0];
        let covar = array![1.0];
        
        let mut samples = Vec::new();
        for _ in 0..100 {
            let sample = sample_gaussian(&mean, &covar, &mut rng).unwrap();
            samples.push(sample[0]);
        }
        
        let sample_mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        // Mean should be roughly around 5.0
        assert!((sample_mean - 5.0).abs() < 1.0);
    }
}
