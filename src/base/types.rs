//! Common types used throughout the library

use ndarray::{Array1, Array2};

/// Type alias for state transition matrix
pub type TransitionMatrix = Array2<f64>;

/// Type alias for initial state probabilities
pub type InitialProbs = Array1<f64>;

/// Type alias for observation sequences
pub type Observations = Array2<f64>;

/// Type alias for state sequences
pub type States = Array1<usize>;

/// Covariance type for Gaussian models
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum CovarianceType {
    /// Full covariance matrix
    Full,
    /// Diagonal covariance matrix
    #[default]
    Diagonal,
    /// Spherical covariance (single variance value)
    Spherical,
    /// Tied covariance (same for all states)
    Tied,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_type_default() {
        assert_eq!(CovarianceType::default(), CovarianceType::Diagonal);
    }

    #[test]
    fn test_covariance_type_equality() {
        assert_eq!(CovarianceType::Full, CovarianceType::Full);
        assert_ne!(CovarianceType::Full, CovarianceType::Diagonal);
    }

    #[test]
    fn test_type_aliases() {
        let transition: TransitionMatrix = Array2::zeros((3, 3));
        assert_eq!(transition.shape(), &[3, 3]);

        let initial: InitialProbs = Array1::zeros(3);
        assert_eq!(initial.len(), 3);

        let obs: Observations = Array2::zeros((10, 2));
        assert_eq!(obs.shape(), &[10, 2]);

        let states: States = Array1::zeros(10);
        assert_eq!(states.len(), 10);
    }
}
