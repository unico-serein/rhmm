//! Error types for the rhmm library

use thiserror::Error;

/// Result type alias for HMM operations
pub type Result<T> = std::result::Result<T, HmmError>;

/// Error types that can occur in HMM operations
#[derive(Error, Debug)]
pub enum HmmError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid probability: {0}")]
    InvalidProbability(String),

    #[error("Model not fitted: {0}")]
    ModelNotFitted(String),

    #[error("Convergence failed: {0}")]
    ConvergenceError(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invalid_parameter_error() {
        let err = HmmError::InvalidParameter("test parameter".to_string());
        assert_eq!(err.to_string(), "Invalid parameter: test parameter");
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let err = HmmError::DimensionMismatch {
            expected: 3,
            actual: 2,
        };
        assert_eq!(err.to_string(), "Dimension mismatch: expected 3, got 2");
    }

    #[test]
    fn test_invalid_probability_error() {
        let err = HmmError::InvalidProbability("negative value".to_string());
        assert_eq!(err.to_string(), "Invalid probability: negative value");
    }

    #[test]
    fn test_model_not_fitted_error() {
        let err = HmmError::ModelNotFitted("must fit first".to_string());
        assert_eq!(err.to_string(), "Model not fitted: must fit first");
    }

    #[test]
    fn test_convergence_error() {
        let err = HmmError::ConvergenceError("max iterations reached".to_string());
        assert_eq!(
            err.to_string(),
            "Convergence failed: max iterations reached"
        );
    }

    #[test]
    fn test_invalid_state_error() {
        let err = HmmError::InvalidState("state out of bounds".to_string());
        assert_eq!(err.to_string(), "Invalid state: state out of bounds");
    }

    #[test]
    fn test_numerical_error() {
        let err = HmmError::NumericalError("overflow detected".to_string());
        assert_eq!(err.to_string(), "Numerical error: overflow detected");
    }

    #[test]
    fn test_result_type() {
        let ok_result: Result<i32> = Ok(42);
        assert!(ok_result.is_ok());
        assert_eq!(ok_result.unwrap(), 42);

        let err_result: Result<i32> = Err(HmmError::InvalidParameter("test".to_string()));
        assert!(err_result.is_err());
    }

    #[test]
    fn test_error_debug() {
        let err = HmmError::InvalidParameter("test".to_string());
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("InvalidParameter"));
    }
}
