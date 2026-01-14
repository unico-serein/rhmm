//! Parameter validation utilities

use ndarray::{Array1, Array2};
use crate::errors::{Result, HmmError};

/// Validate that probabilities sum to 1
pub fn validate_probability_vector(probs: &Array1<f64>, name: &str) -> Result<()> {
    let sum: f64 = probs.sum();
    if (sum - 1.0).abs() > 1e-6 {
        return Err(HmmError::InvalidProbability(format!(
            "{} must sum to 1.0, got {}",
            name, sum
        )));
    }

    for &p in probs.iter() {
        if p < 0.0 || p > 1.0 {
            return Err(HmmError::InvalidProbability(format!(
                "{} contains invalid probability: {}",
                name, p
            )));
        }
    }

    Ok(())
}

/// Validate that each row of a matrix sums to 1 (stochastic matrix)
pub fn validate_transition_matrix(matrix: &Array2<f64>) -> Result<()> {
    if matrix.nrows() != matrix.ncols() {
        return Err(HmmError::InvalidParameter(
            "Transition matrix must be square".to_string(),
        ));
    }

    for i in 0..matrix.nrows() {
        let row_sum: f64 = matrix.row(i).sum();
        if (row_sum - 1.0).abs() > 1e-6 {
            return Err(HmmError::InvalidProbability(format!(
                "Row {} of transition matrix must sum to 1.0, got {}",
                i, row_sum
            )));
        }

        for &p in matrix.row(i).iter() {
            if p < 0.0 || p > 1.0 {
                return Err(HmmError::InvalidProbability(format!(
                    "Transition matrix contains invalid probability: {}",
                    p
                )));
            }
        }
    }

    Ok(())
}

/// Validate observation dimensions
pub fn validate_observations(
    observations: &Array2<f64>,
    expected_features: usize,
) -> Result<()> {
    if observations.ncols() != expected_features {
        return Err(HmmError::DimensionMismatch {
            expected: expected_features,
            actual: observations.ncols(),
        });
    }

    if observations.nrows() == 0 {
        return Err(HmmError::InvalidParameter(
            "Observations cannot be empty".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_validate_probability_vector_valid() {
        let probs = array![0.3, 0.5, 0.2];
        assert!(validate_probability_vector(&probs, "test").is_ok());
    }

    #[test]
    fn test_validate_probability_vector_not_sum_to_one() {
        let probs = array![0.3, 0.5, 0.3];
        assert!(validate_probability_vector(&probs, "test").is_err());
    }

    #[test]
    fn test_validate_probability_vector_negative() {
        let probs = array![-0.1, 0.6, 0.5];
        assert!(validate_probability_vector(&probs, "test").is_err());
    }

    #[test]
    fn test_validate_probability_vector_greater_than_one() {
        let probs = array![0.5, 0.5, 1.1];
        assert!(validate_probability_vector(&probs, "test").is_err());
    }

    #[test]
    fn test_validate_transition_matrix_valid() {
        let matrix = array![
            [0.7, 0.3],
            [0.4, 0.6]
        ];
        assert!(validate_transition_matrix(&matrix).is_ok());
    }

    #[test]
    fn test_validate_transition_matrix_not_square() {
        let matrix = array![
            [0.7, 0.3],
            [0.4, 0.6],
            [0.5, 0.5]
        ];
        assert!(validate_transition_matrix(&matrix).is_err());
    }

    #[test]
    fn test_validate_transition_matrix_row_not_sum_to_one() {
        let matrix = array![
            [0.7, 0.2],
            [0.4, 0.6]
        ];
        assert!(validate_transition_matrix(&matrix).is_err());
    }

    #[test]
    fn test_validate_observations_valid() {
        let obs = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        assert!(validate_observations(&obs, 2).is_ok());
    }

    #[test]
    fn test_validate_observations_wrong_features() {
        let obs = array![
            [1.0, 2.0],
            [3.0, 4.0]
        ];
        assert!(validate_observations(&obs, 3).is_err());
    }

    #[test]
    fn test_validate_observations_empty() {
        let obs = Array2::<f64>::zeros((0, 2));
        assert!(validate_observations(&obs, 2).is_err());
    }
}
