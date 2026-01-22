//! Sequence handling utilities for multiple sequences

use crate::errors::{HmmError, Result};
use ndarray::{Array2, ArrayView2};

/// Validate lengths parameter
///
/// # Arguments
///
/// * `observations` - Observation array
/// * `lengths` - Sequence lengths
///
/// # Returns
///
/// Result indicating if lengths are valid
pub fn validate_lengths(observations: &Array2<f64>, lengths: &[usize]) -> Result<()> {
    if lengths.is_empty() {
        return Err(HmmError::InvalidParameter(
            "Lengths array cannot be empty".to_string(),
        ));
    }

    let total_length: usize = lengths.iter().sum();
    if total_length != observations.nrows() {
        return Err(HmmError::InvalidParameter(format!(
            "Sum of lengths ({}) does not match number of observations ({})",
            total_length,
            observations.nrows()
        )));
    }

    for &length in lengths {
        if length == 0 {
            return Err(HmmError::InvalidParameter(
                "Sequence length cannot be zero".to_string(),
            ));
        }
    }

    Ok(())
}

/// Split observations into multiple sequences based on lengths
///
/// # Arguments
///
/// * `observations` - Concatenated observation sequences
/// * `lengths` - Length of each sequence
///
/// # Returns
///
/// Vector of observation sequences
pub fn split_sequences<'a>(
    observations: &'a Array2<f64>,
    lengths: &[usize],
) -> Result<Vec<ArrayView2<'a, f64>>> {
    validate_lengths(observations, lengths)?;

    let mut sequences = Vec::with_capacity(lengths.len());
    let mut start = 0;

    for &length in lengths {
        let end = start + length;
        let sequence = observations.slice(ndarray::s![start..end, ..]);
        sequences.push(sequence);
        start = end;
    }

    Ok(sequences)
}

/// Get default lengths (single sequence)
///
/// # Arguments
///
/// * `n_samples` - Total number of samples
///
/// # Returns
///
/// Vector with single length
pub fn default_lengths(n_samples: usize) -> Vec<usize> {
    vec![n_samples]
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_validate_lengths_valid() {
        let obs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let lengths = vec![2, 2];
        assert!(validate_lengths(&obs, &lengths).is_ok());
    }

    #[test]
    fn test_validate_lengths_empty() {
        let obs = array![[1.0, 2.0]];
        let lengths: Vec<usize> = vec![];
        assert!(validate_lengths(&obs, &lengths).is_err());
    }

    #[test]
    fn test_validate_lengths_sum_mismatch() {
        let obs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let lengths = vec![2, 2]; // Sum is 4, but obs has 3 rows
        assert!(validate_lengths(&obs, &lengths).is_err());
    }

    #[test]
    fn test_validate_lengths_zero_length() {
        let obs = array![[1.0, 2.0], [3.0, 4.0]];
        let lengths = vec![0, 2];
        assert!(validate_lengths(&obs, &lengths).is_err());
    }

    #[test]
    fn test_split_sequences_valid() {
        let obs = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let lengths = vec![2, 3];
        let sequences = split_sequences(&obs, &lengths).unwrap();

        assert_eq!(sequences.len(), 2);
        assert_eq!(sequences[0].nrows(), 2);
        assert_eq!(sequences[1].nrows(), 3);
        assert_eq!(sequences[0][[0, 0]], 1.0);
        assert_eq!(sequences[1][[0, 0]], 5.0);
    }

    #[test]
    fn test_split_sequences_single() {
        let obs = array![[1.0, 2.0], [3.0, 4.0]];
        let lengths = vec![2];
        let sequences = split_sequences(&obs, &lengths).unwrap();

        assert_eq!(sequences.len(), 1);
        assert_eq!(sequences[0].nrows(), 2);
    }

    #[test]
    fn test_default_lengths() {
        let lengths = default_lengths(10);
        assert_eq!(lengths, vec![10]);
    }
}
