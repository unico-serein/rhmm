//! Utility functions for HMM operations

pub mod normalization;
pub mod validation;
pub mod sampling;

#[cfg(feature = "polars")]
pub mod polars;

// Re-export commonly used validation functions
pub use validation::{validate_observations, validate_probability_vector, validate_transition_matrix};
pub use normalization::{normalize_vector, normalize_matrix_rows, log_normalize, exp_normalize};
