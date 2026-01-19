//! # rhmm - Rust Hidden Markov Models
//!
//! A Rust library for Hidden Markov Models, inspired by Python's hmmlearn.
//!
//! ## Features
//!
//! - Multiple HMM model types (Gaussian, GMM, Multinomial)
//! - Standard HMM algorithms (Forward, Backward, Viterbi, Baum-Welch)
//! - Efficient implementation using ndarray
//!
//! ## Example
//!
//! ```rust,ignore
//! use rhmm::models::GaussianHMM;
//!
//! let hmm = GaussianHMM::new(3);
//! ```

pub mod algorithms;
pub mod base;
pub mod errors;
pub mod models;
pub mod utils;

// Re-export commonly used types
pub use base::HiddenMarkovModel;
pub use errors::{HmmError, Result};
