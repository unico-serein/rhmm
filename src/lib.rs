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

pub mod base;
pub mod models;
pub mod algorithms;
pub mod utils;
pub mod errors;

// Re-export commonly used types
pub use errors::{HmmError, Result};
pub use base::HiddenMarkovModel;
