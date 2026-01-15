//! HMM model implementations

pub mod beta;
pub mod gaussian;
pub mod gmm;
pub mod multinomial;

pub use beta::BetaHMM;
pub use gaussian::GaussianHMM;
pub use gmm::GMMHMM;
pub use multinomial::MultinomialHMM;
