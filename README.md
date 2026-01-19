# rhmm - Rust Hidden Markov Models

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust library for Hidden Markov Models (HMM), inspired by Python's hmmlearn. This library provides efficient implementations of various HMM models and algorithms using `ndarray` for numerical computations.

## 🚀 Features

- **Multiple HMM Model Types**
  - **Gaussian HMM**: Models continuous data with Gaussian emission distributions
  - **Beta HMM**: Models data in the range [0, 1] (e.g., conversion rates, proportions)
  - **Multinomial HMM(Not implemented yet, coming soon)**: Models discrete categorical data
  - **Gaussian Mixture Model HMM (GMM-HMM) (Not implemented yet, coming soon)**: Models complex continuous distributions

- **Standard HMM Algorithms**
  - **Forward Algorithm**: Compute observation probabilities
  - **Backward Algorithm**: Compute backward probabilities
  - **Viterbi Algorithm**: Find the most likely state sequence
  - **Baum-Welch Algorithm**: Train model parameters using EM

- **Efficient Implementation**
  - Built on `ndarray` for fast numerical operations
  - Support for multiple covariance types (diagonal, spherical, full, tied)
  - Robust numerical stability with log-space computations

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rhmm = "0.0.1"
```

Or install from source:

```bash
git clone https://github.com/yourusername/rhmm.git
cd rhmm
cargo build --release
```

## 🔧 Dependencies

- `ndarray` - N-dimensional arrays
- `ndarray-linalg` - Linear algebra operations
- `rand` - Random number generation
- `rand_distr` - Probability distributions
- `thiserror` - Error handling
- `serde` - Serialization support

## 📖 Quick Start

### Gaussian HMM Example

```rust
use ndarray::array;
use rhmm::models::GaussianHMM;
use rhmm::base::HiddenMarkovModel;

fn main() {
    // Create training data
    let observations = array![
        [0.5, 1.0],
        [0.6, 1.1],
        [5.0, 6.0],
        [5.1, 6.2],
    ];

    // Create and train model with 2 hidden states
    let mut model = GaussianHMM::new(2);
    model.fit(&observations, None).unwrap();

    // Predict hidden states
    let states = model.predict(&observations).unwrap();
    println!("Predicted states: {:?}", states);

    // Calculate log-likelihood
    let log_prob = model.score(&observations).unwrap();
    println!("Log probability: {:.4}", log_prob);

    // Generate synthetic data
    let (sampled_obs, sampled_states) = model.sample(10).unwrap();
    println!("Generated {} samples", sampled_obs.nrows());
}
```

### Beta HMM Example (Conversion Rate Analysis)

```rust
use ndarray::array;
use rhmm::models::BetaHMM;
use rhmm::base::HiddenMarkovModel;

fn main() {
    // Conversion rates (values between 0 and 1)
    let observations = array![
        [0.12, 0.15],  // Low conversion
        [0.10, 0.13],  // Low conversion
        [0.75, 0.82],  // High conversion
        [0.78, 0.85],  // High conversion
    ];

    // Create and train model
    let mut model = BetaHMM::new(2);
    model.fit(&observations, None).unwrap();

    // Predict states
    let states = model.predict(&observations).unwrap();
    println!("States: {:?}", states);

    // Get learned parameters
    if let (Some(alphas), Some(betas)) = (model.alphas(), model.betas()) {
        println!("Alpha parameters: {:?}", alphas);
        println!("Beta parameters: {:?}", betas);
    }
}
```

## 🎯 Use Cases

### Gaussian HMM
- **Speech Recognition**: Model acoustic features
- **Financial Markets**: Detect market regimes (bull/bear)
- **Sensor Data**: Analyze time-series sensor readings
- **Bioinformatics**: Gene sequence analysis

### Beta HMM
- **E-commerce**: Conversion rate analysis
- **Marketing**: Click-through rate modeling
- **Finance**: Market share dynamics
- **Quality Control**: Success rate tracking

### Multinomial HMM
- **Natural Language Processing**: Part-of-speech tagging
- **Bioinformatics**: DNA sequence modeling
- **User Behavior**: Clickstream analysis
- **Weather Modeling**: Discrete weather states

## 📚 API Overview

### Core Trait: `HiddenMarkovModel`

All HMM models implement this trait:

```rust
pub trait HiddenMarkovModel {
    /// Get the number of hidden states
    fn n_states(&self) -> usize;

    /// Get the number of features/dimensions
    fn n_features(&self) -> usize;

    /// Fit the model to observed data
    fn fit(&mut self, observations: &Array2<f64>, lengths: Option<&[usize]>) -> Result<()>;

    /// Predict the most likely state sequence (Viterbi)
    fn predict(&self, observations: &Array2<f64>) -> Result<Array1<usize>>;

    /// Compute the log probability of observations
    fn score(&self, observations: &Array2<f64>) -> Result<f64>;

    /// Sample from the model
    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)>;

    /// Decode the most likely state sequence
    fn decode(&self, observations: &Array2<f64>) -> Result<(f64, Array1<usize>)>;
}
```

### Model Types

#### GaussianHMM
```rust
let model = GaussianHMM::new(n_states);
let model = GaussianHMM::with_covariance_type(n_states, CovarianceType::Diagonal);
```

#### BetaHMM
```rust
let model = BetaHMM::new(n_states);
```

#### MultinomialHMM
```rust
let model = MultinomialHMM::new(n_states, n_features);
```

#### GaussianMixtureHMM
```rust
let model = GaussianMixtureHMM::new(n_states, n_mix);
```

## 🔬 Examples

Run the included examples:

```bash
# Beta HMM example (conversion rate analysis)
cargo run --example beta_hmm_example

# Polars integration example
cargo run --example polars_example
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test integration_tests
```

## 📊 Performance

The library is optimized for performance:
- Uses `ndarray` for efficient numerical operations
- Log-space computations for numerical stability
- Vectorized operations where possible
- Minimal allocations in hot paths

## 🛠️ Advanced Usage

### Multiple Sequences

Train on multiple sequences of different lengths:

```rust
let observations = array![/* concatenated sequences */];
let lengths = vec![10, 15, 20]; // Length of each sequence
model.fit(&observations, Some(&lengths)).unwrap();
```

### Custom Initialization

```rust
let mut model = GaussianHMM::new(3);
// Set custom initial parameters before fitting
// model.set_start_prob(...);
// model.set_transition_matrix(...);
model.fit(&observations, None).unwrap();
```

### Covariance Types

```rust
use rhmm::base::CovarianceType;

// Diagonal covariance (default)
let model = GaussianHMM::with_covariance_type(3, CovarianceType::Diagonal);

// Spherical covariance (single variance)
let model = GaussianHMM::with_covariance_type(3, CovarianceType::Spherical);

// Full covariance matrix
let model = GaussianHMM::with_covariance_type(3, CovarianceType::Full);
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [hmmlearn](https://github.com/hmmlearn/hmmlearn) (Python)
- Built with [ndarray](https://github.com/rust-ndarray/ndarray)

## 📧 Contact

- **Author**: YanXu Fu
- **Email**: unico-serein@hotmail.com
- **GitHub**: [@unico-serein](https://github.com/unico-serein)

## 🗺️ Roadmap

- [ ] Add more emission distributions (Poisson, Exponential)
- [ ] Implement parallel training for multiple sequences
- [ ] Add model serialization/deserialization
- [ ] Improve documentation with more examples
- [ ] Add benchmarks
- [ ] Support for GPU acceleration
---

**Star ⭐ this repository if you find it helpful!**
