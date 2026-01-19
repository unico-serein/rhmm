# RHMM - Rust Hidden Markov Models

[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A comprehensive and efficient Hidden Markov Model (HMM) library implemented in Rust, supporting multiple emission distributions and advanced algorithms.

## 🌟 Features

- **Multiple Emission Distributions**
  - 🔢 **Gaussian HMM**: For continuous data with normal distributions
  - 📊 **Beta HMM**: For data in range [0, 1] (proportions, rates, probabilities)
  - 🎯 **Multinomial HMM**: For discrete observations
  - 🔀 **Gaussian Mixture Model HMM**: For complex multimodal distributions

- **Core Algorithms**
  - ✅ Forward Algorithm (probability computation)
  - ✅ Backward Algorithm (probability computation)
  - ✅ Viterbi Algorithm (most likely state sequence)
  - ✅ Baum-Welch Algorithm (parameter estimation via EM)

- **Robust Implementation**
  - 🛡️ Comprehensive parameter validation
  - 🧪 Extensive test coverage
  - 📝 Well-documented API
  - ⚡ Efficient numerical computations with `ndarray`

## 📦 Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rhmm = "0.1.0"
ndarray = "0.15"
```

### Optional Features

#### Polars Support
To enable Polars integration for seamless DataFrame and Series support:

```toml
[dependencies]
rhmm = { version = "0.1.0", features = ["polars"] }
polars = "0.35"
```

This enables:
- Direct training with Polars Series and DataFrame
- Automatic type conversion between Polars and ndarray
- Support for selected columns from DataFrame
- Synthetic data generation as Polars DataFrame

## 🚀 Quick Start

### Gaussian HMM Example

```rust
use ndarray::array;
use rhmm::models::GaussianHMM;
use rhmm::base::HiddenMarkovModel;

fn main() {
    // Create training data: temperature readings
    let observations = array![
        [20.5, 21.0],  // Cool period
        [21.2, 20.8],
        [30.1, 29.8],  // Hot period
        [31.5, 30.2],
        [20.0, 21.5],  // Cool period again
    ];

    // Create and train a Gaussian HMM with 2 states
    let mut model = GaussianHMM::new(2);
    model.fit(&observations, None).unwrap();

    // Predict the most likely state sequence
    let states = model.predict(&observations).unwrap();
    println!("Predicted states: {:?}", states);

    // Calculate log-likelihood
    let log_prob = model.score(&observations).unwrap();
    println!("Log-likelihood: {:.4}", log_prob);

    // Generate synthetic data
    let (sampled_obs, sampled_states) = model.sample(10).unwrap();
    println!("Generated {} samples", sampled_obs.nrows());
}
```

### Beta HMM Example

Perfect for modeling conversion rates, success rates, or any data in [0, 1]:

```rust
use ndarray::array;
use rhmm::models::BetaHMM;
use rhmm::base::HiddenMarkovModel;

fn main() {
    // Conversion rates: [mobile_rate, desktop_rate]
    let observations = array![
        [0.12, 0.15],  // Low conversion period
        [0.10, 0.13],
        [0.75, 0.82],  // High conversion period
        [0.78, 0.85],
        [0.11, 0.14],  // Back to low
    ];

    let mut model = BetaHMM::new(2);
    model.fit(&observations, None).unwrap();

    // Predict states for new data
    let new_data = array![[0.80, 0.85]];
    let states = model.predict(&new_data).unwrap();
    println!("Predicted state: {:?}", states);
}
```

## 📚 Documentation

### Available Models

#### 1. GaussianHMM

For continuous data following normal distributions.

```rust
use rhmm::models::GaussianHMM;
use rhmm::base::CovarianceType;

// Create with default diagonal covariance
let mut model = GaussianHMM::new(3);

// Or specify covariance type
let mut model = GaussianHMM::with_covariance_type(
    3, 
    CovarianceType::Spherical
);
```

**Covariance Types:**
- `Diagonal`: Independent features (default)
- `Spherical`: Single variance for all features
- `Full`: Full covariance matrix (planned)
- `Tied`: Shared covariance across states (planned)

#### 2. BetaHMM

For data in range (0, 1), such as:
- Conversion rates
- Success probabilities
- Market shares
- Proportions

```rust
use rhmm::models::BetaHMM;

let mut model = BetaHMM::new(2);

// Get learned parameters
if let (Some(alphas), Some(betas)) = (model.alphas(), model.betas()) {
    println!("Alpha parameters: {:?}", alphas);
    println!("Beta parameters: {:?}", betas);
}

// Compute statistics
let means = model.compute_means();
let variances = model.compute_variances();
```

#### 3. MultinomialHMM

For discrete observations.

```rust
use rhmm::models::MultinomialHMM;

let model = MultinomialHMM::new(
    3,  // number of states
    10  // number of possible observations
);
```

#### 4. GMMHMM

Gaussian Mixture Model HMM for complex multimodal distributions.

```rust
use rhmm::models::GMMHMM;

let model = GMMHMM::new(
    3,  // number of states
    2   // number of mixture components per state
);
```

### Core Trait: HiddenMarkovModel

All models implement the `HiddenMarkovModel` trait:

```rust
pub trait HiddenMarkovModel {
    /// Get number of hidden states
    fn n_states(&self) -> usize;
    
    /// Get number of features
    fn n_features(&self) -> usize;
    
    /// Train the model
    fn fit(&mut self, observations: &Array2<f64>, lengths: Option<&[usize]>) -> Result<()>;
    
    /// Predict most likely state sequence (Viterbi)
    fn predict(&self, observations: &Array2<f64>) -> Result<Array1<usize>>;
    
    /// Compute log probability (Forward algorithm)
    fn score(&self, observations: &Array2<f64>) -> Result<f64>;
    
    /// Generate synthetic samples
    fn sample(&self, n_samples: usize) -> Result<(Array2<f64>, Array1<usize>)>;
    
    /// Decode state sequence with probability
    fn decode(&self, observations: &Array2<f64>) -> Result<(f64, Array1<usize>)>;
}
```

### Utility Functions

The library provides helpful utility functions in the `utils` module:

```rust
use rhmm::utils::{
    normalize_vector,
    normalize_matrix_rows,
    validate_probability_vector,
    validate_transition_matrix,
    validate_observations,
};

// Normalize probabilities
let probs = array![0.3, 0.5, 0.2];
let normalized = normalize_vector(probs);

// Validate parameters
validate_probability_vector(&probs, "Initial probabilities")?;
validate_transition_matrix(&trans_matrix)?;
validate_observations(&observations, n_features)?;
```

## 🎯 Use Cases

### 1. **Financial Markets**
- Stock price regime detection (bull/bear markets)
- Volatility clustering analysis
- Trading signal generation

### 2. **E-commerce**
- Customer behavior segmentation
- Conversion rate analysis
- Churn prediction

### 3. **Natural Language Processing**
- Part-of-speech tagging
- Named entity recognition
- Speech recognition

### 4. **Bioinformatics**
- Gene sequence analysis
- Protein structure prediction
- DNA motif discovery

### 5. **Time Series Analysis**
- Anomaly detection
- Regime switching models
- Seasonal pattern recognition

## 📖 Examples

Run the included examples:

```bash
# Beta HMM example (conversion rate analysis)
cargo run --example beta_hmm_example
```

The example demonstrates:
- Training a Beta HMM on conversion rate data
- Examining learned parameters
- Predicting hidden states
- Generating synthetic data
- Evaluating model performance

## 🏗️ Project Structure

```
rhmm/
├── src/
│   ├── lib.rs                 # Library entry point
│   ├── base/                  # Core traits and types
│   │   ├── hmm.rs            # HiddenMarkovModel trait
│   │   └── types.rs          # Common types
│   ├── models/               # HMM implementations
│   │   ├── gaussian.rs       # Gaussian HMM
│   │   ├── beta.rs           # Beta HMM
│   │   ├── gmm.rs            # GMM HMM
│   │   └── multinomial.rs    # Multinomial HMM
│   ├── algorithms/           # Core algorithms
│   │   ├── forward.rs        # Forward algorithm
│   │   ├── backward.rs       # Backward algorithm
│   │   ├── viterbi.rs        # Viterbi algorithm
│   │   └── baum_welch.rs     # Baum-Welch (EM)
│   ├── utils/                # Utility functions
│   │   ├── normalization.rs  # Probability normalization
│   │   ├── validation.rs     # Parameter validation
│   │   └── sampling.rs       # Sampling utilities
│   └── errors.rs             # Error types
├── examples/                 # Usage examples
├── tests/                    # Integration tests
└── Cargo.toml
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
cargo test

# Run with output
cargo test -- --nocapture

# Run specific test
cargo test test_gaussian_hmm
```

## 🔧 Development

### Building

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Check for errors
cargo check
```

### Code Quality

```bash
# Format code
cargo fmt

# Run linter
cargo clippy

# Generate documentation
cargo doc --open
```

## 📊 Performance

The library is optimized for performance using:
- Efficient matrix operations with `ndarray`
- Numerical stability in log-space computations
- Vectorized operations where possible
- Minimal allocations in hot paths

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines

1. Write tests for new features
2. Update documentation
3. Follow Rust naming conventions
4. Run `cargo fmt` and `cargo clippy`
5. Ensure all tests pass

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [ndarray](https://github.com/rust-ndarray/ndarray) for efficient numerical computing
- Inspired by [hmmlearn](https://github.com/hmmlearn/hmmlearn) (Python)
- Thanks to the Rust community for excellent tooling and libraries

## 📮 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/rhmm/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/rhmm/discussions)

## 🗺️ Roadmap

- [ ] Full covariance matrix support for Gaussian HMM
- [ ] Tied covariance support
- [ ] Parallel training for multiple sequences
- [ ] Model serialization/deserialization
- [ ] Online learning capabilities
- [ ] More emission distributions (Poisson, Exponential, etc.)
- [ ] GPU acceleration support
- [ ] Python bindings via PyO3

---

**Made with ❤️ in Rust**
