//! Polars Integration Example
//!
//! This example demonstrates how to use RHMM with Polars Series and DataFrame.
//! Run with: cargo run --example polars_example --features polars

#[cfg(feature = "polars")]
use polars::prelude::*;
#[cfg(feature = "polars")]
use rhmm::models::{GaussianHMM, GMMHMM};
#[cfg(feature = "polars")]
use rhmm::base::HiddenMarkovModel;
#[cfg(feature = "polars")]
use rhmm::utils::polars::{series_to_array2, dataframe_to_array2};

#[cfg(not(feature = "polars"))]
fn main() {
    println!("This example requires the 'polars' feature to be enabled.");
    println!("Please run with: cargo run --example polars_example --features polars");
}

#[cfg(feature = "polars")]
fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           RHMM with Polars Integration Example            ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Example 1: Using Series with Gaussian HMM
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 1: Gaussian HMM with Polars Series");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create temperature data using Polars Series
    let temp_series1 = Series::new("temperature_1", vec![20.5, 21.2, 20.8, 30.1, 29.8, 31.5, 20.0]);
    let temp_series2 = Series::new("temperature_2", vec![21.0, 20.8, 21.5, 29.8, 30.2, 30.8, 21.5]);
    
    println!("📊 Created Polars Series:");
    println!("   Series 1: {:?}", temp_series1.head(Some(3)));
    println!("   Series 2: {:?}", temp_series2.head(Some(3)));
    println!("   Data shape: {} samples × {} features\n", temp_series1.len(), 2);

    // Create and train Gaussian HMM
    let mut gaussian_hmm = GaussianHMM::new(2);
    
    println!("🔧 Training Gaussian HMM with 2 states...");
    match gaussian_hmm.fit_from_series(&[temp_series1.clone(), temp_series2.clone()], None) {
        Ok(_) => println!("✓ Training completed successfully!\n"),
        Err(e) => {
            println!("✗ Training failed: {:?}\n", e);
            return;
        }
    }

    // Predict using Series
    println!("🔮 Predicting states using Polars Series...");
    match gaussian_hmm.predict_from_series(&[temp_series1, temp_series2]) {
        Ok(predicted_states) => {
            println!("✓ Predicted states: {:?}", predicted_states);
            
            // Add predicted states to original data
            let mut df_with_states = DataFrame::new(vec![
                Series::new("temp_1", vec![20.5, 21.2, 20.8, 30.1, 29.8, 31.5, 20.0]),
                Series::new("temp_2", vec![21.0, 20.8, 21.5, 29.8, 30.2, 30.8, 21.5]),
                predicted_states,
            ]).unwrap();
            
            println!("\n📋 Data with predicted states:");
            println!("{}", df_with_states);
        },
        Err(e) => println!("✗ Prediction failed: {:?}", e),
    }

    // Example 2: Using DataFrame with GMM HMM
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 2: GMM HMM with Polars DataFrame");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a more complex dataset using DataFrame
    let df = DataFrame::new(vec![
        Series::new("feature_1", vec![1.0, 1.2, 0.9, 5.0, 5.1, 4.9, 1.1, 0.8]),
        Series::new("feature_2", vec![2.0, 1.8, 2.1, 6.0, 5.9, 6.1, 2.2, 1.9]),
        Series::new("feature_3", vec![0.5, 0.7, 0.4, 4.5, 4.6, 4.4, 0.6, 0.3]),
    ]).unwrap();

    println!("📊 Created Polars DataFrame:");
    println!("{}", df.head(Some(5)));
    println!("   Shape: {:?}\n", df.shape());

    // Create and train GMM HMM
    let mut gmm_hmm = GMMHMM::new(2, 2);
    
    println!("🔧 Training GMM HMM with 2 states and 2 mixture components...");
    match gmm_hmm.fit_from_dataframe(&df, None, None) {
        Ok(_) => println!("✓ Training completed successfully!\n"),
        Err(e) => {
            println!("✗ Training failed: {:?}\n", e);
            return;
        }
    }

    // Predict using DataFrame
    println!("🔮 Predicting states using DataFrame...");
    match gmm_hmm.predict_from_dataframe(&df, None) {
        Ok(predicted_states) => {
            println!("✓ Predicted states: {:?}", predicted_states);
            
            // Add to DataFrame
            let mut df_with_states = df.clone();
            df_with_states.with_column(predicted_states).unwrap();
            
            println!("\n📋 DataFrame with predicted states:");
            println!("{}", df_with_states);
        },
        Err(e) => println!("✗ Prediction failed: {:?}", e),
    }

    // Example 3: Advanced usage - selected columns and scoring
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 3: Advanced Usage - Selected Columns and Scoring");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Use only specific columns
    let selected_columns = ["feature_1", "feature_3"];
    
    println!("📊 Using selected columns: {:?}", selected_columns);
    
    // Train on selected columns
    let mut gmm_hmm2 = GMMHMM::new(2, 2);
    match gmm_hmm2.fit_from_dataframe(&df, Some(&selected_columns), None) {
        Ok(_) => println!("✓ Training on selected columns completed!\n"),
        Err(e) => {
            println!("✗ Training failed: {:?}\n", e);
            return;
        }
    }

    // Score the data
    println!("📈 Computing log-likelihood...");
    match gmm_hmm2.score_from_dataframe(&df, Some(&selected_columns)) {
        Ok(log_prob) => println!("✓ Log-likelihood: {:.4}", log_prob),
        Err(e) => println!("✗ Scoring failed: {:?}", e),
    }

    // Generate synthetic samples
    println!("\n🎲 Generating synthetic samples...");
    match gmm_hmm2.sample_to_dataframe(5, "synthetic_") {
        Ok(synthetic_df) => {
            println!("✓ Generated synthetic data:");
            println!("{}", synthetic_df);
        },
        Err(e) => println!("✗ Sampling failed: {:?}", e),
    }

    // Example 4: Manual conversion and comparison
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Example 4: Manual Conversion and Comparison");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Manual conversion
    let manual_array = dataframe_to_array2(&df, Some(&selected_columns)).unwrap();
    println!("📊 Manual conversion to ndarray:");
    println!("   Shape: {:?}", manual_array.shape());
    println!("   First few rows:");
    for i in 0..3.min(manual_array.nrows()) {
        print!("   Row {}: [", i);
        for j in 0..manual_array.ncols() {
            print!("{:.1}", manual_array[[i, j]]);
            if j < manual_array.ncols() - 1 { print!(", "); }
        }
        println!("]");
    }

    // Compare results
    let polars_states = gmm_hmm2.predict_from_dataframe(&df, Some(&selected_columns)).unwrap();
    let ndarray_states = gmm_hmm2.predict(&manual_array).unwrap();
    
    println!("\n🔍 Comparing results:");
    println!("   Polars prediction: {:?}", polars_states);
    println!("   Ndarray prediction: {:?}", ndarray_states);
    
    let polars_score = gmm_hmm2.score_from_dataframe(&df, Some(&selected_columns)).unwrap();
    let ndarray_score = gmm_hmm2.score(&manual_array).unwrap();
    
    println!("   Polars score: {:.4}", polars_score);
    println!("   Ndarray score: {:.4}", ndarray_score);
    
    if (polars_score - ndarray_score).abs() < 1e-10 {
        println!("✓ Results match perfectly!");
    } else {
        println!("⚠️  Small difference in scores: {:.2e}", (polars_score - ndarray_score).abs());
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║              Polars Integration Example Complete! ✓       ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n💡 Key Features Demonstrated:");
    println!("   • Direct training with Polars Series and DataFrame");
    println!("   • Prediction with automatic conversion");
    println!("   • Support for selected columns");
    println!("   • Synthetic data generation as DataFrame");
    println!("   • Seamless integration with existing ndarray API");
    println!("\n📚 Use Cases:");
    println!("   • Time series analysis with Polars");
    println!("   • Financial data modeling");
    println!("   • Sensor data processing");
    println!("   • Any Polars-based data pipeline");
}