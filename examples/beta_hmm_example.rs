//! Beta HMM Example
//!
//! This example demonstrates how to use the Beta Hidden Markov Model
//! to model data in the range [0, 1], such as conversion rates or proportions.
//!
//! Run this example with:
//! ```bash
//! cargo run --example beta_hmm_example
//! ```

use ndarray::array;
use rhmm::models::BetaHMM;
use rhmm::base::HiddenMarkovModel;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║           Beta HMM Example - Conversion Rate Analysis     ║");
    println!("╚═══════════════════════════════════════════════════════════╝\n");

    // Scenario: We're analyzing daily conversion rates for an e-commerce website
    // The conversion rates follow two hidden states:
    // - State 0: "Low conversion period" (around 10-20%)
    // - State 1: "High conversion period" (around 70-85%)

    println!("📊 Scenario: E-commerce Conversion Rate Analysis");
    println!("   We have 15 days of conversion rate data (values between 0 and 1)\n");

    // Create training data: conversion rates over 15 days
    // Each row represents a day, with 2 features:
    // - Feature 1: Mobile conversion rate
    // - Feature 2: Desktop conversion rate
    let observations = array![
        [0.12, 0.15],  // Day 1: Low conversion
        [0.10, 0.13],  // Day 2: Low conversion
        [0.14, 0.16],  // Day 3: Low conversion
        [0.75, 0.82],  // Day 4: High conversion (campaign started)
        [0.78, 0.85],  // Day 5: High conversion
        [0.80, 0.83],  // Day 6: High conversion
        [0.76, 0.81],  // Day 7: High conversion
        [0.11, 0.14],  // Day 8: Low conversion (campaign ended)
        [0.13, 0.12],  // Day 9: Low conversion
        [0.15, 0.17],  // Day 10: Low conversion
        [0.79, 0.84],  // Day 11: High conversion (new campaign)
        [0.82, 0.86],  // Day 12: High conversion
        [0.77, 0.80],  // Day 13: High conversion
        [0.12, 0.15],  // Day 14: Low conversion (campaign ended)
        [0.14, 0.13],  // Day 15: Low conversion
    ];

    println!("✓ Loaded {} days of data with {} features per day",
             observations.nrows(), observations.ncols());
    println!("  Features: [Mobile Rate, Desktop Rate]\n");

    // Step 1: Create and train the model
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 1: Training Beta HMM with 2 hidden states");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut model = BetaHMM::new(2);
    
    match model.fit(&observations, None) {
        Ok(_) => println!("✓ Model training completed successfully!\n"),
        Err(e) => {
            eprintln!("✗ Training failed: {:?}", e);
            return;
        }
    }

    // Step 2: Examine learned parameters
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 2: Learned Model Parameters");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    if let (Some(alphas), Some(betas)) = (model.alphas(), model.betas()) {
        println!("📈 Beta Distribution Parameters:");
        println!("   Alpha (shape parameter 1):");
        for i in 0..2 {
            println!("      State {}: [{:.2}, {:.2}]", i, alphas[[i, 0]], alphas[[i, 1]]);
        }
        println!("\n   Beta (shape parameter 2):");
        for i in 0..2 {
            println!("      State {}: [{:.2}, {:.2}]", i, betas[[i, 0]], betas[[i, 1]]);
        }

        if let Some(means) = model.compute_means() {
            println!("\n📊 Expected Conversion Rates (Mean) per State:");
            for i in 0..2 {
                println!("      State {}: Mobile={:.1}%, Desktop={:.1}%",
                         i, means[[i, 0]] * 100.0, means[[i, 1]] * 100.0);
            }
        }

        if let Some(vars) = model.compute_variances() {
            println!("\n📉 Variance per State:");
            for i in 0..2 {
                println!("      State {}: Mobile={:.4}, Desktop={:.4}",
                         i, vars[[i, 0]], vars[[i, 1]]);
            }
        }
    }

    if let Some(trans_mat) = model.transition_matrix() {
        println!("\n🔄 State Transition Probabilities:");
        println!("   From State 0 → State 0: {:.1}%", trans_mat[[0, 0]] * 100.0);
        println!("   From State 0 → State 1: {:.1}%", trans_mat[[0, 1]] * 100.0);
        println!("   From State 1 → State 0: {:.1}%", trans_mat[[1, 0]] * 100.0);
        println!("   From State 1 → State 1: {:.1}%", trans_mat[[1, 1]] * 100.0);
    }

    if let Some(start_prob) = model.start_prob() {
        println!("\n🎯 Initial State Probabilities:");
        println!("   State 0 (Low): {:.1}%", start_prob[0] * 100.0);
        println!("   State 1 (High): {:.1}%", start_prob[1] * 100.0);
    }

    // Step 3: Predict hidden states
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 3: Predict Hidden States (Viterbi Algorithm)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    match model.predict(&observations) {
        Ok(states) => {
            println!("🔍 Predicted State Sequence:");
            println!("   Day | Mobile | Desktop | State | Interpretation");
            println!("   ----+--------+---------+-------+------------------");
            for (day, (obs_row, &state)) in observations.outer_iter().zip(states.iter()).enumerate() {
                let state_name = if state == 0 { "Low " } else { "High" };
                println!("   {:2}  | {:.1}%  |  {:.1}%  |   {}  | {} conversion",
                         day + 1,
                         obs_row[0] * 100.0,
                         obs_row[1] * 100.0,
                         state,
                         state_name);
            }
        }
        Err(e) => eprintln!("\n✗ Prediction failed: {:?}", e),
    }

    // Step 4: Calculate log-likelihood
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 4: Model Evaluation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    match model.score(&observations) {
        Ok(log_prob) => {
            println!("📊 Log-likelihood of observed data: {:.4}", log_prob);
            println!("   (Higher values indicate better fit)\n");
        }
        Err(e) => eprintln!("✗ Scoring failed: {:?}\n", e),
    }

    // Step 5: Generate synthetic data
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 5: Generate Synthetic Data");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("🎲 Sampling 10 new days from the learned model...\n");
    
    match model.sample(10) {
        Ok((sampled_obs, sampled_states)) => {
            println!("Generated Data:");
            println!("   Day | Mobile | Desktop | State | Type");
            println!("   ----+--------+---------+-------+------");
            for (day, (obs_row, &state)) in sampled_obs.outer_iter().zip(sampled_states.iter()).enumerate() {
                let state_name = if state == 0 { "Low " } else { "High" };
                println!("   {:2}  | {:.1}%  |  {:.1}%  |   {}  | {}",
                         day + 1,
                         obs_row[0] * 100.0,
                         obs_row[1] * 100.0,
                         state,
                         state_name);
            }
        }
        Err(e) => eprintln!("✗ Sampling failed: {:?}", e),
    }

    // Step 6: Predict on new data
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Step 6: Predict on New Unseen Data");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let new_data = array![
        [0.11, 0.14],  // Looks like low conversion
        [0.81, 0.87],  // Looks like high conversion
        [0.13, 0.15],  // Looks like low conversion
    ];

    println!("🔮 Predicting states for 3 new days:");
    match model.predict(&new_data) {
        Ok(states) => {
            for (day, (obs_row, &state)) in new_data.outer_iter().zip(states.iter()).enumerate() {
                let state_name = if state == 0 { "Low conversion" } else { "High conversion" };
                println!("   Day {}: [{:.1}%, {:.1}%] → State {} ({})",
                         day + 1,
                         obs_row[0] * 100.0,
                         obs_row[1] * 100.0,
                         state,
                         state_name);
            }
        }
        Err(e) => eprintln!("✗ Prediction failed: {:?}", e),
    }

    println!("\n╔═══════════════════════════════════════════════════════════╗");
    println!("║                    Example Completed! ✓                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("\n💡 Key Takeaways:");
    println!("   • Beta HMM successfully identified two conversion rate states");
    println!("   • The model learned transition patterns between states");
    println!("   • Can predict states for new unseen data");
    println!("   • Can generate synthetic data following learned patterns");
    println!("\n📚 Use Cases:");
    println!("   • Conversion rate analysis");
    println!("   • Market share modeling");
    println!("   • Success rate tracking");
    println!("   • Any proportion/rate data in [0,1] range\n");
}
