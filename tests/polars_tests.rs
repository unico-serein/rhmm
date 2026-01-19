//! Polars integration tests
//!
//! Run with: cargo test --test polars_tests --features polars

#[cfg(feature = "polars")]
use polars::prelude::*;
#[cfg(feature = "polars")]
use rhmm::models::{GaussianHMM, GMMHMM};
#[cfg(feature = "polars")]
use rhmm::base::HiddenMarkovModel;
#[cfg(feature = "polars")]
use rhmm::utils::polars::{
    series_to_array1, series_to_array2, dataframe_to_array2, 
    array1_to_series, array2_to_dataframe
};

#[cfg(not(feature = "polars"))]
#[test]
fn test_polars_not_enabled() {
    println!("Polars feature is not enabled. Skipping Polars tests.");
}

#[cfg(feature = "polars")]
#[test]
fn test_gaussian_hmm_fit_from_series() {
    let series1 = Series::new("feature_1", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let series2 = Series::new("feature_2", vec![2.0, 3.0, 4.0, 5.0, 6.0]);
    
    let mut hmm = GaussianHMM::new(2);
    
    // Test training from series
    let result = hmm.fit_from_series(&[series1.clone(), series2.clone()], None);
    assert!(result.is_ok());
    assert!(hmm.is_fitted());
    
    // Test prediction from series
    let predicted = hmm.predict_from_series(&[series1, series2]);
    assert!(predicted.is_ok());
    let states = predicted.unwrap();
    assert_eq!(states.len(), 5);
}

#[cfg(feature = "polars")]
#[test]
fn test_gmm_hmm_fit_from_dataframe() {
    let df = DataFrame::new(vec![
        Series::new("col1", vec![1.0, 2.0, 3.0, 4.0]),
        Series::new("col2", vec![2.0, 3.0, 4.0, 5.0]),
        Series::new("col3", vec![3.0, 4.0, 5.0, 6.0]),
    ]).unwrap();
    
    let mut gmm = GMMHMM::new(2, 2);
    
    // Test training from dataframe
    let result = gmm.fit_from_dataframe(&df, None, None);
    assert!(result.is_ok());
    assert!(gmm.is_fitted());
    
    // Test prediction from dataframe
    let predicted = gmm.predict_from_dataframe(&df, None);
    assert!(predicted.is_ok());
    let states = predicted.unwrap();
    assert_eq!(states.len(), 4);
}

#[cfg(feature = "polars")]
#[test]
fn test_selected_columns() {
    let df = DataFrame::new(vec![
        Series::new("temp", vec![20.0, 21.0, 22.0, 23.0]),
        Series::new("humidity", vec![60.0, 65.0, 70.0, 75.0]),
        Series::new("pressure", vec![1013.0, 1014.0, 1015.0, 1016.0]),
        Series::new("noise", vec![0.1, 0.2, 0.3, 0.4]), // This should be ignored
    ]).unwrap();
    
    let mut hmm = GaussianHMM::new(2);
    
    // Train using only selected columns
    let selected = ["temp", "humidity", "pressure"];
    let result = hmm.fit_from_dataframe(&df, Some(&selected), None);
    assert!(result.is_ok());
    
    // Predict using same selected columns
    let predicted = hmm.predict_from_dataframe(&df, Some(&selected));
    assert!(predicted.is_ok());
}

#[cfg(feature = "polars")]
#[test]
fn test_score_from_polars() {
    let series1 = Series::new("x", vec![1.0, 2.0, 3.0]);
    let series2 = Series::new("y", vec![2.0, 3.0, 4.0]);
    
    let mut hmm = GaussianHMM::new(2);
    hmm.fit_from_series(&[series1.clone(), series2.clone()], None).unwrap();
    
    // Test scoring from series
    let score = hmm.score_from_series(&[series1, series2]);
    assert!(score.is_ok());
    let log_prob = score.unwrap();
    assert!(log_prob.is_finite());
}

#[cfg(feature = "polars")]
#[test]
fn test_sample_to_dataframe() {
    let series1 = Series::new("x", vec![1.0, 2.0, 3.0]);
    let series2 = Series::new("y", vec![2.0, 3.0, 4.0]);
    
    let mut hmm = GaussianHMM::new(2);
    hmm.fit_from_series(&[series1, series2], None).unwrap();
    
    // Test sampling to dataframe
    let result = hmm.sample_to_dataframe(5, "feature_");
    assert!(result.is_ok());
    
    let df = result.unwrap();
    assert_eq!(df.shape(), (5, 3)); // 2 features + 1 states column
    assert!(df.column("feature_0").is_ok());
    assert!(df.column("feature_1").is_ok());
    assert!(df.column("states").is_ok());
}

#[cfg(feature = "polars")]
#[test]
fn test_conversion_functions() {
    // Test series_to_array1
    let series = Series::new("test", vec![1.0, 2.0, 3.0]);
    let array1 = series_to_array1(&series).unwrap();
    assert_eq!(array1.len(), 3);
    assert_eq!(array1[0], 1.0);
    
    // Test series_to_array2
    let series1 = Series::new("col1", vec![1.0, 2.0]);
    let series2 = Series::new("col2", vec![3.0, 4.0]);
    let array2 = series_to_array2(&[series1, series2]).unwrap();
    assert_eq!(array2.shape(), &[2, 2]);
    assert_eq!(array2[[0, 0]], 1.0);
    assert_eq!(array2[[0, 1]], 3.0);
    
    // Test array1_to_series
    let array = ndarray::Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let series = array1_to_series(&array, "result");
    assert_eq!(series.len(), 3);
    assert_eq!(series.name(), "result");
    
    // Test array2_to_dataframe
    let array = ndarray::Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let df = array2_to_dataframe(&array, "col_").unwrap();
    assert_eq!(df.shape(), (2, 2));
    assert_eq!(df.get_column_names(), &["col_0", "col_1"]);
}

#[cfg(feature = "polars")]
#[test]
fn test_dataframe_to_array2_all_numeric() {
    let df = DataFrame::new(vec![
        Series::new("a", vec![1.0, 2.0]),
        Series::new("b", vec![3.0, 4.0]),
        Series::new("c", vec![5.0, 6.0]),
    ]).unwrap();
    
    let result = dataframe_to_array2(&df, None).unwrap();
    assert_eq!(result.shape(), &[2, 3]);
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[0, 2]], 5.0);
}

#[cfg(feature = "polars")]
#[test]
fn test_dataframe_to_array2_selected_columns() {
    let df = DataFrame::new(vec![
        Series::new("temp", vec![20.0, 21.0]),
        Series::new("humidity", vec![60.0, 65.0]),
        Series::new("pressure", vec![1013.0, 1014.0]),
    ]).unwrap();
    
    let selected = ["temp", "pressure"];
    let result = dataframe_to_array2(&df, Some(&selected)).unwrap();
    assert_eq!(result.shape(), &[2, 2]);
    assert_eq!(result[[0, 0]], 20.0);
    assert_eq!(result[[0, 1]], 1013.0);
}

#[cfg(feature = "polars")]
#[test]
fn test_different_numeric_types() {
    // Test with different numeric types
    let df = DataFrame::new(vec![
        Series::new("int32", vec![1i32, 2i32, 3i32]),
        Series::new("int64", vec![4i64, 5i64, 6i64]),
        Series::new("float32", vec![7.0f32, 8.0f32, 9.0f32]),
        Series::new("float64", vec![10.0f64, 11.0f64, 12.0f64]),
    ]).unwrap();
    
    let mut hmm = GaussianHMM::new(2);
    let result = hmm.fit_from_dataframe(&df, None, None);
    assert!(result.is_ok());
    
    let predicted = hmm.predict_from_dataframe(&df, None).unwrap();
    assert_eq!(predicted.len(), 3);
}

#[cfg(feature = "polars")]
#[test]
fn test_consistency_with_ndarray() {
    // Test that Polars methods give same results as ndarray methods
    let array = ndarray::Array2::from_shape_vec((4, 2), vec![
        1.0, 2.0,
        2.0, 3.0,
        5.0, 6.0,
        6.0, 7.0,
    ]).unwrap();
    
    let df = DataFrame::new(vec![
        Series::new("x", vec![1.0, 2.0, 5.0, 6.0]),
        Series::new("y", vec![2.0, 3.0, 6.0, 7.0]),
    ]).unwrap();
    
    let mut hmm1 = GaussianHMM::new(2);
    let mut hmm2 = GaussianHMM::new(2);
    
    // Train both models
    hmm1.fit(&array, None).unwrap();
    hmm2.fit_from_dataframe(&df, None, None).unwrap();
    
    // Compare predictions
    let states1 = hmm1.predict(&array).unwrap();
    let states2 = hmm2.predict_from_dataframe(&df, None).unwrap();
    
    // Convert Polars series to Vec for comparison
    let states2_vec: Vec<usize> = states2.u32().unwrap().into_no_null_iter()
        .map(|x| x as usize).collect();
    
    assert_eq!(states1.as_slice().unwrap(), states2_vec.as_slice());
    
    // Compare scores
    let score1 = hmm1.score(&array).unwrap();
    let score2 = hmm2.score_from_dataframe(&df, None).unwrap();
    
    assert!((score1 - score2).abs() < 1e-10);
}