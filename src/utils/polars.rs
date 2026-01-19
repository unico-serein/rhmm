//! Polars Series and DataFrame support for RHMM
//!
//! This module provides conversion utilities between Polars Series/DataFrame
//! and ndarray for seamless integration with HMM models.

#[cfg(feature = "polars")]
use polars::prelude::*;
#[cfg(feature = "polars")]
use ndarray::{Array1, Array2};
use crate::errors::{Result, HmmError};

/// Convert Polars Series to ndarray Array1
#[cfg(feature = "polars")]
pub fn series_to_array1(series: &Series) -> Result<Array1<f64>> {
    match series.dtype() {
        DataType::Float64 => {
            let values: Vec<f64> = series.f64()
                .ok_or_else(|| HmmError::InvalidParameter("Failed to get f64 values from series".to_string()))?
                .into_no_null_iter()
                .collect();
            Ok(Array1::from_vec(values))
        },
        DataType::Float32 => {
            let values: Vec<f32> = series.f32()
                .ok_or_else(|| HmmError::InvalidParameter("Failed to get f32 values from series".to_string()))?
                .into_no_null_iter()
                .collect();
            Ok(Array1::from_vec(values.iter().map(|&x| x as f64).collect()))
        },
        DataType::Int64 => {
            let values: Vec<i64> = series.i64()
                .ok_or_else(|| HmmError::InvalidParameter("Failed to get i64 values from series".to_string()))?
                .into_no_null_iter()
                .collect();
            Ok(Array1::from_vec(values.iter().map(|&x| x as f64).collect()))
        },
        DataType::Int32 => {
            let values: Vec<i32> = series.i32()
                .ok_or_else(|| HmmError::InvalidParameter("Failed to get i32 values from series".to_string()))?
                .into_no_null_iter()
                .collect();
            Ok(Array1::from_vec(values.iter().map(|&x| x as f64).collect()))
        },
        _ => Err(HmmError::InvalidParameter(
            format!("Unsupported data type: {:?}. Only numeric types are supported.", series.dtype())
        ))
    }
}

/// Convert multiple Polars Series to ndarray Array2
#[cfg(feature = "polars")]
pub fn series_to_array2(series_list: &[Series]) -> Result<Array2<f64>> {
    if series_list.is_empty() {
        return Err(HmmError::InvalidParameter("No series provided".to_string()));
    }
    
    let n_samples = series_list[0].len();
    let n_features = series_list.len();
    
    // Check that all series have the same length
    for (i, series) in series_list.iter().enumerate() {
        if series.len() != n_samples {
            return Err(HmmError::InvalidParameter(
                format!("Series {} has length {} but expected {}", i, series.len(), n_samples)
            ));
        }
    }
    
    let mut array = Array2::zeros((n_samples, n_features));
    
    for (j, series) in series_list.iter().enumerate() {
        match series.dtype() {
            DataType::Float64 => {
                let values = series.f64()
                    .ok_or_else(|| HmmError::InvalidParameter("Failed to get f64 values".to_string()))?;
                for i in 0..n_samples {
                    array[[i, j]] = values.get(i).unwrap_or(0.0);
                }
            },
            DataType::Float32 => {
                let values = series.f32()
                    .ok_or_else(|| HmmError::InvalidParameter("Failed to get f32 values".to_string()))?;
                for i in 0..n_samples {
                    array[[i, j]] = values.get(i).unwrap_or(0.0) as f64;
                }
            },
            DataType::Int64 => {
                let values = series.i64()
                    .ok_or_else(|| HmmError::InvalidParameter("Failed to get i64 values".to_string()))?;
                for i in 0..n_samples {
                    array[[i, j]] = values.get(i).unwrap_or(0) as f64;
                }
            },
            DataType::Int32 => {
                let values = series.i32()
                    .ok_or_else(|| HmmError::InvalidParameter("Failed to get i32 values".to_string()))?;
                for i in 0..n_samples {
                    array[[i, j]] = values.get(i).unwrap_or(0) as f64;
                }
            },
            _ => return Err(HmmError::InvalidParameter(
                format!("Unsupported data type: {:?}. Only numeric types are supported.", series.dtype())
            ))
        }
    }
    
    Ok(array)
}

/// Convert Polars DataFrame to ndarray Array2
#[cfg(feature = "polars")]
pub fn dataframe_to_array2(df: &DataFrame, columns: Option<&[&str]>) -> Result<Array2<f64>> {
    let selected_columns = match columns {
        Some(cols) => {
            // Use specified columns
            let mut series_list = Vec::new();
            for &col_name in cols {
                let series = df.column(col_name)
                    .map_err(|e| HmmError::InvalidParameter(format!("Column '{}' not found: {}", col_name, e)))?;
                series_list.push(series.clone());
            }
            series_list
        },
        None => {
            // Use all numeric columns
            let mut series_list = Vec::new();
            for (col_name, series) in df.get_columns().iter().enumerate() {
                if series.dtype().is_numeric() {
                    series_list.push(series.clone());
                }
            }
            if series_list.is_empty() {
                return Err(HmmError::InvalidParameter("No numeric columns found in DataFrame".to_string()));
            }
            series_list
        }
    };
    
    series_to_array2(&selected_columns)
}

/// Convert ndarray Array1 to Polars Series
#[cfg(feature = "polars")]
pub fn array1_to_series(array: &Array1<f64>, name: &str) -> Series {
    Series::new(name, array.to_vec())
}

/// Convert ndarray Array2 to Polars DataFrame
#[cfg(feature = "polars")]
pub fn array2_to_dataframe(array: &Array2<f64>, column_prefix: &str) -> Result<DataFrame> {
    let n_features = array.ncols();
    let mut series_list = Vec::new();
    
    for j in 0..n_features {
        let column_name = format!("{}{}", column_prefix, j);
        let column_data: Vec<f64> = array.column(j).to_vec();
        series_list.push(Series::new(&column_name, column_data));
    }
    
    DataFrame::new(series_list)
        .map_err(|e| HmmError::InvalidParameter(format!("Failed to create DataFrame: {}", e)))
}

/// Helper function to check if Polars feature is enabled
pub fn is_polars_enabled() -> bool {
    cfg!(feature = "polars")
}

#[cfg(test)]
#[cfg(feature = "polars")]
mod tests {
    use super::*;
    
    #[test]
    fn test_series_to_array1_float64() {
        let series = Series::new("test", vec![1.0, 2.0, 3.0]);
        let result = series_to_array1(&series).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }
    
    #[test]
    fn test_series_to_array1_int32() {
        let series = Series::new("test", vec![1i32, 2i32, 3i32]);
        let result = series_to_array1(&series).unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }
    
    #[test]
    fn test_series_to_array2() {
        let series1 = Series::new("col1", vec![1.0, 2.0, 3.0]);
        let series2 = Series::new("col2", vec![4.0, 5.0, 6.0]);
        let series_list = vec![series1, series2];
        
        let result = series_to_array2(&series_list).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 4.0);
        assert_eq!(result[[2, 0]], 3.0);
        assert_eq!(result[[2, 1]], 6.0);
    }
    
    #[test]
    fn test_array1_to_series() {
        let array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let series = array1_to_series(&array, "test");
        assert_eq!(series.len(), 3);
        assert_eq!(series.name(), "test");
        assert_eq!(series.f64().unwrap().get(0).unwrap(), 1.0);
    }
    
    #[test]
    fn test_array2_to_dataframe() {
        let array = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let df = array2_to_dataframe(&array, "feature_").unwrap();
        
        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.get_column_names(), &["feature_0", "feature_1"]);
        assert_eq!(df.column("feature_0").unwrap().f64().unwrap().get(0).unwrap(), 1.0);
        assert_eq!(df.column("feature_1").unwrap().f64().unwrap().get(0).unwrap(), 2.0);
    }
    
    #[test]
    fn test_dataframe_to_array2_all_columns() {
        let df = DataFrame::new(vec![
            Series::new("col1", vec![1.0, 2.0, 3.0]),
            Series::new("col2", vec![4.0, 5.0, 6.0]),
            Series::new("col3", vec![7.0, 8.0, 9.0]),
        ]).unwrap();
        
        let result = dataframe_to_array2(&df, None).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 2]], 7.0);
        assert_eq!(result[[2, 1]], 8.0);
    }
    
    #[test]
    fn test_dataframe_to_array2_selected_columns() {
        let df = DataFrame::new(vec![
            Series::new("col1", vec![1.0, 2.0, 3.0]),
            Series::new("col2", vec![4.0, 5.0, 6.0]),
            Series::new("col3", vec![7.0, 8.0, 9.0]),
        ]).unwrap();
        
        let result = dataframe_to_array2(&df, Some(&["col1", "col3"])).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 7.0);
        assert_eq!(result[[2, 0]], 3.0);
        assert_eq!(result[[2, 1]], 9.0);
    }
    
    #[test]
    fn test_is_polars_enabled() {
        assert!(is_polars_enabled());
    }
}