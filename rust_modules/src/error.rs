//! Error handling for astrobiology Rust extensions
//! 
//! This module provides comprehensive error handling with proper Python
//! exception mapping and detailed error messages for debugging.

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use thiserror::Error;

/// Result type alias for astrobiology operations
pub type Result<T> = std::result::Result<T, AstrobiologyError>;

/// Comprehensive error types for astrobiology operations
#[derive(Error, Debug)]
pub enum AstrobiologyError {
    #[error("Invalid tensor dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: String, actual: String },
    
    #[error("Memory allocation failed: {message}")]
    MemoryError { message: String },
    
    #[error("Tensor operation failed: {operation} - {details}")]
    TensorOperationError { operation: String, details: String },
    
    #[error("Invalid input data: {message}")]
    InvalidInput { message: String },
    
    #[error("Numerical computation error: {message}")]
    NumericalError { message: String },
    
    #[error("Python integration error: {message}")]
    PythonIntegrationError { message: String },
    
    #[error("Performance optimization failed: {message}")]
    OptimizationError { message: String },
    
    #[error("Internal error: {message}")]
    InternalError { message: String },

    #[error("Initialization error: {message}")]
    InitializationError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Service unavailable: {service}")]
    ServiceUnavailable { service: String },

    #[error("Model not found: {model_name}")]
    ModelNotFound { model_name: String },
}

impl From<AstrobiologyError> for PyErr {
    fn from(err: AstrobiologyError) -> PyErr {
        match err {
            AstrobiologyError::InvalidDimensions { .. } => {
                PyRuntimeError::new_err(format!("Datacube dimension error: {}", err))
            }
            AstrobiologyError::MemoryError { .. } => {
                PyRuntimeError::new_err(format!("Memory allocation error: {}", err))
            }
            AstrobiologyError::TensorOperationError { .. } => {
                PyRuntimeError::new_err(format!("Tensor operation failed: {}", err))
            }
            AstrobiologyError::InvalidInput { .. } => {
                PyRuntimeError::new_err(format!("Invalid input: {}", err))
            }
            AstrobiologyError::NumericalError { .. } => {
                PyRuntimeError::new_err(format!("Numerical computation error: {}", err))
            }
            AstrobiologyError::PythonIntegrationError { .. } => {
                PyRuntimeError::new_err(format!("Python integration error: {}", err))
            }
            AstrobiologyError::OptimizationError { .. } => {
                PyRuntimeError::new_err(format!("Optimization error: {}", err))
            }
            AstrobiologyError::InternalError { .. } => {
                PyRuntimeError::new_err(format!("Internal error: {}", err))
            }
            AstrobiologyError::InitializationError { .. } => {
                PyRuntimeError::new_err(format!("Initialization error: {}", err))
            }
            AstrobiologyError::NetworkError { .. } => {
                PyRuntimeError::new_err(format!("Network error: {}", err))
            }
            AstrobiologyError::ServiceUnavailable { .. } => {
                PyRuntimeError::new_err(format!("Service unavailable: {}", err))
            }
            AstrobiologyError::ModelNotFound { .. } => {
                PyRuntimeError::new_err(format!("Model not found: {}", err))
            }
        }
    }
}

/// Validate tensor dimensions for datacube operations
pub fn validate_datacube_dimensions(
    shape: &[usize],
    expected_dims: usize,
    operation: &str,
) -> Result<()> {
    if shape.len() != expected_dims {
        return Err(AstrobiologyError::InvalidDimensions {
            expected: format!("{} dimensions", expected_dims),
            actual: format!("{} dimensions: {:?}", shape.len(), shape),
        });
    }
    
    // Check for reasonable dimension sizes
    for (i, &dim_size) in shape.iter().enumerate() {
        if dim_size == 0 {
            return Err(AstrobiologyError::InvalidInput {
                message: format!("Dimension {} has size 0 in operation '{}'", i, operation),
            });
        }
        
        // Prevent extremely large allocations (>100GB per tensor)
        let total_elements: usize = shape.iter().product();
        if total_elements > 25_000_000_000 {  // ~100GB for f32
            return Err(AstrobiologyError::MemoryError {
                message: format!(
                    "Tensor too large: {} elements ({:.1} GB) in operation '{}'",
                    total_elements,
                    total_elements as f64 * 4.0 / 1e9,
                    operation
                ),
            });
        }
    }
    
    Ok(())
}

/// Validate transpose dimensions
pub fn validate_transpose_dims(
    original_dims: usize,
    transpose_dims: &[usize],
) -> Result<()> {
    if transpose_dims.len() != original_dims {
        return Err(AstrobiologyError::InvalidInput {
            message: format!(
                "Transpose dimensions length {} doesn't match tensor dimensions {}",
                transpose_dims.len(),
                original_dims
            ),
        });
    }
    
    // Check that all dimensions are valid indices
    for &dim in transpose_dims {
        if dim >= original_dims {
            return Err(AstrobiologyError::InvalidInput {
                message: format!(
                    "Transpose dimension {} is out of range for {}-dimensional tensor",
                    dim,
                    original_dims
                ),
            });
        }
    }
    
    // Check that all dimensions are unique
    let mut sorted_dims = transpose_dims.to_vec();
    sorted_dims.sort_unstable();
    for i in 1..sorted_dims.len() {
        if sorted_dims[i] == sorted_dims[i - 1] {
            return Err(AstrobiologyError::InvalidInput {
                message: format!(
                    "Duplicate dimension {} in transpose operation",
                    sorted_dims[i]
                ),
            });
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_datacube_dimensions() {
        // Valid 7D datacube
        assert!(validate_datacube_dimensions(&[8, 12, 24, 4, 37, 64, 128], 7, "test").is_ok());
        
        // Invalid dimensions count
        assert!(validate_datacube_dimensions(&[8, 12, 24], 7, "test").is_err());
        
        // Zero dimension
        assert!(validate_datacube_dimensions(&[8, 0, 24, 4, 37, 64, 128], 7, "test").is_err());
    }
    
    #[test]
    fn test_validate_transpose_dims() {
        // Valid transpose
        assert!(validate_transpose_dims(7, &[0, 2, 1, 3, 4, 5, 6]).is_ok());
        
        // Invalid length
        assert!(validate_transpose_dims(7, &[0, 1, 2]).is_err());
        
        // Out of range dimension
        assert!(validate_transpose_dims(7, &[0, 1, 2, 3, 4, 5, 7]).is_err());
        
        // Duplicate dimension
        assert!(validate_transpose_dims(7, &[0, 1, 1, 3, 4, 5, 6]).is_err());
    }
}
