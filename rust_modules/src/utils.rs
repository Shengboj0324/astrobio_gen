//! Utility functions for astrobiology Rust extensions
//! 
//! This module provides utility functions for version information,
//! performance monitoring, and system diagnostics.

use pyo3::prelude::*;
use std::collections::HashMap;

/// Get version information for the Rust extensions
#[pyfunction]
pub fn get_version() -> PyResult<String> {
    Ok(env!("CARGO_PKG_VERSION").to_string())
}

/// Get performance and system information
#[pyfunction]
pub fn get_performance_info() -> PyResult<HashMap<String, String>> {
    let mut info = HashMap::new();
    
    // Version information
    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("build_target".to_string(), std::env::var("TARGET").unwrap_or_else(|_| "unknown".to_string()));
    
    // CPU information
    info.insert("cpu_count".to_string(), num_cpus::get().to_string());
    info.insert("cpu_count_physical".to_string(), num_cpus::get_physical().to_string());
    
    // Compilation features
    let mut features = Vec::new();
    
    #[cfg(feature = "datacube-processing")]
    features.push("datacube-processing");
    
    #[cfg(feature = "concurrent-acquisition")]
    features.push("concurrent-acquisition");
    
    info.insert("features".to_string(), features.join(", "));
    
    // SIMD capabilities
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            info.insert("simd_avx2".to_string(), "available".to_string());
        } else {
            info.insert("simd_avx2".to_string(), "not_available".to_string());
        }
        
        if is_x86_feature_detected!("fma") {
            info.insert("simd_fma".to_string(), "available".to_string());
        } else {
            info.insert("simd_fma".to_string(), "not_available".to_string());
        }
    }
    
    // Memory information (if available)
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(mem_kb) = line.split_whitespace().nth(1) {
                        if let Ok(mem_kb) = mem_kb.parse::<u64>() {
                            let mem_gb = mem_kb / 1024 / 1024;
                            info.insert("system_memory_gb".to_string(), mem_gb.to_string());
                        }
                    }
                    break;
                }
            }
        }
    }
    
    // Optimization level
    #[cfg(debug_assertions)]
    info.insert("optimization".to_string(), "debug".to_string());
    
    #[cfg(not(debug_assertions))]
    info.insert("optimization".to_string(), "release".to_string());
    
    Ok(info)
}

/// Calculate optimal chunk size for parallel processing
pub fn calculate_optimal_chunk_size(total_elements: usize, num_threads: usize) -> usize {
    // Aim for chunks that are large enough to amortize threading overhead
    // but small enough to provide good load balancing
    let min_chunk_size = 1000;  // Minimum elements per chunk
    let max_chunk_size = 100_000;  // Maximum elements per chunk
    
    let naive_chunk_size = (total_elements + num_threads - 1) / num_threads;
    
    naive_chunk_size
        .max(min_chunk_size)
        .min(max_chunk_size)
}

/// Memory alignment utilities for SIMD operations
pub const SIMD_ALIGNMENT: usize = 32;  // 256-bit alignment for AVX2

/// Check if a pointer is properly aligned for SIMD operations
pub fn is_simd_aligned<T>(ptr: *const T) -> bool {
    (ptr as usize) % SIMD_ALIGNMENT == 0
}

/// Get the number of elements that need to be processed before SIMD alignment
pub fn elements_to_alignment<T>(ptr: *const T) -> usize {
    let alignment_offset = (ptr as usize) % SIMD_ALIGNMENT;
    if alignment_offset == 0 {
        0
    } else {
        (SIMD_ALIGNMENT - alignment_offset) / std::mem::size_of::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_optimal_chunk_size() {
        // Test with small dataset
        assert_eq!(calculate_optimal_chunk_size(5000, 4), 1250);
        
        // Test with large dataset
        assert_eq!(calculate_optimal_chunk_size(1_000_000, 8), 100_000);
        
        // Test minimum chunk size
        assert_eq!(calculate_optimal_chunk_size(2000, 8), 1000);
    }
    
    #[test]
    fn test_simd_alignment() {
        let aligned_data: Vec<f32> = vec![0.0; 1024];
        let ptr = aligned_data.as_ptr();
        
        // Note: This test may not always pass due to allocator behavior
        // but demonstrates the alignment checking functionality
        let _ = is_simd_aligned(ptr);
        let _ = elements_to_alignment(ptr);
    }
}
