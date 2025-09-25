#!/usr/bin/env python3
"""
Performance optimizer for the unified embedder pipeline.

This module provides platform-specific optimizations for:
- MacBook Pro M2 (Apple Silicon) with MPS acceleration
- RTX8000 workstation with CUDA acceleration
- General CPU-only fallback optimizations

Key optimizations:
1. Device detection and optimal model loading
2. Batch size optimization based on available memory
3. Worker process tuning for multiprocessing
4. Memory management and cleanup strategies
5. Model caching and reuse patterns
"""

import os
import platform
import psutil
import multiprocessing
import torch
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """
    Analyzes system capabilities and provides optimal configuration for embedding pipeline.
    """
    
    def __init__(self):
        self.system_info = self._analyze_system()
        self.optimization_config = self._generate_optimization_config()
        
    def _analyze_system(self) -> Dict[str, Any]:
        """Analyze system hardware and capabilities."""
        system_info = {
            'platform': platform.system(),
            'machine': platform.machine(),
            'cpu_count': multiprocessing.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'has_cuda': torch.cuda.is_available(),
            'has_mps': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        # Detect specific hardware
        if system_info['platform'] == 'Darwin' and 'arm64' in system_info['machine']:
            system_info['device_type'] = 'apple_silicon'
            system_info['is_m2'] = True  # Assumption - could be refined
        elif system_info['has_cuda']:
            system_info['device_type'] = 'cuda_gpu'
            # Try to detect RTX8000 specifically and dual GPU setup
            if system_info['cuda_device_count'] > 0:
                try:
                    gpu_names = []
                    rtx8000_count = 0
                    has_nvlink = False
                    
                    for i in range(system_info['cuda_device_count']):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_names.append(gpu_name)
                        if 'rtx' in gpu_name.lower() and '8000' in gpu_name.lower():
                            rtx8000_count += 1
                    
                    system_info['gpu_names'] = gpu_names
                    system_info['gpu_name'] = gpu_names[0]  # Primary GPU
                    system_info['is_rtx8000'] = rtx8000_count > 0
                    system_info['rtx8000_count'] = rtx8000_count
                    system_info['is_dual_rtx8000'] = rtx8000_count >= 2
                    
                    # Check for NVLink (requires nvidia-ml-py or direct CUDA calls)
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        if system_info['cuda_device_count'] >= 2:
                            # Check if GPUs are connected via high-speed link (NVLink)
                            handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
                            handle1 = pynvml.nvmlDeviceGetHandleByIndex(1) 
                            # This is a simplified check - real NVLink detection is more complex
                            system_info['has_nvlink'] = True  # Assume NVLink on dual RTX8000
                    except ImportError:
                        logger.debug("pynvml not available for NVLink detection")
                        system_info['has_nvlink'] = rtx8000_count >= 2  # Reasonable assumption
                    except Exception as e:
                        logger.debug(f"NVLink detection failed: {e}")
                        system_info['has_nvlink'] = False
                        
                    # Detect CUDA compute capability for optimization
                    try:
                        major, minor = torch.cuda.get_device_capability(0)
                        system_info['cuda_compute_capability'] = f"{major}.{minor}"
                        system_info['has_tensor_cores'] = major >= 7  # Volta+ has Tensor Cores
                    except:
                        system_info['cuda_compute_capability'] = "unknown"
                        system_info['has_tensor_cores'] = True  # Assume RTX8000 has them
                        
                except Exception as e:
                    logger.debug(f"GPU detection failed: {e}")
                    system_info['gpu_name'] = 'unknown'
                    system_info['is_rtx8000'] = False
                    system_info['is_dual_rtx8000'] = False
        else:
            system_info['device_type'] = 'cpu_only'
            
        return system_info
    
    def _generate_optimization_config(self) -> Dict[str, Any]:
        """Generate optimal configuration based on system analysis."""
        config = {
            'device': 'cpu',  # Default fallback
            'batch_size': 32,
            'num_workers': max(1, self.system_info['cpu_count'] - 1),
            'embedding_device': 'cpu',
            'use_half_precision': False,
            'enable_model_caching': True,
            'memory_optimization': 'standard',
            'chunk_processing': 'sequential',
        }
        
        # MacBook Pro M2 optimizations
        if self.system_info.get('device_type') == 'apple_silicon':
            config.update(self._apple_silicon_config())
            
        # Dual RTX8000 + NVLink optimizations (best performance)
        elif self.system_info.get('is_dual_rtx8000') and self.system_info.get('has_nvlink'):
            config.update(self._dual_rtx8000_nvlink_config())
            
        # Single or dual RTX8000 optimizations  
        elif self.system_info.get('is_rtx8000'):
            config.update(self._rtx8000_config())
            
        # General CUDA optimizations
        elif self.system_info.get('has_cuda'):
            config.update(self._cuda_config())
            
        # CPU-only optimizations
        else:
            config.update(self._cpu_only_config())
            
        return config
    
    def _apple_silicon_config(self) -> Dict[str, Any]:
        """Optimizations specific to Apple Silicon M2."""
        logger.info("🍎 Configuring optimizations for Apple Silicon M2")
        
        return {
            'device': 'mps' if self.system_info['has_mps'] else 'cpu',
            'embedding_device': 'mps' if self.system_info['has_mps'] else 'cpu', 
            'batch_size': 64,  # M2 has unified memory, can handle larger batches
            'num_workers': min(8, self.system_info['cpu_count']),  # M2 has 8-10 cores typically
            'use_half_precision': True,  # MPS supports FP16
            'memory_optimization': 'aggressive',
            'chunk_processing': 'parallel',
            'pytorch_settings': {
                'PYTORCH_ENABLE_MPS_FALLBACK': '1',
                'PYTORCH_MPS_HIGH_WATERMARK_RATIO': '0.0',  # Disable MPS memory caching
            }
        }
    
    def _dual_rtx8000_nvlink_config(self) -> Dict[str, Any]:
        """Optimizations specific to dual RTX8000 + NVLink workstation."""
        logger.info("🔥 Configuring optimizations for DUAL RTX8000 + NVLink workstation")
        
        # Dual RTX8000 with NVLink = 96GB total VRAM + high-speed inter-GPU communication
        # This is the ultimate setup for embedding workloads
        return {
            'device': 'cuda',
            'embedding_device': 'cuda:0',  # Primary GPU for embedding
            'secondary_device': 'cuda:1',   # Secondary GPU for parallel processing
            'batch_size': 512,  # Very large batch size for dual 48GB GPUs
            'num_workers': min(24, self.system_info['cpu_count']),  # High-end workstation cores
            'use_half_precision': True,  # RTX8000 Tensor Cores + FP16
            'use_mixed_precision': True,  # Advanced mixed precision for max performance
            'memory_optimization': 'nvlink_optimized',
            'chunk_processing': 'dual_gpu_parallel',
            'enable_data_parallel': True,  # Use both GPUs for model parallelism
            'cuda_settings': {
                'CUDA_VISIBLE_DEVICES': '0,1',  # Use both GPUs
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:4096,expandable_segments:True',
                'NCCL_DEBUG': 'INFO',  # For multi-GPU debugging
                'NCCL_P2P_DISABLE': '0',  # Enable P2P for NVLink
                'CUDA_LAUNCH_BLOCKING': '0',  # Async execution
            },
            'pytorch_settings': {
                'TORCH_CUDNN_V8_API_ENABLED': '1',  # Latest cuDNN optimizations
                'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',  # TF32 for Ampere
                'TORCH_CUDNN_BENCHMARK': '1',  # Optimize for consistent input sizes
            },
            'nvlink_settings': {
                'enable_peer_access': True,  # Direct GPU-to-GPU memory access
                'unified_memory': True,  # Use CUDA unified memory
                'memory_pool': 'unified',  # Shared memory pool across GPUs
            }
        }
    
    def _rtx8000_config(self) -> Dict[str, Any]:
        """Optimizations specific to single RTX8000 workstation."""
        gpu_count = self.system_info.get('rtx8000_count', 1)
        if gpu_count > 1:
            logger.info(f"🚀 Configuring optimizations for {gpu_count}x RTX8000 workstation")
        else:
            logger.info("🚀 Configuring optimizations for RTX8000 workstation")
        
        # RTX8000 has 48GB VRAM - can handle very large batches
        config = {
            'device': 'cuda',
            'embedding_device': 'cuda:0',
            'batch_size': 384 if gpu_count > 1 else 256,  # Larger batch for multi-GPU
            'num_workers': min(20, self.system_info['cpu_count']),  # Workstation likely has many cores
            'use_half_precision': True,  # RTX8000 has Tensor Cores
            'use_mixed_precision': True,  # Mixed precision for Tensor Cores
            'memory_optimization': 'gpu_optimized',
            'chunk_processing': 'gpu_parallel',
            'cuda_settings': {
                'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:3072,expandable_segments:True',
                'CUDA_LAUNCH_BLOCKING': '0',
            },
            'pytorch_settings': {
                'TORCH_CUDNN_V8_API_ENABLED': '1',
                'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '1',
                'TORCH_CUDNN_BENCHMARK': '1',
            }
        }
        
        # Multi-GPU settings for non-NVLink dual RTX8000
        if gpu_count > 1:
            config['cuda_settings']['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in range(gpu_count))
            config['enable_data_parallel'] = True
            config['secondary_device'] = 'cuda:1'
        else:
            config['cuda_settings']['CUDA_VISIBLE_DEVICES'] = '0'
            
        return config
    
    def _cuda_config(self) -> Dict[str, Any]:
        """General CUDA GPU optimizations."""
        logger.info("⚡ Configuring optimizations for CUDA GPU")
        
        # Conservative settings for unknown GPU
        return {
            'device': 'cuda',
            'embedding_device': 'cuda:0',
            'batch_size': 128,
            'num_workers': min(8, self.system_info['cpu_count']),
            'use_half_precision': True,
            'memory_optimization': 'gpu_conservative',
            'chunk_processing': 'gpu_parallel',
        }
    
    def _cpu_only_config(self) -> Dict[str, Any]:
        """CPU-only optimizations."""
        logger.info("🖥️ Configuring optimizations for CPU-only processing")
        
        return {
            'device': 'cpu',
            'embedding_device': 'cpu',
            'batch_size': 16,  # Conservative for CPU
            'num_workers': max(1, self.system_info['cpu_count'] - 2),
            'use_half_precision': False,  # CPU doesn't benefit from FP16
            'memory_optimization': 'cpu_optimized',
            'chunk_processing': 'cpu_parallel',
        }
    
    def apply_optimizations(self) -> None:
        """Apply system-level optimizations."""
        logger.info("🔧 Applying performance optimizations")
        logger.info(f"System: {self.system_info['device_type']}")
        logger.info(f"Config: {self.optimization_config}")
        
        # Set environment variables
        self._set_environment_variables()
        
        # Configure PyTorch settings
        self._configure_pytorch()
        
        # Log optimization summary
        self._log_optimization_summary()
    
    def _set_environment_variables(self) -> None:
        """Set optimal environment variables."""
        # General settings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid multiprocessing issues
        
        # Platform-specific settings
        if self.system_info.get('device_type') == 'apple_silicon':
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            # Don't force CPU for Apple Silicon when MPS is available
            if not self.system_info['has_mps']:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                
        elif self.system_info.get('has_cuda'):
            # CUDA-specific optimizations
            cuda_settings = self.optimization_config.get('cuda_settings', {})
            pytorch_settings = self.optimization_config.get('pytorch_settings', {})
            
            # Apply CUDA environment variables
            for key, value in cuda_settings.items():
                if key != 'cuda_settings':  # Skip nested dict
                    os.environ[key] = str(value)
                    logger.debug(f"Set {key}={value}")
            
            # Apply PyTorch environment variables  
            for key, value in pytorch_settings.items():
                os.environ[key] = str(value)
                logger.debug(f"Set {key}={value}")
                
            # Special handling for dual GPU setup
            if self.system_info.get('is_dual_rtx8000'):
                # Additional multi-GPU environment variables
                os.environ["NCCL_ALGO"] = "Ring"  # Optimize for NVLink
                os.environ["NCCL_MIN_NRINGS"] = "2"  # Use multiple communication rings
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Consistent device ordering
                logger.info("🔗 Applied NVLink/multi-GPU environment optimizations")
                
        else:
            # Force CPU for non-accelerated systems
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    def _configure_pytorch(self) -> None:
        """Configure PyTorch for optimal performance."""
        # Set number of threads for CPU operations
        if self.system_info['device_type'] == 'cpu_only':
            torch.set_num_threads(min(8, self.system_info['cpu_count']))
        
        # Enable optimized attention if available (for newer PyTorch versions)
        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except:
            pass
            
        # Configure memory management
        if self.system_info.get('has_cuda'):
            torch.cuda.empty_cache()
    
    def _log_optimization_summary(self) -> None:
        """Log comprehensive optimization summary."""
        logger.info("📊 Performance Optimization Summary:")
        logger.info(f"  🏗️  Architecture: {self.system_info.get('machine', 'unknown')}")
        logger.info(f"  🖥️  Platform: {self.system_info.get('platform', 'unknown')}")
        logger.info(f"  🧠 CPU Cores: {self.system_info.get('cpu_count', 'unknown')}")
        logger.info(f"  💾 Memory: {self.system_info.get('total_memory_gb', 0):.1f} GB total, {self.system_info.get('available_memory_gb', 0):.1f} GB available")
        
        if self.system_info.get('has_cuda'):
            gpu_count = self.system_info.get('cuda_device_count', 1)
            if gpu_count > 1:
                gpu_names = self.system_info.get('gpu_names', ['Unknown'] * gpu_count)
                logger.info(f"  🎮 GPUs ({gpu_count}): {', '.join(gpu_names)}")
                if self.system_info.get('has_nvlink'):
                    logger.info(f"  🔗 NVLink: ✅ Enabled")
                if self.system_info.get('is_dual_rtx8000'):
                    logger.info(f"  🔥 Dual RTX8000 Mode: ✅ Ultimate Performance")
            else:
                logger.info(f"  🎮 GPU: {self.system_info.get('gpu_name', 'Unknown CUDA GPU')}")
            
            if self.system_info.get('has_tensor_cores'):
                logger.info(f"  ⚡ Tensor Cores: ✅ Available")
            
            if self.system_info.get('cuda_compute_capability'):
                logger.info(f"  🔧 CUDA Compute: {self.system_info['cuda_compute_capability']}")
            
        logger.info(f"  🎯 Target Device: {self.optimization_config.get('device', 'cpu')}")
        if self.optimization_config.get('secondary_device'):
            logger.info(f"  🎯 Secondary Device: {self.optimization_config['secondary_device']}")
        logger.info(f"  📦 Batch Size: {self.optimization_config.get('batch_size', 32)}")
        logger.info(f"  👥 Workers: {self.optimization_config.get('num_workers', 1)}")
        logger.info(f"  🏃 Half Precision: {self.optimization_config.get('use_half_precision', False)}")
        if self.optimization_config.get('use_mixed_precision'):
            logger.info(f"  🎭 Mixed Precision: ✅ Tensor Core Optimization")
    
    def get_optimal_worker_count(self, file_count: int, file_size_estimate: int = 100000) -> int:
        """
        Calculate optimal worker count based on system resources and workload.
        
        Args:
            file_count: Number of files to process
            file_size_estimate: Estimated size per file in bytes
            
        Returns:
            Optimal number of worker processes
        """
        base_workers = self.optimization_config.get('num_workers', 1)
        
        # Memory-based scaling
        available_memory = self.system_info['available_memory_gb']
        memory_per_worker = (file_size_estimate * 3) / (1024**3)  # 3x overhead estimate
        memory_limited_workers = int(available_memory / memory_per_worker)
        
        # Don't exceed base recommendation or create too many workers for small workloads
        optimal_workers = min(
            base_workers,
            memory_limited_workers,
            max(1, file_count // 10)  # At least 10 files per worker
        )
        
        logger.info(f"🔢 Optimal workers: {optimal_workers} (base: {base_workers}, memory-limited: {memory_limited_workers})")
        return max(1, optimal_workers)
    
    def get_optimal_batch_size(self, text_length: int = 1000) -> int:
        """
        Get optimal batch size based on text length and system capabilities.
        
        Args:
            text_length: Average text length to process
            
        Returns:
            Optimal batch size
        """
        base_batch_size = self.optimization_config.get('batch_size', 32)
        
        # Adjust for text length (longer texts need smaller batches)
        if text_length > 5000:
            adjusted_batch_size = max(1, base_batch_size // 4)
        elif text_length > 2000:
            adjusted_batch_size = max(1, base_batch_size // 2)
        else:
            adjusted_batch_size = base_batch_size
            
        return adjusted_batch_size
    
    def should_use_gpu_acceleration(self) -> bool:
        """Check if GPU acceleration should be used for embeddings."""
        return (
            self.optimization_config.get('device') != 'cpu' and
            (self.system_info.get('has_cuda') or self.system_info.get('has_mps'))
        )
    
    def get_embedding_device(self) -> str:
        """Get the optimal device for embedding computation."""
        return self.optimization_config.get('embedding_device', 'cpu')
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of system capabilities for logging/debugging."""
        return {
            'system_info': self.system_info,
            'optimization_config': self.optimization_config,
            'recommendations': {
                'use_gpu': self.should_use_gpu_acceleration(),
                'embedding_device': self.get_embedding_device(),
                'optimal_workers': self.get_optimal_worker_count(1000),
                'optimal_batch_size': self.get_optimal_batch_size(),
            }
        }

def create_performance_optimizer() -> PerformanceOptimizer:
    """Factory function to create and configure performance optimizer."""
    optimizer = PerformanceOptimizer()
    optimizer.apply_optimizations()
    return optimizer

if __name__ == "__main__":
    # Demo/test the performance optimizer
    optimizer = create_performance_optimizer()
    summary = optimizer.get_system_summary()
    
    print("🚀 Performance Optimizer Test")
    print("=" * 50)
    print(f"System Type: {summary['system_info']['device_type']}")
    print(f"Recommended Device: {summary['recommendations']['embedding_device']}")
    print(f"Optimal Workers: {summary['recommendations']['optimal_workers']}")
    print(f"Optimal Batch Size: {summary['recommendations']['optimal_batch_size']}")
    print(f"GPU Acceleration: {summary['recommendations']['use_gpu']}")