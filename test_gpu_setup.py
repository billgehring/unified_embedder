#!/usr/bin/env python3
"""
GPU Setup Testing Utility for Ubuntu RTX8000 Workstation

This script tests and validates the GPU configuration for optimal
embedding performance on the dual RTX8000 + NVLink setup.
"""

import sys
import torch
import subprocess
import platform
from performance_optimizer import create_performance_optimizer

def test_nvidia_setup():
    """Test NVIDIA driver and CUDA installation."""
    print("🔍 Testing NVIDIA Setup")
    print("=" * 50)
    
    # Test nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA drivers installed and working")
            print(f"nvidia-smi output preview:")
            # Show first few lines of nvidia-smi output
            lines = result.stdout.split('\n')[:6]
            for line in lines:
                if line.strip():
                    print(f"   {line}")
        else:
            print("❌ NVIDIA drivers not working properly")
            print(f"Error: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("⚠️  nvidia-smi timeout")
    except FileNotFoundError:
        print("❌ nvidia-smi not found - install NVIDIA drivers")
    except Exception as e:
        print(f"❌ Error testing nvidia-smi: {e}")
    
    print()

def test_pytorch_cuda():
    """Test PyTorch CUDA installation."""
    print("🔍 Testing PyTorch CUDA Setup")
    print("=" * 50)
    
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            compute_cap = torch.cuda.get_device_capability(i)
            print(f"GPU {i}: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            print(f"   Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
            
            # Test basic CUDA operations
            try:
                device = torch.device(f'cuda:{i}')
                test_tensor = torch.randn(1000, 1000, device=device)
                result = torch.matmul(test_tensor, test_tensor.t())
                print(f"   ✅ CUDA operations working on GPU {i}")
            except Exception as e:
                print(f"   ❌ CUDA operations failed on GPU {i}: {e}")
    else:
        print("❌ CUDA not available in PyTorch")
        print("   Install PyTorch with CUDA support:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print()

def test_tesseract():
    """Test Tesseract OCR installation."""
    print("🔍 Testing Tesseract OCR")
    print("=" * 50)
    
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Tesseract installed: {version_line}")
        else:
            print("❌ Tesseract not working properly")
    except FileNotFoundError:
        print("❌ Tesseract not found")
        print("   Install with: sudo apt-get install tesseract-ocr")
    except Exception as e:
        print(f"❌ Error testing Tesseract: {e}")
    
    print()

def test_performance_optimizer():
    """Test the performance optimizer configuration."""
    print("🔍 Testing Performance Optimizer")
    print("=" * 50)
    
    try:
        optimizer = create_performance_optimizer()
        summary = optimizer.get_system_summary()
        
        print("System Configuration:")
        print(f"  Device Type: {summary['system_info']['device_type']}")
        print(f"  GPU Acceleration: {summary['recommendations']['use_gpu']}")
        print(f"  Target Device: {summary['recommendations']['embedding_device']}")
        print(f"  Optimal Workers: {summary['recommendations']['optimal_workers']}")
        print(f"  Optimal Batch Size: {summary['recommendations']['optimal_batch_size']}")
        
        # Special checks for RTX8000
        if summary['system_info'].get('is_dual_rtx8000'):
            print("🔥 DUAL RTX8000 DETECTED - Ultimate Performance Mode!")
            if summary['system_info'].get('has_nvlink'):
                print("🔗 NVLink detected - Maximum throughput enabled")
        elif summary['system_info'].get('is_rtx8000'):
            rtx_count = summary['system_info'].get('rtx8000_count', 1)
            print(f"🚀 RTX8000 detected ({rtx_count}x) - High Performance Mode!")
        
    except Exception as e:
        print(f"❌ Performance optimizer failed: {e}")
    
    print()

def test_nvlink():
    """Test NVLink connectivity if available."""
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        return
        
    print("🔍 Testing NVLink Connectivity")
    print("=" * 50)
    
    try:
        # Test peer-to-peer access
        can_access_peer = torch.cuda.can_device_access_peer(0, 1)
        print(f"P2P Access (GPU 0 -> GPU 1): {can_access_peer}")
        
        if can_access_peer:
            # Test actual P2P transfer speed
            device0 = torch.device('cuda:0')
            device1 = torch.device('cuda:1')
            
            # Create test data
            test_size = 100 * 1024 * 1024  # 100MB
            data = torch.randn(test_size // 4, device=device0)  # 4 bytes per float32
            
            # Warm up
            for _ in range(3):
                data_copy = data.to(device1)
                torch.cuda.synchronize()
            
            # Time the transfer
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            data_copy = data.to(device1)
            end_event.record()
            torch.cuda.synchronize()
            
            transfer_time = start_event.elapsed_time(end_event)  # milliseconds
            transfer_rate = (test_size / (1024**3)) / (transfer_time / 1000)  # GB/s
            
            print(f"P2P Transfer Rate: {transfer_rate:.2f} GB/s")
            
            if transfer_rate > 20:  # NVLink should be much faster than PCIe
                print("🔗 ✅ High-speed interconnect detected (likely NVLink)")
            else:
                print("⚠️  Standard PCIe speed detected")
        else:
            print("❌ P2P access not available")
            
    except Exception as e:
        print(f"❌ NVLink test failed: {e}")
    
    print()

def main():
    """Run all GPU setup tests."""
    print("🧪 GPU Setup Testing for Ubuntu RTX8000 Workstation")
    print("=" * 70)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    print()
    
    test_nvidia_setup()
    test_pytorch_cuda()
    test_tesseract()
    test_performance_optimizer()
    test_nvlink()
    
    print("🏁 GPU Setup Testing Complete")
    print("=" * 70)

if __name__ == "__main__":
    main()