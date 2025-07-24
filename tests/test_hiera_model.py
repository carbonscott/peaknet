#!/usr/bin/env python3
"""
Comprehensive test script for Hiera model implementation in PeakNet.

This script tests the Hiera model functionality including:
- Model creation for all variants
- Forward pass with 2D image and 3D video data
- Output shape validation
- Training/inference mode switching
- Error handling

Usage:
    python tests/test_hiera_model.py
    pytest tests/test_hiera_model.py
"""

import sys
import os
import time
import traceback
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn

# Add the project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from peaknet.modeling.hiera import (
        Hiera,
        hiera_tiny_224,
        hiera_small_224, 
        hiera_base_224,
        hiera_base_plus_224,
        hiera_large_224,
        hiera_huge_224,
        hiera_base_16x224,
        hiera_base_plus_16x224,
        hiera_large_16x224,
        hiera_huge_16x224,
    )
    print("✅ Successfully imported Hiera models")
except ImportError as e:
    print(f"❌ Failed to import Hiera models: {e}")
    traceback.print_exc()
    sys.exit(1)


class HieraModelTester:
    """Comprehensive test suite for Hiera models."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")
        
        # Test results storage
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }
        
        # Model factory functions for testing
        self.image_models = {
            'hiera_tiny_224': hiera_tiny_224,
            'hiera_small_224': hiera_small_224,
            'hiera_base_224': hiera_base_224,
            'hiera_base_plus_224': hiera_base_plus_224,
            'hiera_large_224': hiera_large_224,
            'hiera_huge_224': hiera_huge_224,
        }
        
        self.video_models = {
            'hiera_base_16x224': hiera_base_16x224,
            'hiera_base_plus_16x224': hiera_base_plus_16x224,
            'hiera_large_16x224': hiera_large_16x224,
            'hiera_huge_16x224': hiera_huge_16x224,
        }

    def log_test(self, test_name: str, passed: bool, message: str = ""):
        """Log test results."""
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
        if message:
            print(f"    {message}")
        
        if passed:
            self.results['passed'] += 1
        else:
            self.results['failed'] += 1
            self.results['errors'].append(f"{test_name}: {message}")

    def test_model_creation(self) -> bool:
        """Test basic model creation for all variants."""
        print("\n🧪 Testing Model Creation...")
        
        all_passed = True
        
        # Test image models
        for name, model_func in self.image_models.items():
            try:
                model = model_func(pretrained=False, num_classes=10)
                param_count = sum(p.numel() for p in model.parameters())
                self.log_test(f"Create {name}", True, f"Parameters: {param_count:,}")
            except Exception as e:
                self.log_test(f"Create {name}", False, str(e))
                all_passed = False
        
        # Test video models  
        for name, model_func in self.video_models.items():
            try:
                model = model_func(pretrained=False, num_classes=10)
                param_count = sum(p.numel() for p in model.parameters())
                self.log_test(f"Create {name}", True, f"Parameters: {param_count:,}")
            except Exception as e:
                self.log_test(f"Create {name}", False, str(e))
                all_passed = False
                
        return all_passed

    def test_2d_image_forward(self) -> bool:
        """Test forward pass with 2D image data."""
        print("\n🖼️  Testing 2D Image Forward Pass...")
        
        all_passed = True
        batch_size = 2
        
        for name, model_func in self.image_models.items():
            try:
                # Create model
                model = model_func(pretrained=False, num_classes=1000)
                model.to(self.device)
                model.eval()
                
                # Create dummy image data [B, C, H, W]
                x = torch.randn(batch_size, 3, 224, 224, device=self.device)
                
                # Time the forward pass
                start_time = time.time()
                with torch.no_grad():
                    output = model(x)
                forward_time = time.time() - start_time
                
                # Check output shape
                expected_shape = (batch_size, 1000)
                if output.shape == expected_shape:
                    self.log_test(
                        f"{name} forward", True,
                        f"Shape: {output.shape}, Time: {forward_time:.3f}s"
                    )
                else:
                    self.log_test(
                        f"{name} forward", False,
                        f"Expected {expected_shape}, got {output.shape}"
                    )
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"{name} forward", False, str(e))
                all_passed = False
                
        return all_passed

    def test_3d_video_forward(self) -> bool:
        """Test forward pass with 3D video data."""
        print("\n🎬 Testing 3D Video Forward Pass...")
        
        all_passed = True
        batch_size = 1  # Use smaller batch for video due to memory
        
        for name, model_func in self.video_models.items():
            try:
                # Create model
                model = model_func(pretrained=False, num_classes=400)
                model.to(self.device)
                model.eval()
                
                # Create dummy video data [B, C, T, H, W]
                x = torch.randn(batch_size, 3, 16, 224, 224, device=self.device)
                
                # Time the forward pass
                start_time = time.time()
                with torch.no_grad():
                    output = model(x)
                forward_time = time.time() - start_time
                
                # Check output shape
                expected_shape = (batch_size, 400)
                if output.shape == expected_shape:
                    self.log_test(
                        f"{name} forward", True,
                        f"Shape: {output.shape}, Time: {forward_time:.3f}s"
                    )
                else:
                    self.log_test(
                        f"{name} forward", False,
                        f"Expected {expected_shape}, got {output.shape}"
                    )
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"{name} forward", False, str(e))
                all_passed = False
                
        return all_passed

    def test_training_inference_modes(self) -> bool:
        """Test training and inference mode switching."""
        print("\n🔄 Testing Training/Inference Modes...")
        
        try:
            model = hiera_tiny_224(pretrained=False, num_classes=10)
            model.to(self.device)
            
            x = torch.randn(1, 3, 224, 224, device=self.device)
            
            # Test training mode
            model.train()
            train_output = model(x)
            
            # Test inference mode
            model.eval()
            with torch.no_grad():
                eval_output = model(x)
            
            # Both should work and have same shape
            if train_output.shape == eval_output.shape == (1, 10):
                self.log_test("Training/Inference modes", True, 
                             f"Both modes work, shape: {train_output.shape}")
                return True
            else:
                self.log_test("Training/Inference modes", False,
                             f"Shape mismatch: train={train_output.shape}, eval={eval_output.shape}")
                return False
                
        except Exception as e:
            self.log_test("Training/Inference modes", False, str(e))
            return False

    def test_custom_parameters(self) -> bool:
        """Test model creation with custom parameters."""
        print("\n⚙️  Testing Custom Parameters...")
        
        all_passed = True
        
        # Test custom configurations
        custom_configs = [
            {"num_classes": 5, "embed_dim": 64, "num_heads": 1},
            {"num_classes": 100, "stages": (1, 2, 4, 1)},
            {"input_size": (128, 128), "patch_stride": (2, 2)},
        ]
        
        for i, config in enumerate(custom_configs):
            try:
                model = Hiera(**config)
                param_count = sum(p.numel() for p in model.parameters())
                self.log_test(f"Custom config {i+1}", True, f"Parameters: {param_count:,}")
            except Exception as e:
                self.log_test(f"Custom config {i+1}", False, str(e))
                all_passed = False
                
        return all_passed

    def test_different_input_sizes(self) -> bool:
        """Test models with different input sizes."""
        print("\n📐 Testing Different Input Sizes...")
        
        all_passed = True
        
        # Test configurations: (input_size, model_config, test_input_shape)
        test_configs = [
            # Standard 224x224 model
            ((224, 224), {"num_classes": 10}, (1, 3, 224, 224)),
            ((224, 224), {"num_classes": 10}, (2, 3, 224, 224)),  # Different batch size
            
            # Properly configured 128x128 model
            ((128, 128), {"num_classes": 10}, (1, 3, 128, 128)),
            
            # Custom 320x320 model
            ((320, 320), {"num_classes": 10}, (1, 3, 320, 320)),
        ]
        
        for input_size, model_config, test_shape in test_configs:
            try:
                # Create model with correct input_size configuration
                model = Hiera(input_size=input_size, **model_config)
                model.to(self.device)
                model.eval()
                
                x = torch.randn(*test_shape, device=self.device)
                with torch.no_grad():
                    output = model(x)
                    
                expected_batch = test_shape[0]
                if output.shape[0] == expected_batch and output.shape[1] == model_config["num_classes"]:
                    self.log_test(f"Input size {test_shape} (model: {input_size})", True, 
                                f"Output: {output.shape}")
                else:
                    self.log_test(f"Input size {test_shape} (model: {input_size})", False, 
                                f"Expected batch {expected_batch}, got {output.shape}")
                    all_passed = False
                    
            except Exception as e:
                self.log_test(f"Input size {test_shape} (model: {input_size})", False, str(e))
                all_passed = False
                
        return all_passed

    def test_memory_usage(self) -> bool:
        """Basic memory usage test."""
        print("\n💾 Testing Memory Usage...")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
                
                model = hiera_base_224(pretrained=False, num_classes=1000)
                model.to(self.device)
                
                x = torch.randn(1, 3, 224, 224, device=self.device)
                output = model(x)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_mb = (peak_memory - initial_memory) / 1024 / 1024
                
                self.log_test("Memory usage", True, f"Peak usage: {memory_mb:.1f} MB")
            else:
                self.log_test("Memory usage", True, "CPU mode - skipped GPU memory test")
                
            return True
            
        except Exception as e:
            self.log_test("Memory usage", False, str(e))
            return False

    def test_error_cases(self) -> bool:
        """Test error handling with invalid inputs."""
        print("\n🚨 Testing Error Cases...")
        
        model = hiera_tiny_224(pretrained=False, num_classes=10)
        model.to(self.device)
        model.eval()
        
        # Test invalid input dimensions
        invalid_inputs = [
            torch.randn(1, 2, 224, 224, device=self.device),  # Wrong channels
            torch.randn(1, 3, 224, device=self.device),      # Missing dimension
        ]
        
        errors_caught = 0
        for i, x in enumerate(invalid_inputs):
            try:
                with torch.no_grad():
                    output = model(x)
                # If we get here, the model didn't catch the error
                self.log_test(f"Error case {i+1}", False, "Expected error but model ran")
            except Exception:
                # Expected behavior
                errors_caught += 1
                
        if errors_caught == len(invalid_inputs):
            self.log_test("Error handling", True, f"Caught {errors_caught} expected errors")
            return True
        else:
            self.log_test("Error handling", False, 
                         f"Only caught {errors_caught}/{len(invalid_inputs)} expected errors")
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("🚀 Starting Hiera Model Tests...")
        print("=" * 50)
        
        test_functions = [
            self.test_model_creation,
            self.test_2d_image_forward,
            self.test_3d_video_forward,
            self.test_training_inference_modes,
            self.test_custom_parameters,
            self.test_different_input_sizes,
            self.test_memory_usage,
            self.test_error_cases,
        ]
        
        all_passed = True
        for test_func in test_functions:
            try:
                test_passed = test_func()
                all_passed = all_passed and test_passed
            except Exception as e:
                print(f"❌ Test function {test_func.__name__} failed: {e}")
                all_passed = False
        
        return all_passed

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📈 Success Rate: {self.results['passed']/(self.results['passed']+self.results['failed'])*100:.1f}%")
        
        if self.results['errors']:
            print("\n🔍 Failed Tests:")
            for error in self.results['errors']:
                print(f"  • {error}")
                
        overall_success = self.results['failed'] == 0
        status = "🎉 ALL TESTS PASSED!" if overall_success else "⚠️  SOME TESTS FAILED"
        print(f"\n{status}")
        
        return overall_success


def main():
    """Main test execution function."""
    print("🧪 Hiera Model Test Suite")
    print("=" * 30)
    
    tester = HieraModelTester()
    success = tester.run_all_tests()
    overall_success = tester.print_summary()
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)


# Pytest-compatible test functions
def test_hiera_model_creation():
    tester = HieraModelTester()
    assert tester.test_model_creation(), "Model creation test failed"

def test_hiera_2d_forward():
    tester = HieraModelTester()
    assert tester.test_2d_image_forward(), "2D forward pass test failed"

def test_hiera_3d_forward():
    tester = HieraModelTester()
    assert tester.test_3d_video_forward(), "3D forward pass test failed"

def test_hiera_modes():
    tester = HieraModelTester()
    assert tester.test_training_inference_modes(), "Training/inference modes test failed"

def test_hiera_custom_params():
    tester = HieraModelTester()
    assert tester.test_custom_parameters(), "Custom parameters test failed"


if __name__ == "__main__":
    main()