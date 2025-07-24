#!/usr/bin/env python3
"""
Test script for Hiera Segmentation implementation.

This script tests the basic functionality of the HieraSegmentation model
including model creation, forward pass, and output shape validation.
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
    from peaknet.modeling.hiera_segmentation import (
        HieraSegmentation,
        hiera_seg_tiny_224,
        hiera_seg_small_224,
        hiera_seg_base_224,
        hiera_seg_base_plus_224,
        hiera_seg_large_224,
    )
    print("✅ Successfully imported Hiera Segmentation models")
except ImportError as e:
    print(f"❌ Failed to import Hiera Segmentation models: {e}")
    traceback.print_exc()
    sys.exit(1)


class HieraSegmentationTester:
    """Test suite for Hiera Segmentation models."""
    
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
        self.models = {
            'hiera_seg_tiny_224': hiera_seg_tiny_224,
            'hiera_seg_small_224': hiera_seg_small_224,
            'hiera_seg_base_224': hiera_seg_base_224,
            'hiera_seg_base_plus_224': hiera_seg_base_plus_224,
            'hiera_seg_large_224': hiera_seg_large_224,
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
        """Test basic model creation for all segmentation variants."""
        print("\n🧪 Testing Segmentation Model Creation...")
        
        all_passed = True
        
        for name, model_func in self.models.items():
            try:
                model = model_func(pretrained=False, in_chans=3, num_classes=2)
                param_count = sum(p.numel() for p in model.parameters())
                self.log_test(f"Create {name}", True, f"Parameters: {param_count:,}")
            except Exception as e:
                self.log_test(f"Create {name}", False, str(e))
                all_passed = False
                
        return all_passed

    def test_custom_segmentation_model(self) -> bool:
        """Test custom segmentation model creation."""
        print("\n⚙️ Testing Custom Segmentation Model...")
        
        try:
            model = HieraSegmentation(
                num_classes=5,  # Multi-class segmentation
                input_size=(224, 224),
                in_chans=1,  # Single channel for X-ray
                embed_dim=64,
                num_heads=1,
                stages=(1, 1, 2, 1),
                decoder_embed_dim=256,
                decoder_depth=4,
                decoder_num_heads=8,
                q_pool=2,
            )
            param_count = sum(p.numel() for p in model.parameters())
            self.log_test("Custom segmentation model", True, 
                         f"Parameters: {param_count:,}, Classes: {model.num_classes}")
            return True
        except Exception as e:
            self.log_test("Custom segmentation model", False, str(e))
            return False

    def test_forward_pass(self) -> bool:
        """Test forward pass with different configurations."""
        print("\n🎯 Testing Segmentation Forward Pass...")
        
        all_passed = True
        
        # Test cases: (batch_size, channels, height, width, num_classes)
        test_cases = [
            (1, 3, 224, 224, 2),    # Single image, RGB, binary segmentation
            (2, 1, 224, 224, 5),    # Batch, single-channel, multi-class
            (1, 3, 224, 224, 10),   # Single image, many classes
        ]
        
        for batch_size, channels, height, width, num_classes in test_cases:
            try:
                model = hiera_seg_tiny_224(
                    pretrained=False, 
                    in_chans=channels, 
                    num_classes=num_classes
                )
                model.to(self.device)
                model.eval()
                
                x = torch.randn(batch_size, channels, height, width, device=self.device)
                
                start_time = time.time()
                with torch.no_grad():
                    output = model(x)
                forward_time = time.time() - start_time
                
                expected_shape = (batch_size, num_classes, height, width)
                if output.shape == expected_shape:
                    self.log_test(
                        f"Forward pass {batch_size}x{channels}x{height}x{width} → {num_classes} classes",
                        True,
                        f"Output: {output.shape}, Time: {forward_time:.3f}s"
                    )
                else:
                    self.log_test(
                        f"Forward pass {batch_size}x{channels}x{height}x{width} → {num_classes} classes",
                        False,
                        f"Expected {expected_shape}, got {output.shape}"
                    )
                    all_passed = False
                    
            except Exception as e:
                self.log_test(
                    f"Forward pass {batch_size}x{channels}x{height}x{width} → {num_classes} classes",
                    False, str(e)
                )
                all_passed = False
                
        return all_passed

    def test_single_channel_xray(self) -> bool:
        """Test single-channel X-ray compatibility."""
        print("\n🔬 Testing Single-Channel X-ray Compatibility...")
        
        try:
            # Create model for single-channel X-ray segmentation
            model = hiera_seg_tiny_224(
                pretrained=False, 
                in_chans=1, 
                num_classes=3  # background, peak, noise
            )
            model.to(self.device)
            model.eval()
            
            # Test with single-channel input (X-ray data)
            x = torch.randn(2, 1, 224, 224, device=self.device)
            
            with torch.no_grad():
                output = model(x)
            
            expected_shape = (2, 3, 224, 224)
            if output.shape == expected_shape:
                self.log_test("Single-channel X-ray segmentation", True, 
                             f"Output shape: {output.shape}")
                return True
            else:
                self.log_test("Single-channel X-ray segmentation", False,
                             f"Expected {expected_shape}, got {output.shape}")
                return False
                
        except Exception as e:
            self.log_test("Single-channel X-ray segmentation", False, str(e))
            return False

    def test_output_properties(self) -> bool:
        """Test segmentation output properties."""
        print("\n📊 Testing Segmentation Output Properties...")
        
        try:
            model = hiera_seg_tiny_224(pretrained=False, in_chans=3, num_classes=4)
            model.to(self.device)
            model.eval()
            
            x = torch.randn(1, 3, 224, 224, device=self.device)
            
            with torch.no_grad():
                output = model(x)
            
            # Check output properties
            checks = []
            
            # Shape check
            checks.append(("Shape", output.shape == (1, 4, 224, 224)))
            
            # Data type check
            checks.append(("Data type", output.dtype == torch.float32))
            
            # Gradient check
            checks.append(("Requires grad", output.requires_grad == False))  # eval mode
            
            # Value range check (logits can be any real number)
            checks.append(("Finite values", torch.isfinite(output).all().item()))
            
            all_checks_passed = all(passed for _, passed in checks)
            
            check_details = ", ".join([f"{name}: {'✓' if passed else '✗'}" for name, passed in checks])
            self.log_test("Segmentation output properties", all_checks_passed, check_details)
            
            return all_checks_passed
            
        except Exception as e:
            self.log_test("Segmentation output properties", False, str(e))
            return False

    def test_memory_usage(self) -> bool:
        """Test memory usage for segmentation."""
        print("\n💾 Testing Segmentation Memory Usage...")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
                
                model = hiera_seg_tiny_224(pretrained=False, in_chans=3, num_classes=2)
                model.to(self.device)
                
                x = torch.randn(1, 3, 224, 224, device=self.device)
                output = model(x)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_mb = (peak_memory - initial_memory) / 1024 / 1024
                
                self.log_test("Segmentation memory usage", True, f"Peak usage: {memory_mb:.1f} MB")
            else:
                self.log_test("Segmentation memory usage", True, "CPU mode - skipped GPU memory test")
                
            return True
            
        except Exception as e:
            self.log_test("Segmentation memory usage", False, str(e))
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("🚀 Starting Hiera Segmentation Tests...")
        print("=" * 50)
        
        test_functions = [
            self.test_model_creation,
            self.test_custom_segmentation_model,
            self.test_forward_pass,
            self.test_single_channel_xray,
            self.test_output_properties,
            self.test_memory_usage,
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
        print("📊 SEGMENTATION TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        
        if self.results['passed'] + self.results['failed'] > 0:
            success_rate = self.results['passed']/(self.results['passed']+self.results['failed'])*100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        if self.results['errors']:
            print("\n🔍 Failed Tests:")
            for error in self.results['errors']:
                print(f"  • {error}")
                
        overall_success = self.results['failed'] == 0
        status = "🎉 ALL SEGMENTATION TESTS PASSED!" if overall_success else "⚠️ SOME SEGMENTATION TESTS FAILED"
        print(f"\n{status}")
        
        return overall_success


def main():
    """Main test execution function."""
    print("🧪 Hiera Segmentation Test Suite")
    print("=" * 35)
    
    tester = HieraSegmentationTester()
    success = tester.run_all_tests()
    overall_success = tester.print_summary()
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()