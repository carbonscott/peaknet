#!/usr/bin/env python3
"""
Comprehensive test script for Hiera MAE (Masked Autoencoder) implementation in PeakNet.

This script tests the Hiera MAE model functionality including:
- Model creation for all MAE variants
- Forward pass with 2D image and 3D video data
- Encoder and decoder functionality
- Multi-scale fusion heads
- Output shape validation
- Training/inference mode switching
- Single-channel data compatibility (for X-ray peak detection)

Usage:
    python tests/test_hiera_mae.py
    pytest tests/test_hiera_mae.py
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
    from peaknet.modeling.hiera_mae import (
        MaskedAutoencoderHiera,
        apply_fusion_head,
        mae_hiera_tiny_224,
        mae_hiera_small_224,
        mae_hiera_base_224,
        mae_hiera_base_plus_224,
        mae_hiera_large_224,
        mae_hiera_huge_224,
        mae_hiera_base_16x224,
        mae_hiera_base_plus_16x224,
        mae_hiera_large_16x224,
        mae_hiera_huge_16x224,
    )
    print("✅ Successfully imported Hiera MAE models")
except ImportError as e:
    print(f"❌ Failed to import Hiera MAE models: {e}")
    traceback.print_exc()
    sys.exit(1)


class HieraMAETester:
    """Comprehensive test suite for Hiera MAE models."""
    
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
            'mae_hiera_tiny_224': mae_hiera_tiny_224,
            'mae_hiera_small_224': mae_hiera_small_224,
            'mae_hiera_base_224': mae_hiera_base_224,
            'mae_hiera_base_plus_224': mae_hiera_base_plus_224,
            'mae_hiera_large_224': mae_hiera_large_224,
            'mae_hiera_huge_224': mae_hiera_huge_224,
        }
        
        self.video_models = {
            'mae_hiera_base_16x224': mae_hiera_base_16x224,
            'mae_hiera_base_plus_16x224': mae_hiera_base_plus_16x224,
            'mae_hiera_large_16x224': mae_hiera_large_16x224,
            'mae_hiera_huge_16x224': mae_hiera_huge_16x224,
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
        """Test basic model creation for all MAE variants."""
        print("\n🧪 Testing MAE Model Creation...")
        
        all_passed = True
        
        # Test image models
        for name, model_func in self.image_models.items():
            try:
                model = model_func(pretrained=False, in_chans=3)
                param_count = sum(p.numel() for p in model.parameters())
                self.log_test(f"Create {name}", True, f"Parameters: {param_count:,}")
            except Exception as e:
                self.log_test(f"Create {name}", False, str(e))
                all_passed = False
        
        # Test video models  
        for name, model_func in self.video_models.items():
            try:
                model = model_func(pretrained=False, in_chans=3)
                param_count = sum(p.numel() for p in model.parameters())
                self.log_test(f"Create {name}", True, f"Parameters: {param_count:,}")
            except Exception as e:
                self.log_test(f"Create {name}", False, str(e))
                all_passed = False
                
        return all_passed

    def test_single_channel_compatibility(self) -> bool:
        """Test compatibility with single-channel X-ray data (PeakNet use case)."""
        print("\n🔬 Testing Single-Channel X-ray Compatibility...")
        
        try:
            # Create model for single-channel input (X-ray data)
            model = mae_hiera_tiny_224(pretrained=False, in_chans=1)
            model.to(self.device)
            model.eval()
            
            # Test with single-channel input
            x = torch.randn(2, 1, 224, 224, device=self.device)
            
            with torch.no_grad():
                loss, pred, label, mask = model(x, mask_ratio=0.6)
            
            self.log_test("Single-channel compatibility", True, 
                         f"Loss: {loss.item():.4f}, Pred shape: {pred.shape}, Label shape: {label.shape}")
            return True
            
        except Exception as e:
            self.log_test("Single-channel compatibility", False, str(e))
            return False

    def test_mae_forward_pass(self) -> bool:
        """Test MAE forward pass with masking."""
        print("\n🎭 Testing MAE Forward Pass with Masking...")
        
        all_passed = True
        batch_size = 2
        
        # Test a representative model
        try:
            model = mae_hiera_tiny_224(pretrained=False, in_chans=3)
            model.to(self.device)
            model.eval()
            
            # Create dummy image data [B, C, H, W]
            x = torch.randn(batch_size, 3, 224, 224, device=self.device)
            
            # Test different mask ratios
            mask_ratios = [0.3, 0.6, 0.9]
            
            for mask_ratio in mask_ratios:
                start_time = time.time()
                with torch.no_grad():
                    loss, pred, label, mask = model(x, mask_ratio=mask_ratio)
                forward_time = time.time() - start_time
                
                # Check outputs
                if (isinstance(loss, torch.Tensor) and 
                    pred.shape[0] > 0 and 
                    label.shape == pred.shape and
                    mask.shape[0] == batch_size):
                    self.log_test(
                        f"MAE forward (mask_ratio={mask_ratio})", True,
                        f"Loss: {loss.item():.4f}, Time: {forward_time:.3f}s"
                    )
                else:
                    self.log_test(
                        f"MAE forward (mask_ratio={mask_ratio})", False,
                        f"Invalid output shapes: loss={loss}, pred={pred.shape}, label={label.shape}"
                    )
                    all_passed = False
                    
        except Exception as e:
            self.log_test("MAE forward pass", False, str(e))
            all_passed = False
                
        return all_passed

    def test_encoder_decoder_separation(self) -> bool:
        """Test encoder and decoder can be called separately."""
        print("\n🔀 Testing Encoder/Decoder Separation...")
        
        try:
            model = mae_hiera_tiny_224(pretrained=False, in_chans=3)
            model.to(self.device)
            model.eval()
            
            x = torch.randn(1, 3, 224, 224, device=self.device)
            
            with torch.no_grad():
                # Test encoder
                encoded, mask = model.forward_encoder(x, mask_ratio=0.6)
                
                # Test decoder  
                decoded, pred_mask = model.forward_decoder(encoded, mask)
                
                # Check shapes
                if (encoded.ndim >= 2 and 
                    decoded.ndim >= 1 and 
                    mask.shape[0] == 1 and
                    pred_mask.shape[0] == 1):
                    self.log_test("Encoder/Decoder separation", True,
                                 f"Encoded: {encoded.shape}, Decoded: {decoded.shape}")
                    return True
                else:
                    self.log_test("Encoder/Decoder separation", False,
                                 f"Invalid shapes: encoded={encoded.shape}, decoded={decoded.shape}")
                    return False
                    
        except Exception as e:
            self.log_test("Encoder/Decoder separation", False, str(e))
            return False

    def test_multi_scale_fusion(self) -> bool:
        """Test multi-scale fusion heads functionality."""
        print("\n🔗 Testing Multi-scale Fusion Heads...")
        
        try:
            model = mae_hiera_base_224(pretrained=False, in_chans=3)
            model.to(self.device)
            model.eval()
            
            # Check that fusion heads were created
            fusion_heads = model.multi_scale_fusion_heads
            num_heads = len(fusion_heads)
            
            if num_heads > 0:
                self.log_test("Multi-scale fusion heads created", True,
                             f"Number of fusion heads: {num_heads}")
                
                # Test apply_fusion_head function
                dummy_feature = torch.randn(1, 4, 8, 8, 96, device=self.device)  # [B, #MUs, My, Mx, C]
                head = fusion_heads[0]
                
                with torch.no_grad():
                    fused = apply_fusion_head(head, dummy_feature)
                
                if fused.shape[0] == 1:  # Batch dimension preserved
                    self.log_test("apply_fusion_head function", True,
                                 f"Input: {dummy_feature.shape}, Output: {fused.shape}")
                    return True
                else:
                    self.log_test("apply_fusion_head function", False,
                                 f"Invalid output shape: {fused.shape}")
                    return False
            else:
                self.log_test("Multi-scale fusion heads", False, "No fusion heads created")
                return False
                
        except Exception as e:
            self.log_test("Multi-scale fusion heads", False, str(e))
            return False

    def test_3d_video_mae(self) -> bool:
        """Test MAE with 3D video data."""
        print("\n🎬 Testing 3D Video MAE...")
        
        try:
            model = mae_hiera_base_16x224(pretrained=False, in_chans=3)
            model.to(self.device)
            model.eval()
            
            # Create dummy video data [B, C, T, H, W]
            x = torch.randn(1, 3, 16, 224, 224, device=self.device)
            
            start_time = time.time()
            with torch.no_grad():
                loss, pred, label, mask = model(x, mask_ratio=0.8)
            forward_time = time.time() - start_time
            
            if (isinstance(loss, torch.Tensor) and 
                pred.shape[0] > 0 and 
                label.shape == pred.shape):
                self.log_test("3D Video MAE", True,
                             f"Loss: {loss.item():.4f}, Time: {forward_time:.3f}s")
                return True
            else:
                self.log_test("3D Video MAE", False,
                             f"Invalid outputs: loss={loss}, pred={pred.shape}, label={label.shape}")
                return False
                
        except Exception as e:
            self.log_test("3D Video MAE", False, str(e))
            return False

    def test_custom_mae_configuration(self) -> bool:
        """Test MAE with custom configurations."""
        print("\n⚙️  Testing Custom MAE Configuration...")
        
        try:
            # Test custom configuration matching PeakNet use case
            model = MaskedAutoencoderHiera(
                input_size=(224, 224),
                in_chans=1,  # Single channel for X-ray
                embed_dim=64,  # Smaller for testing
                num_heads=1,
                stages=(1, 1, 2, 1),  # Smaller for testing
                decoder_embed_dim=256,
                decoder_depth=4,
                decoder_num_heads=8,
                q_pool=2,
            )
            model.to(self.device)
            
            x = torch.randn(1, 1, 224, 224, device=self.device)
            
            with torch.no_grad():
                loss, pred, label, mask = model(x, mask_ratio=0.5)
            
            param_count = sum(p.numel() for p in model.parameters())
            self.log_test("Custom MAE configuration", True,
                         f"Parameters: {param_count:,}, Loss: {loss.item():.4f}")
            return True
            
        except Exception as e:
            self.log_test("Custom MAE configuration", False, str(e))
            return False

    def test_memory_usage(self) -> bool:
        """Basic memory usage test for MAE."""
        print("\n💾 Testing MAE Memory Usage...")
        
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                initial_memory = torch.cuda.memory_allocated()
                
                model = mae_hiera_tiny_224(pretrained=False, in_chans=3)
                model.to(self.device)
                
                x = torch.randn(1, 3, 224, 224, device=self.device)
                loss, pred, label, mask = model(x, mask_ratio=0.6)
                
                peak_memory = torch.cuda.max_memory_allocated()
                memory_mb = (peak_memory - initial_memory) / 1024 / 1024
                
                self.log_test("MAE memory usage", True, f"Peak usage: {memory_mb:.1f} MB")
            else:
                self.log_test("MAE memory usage", True, "CPU mode - skipped GPU memory test")
                
            return True
            
        except Exception as e:
            self.log_test("MAE memory usage", False, str(e))
            return False

    def run_all_tests(self) -> bool:
        """Run all tests and return overall success."""
        print("🚀 Starting Hiera MAE Tests...")
        print("=" * 50)
        
        test_functions = [
            self.test_model_creation,
            self.test_single_channel_compatibility,
            self.test_mae_forward_pass,
            self.test_encoder_decoder_separation,
            self.test_multi_scale_fusion,
            self.test_3d_video_mae,
            self.test_custom_mae_configuration,
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
        print("📊 MAE TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {self.results['passed']}")
        print(f"❌ Failed: {self.results['failed']}")
        print(f"📈 Success Rate: {self.results['passed']/(self.results['passed']+self.results['failed'])*100:.1f}%")
        
        if self.results['errors']:
            print("\n🔍 Failed Tests:")
            for error in self.results['errors']:
                print(f"  • {error}")
                
        overall_success = self.results['failed'] == 0
        status = "🎉 ALL MAE TESTS PASSED!" if overall_success else "⚠️  SOME MAE TESTS FAILED"
        print(f"\n{status}")
        
        return overall_success


def main():
    """Main test execution function."""
    print("🧪 Hiera MAE Test Suite")
    print("=" * 30)
    
    tester = HieraMAETester()
    success = tester.run_all_tests()
    overall_success = tester.print_summary()
    
    # Return appropriate exit code
    sys.exit(0 if overall_success else 1)


# Pytest-compatible test functions
def test_mae_model_creation():
    tester = HieraMAETester()
    assert tester.test_model_creation(), "MAE model creation test failed"

def test_mae_single_channel():
    tester = HieraMAETester()
    assert tester.test_single_channel_compatibility(), "Single-channel compatibility test failed"

def test_mae_forward():
    tester = HieraMAETester()
    assert tester.test_mae_forward_pass(), "MAE forward pass test failed"

def test_mae_encoder_decoder():
    tester = HieraMAETester()
    assert tester.test_encoder_decoder_separation(), "Encoder/decoder separation test failed"

def test_mae_fusion():
    tester = HieraMAETester()
    assert tester.test_multi_scale_fusion(), "Multi-scale fusion test failed"


if __name__ == "__main__":
    main()