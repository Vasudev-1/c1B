import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, QConfig, default_observer
import numpy as np
from typing import Union
import logging

from .utils import debug_print

class ModelQuantizer:
    """Advanced model quantization utilities"""
    
    @staticmethod
    def quantize_model(model: nn.Module, 
                      bits: int = 8,
                      backend: str = 'fbgemm') -> nn.Module:
        """Quantize a PyTorch model"""
        
        debug_print(f"Starting {bits}-bit quantization with {backend} backend...")
        
        try:
            if bits == 8:
                # Dynamic quantization for 8-bit
                quantized_model = quantize_dynamic(
                    model,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
            elif bits == 16:
                # Half precision for 16-bit
                quantized_model = model.half()
            else:
                raise ValueError(f"Unsupported quantization bits: {bits}")
            
            # Estimate size reduction
            original_size = ModelQuantizer._calculate_model_size(model)
            quantized_size = ModelQuantizer._calculate_model_size(quantized_model)
            size_reduction = (1 - quantized_size / original_size) * 100
            
            debug_print(f"Quantization completed:")
            debug_print(f"  Original size: {original_size:.2f} MB")
            debug_print(f"  Quantized size: {quantized_size:.2f} MB")
            debug_print(f"  Size reduction: {size_reduction:.1f}%")
            
            return quantized_model
            
        except Exception as e:
            logging.error(f"Quantization failed: {e}")
            debug_print("Returning original model due to quantization failure")
            return model
    
    @staticmethod
    def _calculate_model_size(model: nn.Module) -> float:
        """Calculate model size in MB"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    @staticmethod
    def benchmark_quantized_model(original_model: nn.Module, 
                                quantized_model: nn.Module,
                                test_data: list) -> dict:
        """Benchmark quantized model against original"""
        
        import time
        
        # Speed benchmark
        start_time = time.time()
        original_outputs = original_model.encode(test_data)
        original_time = time.time() - start_time
        
        start_time = time.time()
        quantized_outputs = quantized_model.encode(test_data)
        quantized_time = time.time() - start_time
        
        # Accuracy benchmark (cosine similarity)
        cosine_similarities = []
        for orig, quant in zip(original_outputs, quantized_outputs):
            cosine_sim = np.dot(orig, quant) / (np.linalg.norm(orig) * np.linalg.norm(quant))
            cosine_similarities.append(cosine_sim)
        
        avg_similarity = np.mean(cosine_similarities)
        
        results = {
            'original_inference_time': original_time,
            'quantized_inference_time': quantized_time,
            'speedup': original_time / quantized_time,
            'average_cosine_similarity': avg_similarity,
            'similarity_std': np.std(cosine_similarities),
            'accuracy_retention': avg_similarity * 100  # percentage
        }
        
        debug_print(f"Quantization benchmark:")
        debug_print(f"  Speedup: {results['speedup']:.2f}x")
        debug_print(f"  Accuracy retention: {results['accuracy_retention']:.1f}%")
        
        return results

# Convenience function for the main embedding module
def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    """Convenience function for quantizing models"""
    return ModelQuantizer.quantize_model(model, bits)
