import torch  # Add this import
import torch.nn as nn
from torch.quantization import quantize_dynamic
from .utils import debug, error

def quantize_model(model: nn.Module, bits: int = 8) -> nn.Module:
    if bits == 8:
        try:
            qm = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            debug("Applied 8-bit quantization")
            return qm
        except Exception as e:
            error("Quantization failed", e)
    return model