"""
Model Quantization Module
Implements lightweight weight quantization for model obfuscation.
Supports PTQ (Post-Training Quantization) and FP8 strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    bits: int = 8
    scheme: str = "symmetric"
    method: str = "ptq"
    per_channel: bool = False
    group_size: int = 128


class ModelQuantizer:
    """
    Model Quantization for IP Protection.

    Reduces model precision to resist model inversion attacks.
    Supported methods:
    - PTQ (Post-Training Quantization): 8-bit, 4-bit
    - FP8: Floating point 8-bit
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.config = config or QuantizationConfig(bits=8)
        self.quantizers: Dict[str, torch.nn.Module] = {}

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        bits: int = 8
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize a single tensor.

        Args:
            tensor: Input tensor to quantize
            bits: Number of bits for quantization

        Returns:
            Tuple of (quantized, scale, zero_point)
        """
        qmin = -(2 ** (bits - 1))
        qmax = (2 ** (bits - 1)) - 1

        if self.config.scheme == "symmetric":
            scale = tensor.abs().max() / (qmax - 1)
            quantized = torch.round(tensor / scale)
            zero_point = torch.tensor(0, dtype=torch.long)
        else:
            scale = (tensor.max() - tensor.min()) / (qmax - qmin)
            zero_point = torch.round(-tensor.min() / scale)
            quantized = torch.round(tensor / scale + zero_point)

        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)
        return quantized, scale, zero_point

    def dequantize_tensor(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize a tensor back to float."""
        if self.config.scheme == "symmetric":
            return quantized.float() * scale
        else:
            return (quantized.float() - zero_point) * scale

    def quantize_model(self, model: nn.Module) -> Dict[str, nn.Module]:
        """
        Quantize entire model.

        Args:
            model: PyTorch model to quantize

        Returns:
            Dictionary mapping layer names to quantizers
        """
        quantized_state = {}
        quantizers = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                continue

            quantized, scale, zero_point = self.quantize_tensor(
                param.data,
                bits=self.config.bits
            )

            quantized_state[name] = (quantized, scale, zero_point)
            quantizers[name] = {
                'scale': scale,
                'zero_point': zero_point,
                'bits': self.config.bits
            }

            logger.info(f"Quantized layer: {name}")

        return quantized_state

    def apply_quantization(
        self,
        model: nn.Module,
        quantized_state: Dict[str, Tuple]
    ) -> nn.Module:
        """
        Apply quantized weights to model.

        Args:
            model: Model to apply quantization to
            quantized_state: Dict from quantize_model

        Returns:
            Model with quantized weights
        """
        for name, (quantized, scale, zero_point) in quantized_state.items():
            dequantized = self.dequantize_tensor(quantized, scale, zero_point)
            param = model.get_parameter(name)
            param.data = dequantized

        return model

    def get_model_size(self, model: nn.Module, bits: int = 32) -> float:
        """
        Calculate model size in MB.

        Args:
            model: PyTorch model
            bits: Original bit precision (default: 32)

        Returns:
            Model size in megabytes
        """
        total_params = 0
        for param in model.parameters():
            total_params += param.numel()

        original_size_mb = (total_params * bits) / (8 * 1024 * 1024)
        quantized_size_mb = (total_params * self.config.bits) / (8 * 1024 * 1024)

        return {
            'original_mb': original_size_mb,
            'quantized_mb': quantized_size_mb,
            'compression_ratio': original_size_mb / quantized_size_mb
        }


class FP8Quantizer(ModelQuantizer):
    """Floating point 8-bit quantization."""

    def __init__(self):
        super().__init__(QuantizationConfig(bits=8, scheme="fp8"))

    def quantize_fp8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize to FP8 format."""
        exp_min = -126
        exp_max = 126

        tensor_flat = tensor.flatten()
        max_val = tensor_flat.max().item()
        min_val = tensor_flat.min().item()
        max_abs = max(abs(max_val), abs(min_val))

        scale = 2.0 ** (exp_max - 8)
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)

        return quantized, scale

    def dequantize_fp8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize from FP8 format."""
        return quantized.float() * scale