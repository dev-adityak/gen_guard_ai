"""
Model Loader Module
Handles loading and caching of LLMs for GenGuard-AI.
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    parameters: str
    context_length: int
    description: str


class ModelLoader:
    """
    Model loader with caching and multiple model support.
    
    Supports:
    - Phi-3-mini (3.8B parameters)
    - Gemma-2B (2B parameters)
    - Llama-3.2-1B (1B parameters)
    - Qwen2-0.5B (0.5B parameters) - lightweight fallback
    """

    _instance: Optional['ModelLoader'] = None
    _cache = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def get_available_models() -> List[ModelInfo]:
        """Return list of available models."""
        return [
            ModelInfo(
                name="microsoft/Phi-3-mini-4k-instruct",
                parameters="3.8B",
                context_length=4096,
                description="Microsoft's efficient Phi-3 model, great for privacy tasks"
            ),
            ModelInfo(
                name="google/gemma-2b",
                parameters="2B",
                context_length=8192,
                description="Google's lightweight Gemma model"
            ),
            ModelInfo(
                name="meta-llama/Llama-3.2-1B-Instruct",
                parameters="1B",
                context_length=128000,
                description="Meta's smallest Llama, excellent efficiency"
            ),
            ModelInfo(
                name="Qwen/Qwen2-0.5B-Instruct",
                parameters="0.5B",
                context_length=32768,
                description="Alibaba's ultra-lightweight model for resource-constrained environments"
            ),
        ]

    def load(
        self,
        model_name: str = "Qwen/Qwen2-0.5B-Instruct",
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        max_length: int = 512
    ) -> Tuple[any, any]:
        """
        Load model and tokenizer.
        
        Args:
            model_name: Hugging Face model ID
            device: 'cuda' or 'cpu' (auto-detect if None)
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization (default, memory efficient)
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        cache_key = f"{model_name}_{device}_{load_in_4bit}"
        
        if cache_key in ModelLoader._cache:
            logger.info(f"Using cached model: {model_name}")
            return ModelLoader._cache[cache_key]

        logger.info(f"Loading model: {model_name} on {device}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            quantization_config = None
            if load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )

            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            else:
                model_kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            model.eval()
            
            ModelLoader._cache[cache_key] = (model, tokenizer)
            logger.info(f"Model loaded successfully: {model_name}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise ModelLoadingError(f"Could not load model: {model_name}") from e

    def unload(self, model_name: str = None):
        """Unload model from cache to free memory."""
        if model_name:
            keys_to_remove = [k for k in ModelLoader._cache if model_name in k]
            for key in keys_to_remove:
                del ModelLoader._cache[key]
                logger.info(f"Unloaded model: {key}")
        else:
            ModelLoader._cache.clear()
            logger.info("Cleared all model cache")

    def get_memory_usage(self) -> dict:
        """Get current memory usage."""
        if not torch.cuda.is_available():
            return {"device": "cpu", "memory_used_mb": 0}
        
        return {
            "device": "cuda",
            "memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
            "memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
            "max_memory_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024
        }


class ModelLoadingError(Exception):
    """Error raised when model loading fails."""
    pass


def load_model(
    model_name: str = "Qwen/Qwen2-0.5B-Instruct",
    device: str = None,
    **kwargs
) -> Tuple[any, any]:
    """
    Convenience function to load a model.
    
    Example:
        model, tokenizer = load_model("microsoft/Phi-3-mini-4k-instruct")
    """
    loader = ModelLoader()
    return loader.load(model_name, device, **kwargs)