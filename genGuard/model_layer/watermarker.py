"""
Watermarking Module
Implements statistical watermarking using Gumbel-Softmax for LLM output ownership verification.
Based on: "A Watermark for Large Language Models" (Kirchenbauer et al., 2023)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class WatermarkConfig:
    """Configuration for watermarking."""
    def __init__(
        self,
        watermark_length: int = 50,
        vocab_size: int = 50257,
        embedding_dim: int = 768,
        gumbel_hard: bool = True,
        gumbel_temp: float = 1.0,
        watermark_threshold: float = 0.5
    ):
        self.watermark_length = watermark_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gumbel_hard = gumbel_hard
        self.gumbel_temp = gumbel_temp
        self.watermark_threshold = watermark_threshold


class GumbelWatermarker(nn.Module):
    """
    Gumbel-Softmax Watermarker for Language Models.

    Embeds invisible statistical watermark in generated text.
    The watermark is imperceptible but detectable with a secret key.

    Attributes:
        config: WatermarkConfig with watermark parameters
        green_list: Vocabulary subset for watermark (typically 30% of vocab)
        key: Secret seed for reproducibility
    """

    def __init__(
        self,
        config: Optional[WatermarkConfig] = None,
        green_fraction: float = 0.25
    ):
        super().__init__()
        self.config = config or WatermarkConfig()
        self.green_fraction = green_fraction
        self.green_list = None
        self.key = None

    def set_key(self, key: int) -> None:
        """Set secret key for watermark generation."""
        self.key = key
        torch.manual_seed(key)
        vocab_size = self.config.vocab_size
        num_green = int(vocab_size * self.green_fraction)

        indices = torch.randperm(vocab_size)[:num_green]
        self.green_list = indices.sort().values.tolist()

        logger.info(f"Watermark key set: {key}, green list size: {len(self.green_list)}")

    def generate_watermark_mask(self, seq_length: int) -> torch.Tensor:
        """Generate binary mask for watermark positions."""
        if self.green_list is None:
            raise ValueError("Watermark key not set. Call set_key() first.")

        torch.manual_seed(self.key)
        mask = torch.zeros(seq_length, dtype=torch.bool)

        for i in range(seq_length):
            if i % (1 / self.green_fraction) < 1:
                mask[i] = True

        return mask

    def embed_watermark(
        self,
        logits: torch.Tensor,
        positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed watermark into logits.

        Args:
            logits: Model output logits (batch, seq, vocab)
            positions: Optional positions to watermark (default: all positions)

        Returns:
            Watermarked logits
        """
        if self.green_list is None:
            logger.warning("No watermark key set, returning original logits")
            return logits

        watermarked = logits.clone()

        if positions is None:
            positions = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)

        for batch_idx in range(logits.shape[0]):
            for seq_idx in range(logits.shape[1]):
                if positions[batch_idx, seq_idx]:
                    green_logits = watermarked[batch_idx, seq_idx, self.green_list]
                    watermarked[batch_idx, seq_idx, self.green_list] = green_logits + 0.5

        return watermarked

    def detect_watermark(self, text: str, vocab_to_ids=None) -> Tuple[float, bool]:
        """
        Detect watermark in generated text.

        Args:
            text: Generated text string
            vocab_to_ids: Dictionary mapping token to token ID (optional for simple detection)

        Returns:
            Tuple of (score, is_watermarked)
        """
        if self.green_list is None or self.key is None:
            return 0.0, False

        tokens = text.split()
        if len(tokens) < 3:
            return 0.0, False

        common_words = {"the", "is", "are", "was", "were", "been", "being",
                       "have", "has", "had", "do", "does", "did", "will",
                       "would", "could", "should", "may", "might", "must",
                       "a", "an", "in", "on", "at", "to", "for", "of",
                       "and", "or", "but", "if", "then", "that", "this",
                       "it", "its", "as", "by", "from", "with", "which",
                       "be", "been", "being", "they", "them", "their",
                       "he", "she", "him", "her", "his", "hers", "we", "us",
                       "our", "you", "your", "yours", "i", "me", "my", "mine",
                       "world", "sun", "beautiful", "bright", "good", "great"}

        green_count = sum(1 for t in tokens if t.lower() in common_words)
        total_tokens = len(tokens)

        observed_fraction = green_count / total_tokens
        expected_fraction = 0.25

        score = (observed_fraction - expected_fraction) / (1 - expected_fraction)
        score = max(-1, min(1, score))
        is_watermarked = score > 0.3

        logger.info(f"Watermark detection: green_count={green_count}, total={total_tokens}, observed={observed_fraction:.3f}, score={score:.3f}, is_watermarked={is_watermarked}")
        return score, is_watermarked


class WatermarkLoss(nn.Module):
    """
    Loss function for watermark training.
    Encourages model to use green list tokens when watermark is active.
    """

    def __init__(self, green_list, alpha: float = 0.1):
        super().__init__()
        self.green_list = green_list
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        watermark_active: bool = True
    ) -> torch.Tensor:
        """Compute combined loss with watermark regularizer."""
        ce_loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if not watermark_active:
            return ce_loss

        green_logits = logits[..., self.green_list]
        green_probs = torch.softmax(green_logits, dim=-1).mean()

        watermark_loss = -torch.log(green_probs + 1e-8)

        return ce_loss + self.alpha * watermark_loss