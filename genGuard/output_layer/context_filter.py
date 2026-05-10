"""
Context-Aware Filter Module
Uses DistilBERT for context classification and filtering decisions.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

try:
    from transformers import (
        DistilBertTokenizer,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not installed. Context filter will be limited.")


@dataclass
class ClassificationResult:
    """Result of context classification."""
    allowed: bool
    confidence: float
    label: str
    reason: str


class ContextAwareFilter:
    """
    Context-Aware Redaction Filter.

    Determines whether detected PII should be redacted
    based on the context of the text.

    Examples:
    - "Patient John Doe" in medical report -> BLOCKED
    - "John wrote a story" in fiction -> ALLOWED
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_labels: int = 2,
        threshold: float = 0.5
    ):
        self.threshold = threshold
        self.num_labels = num_labels
        self.tokenizer = None
        self.model = None

        if TRANSFORMERS_AVAILABLE:
            if model_path:
                self._load_model(model_path)
            else:
                self._init_default_model()
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("Using fallback rule-based classification")

    def _init_default_model(self) -> None:
        """Initialize default DistilBERT model."""
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=self.num_labels
        )
        self.model.eval()

    def _load_model(self, model_path: str) -> None:
        """Load fine-tuned model."""
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def classify(self, text: str, context: Optional[str] = None) -> ClassificationResult:
        """Classify whether text context allows PII."""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return self._fallback_classify(text, context)

        input_text = text
        if context:
            input_text = f"{context}: {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, predicted_class = torch.max(probs, dim=-1)

        allowed = predicted_class.item() == 1
        confidence_value = confidence.item()
        label = "allowed" if allowed else "blocked"
        reason = self._generate_reason(text, label)

        return ClassificationResult(
            allowed=allowed,
            confidence=confidence_value,
            label=label,
            reason=reason
        )

    def _fallback_classify(self, text: str, context: Optional[str] = None) -> ClassificationResult:
        """Fallback rule-based classification when transformers unavailable."""
        if context:
            blocked_contexts = ["medical", "health", "legal", "financial", "patient", "doctor"]
            allowed_contexts = ["fiction", "story", "creative", "hypothetical", "novel"]

            context_lower = context.lower()
            if any(c in context_lower for c in blocked_contexts):
                return ClassificationResult(
                    allowed=False,
                    confidence=0.9,
                    label="blocked",
                    reason=f"Blocked context: {context}"
                )
            if any(c in context_lower for c in allowed_contexts):
                return ClassificationResult(
                    allowed=True,
                    confidence=0.9,
                    label="allowed",
                    reason=f"Allowed context: {context}"
                )

        high_risk_types = ["ssn", "credit card", "phone", "email"]
        if any(p in text.lower() for p in high_risk_types):
            return ClassificationResult(
                allowed=False,
                confidence=0.7,
                label="blocked",
                reason="High-risk PII detected"
            )

        return ClassificationResult(
            allowed=True,
            confidence=0.5,
            label="allowed",
            reason="No strong redaction signals"
        )

    def _generate_reason(self, text: str, label: str) -> str:
        """Generate explanation for classification."""
        if label == "allowed":
            return "PII allowed in non-sensitive context"
        else:
            return "PII redacted for privacy compliance"

    def should_redact(
        self,
        text: str,
        pii_entities: List,
        context: str = ""
    ) -> Tuple[bool, str]:
        """Determine if PII should be redacted."""
        result = self.classify(text, context)
        return not result.allowed, result.reason

    def fine_tune(
        self,
        train_dataset: List[Tuple[str, int]],
        eval_dataset: Optional[List[Tuple[str, int]]] = None,
        output_dir: str = "./context_filter_model",
        num_epochs: int = 3
    ) -> None:
        """Fine-tune DistilBERT for context classification."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers not available. Cannot fine-tune.")
            return

        self.logger.info(f"Fine-tuning on {len(train_dataset)} examples")

        texts = [t for t, l in train_dataset]
        labels = [l for t, l in train_dataset]

        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        class TextDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        train_dataset_obj = TextDataset(encodings, labels)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=8,
            learning_rate=2e-5,
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset_obj
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.logger.info(f"Model saved to {output_dir}")