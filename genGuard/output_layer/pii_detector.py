"""
PII Detector Module
Uses Microsoft Presidio and spaCy for PII detection and redaction.
"""

import re
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import Counter

logger = logging.getLogger(__name__)

PRESIDIO_AVAILABLE = False
SPACY_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    logger.warning("Presidio not installed. PII detection will be limited.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not installed. NER-based detection will be limited.")


@dataclass
class PIIEntity:
    """Represents a detected PII entity."""
    entity_type: str
    text: str
    start: int
    end: int
    score: float


@dataclass
class PIIAnalysisResult:
    """Result of PII analysis."""
    text: str
    entities: List[PIIEntity]
    has_pii: bool
    entity_counts: Dict[str, int]
    risk_level: str


class PIIDetector:
    """
    PII Detection and Redaction System.

    Combines:
    - Microsoft Presidio for PII detection
    - spaCy for NER
    - Custom regex for domain-specific PII

    Supported entity types:
    - PERSON: Names
    - EMAIL: Email addresses
    - PHONE: Phone numbers
    - SSN: Social Security Numbers
    - CREDIT_CARD: Credit card numbers
    - DATE: Dates
    - LOCATION: Addresses
    """

    def __init__(
        self,
        language: str = "en",
        nlp_model: str = "en_core_web_sm",
        confidence_threshold: float = 0.5,
        supported_entities: Optional[List[str]] = None
    ):
        self.language = language
        self.nlp_model = nlp_model
        self.confidence_threshold = confidence_threshold

        self.supported_entities = supported_entities or [
            "PERSON", "EMAIL", "PHONE_NUMBER", "SSN",
            "CREDIT_CARD", "DATE_TIME", "US_BANK_ROUTING_NUMBER", "US_SSN"
        ]

        self.nlp = None
        self.presidio_analyzer = None
        self.presidio_anonymizer = None
        self.logger = logging.getLogger(__name__)

        self._init_engines()
        self._init_custom_patterns()

    def _init_engines(self) -> None:
        """Initialize Presidio and spaCy engines."""
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.nlp_model)
            except OSError:
                self.logger.warning(f"Downloading spaCy model {self.nlp_model}")
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", self.nlp_model])
                self.nlp = spacy.load(self.nlp_model)

        if PRESIDIO_AVAILABLE:
            self.presidio_analyzer = AnalyzerEngine()
            self.presidio_anonymizer = AnonymizerEngine()

    def _init_custom_patterns(self) -> None:
        """Initialize custom regex patterns for domain-specific PII."""
        self.custom_patterns = {
            "PHONE": {
                "pattern": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "score": 0.9
            },
            "EMAIL": {
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "score": 0.95
            },
            "SSN": {
                "pattern": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
                "score": 0.9
            },
            "CREDIT_CARD": {
                "pattern": r"\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b",
                "score": 0.85
            },
            "IP_ADDRESS": {
                "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                "score": 0.7
            }
        }

    def detect(self, text: str) -> PIIAnalysisResult:
        """Detect PII entities in text."""
        entities = []

        if PRESIDIO_AVAILABLE:
            entities.extend(self._detect_with_presidio(text))

        entities.extend(self._detect_with_patterns(text))
        entities = self._deduplicate_entities(entities)

        entity_counts = Counter(e.entity_type for e in entities)
        has_pii = len(entities) > 0
        risk_level = self._calculate_risk_level(entity_counts)

        return PIIAnalysisResult(
            text=text,
            entities=entities,
            has_pii=has_pii,
            entity_counts=dict(entity_counts),
            risk_level=risk_level
        )

    def _detect_with_presidio(self, text: str) -> List[PIIEntity]:
        """Detect PII using Presidio."""
        entities = []

        results = self.presidio_analyzer.analyze(
            text=text,
            language=self.language,
            entities=self.supported_entities
        )

        for result in results:
            if result.score >= self.confidence_threshold:
                entity = PIIEntity(
                    entity_type=result.entity_type,
                    text=text[result.start:result.end],
                    start=result.start,
                    end=result.end,
                    score=result.score
                )
                entities.append(entity)

        return entities

    def _detect_with_patterns(self, text: str) -> List[PIIEntity]:
        """Detect PII using custom regex patterns."""
        entities = []

        for entity_type, config in self.custom_patterns.items():
            pattern = config["pattern"]
            score = config["score"]

            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = PIIEntity(
                    entity_type=entity_type,
                    text=match.group(),
                    start=match.start(),
                    end=match.end(),
                    score=score
                )
                entities.append(entity)

        return entities

    def _deduplicate_entities(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """Remove overlapping entities, keeping highest score."""
        if not entities:
            return entities

        sorted_entities = sorted(entities, key=lambda e: (e.start, -e.score))

        deduped = []
        for entity in sorted_entities:
            if not deduped or entity.start >= deduped[-1].end:
                deduped.append(entity)
            elif entity.score > deduped[-1].score:
                deduped[-1] = entity

        return deduped

    def _calculate_risk_level(self, entity_counts: Counter) -> str:
        """Calculate risk level based on detected entities."""
        high_risk = {"SSN", "CREDIT_CARD", "MEDICAL"}
        medium_risk = {"EMAIL", "PHONE", "PERSON"}

        types_found = set(entity_counts.keys())

        if types_found & high_risk:
            return "high"
        elif types_found & medium_risk:
            return "medium"
        elif types_found:
            return "low"
        else:
            return "none"

    def redact(
        self,
        text: str,
        replacement: str = "[REDACTED]",
        entity_types: Optional[List[str]] = None
    ) -> str:
        """Redact PII from text."""
        result = self.detect(text)

        if entity_types:
            entities = [e for e in result.entities if e.entity_type in entity_types]
        else:
            entities = result.entities

        redacted_text = text
        for entity in reversed(entities):
            redacted_text = (
                redacted_text[:entity.start] +
                replacement +
                redacted_text[entity.end:]
            )

        return redacted_text