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
    # ========== PII ==========
    - PERSON: Names
    - EMAIL: Email addresses
    - PHONE: Phone numbers
    - INTERNATIONAL_PHONE: International phone numbers
    - SSN: Social Security Numbers
    - AADHAAR: Indian Aadhaar numbers
    - PAN_NUMBER: Indian PAN numbers
    - VOTER_ID: Indian Voter IDs
    - PASSPORT: Passport numbers
    - DRIVER_LICENSE: Driver's license numbers
    - FULL_ADDRESS: Physical addresses
    - GPS_COORDINATES: GPS coordinates

    # ========== Financial Information ==========
    - CREDIT_CARD: Credit card numbers
    - CVV: Card verification values
    - BANK_ACCOUNT: Bank account numbers
    - IFSC_CODE: Indian IFSC codes
    - SWIFT_CODE: SWIFT/BIC codes
    - UPI_ID: UPI payment IDs
    - CRYPTO_WALLET: Cryptocurrency addresses

    # ========== Authentication & Secrets ==========
    - API_KEY: Generic API keys
    - AWS_KEY: AWS access key IDs
    - AWS_SECRET: AWS secret access keys
    - JWT_TOKEN: JSON Web Tokens
    - BEARER_TOKEN: Bearer authorization tokens
    - GENERIC_TOKEN: Various token formats
    - OAUTH_SECRET: OAuth client secrets
    - SSH_KEY: SSH private keys
    - GPG_KEY: GPG private keys
    - PASSWORD: Passwords
    - OTP: One-time passwords
    - PIN: Personal identification numbers
    - SESSION_COOKIE: Session identifiers
    - DATABASE_CREDENTIALS: Database connection strings
    - ENV_SECRET: Environment secrets

    # ========== Network ==========
    - IP_ADDRESS: IPv4 addresses
    - PRIVATE_IP: Private IP ranges
    - MAC_ADDRESS: MAC addresses

    # ========== Medical ==========
    - MEDICAL_RECORD: Medical record numbers
    - INSURANCE_ID: Insurance IDs

    # ========== Company ==========
    - INTERNAL_HOSTNAME: Internal hostnames
    - DATABASE_DUMP: SQL query patterns
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
            "CREDIT_CARD", "DATE_TIME", "US_BANK_ROUTING_NUMBER", "US_SSN",
            "IBAN_CODE", "SWIFT_CODE", "DOMAIN_NAME"
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
        """Initialize custom regex patterns for domain-specific PII and secrets."""
        self.custom_patterns = {
            # ========== Financial Information ==========
            "CREDIT_CARD": {
                "pattern": r"\b(?:[0-9]{4}[-\s]?){3}[0-9]{4}\b",
                "score": 0.95,
                "category": "financial"
            },
            "CVV": {
                "pattern": r"\b(?:cvv|cvc|csc|security.?code)[:\s]*[0-9]{3,4}\b",
                "score": 0.95,
                "category": "financial"
            },
            "BANK_ACCOUNT": {
                "pattern": r"\b(?:account|acct).?(?:no|number|#|:|\s)*[0-9]{8,17}\b",
                "score": 0.85,
                "category": "financial"
            },
            "IFSC_CODE": {
                "pattern": r"\b[A-Z]{4}0[A-Z0-9]{6}\b",
                "score": 0.9,
                "category": "financial"
            },
            "SWIFT_CODE": {
                "pattern": r"\b(?:SWIFT|BIC|IITB|ICIC|HDFC|SBIN|UBIN|UTIB)[A-Z0-9]{8,11}\b",
                "score": 0.9,
                "category": "financial"
            },
            "UPI_ID": {
                "pattern": r"\b[a-zA-Z0-9._-]+@[a-zA-Z0-9]+\b",
                "score": 0.75,
                "category": "financial"
            },
            "CRYPTO_WALLET": {
                "pattern": r"\b(?:0x[a-fA-F0-9]{40}|(?:bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}|xpub[xdefgs123456789ACDEFGHJKLMNPQRSTUVWXYZ]{80,120})\b",
                "score": 0.9,
                "category": "financial"
            },

            # ========== Phone Numbers ==========
            "PHONE": {
                "pattern": r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b",
                "score": 0.9,
                "category": "pii"
            },
            "INTERNATIONAL_PHONE": {
                "pattern": r"\+[1-9]\d{1,14}\b",
                "score": 0.85,
                "category": "pii"
            },

            # ========== Email & Identity ==========
            "EMAIL": {
                "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
                "score": 0.95,
                "category": "pii"
            },
            "SSN": {
                "pattern": r"\b[0-9]{3}-[0-9]{2}-[0-9]{4}\b",
                "score": 0.9,
                "category": "pii"
            },
            "AADHAAR": {
                "pattern": r"\b[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b",
                "score": 0.85,
                "category": "pii"
            },
            "PAN_NUMBER": {
                "pattern": r"\b[A-Z]{5}[0-9]{4}[A-Z]\b",
                "score": 0.85,
                "category": "pii"
            },
            "VOTER_ID": {
                "pattern": r"\b[A-Z]{3}[0-9]{7}\b",
                "score": 0.8,
                "category": "pii"
            },
            "PASSPORT": {
                "pattern": r"\b[A-Z]{1,2}[0-9]{6,9}\b",
                "score": 0.75,
                "category": "pii"
            },
            "DRIVER_LICENSE": {
                "pattern": r"(?i)(?:driver['']?s?[_-]?license|dl[_-]?no|driving[_-]?licence)[:\s]*[A-Z0-9]{5,15}\b",
                "score": 0.85,
                "category": "pii"
            },

            # ========== Authentication & Security Secrets ==========
            "API_KEY": {
                "pattern": r"(?i)(?:api[_-]?key|apikey|api-secret)[:\s=]*['\"]?([a-zA-Z0-9_\-]{20,64})['\"]?",
                "score": 0.9,
                "category": "secret"
            },
            "AWS_KEY": {
                "pattern": r"(?i)(?:aws[_-]?(?:access[_-]?key[_-]?id|secret[_-]?access[_-]?key)|amazon(?:aws)?)['\"]?\s*[:=]\s*['\"]?[A-Z0-9]{20}['\"]?",
                "score": 0.95,
                "category": "secret"
            },
            "AWS_SECRET": {
                "pattern": r"(?i)aws[_-]?secret[_-]?access[_-]?key['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}['\"]?",
                "score": 0.95,
                "category": "secret"
            },
            "JWT_TOKEN": {
                "pattern": r"\beyJ[a-zA-Z0-9_-]+\.eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\b",
                "score": 0.95,
                "category": "secret"
            },
            "BEARER_TOKEN": {
                "pattern": r"(?i)\bbearer[\s-]+(?:eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+|[A-Za-z0-9_.\-]{20,})\b",
                "score": 0.9,
                "category": "secret"
            },
            "GENERIC_TOKEN": {
                "pattern": r"(?i)(?:token|access[_-]?token|refresh[_-]?token|auth[_-]?token)[:\s=]*['\"]?([a-zA-Z0-9_.\-]{20,})['\"]?",
                "score": 0.8,
                "category": "secret"
            },
            "OAUTH_SECRET": {
                "pattern": r"(?i)(?:oauth[_-]?client[_-]?secret|client[_-]?secret|app[_-]?secret)[:\s=]*['\"]?[a-zA-Z0-9]{16,64}['\"]?",
                "score": 0.9,
                "category": "secret"
            },
            "SSH_KEY": {
                "pattern": r"-----BEGIN\s+(?:RSA|DSA|EC|OPENSSH|PPK)?\s+PRIVATE\s+KEY-----",
                "score": 0.95,
                "category": "secret"
            },
            "GPG_KEY": {
                "pattern": r"-----BEGIN\s+PGP\s+PRIVATE\s+KEY\s+BLOCK-----",
                "score": 0.95,
                "category": "secret"
            },
            "PASSWORD": {
                "pattern": r"(?i)(?:password|passwd|pwd|pass)[:\s=]*['\"]?[^'\"]+['\"]?",
                "score": 0.8,
                "category": "secret"
            },
            "OTP": {
                "pattern": r"\b(?:\d{6,8}|one[_-]?time[_-]?password)\b",
                "score": 0.7,
                "category": "secret"
            },
            "PIN": {
                "pattern": r"(?i)(?:pin[:\s]*|enter\s+pin)\d{4,6}\b",
                "score": 0.75,
                "category": "secret"
            },
            "SESSION_COOKIE": {
                "pattern": r"(?i)session[_-]?(?:id|token|cookie)[:\s=]*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
                "score": 0.85,
                "category": "secret"
            },
            "DATABASE_CREDENTIALS": {
                "pattern": r"(?i)(?:postgres|mysql|mongodb|redis|sqlserver)://[^:\s]+:[^@\s]+@",
                "score": 0.95,
                "category": "secret"
            },
            "ENV_SECRET": {
                "pattern": r"(?i)(?:secret|private[_-]?key|access[_-]?token)[:\s=]*['\"]?[a-zA-Z0-9+/=]{20,}['\"]?",
                "score": 0.85,
                "category": "secret"
            },

            # ========== IP & Network ==========
            "IP_ADDRESS": {
                "pattern": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
                "score": 0.7,
                "category": "network"
            },
            "PRIVATE_IP": {
                "pattern": r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b",
                "score": 0.8,
                "category": "network"
            },
            "MAC_ADDRESS": {
                "pattern": r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b",
                "score": 0.7,
                "category": "network"
            },

            # ========== Medical & Health ==========
            "MEDICAL_RECORD": {
                "pattern": r"\b(?:MRN|medical[_-]?record[_-]?number|patient[_-]?id)[:\s]*[0-9]{5,12}\b",
                "score": 0.85,
                "category": "medical"
            },
            "INSURANCE_ID": {
                "pattern": r"\b(?:insurance[_-]?(?:id|number|member[_-]?id)|policy[_-]?no)[:\s]*[A-Z0-9]{6,15}\b",
                "score": 0.8,
                "category": "medical"
            },

            # ========== Location ==========
            "FULL_ADDRESS": {
                "pattern": r"\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|boulevard|blvd| lane|ln|drive|dr|court|ct|way|place|pl)[,.]?\s*(?:apt|suite|ste|unit|#)?\s*\d*,?\s*(?:[A-Z][a-z]+[\s]?){1,2},?\s*[A-Z]{2}\s*\d{5}(-\d{4})?\b",
                "score": 0.8,
                "category": "pii"
            },
            "GPS_COORDINATES": {
                "pattern": r"\b-?[0-9]{1,2}\.[0-9]{4,},?\s*-?[0-9]{1,3}\.[0-9]{4,}\b",
                "score": 0.75,
                "category": "pii"
            },

            # ========== Company Confidential ==========
            "INTERNAL_HOSTNAME": {
                "pattern": r"\b(?:internal|intranet|dev|stage|staging|prod|production)[.-](?:server|host|app)[.-]?(?:[a-z0-9-]+)\b",
                "score": 0.7,
                "category": "company"
            },
            "DATABASE_DUMP": {
                "pattern": r"(?i)(?:insert\s+into|create\s+table|select\s+\*).*from\s+[a-z_]+\s*;?\s*(?:select|insert|update|delete)",
                "score": 0.75,
                "category": "company"
            },

            # ========== Generic Secret Patterns ==========
            "SECRET_STRING": {
                "pattern": r"(?i)(?:secret|private|key|token|credential|auth|api[_-]?key)[:\s=]*['\"]?([a-zA-Z0-9+/=_\-]{20,})['\"]?",
                "score": 0.7,
                "category": "secret"
            },
            "BASE64_SECRET": {
                "pattern": r"\b[a-zA-Z0-9+/]{40,}={0,2}\b",
                "score": 0.6,
                "category": "secret"
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
        high_risk = {
            "SSN", "CREDIT_CARD", "CRYPTO_WALLET", "AWS_KEY", "AWS_SECRET",
            "SSH_KEY", "GPG_KEY", "DATABASE_CREDENTIALS", "PASSWORD", "OTP",
            "JWT_TOKEN", "SECRET_STRING", "PASSPORT", "DRIVER_LICENSE",
            "AADHAAR", "PAN_NUMBER", "MEDICAL_RECORD", "BANK_ACCOUNT", "CVV"
        }
        medium_risk = {
            "EMAIL", "PHONE", "PERSON", "API_KEY", "OAUTH_SECRET",
            "BEARER_TOKEN", "GENERIC_TOKEN", "SESSION_COOKIE", "SWIFT_CODE",
            "IFSC_CODE", "UPI_ID", "INSURANCE_ID", "VOTER_ID", "FULL_ADDRESS",
            "INTERNAL_HOSTNAME", "ENV_SECRET", "BASE64_SECRET", "PRIVATE_IP"
        }
        low_risk = {
            "IP_ADDRESS", "MAC_ADDRESS", "GPS_COORDINATES", "DOMAIN_NAME",
            "INTERNATIONAL_PHONE"
        }

        types_found = set(entity_counts.keys())

        if types_found & high_risk:
            return "critical"
        elif types_found & medium_risk:
            return "high"
        elif types_found & low_risk:
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