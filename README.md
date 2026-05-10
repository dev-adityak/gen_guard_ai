# GenGuard-AI: Privacy-Preserving Generative AI Framework

A lightweight, multi-layered privacy framework for securing small-scale Generative AI models against data leakage.

## Features

- **DP-SGD Training**: Differential privacy with formal (ε,δ) guarantees
- **Federated Learning**: Simulated distributed training
- **PII Detection**: Presidio + spaCy + DistilBERT
- **Context-Aware Redaction**: Smart PII filtering
- **Watermarking**: Statistical ownership verification
- **Model Quantization**: IP protection via obfuscation
- **Privacy Health Monitoring**: Attack simulation and compliance tracking

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from genGuard.pipeline import GenGuardPipeline, PipelineConfig

config = PipelineConfig(privacy_budget=8.0, watermark_key=42)
pipeline = GenGuardPipeline(config=config)

result = pipeline.process_output(
    text="Patient John Doe (email: john@example.com) visit: 555-123-4567",
    context="medical"
)
print(result['output'])
```

## Running the Dashboard

```bash
streamlit run genGuard/app.py
```

## Project Structure

```
genGuard/
├── data_layer/       # Privacy-preserving training
├── output_layer/     # PII detection & redaction
├── model_layer/      # Watermarking & quantization
├── monitoring/       # Attack simulation & audit
├── utils/            # Helper functions
├── config.py         # Centralized configuration
├── pipeline.py       # Unified pipeline
└── app.py            # Streamlit dashboard
```

## License

MIT