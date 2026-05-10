# GenGuard-AI: Implementation Plan

## Project Overview
Privacy-preserving generative AI framework with defense-in-depth approach for resource-constrained environments.

## Phases

### Phase 1: Project Setup ✅ DONE
- [x] `requirements.txt` - All dependencies
- [x] `genGuard/` - Main package with sub-packages
- [x] `genGuard/config.py` - Centralized configuration

### Phase 2: Data Layer (Training Privacy)
- [x] `dpsgd_training.py` - DP-SGD with Opacus integration
- [x] `privacy_tracker.py` - Privacy budget tracking & export
- [x] `fl_simulated.py` - Federated learning with FedAvg

### Phase 3: Output Layer (Real-Time Filtering)
- [x] `pii_detector.py` - Presidio + spaCy PII detection
- [x] `context_filter.py` - DistilBERT context classification
- [x] `redaction_engine.py` - Unified redaction pipeline

### Phase 4: Model Layer (IP Protection)
- [x] `watermarker.py` - Gumbel-Softmax watermarking
- [x] `model_quantizer.py` - PTQ/FP8 quantization

### Phase 5: Monitoring Layer (Compliance)
- [x] `attack_simulator.py` - Jailbreak & MIA testing
- [x] `audit_logger.py` - Compliance event logging
- [x] `privacy_health.py` - Health score computation

### Phase 6: Integration
- [x] `pipeline.py` - Unified GenGuard-AI pipeline
- [x] `app.py` - Streamlit dashboard

### Phase 7: Utilities
- [x] `utils/helpers.py` - Common utilities

## Implementation Status
All core modules implemented. Ready for:
1. Dependency installation
2. Testing individual components
3. Integration with actual LLM models
4. Evaluation benchmarks

## Next Steps
1. Run `pip install -r requirements.txt`
2. Download spaCy model: `python -m spacy download en_core_web_sm`
3. Test individual modules
4. Connect to Phi-3-mini/Gemma-2B for full integration
5. Run privacy/utility benchmarks