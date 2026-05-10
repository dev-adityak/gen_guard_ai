# GenGuard-AI: Privacy-Preserving Generative AI Framework
## M.Tech Project Overview
**Affiliation:** [Your University Name]  
**Domain:** Generative AI, Privacy Engineering, Machine Learning  
**Date:** May 2026

---

## 1. Project Overview
GenGuard-AI is a lightweight, multi-layered privacy framework for securing small-scale Generative AI models against data leakage, optimized for resource-constrained environments (healthcare, banking, education) with limited compute.

It solves the core *privacy paradox* of Generative AI: models need to learn general patterns to be useful, but must forget sensitive individual data to comply with GDPR, HIPAA, and other regulations.

### Simplified Explanation
GenGuard-AI acts as a "privacy shield" for AI tools:
- Trains models without memorizing private user data
- Scans every AI-generated response to hide sensitive details (names, emails, medical info)
- Adds invisible watermarks to prove AI content ownership
- Runs on basic laptops/mid-range GPUs (no supercomputers required)

---

## 2. Problem Statement
Current LLMs frequently leak sensitive data via:
- Memorization of rare training examples (PII, medical records, trade secrets)
- Membership inference attacks (confirming if data was used in training)
- Prompt injection/jailbreak attacks
- Unintended output of private content

Traditional tools (simple anonymization, regex filters) fail against modern GenAI complexity. This project delivers a resource-efficient, layered defense framework for small open-source models.

---

## 3. Key System Components (Defense-in-Depth)
### 3.1 Data Layer (Training Privacy)
- **DP-SGD Training:** Adds calibrated noise to model updates to prevent individual data memorization, with formal (ε,δ) privacy guarantees
- **Simulated Federated Learning:** Trains models locally on distributed nodes (simulated on a single machine for resource constraints) so raw data never leaves its source
- **Secure Aggregation:** Combines model updates without exposing individual node contributions

### 3.2 Model Layer (IP Protection)
- **Statistical Watermarking:** Embeds invisible patterns in generated text to verify ownership and trace leaks
- **Model Obfuscation:** Lightweight weight quantization to resist model inversion attacks

### 3.3 Output Layer (Real-Time Filtering)
- **Hybrid PII Detection:** Combines Presidio/spaCy NER with a fine-tuned DistilBERT classifier
- **Context-Aware Redaction:** Masks PII only when relevant (e.g., redacts patient names in medical reports, allows names in fictional stories)
- **Self-Explanation:** Automatically generates redaction reasons (e.g., "Masked phone number to comply with GDPR")

### 3.4 Monitoring Layer (Compliance)
- **Privacy Budget Tracker:** Logs cumulative privacy loss (ε) during training
- **Attack Simulator:** Tests the model against known jailbreak prompts to generate a privacy health score
- **Audit Logs:** Records all redaction events for compliance reporting

---

## 4. Technology Stack (Lightweight for Limited Resources)
All tools are open-source, free, and optimized for low compute:
| Component | Tool/Library | Rationale |
|-----------|--------------|-----------|
| Base LLM | Phi-3-mini (3.8B) / Gemma-2B | Small, high-performance models for mid-range GPUs/CPUs |
| DP-SGD Training | Opacus (PyTorch) | Industry-standard differential privacy library |
| Federated Learning | Flower (flwr) | Lightweight FL framework with simulated client support |
| PII Detection | Microsoft Presidio + spaCy | Pre-trained NER models for fast, accurate PII detection |
| Safety Classifier | DistilBERT-base-uncased | 66M parameter model for fast, low-resource inference |
| Watermarking | Gumbel-Softmax Implementation | Lightweight statistical watermarking from Kirchenbauer et al. |
| Core Framework | Python 3.9+, PyTorch, Hugging Face Transformers | Standard ML stack with broad community support |
| Optional Interface | Streamlit / FastAPI | Lightweight web UI for testing and dashboards |

---

## 5. Implementation Roadmap
| Phase | Duration | Tasks | Deliverables |
|-------|----------|-------|--------------|
| 1: Setup | Week 1-2 | Environment setup, baseline model testing, dataset collection | Working dev environment, baseline LLM outputs |
| 2: Data Layer | Week 3-5 | DP-SGD training, simulated FL implementation, privacy budget logging | Trained model with (ε,δ) guarantees |
| 3: Output Layer | Week 6-8 | PII detection module, dynamic redaction logic, context awareness | Working output firewall with redaction examples |
| 4: Model Layer | Week 9-10 | Watermarking implementation, model obfuscation | Watermarked model with detection logic |
| 5: Integration | Week 11-12 | Unified pipeline, monitoring dashboard | End-to-end GenGuard-AI prototype |
| 6: Evaluation | Week 13-14 | Privacy/utility/efficiency testing, attack simulations | Evaluation metrics, comparison tables |
| 7: Documentation | Week 15-16 | Thesis writing, code documentation | Final M.Tech thesis, deployable codebase |

---

## 6. Evaluation Metrics
### Privacy Metrics
- Membership Inference Attack (MIA) success rate
- Training data extraction rate
- Privacy budget (ε) consumption

### Utility Metrics
- Model perplexity (language quality)
- Downstream task accuracy (QA, summarization)
- Human evaluation of output fluency

### Efficiency Metrics
- Training time per epoch
- Inference latency (ms/token)
- Memory usage (RAM/GPU)

---

## 7. References
1. Kirchenbauer et al., *A Watermark for Large Language Models*, arXiv:2301.10226, 2023
2. Abadi et al., *Deep Learning with Differential Privacy*, CCS 2016
3. McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
4. Opacus Documentation: https://opacus.ai
5. Flower Documentation: https://flower.dev
6. Presidio Documentation: https://microsoft.github.io/presidio/