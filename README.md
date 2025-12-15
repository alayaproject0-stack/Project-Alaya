# Alaya: The Wake-on-SNN Hybrid Inference System

**Alaya** (Sanskrit: *ālayavijñāna*, "Storehouse Consciousness") is a proof-of-concept implementation of a "Wake-on-SNN" architecture. It demonstrates how a lightweight Spiking Neural Network (SNN) can act as a subconscious filter to significantly reduce the energy consumption of Large Language Models (LLMs) without sacrificing accuracy.

![Energy vs Accuracy Graph](./graph.png)
*(Place your graph image here)*

## 🧠 The Concept

Modern LLMs (like BERT) are powerful but energy-intensive ("System 2" thinking). SNNs are incredibly efficient but less accurate ("System 1" intuition).

**Alaya** combines them:
1.  **The Alaya Layer (SNN)**: Processes all inputs first. It is extremely fast and low-power (approx. 0.02 J/sample).
2.  **The Wake-Up Call**: If Alaya's confidence score falls below a certain threshold (high entropy), it "wakes up" the LLM.
3.  **The Manas Layer (BERT)**: Processes only the difficult samples that Alaya couldn't handle with confidence.

## 🚀 Key Results

Tested on the AG News classification task (N=2000 test samples) on an NVIDIA GPU:

| Model | Accuracy | Energy / Sample | Speed |
| :--- | :--- | :--- | :--- |
| **SNN Only** | 86.70% | 0.027 J | 1.5s |
| **BERT Only** | 89.70% | 0.185 J | 5.5s |
| **Alaya (Hybrid)** | **89.07%** | **0.068 J** | **1.8s** |

- **Energy Efficiency**: **3.6x** more efficient than running BERT alone.
- **Accuracy Loss**: Negligible (**-0.63%** compared to BERT).
- **Wake Rate**: Only **14.9%** of samples required BERT intervention.

## 🛠️ Usage

### Requirements
- Python 3.8+
- PyTorch
- Transformers
- Nvidia-ml-py (for energy monitoring)

```bash
pip install torch transformers datasets nvidia-ml-py
```
