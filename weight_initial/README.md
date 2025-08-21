---
license: apache-2.0
pipeline_tag: text-generation
---

# 🧨 FLAME-MoE

This repository contains the model described in [FLAME-MoE: A Transparent End-to-End Research Platform for Mixture-of-Experts Language Models](https://huggingface.co/papers/2505.20225).

**FLAME-MoE** is a fully open Mixture-of-Experts (MoE) language model suite developed by Carnegie Mellon University. It provides a transparent and reproducible research platform for investigating expert routing, model scaling, and training dynamics in sparse architectures. The suite includes seven decoder-only transformer models ranging from 38M to 1.7B active parameters and reflects production-grade MoE setups with 64 experts per MoE layer, top-8 routing, and shared experts.

---

## 🔍 Model Summary

| Model Name           | Active / Total Params | Layers | MoE Experts (Total/Active/Shared) | Training FLOPs | Tokens Trained |
| -------------------- | --------------------- | ------ | --------------------------------- | -------------- | -------------- |
| FLAME-MoE-38M-100M   | 38M / 100M            | 9      | 64 / 8 / 2                        | 1.0e18         | 4.4B           |
| FLAME-MoE-98M-349M   | 98M / 349M            | 9      | 64 / 8 / 2                        | 3.0e18         | 5.0B           |
| FLAME-MoE-115M-459M  | 115M / 459M           | 12     | 64 / 8 / 2                        | 6.0e18         | 8.7B           |
| FLAME-MoE-290M-1.3B  | 290M / 1.3B           | 9      | 64 / 8 / 2                        | 2.0e19         | 11.4B          |
| FLAME-MoE-419M-2.2B  | 419M / 2.2B           | 15     | 64 / 8 / 2                        | 3.0e19         | 11.9B          |
| FLAME-MoE-721M-3.8B  | 721M / 3.8B           | 12     | 64 / 8 / 2                        | 8.0e19         | 18.4B          |
| FLAME-MoE-1.7B-10.3B | 1.7B / 10.3B          | 18     | 64 / 8 / 2                        | 2.4e20         | 23.1B          |

---

## 📖 Training Details

* **Framework**: Megatron-LM with Expert Parallelism (EP=8), Pipeline Parallelism (PP=1)
* **Data**: Pretrained on DataComp-LM (DCLM)
* **Batch Size**: 1024
* **Sequence Length**: 2048
* **Optimizer**: Adam
* **Scheduler**: WSD (Warmup + Decay)
* **Learning Rate**: Max 3e-4, Min 3e-5
* **Checkpoints**: 10 saved per model across training
* **Hardware**: 32× NVIDIA H100 GPUs

---

## 🛠 Intended Use

FLAME-MoE is developed for **research purposes only**. It supports academic study in:

* Sparse model training dynamics
* Expert routing behavior and specialization
* Scaling laws and compute-optimal design
* Benchmarking and reproducibility in MoE LLMs

It is not intended for commercial deployment or instruction-tuned downstream tasks.

---

## 📂 Access

All models, training scripts, logs, routing traces, and evaluation pipelines are available at:

🔗 [https://github.com/cmu-flame/FLAME-MoE](https://github.com/cmu-flame/FLAME-MoE)