# ET-SCD
# ET-SCD: Evidence-aware Three-way Semantic Contrastive Decoding

Official project repository for **ET-SCD**, a training-free decoding framework for mitigating object hallucinations in Large Vision-Language Models (LVLMs).

## Overview

Large Vision-Language Models often generate hallucinated objects when language priors dominate insufficient visual evidence. To address this issue, we propose **Evidence-aware Three-way Semantic Contrastive Decoding (ET-SCD)**, which introduces uncertainty-aware selective decision-making into contrastive decoding.

ET-SCD incorporates three key components:

- **Object-sensitive selective activation:** Identifies generation steps involving potential object hallucinations, avoiding unnecessary intervention in reliable predictions.
- **Evidence-aware three-way decision-making:** Integrates contrastive evidence and uncertainty estimates to categorize candidate predictions into Accept, Defer, and Reject decisions.
- **Target-aware visual re-query:** Selectively acquires additional visual evidence for ambiguous predictions and refines their decisions through evidence fusion.

The framework aims to reduce object hallucinations while limiting unnecessary computational overhead.

## Evaluation

ET-SCD is evaluated on three representative LVLMs:

- LLaVA-1.5-7B
- Qwen-VL-Chat
- InstructBLIP-7B

Evaluation benchmarks include POPE, CHAIR, and MME-Hall.

## Code Release

**Status: Code preparation in progress.**

The implementation, configuration files, and evaluation scripts are currently being organized for public release.

This repository will be updated with the relevant materials as they become available.

## Citation

Citation information will be updated following publication.
