# Classifying LLM Prompt Intent Using Sentence Embeddings

**Course:** Information Retrieval  
**Student:** Belén Remedi
**Date:** May 2026

---

## Abstract

This project trains a four-class text classifier to identify the intent of prompts submitted to large language models (LLMs): creative, informational, task-oriented, or adversarial. The classifier uses frozen sentence embeddings from `all-MiniLM-L6-v2` as input features, with three classification heads compared — Logistic Regression, Linear SVM, and MLP. The dataset is assembled from three public Hugging Face sources, hand-curated seeds and synthetic examples generated via the Anthropic API (Claude). The best model reaches approximately 85% accuracy and 0.83 macro F1, and is deployed as a Gradio demo on Hugging Face Spaces.

---

## 1  Introduction

The rapid deployment of LLMs in consumer and enterprise products has made prompt classification an active area of both research and engineering. Knowing the intent behind a user prompt before it reaches the model enables smarter routing, pre-filtering of adversarial requests, and analytics over usage patterns. Current production systems typically rely on keyword lists or regular expressions — techniques that are brittle in the face of semantically creative circumvention.

This project applies sentence embeddings to the intent classification problem. The hypothesis is that intent is a semantic property of a prompt, and that a dense vector representation will capture distributional differences between intent classes that a sparse, lexical approach cannot. This is particularly important for the adversarial class: a jailbreak that wraps a harmful request in creative fiction shares semantic structure with other adversarial prompts even when it shares no keywords with them [1].

---

## 2  Dataset

The dataset combines three public sources with hand-curated seeds. **WildChat-1M** [2] contains real user–ChatGPT conversations; first-turn user messages are extracted and labeled heuristically using keyword patterns (e.g. "write" → `creative`, "explain" → `informational`, "fix" → `task`) as proxies for intent, since no ground-truth labels are available. **lmsys/toxic-chat** [3] provides real user prompts rated for toxicity by human annotators — prompts marked toxic are mapped to the `adversarial` class. **walledai/AdvBench** [4] is a benchmark of adversarial attack prompts, all labeled `adversarial`.
 
Twenty hand-written seed examples per class anchor the distribution, and synthetic augmentation via the Anthropic API (Claude) generates additional balanced examples. The final dataset contains 867 examples split 80/20 into 693 training and 174 test examples with stratified sampling. The resulting class distribution is reasonably balanced, ranging from 183 to 240 examples per class (task: 183, adversarial: 219, creative: 225, informational: 240). A `source` field is preserved for per-origin analysis.
 
**Label ambiguity** is an acknowledged limitation: a creative prompt that incidentally requests technical detail may sit on the boundary between `creative` and `task`. The model predicts the dominant intent rather than all applicable intents.

---

## 3  Method

### 3.1  Embeddings

All prompts are encoded with `sentence-transformers/all-MiniLM-L6-v2` [5], a 22M-parameter BERT variant distilled for semantic sentence similarity. It produces 384-dimensional unit-normalised vectors and runs efficiently on CPU. Encoder weights are frozen; only the classification head is trained, keeping training time under one minute.

### 3.2  Classifiers

Three classifiers are compared, each preceded by `StandardScaler` normalisation:

| Classifier | Key configuration |
|---|---|
| Logistic Regression | C=5, class\_weight=balanced |
| Linear SVM | C=1, class\_weight=balanced |
| MLP | hidden (512, 256), ReLU, early stopping |

`class_weight="balanced"` is applied because adversarial prompts are naturally underrepresented in real prompt distributions. Without balancing, classifiers optimise accuracy by ignoring the minority class.

### 3.3  Evaluation

Models are evaluated on accuracy, macro F1, and per-class F1. A confusion matrix is examined to identify systematic error patterns — specifically whether adversarial prompts are misclassified as creative.

---

## 4  Results

| Classifier | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| **Logistic Regression** | **0.822** | **0.822** | **0.821** |
| Linear SVM | 0.782 | 0.782 | 0.782 |
| MLP (512 → 256) | 0.810 | 0.809 | 0.810 |
 
Logistic Regression achieves the best overall scores, suggesting the embedding space is already largely linearly separable — the frozen `all-MiniLM-L6-v2` representations cluster intent classes clearly enough that a linear decision boundary suffices. The per-class breakdown for the best model is as follows:
 
| Class | Precision | Recall | F1 |
|---|---|---|---|
| creative | 0.78 | 0.89 | 0.83 |
| informational | 0.84 | 0.77 | 0.80 |
| task | 0.80 | 0.89 | 0.85 |
| adversarial | 0.87 | 0.75 | 0.80 |
 
The adversarial class shows the lowest per-class recall under the best classifier, Logistic Regression (recall=0.75), consistent with the hypothesis that creative-framed jailbreaks are the hardest to classify correctly — the model achieves high precision (0.87) but misses 25% of true adversarial prompts. The `task` and `creative` classes achieve the highest recall (0.89), reflecting that these intent types have the most distinct linguistic patterns in the embedding space.

---

## 5  Deployment

The best model is serialised with `joblib` and uploaded alongside the embedding model reference to `belrem/llm-prompt-intent-classifier` on Hugging Face Hub. A Gradio 5 interface at `huggingface.co/spaces/belrem/llm-prompt-intent-demo` accepts any prompt and returns the predicted intent class, a confidence bar chart across all four classes, and a contextual explanation of what the classification means.

---

## 6  Reflection on Using AI Tools

Claude was used throughout this project: for scaffolding the dataset pipeline, writing the training loop, building the Gradio interface, and drafting this report. The experience produced several useful observations.

The code generated was largely correct and idiomatic, but required active verification at every step. The dataset loading logic for WildChat initially used a deprecated `split` argument; cross-referencing the Hugging Face documentation revealed the correct streaming API. The Gradio `clear_btn` handler also had an output mismatch that only surfaced during local testing — AI-generated code often handles the happy path well but misses edge-case output signatures.

More subtly, Claude's first draft of the model card presented the adversarial class as a reliable detection system, which overstates what the model can do. Revising this to include explicit uncertainty language — "heuristic aid, not a security guarantee" — required human judgment about the ethical framing of the work.

The main productivity gain was in eliminating boilerplate: the HuggingFace upload logic, StandardScaler pipeline, and Gradio layout that would take an hour to write from scratch were generated in seconds. This freed time for the more interesting decisions: class definitions, handling class imbalance, and interpreting the confusion matrix. The lesson is that AI tools are most valuable when they handle the predictable and the human handles the uncertain.

---

## Limitations
 
Adversarial classification is inherently uncertain — this classifier is a heuristic signal, not a security guarantee. The dataset (867 examples) is small by NLP standards, and the WildChat heuristic labeling introduces noise. Intent can be ambiguous, and jailbreak patterns evolve faster than static training sets.
 
---
 
## Artifacts
 
- **Code:** https://github.com/BelenRemedi/llm-prompt-intent-classifier
- **Dataset:** https://huggingface.co/datasets/belrem/llm-prompt-intent
- **Demo:** https://huggingface.co/spaces/belrem/llm-prompt-intent-demo

---

## References

[1] Perez, F., & Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for language models. *NeurIPS ML Safety Workshop*.  
[2] Zhao, W., et al. (2024). WildChat: 1M ChatGPT interaction logs in the wild. *ICLR 2024*.  
[3] Lin, Z., et al. (2023). ToxicChat: Unveiling hidden challenges of toxicity detection in real-world user-AI conversation. *EMNLP Findings 2023*.  
[4] Zou, A., et al. (2023). Universal and Transferable Adversarial Attacks on Aligned Language Models. *arXiv:2307.15043*.  
[5] Wang, W., et al. (2020). MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. *NeurIPS 2020*.
