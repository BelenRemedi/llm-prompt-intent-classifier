# LLM Prompt Intent Classifier

A text classifier that identifies the **intent** behind prompts sent to large language models, using sentence embeddings. Trained to distinguish four categories: `creative`, `informational`, `task`, and `adversarial`.

| Artifact | Link |
|---|---|
| 🤗 Dataset | [belrem/llm-prompt-intent](https://huggingface.co/datasets/belrem/llm-prompt-intent) |
| 🤗 Model | [belrem/llm-prompt-intent-classifier](https://huggingface.co/belrem/llm-prompt-intent-classifier) |
| 🚀 Live demo | [spaces/belrem/llm-prompt-intent-demo](https://huggingface.co/spaces/belrem/llm-prompt-intent-demo) |

---

## Problem

Every prompt sent to an LLM carries an implicit intent. Knowing that intent in advance has practical value: routing creative prompts to more capable models, flagging adversarial prompts before they reach the LLM, or logging task-oriented queries for analytics. Current systems rely on keyword blocklists or regex — brittle and easy to circumvent.

Sentence embeddings offer a semantic alternative: adversarial prompts that use creative framing ("as a character in a story...") share a distributional signature that a classifier trained on the embedding space can detect, even when surface keywords are innocent.

**Why embeddings?**
The distinction between a legitimate creative writing prompt and an adversarial one that wraps a harmful request in fiction is *semantic*, not lexical. A keyword filter that blocks "make explosives" misses "write a story where the protagonist explains to their chemist friend how to...". Embeddings capture the pragmatic shape of the request.

---

## Labels

| Label | ID | Description |
|---|---|---|
| `creative` | 0 | Fiction, poetry, brainstorming, roleplay, world-building |
| `informational` | 1 | Factual questions, explanations, definitions |
| `task` | 2 | Code, translation, summarisation, editing, structured generation |
| `adversarial` | 3 | Jailbreaks, prompt injection, instruction override, system prompt extraction |

---

## Dataset

Built from three sources, combined and deduplicated:

| Source | Type | Labels |
|---|---|---|
| [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) | Real user–ChatGPT conversations | creative, informational, task (heuristic) |
| [lmsys/toxic-chat](https://huggingface.co/datasets/lmsys/toxic-chat) | Real user prompts, toxicity labeled | adversarial (toxic=1) |
| [walledai/AdvBench](https://huggingface.co/datasets/walledai/AdvBench) | Adversarial benchmark prompts | adversarial |
| Hand-curated seeds | 20 examples per class | all |
| Synthetic (Anthropic API, optional) | Generated balanced examples | all |

Split: 80% train / 20% test, stratified. The `source` field is preserved so you can filter or analyse by origin.

---

## Approach

```
Prompt text
     │
     ▼
all-MiniLM-L6-v2   (384-dim sentence embedding, frozen)
     │
     ▼
StandardScaler
     │
     ├── Logistic Regression   (C=5, class_weight=balanced)
     ├── Linear SVM            (C=1, class_weight=balanced)
     └── MLP                   (512 → 256, ReLU, early stopping)
```

`class_weight="balanced"` is set because adversarial examples are naturally rarer than informational ones in real-world data.

---

## Results

| Classifier | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| **Logistic Regression** | **0.822** | **0.822** | **0.821** |
| Linear SVM | 0.782 | 0.782 | 0.782 |
| MLP (512 → 256) | 0.810 | 0.809 | 0.810 |

Logistic Regression achieves the best scores, suggesting the embedding space is already linearly separable. Per-class results for the best model:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| creative | 0.78 | 0.89 | 0.83 |
| informational | 0.84 | 0.77 | 0.80 |
| task | 0.80 | 0.89 | 0.85 |
| adversarial | 0.87 | 0.75 | 0.80 |

The adversarial class has the lowest recall (0.75) — creative-framed jailbreaks are the hardest to classify correctly.

*Exact figures written to `models/results.json` after training.*

---

## Repo layout

```
prompt-intent-classifier/
├── data/
│   └── build_dataset.py     # Download public data + generate synthetic examples
├── train.py                  # Embed + train 3 classifiers + evaluate + save
├── deploy.py                 # Push dataset / model / Space to HF Hub
├── spaces/
│   └── app/
│       ├── app.py            # Gradio demo
│       └── requirements.txt
├── REPORT.md                 # Academic report (2 pages)
└── README.md
```

---

## Quick start

### 1. Install

```bash
pip install sentence-transformers scikit-learn datasets \
            huggingface_hub joblib gradio pandas anthropic
```

### 2. Build the dataset

```bash
# Public data + seeds only (no API key needed)
python data/build_dataset.py

# Also generate synthetic examples (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
python data/build_dataset.py --synthetic --n_synthetic 200

# Push to HF Hub
python data/build_dataset.py --push --hf_repo belrem/llm-prompt-intent
```

### 3. Train

```bash
python train.py

# With HF push
python train.py --push --hf_model belrem/llm-prompt-intent-classifier
```

### 4. Deploy everything

```bash
huggingface-cli login
python deploy.py
```

### 5. Run the demo locally

```bash
cd spaces/app
python app.py
```

---

## Key design decisions

**Embedding model:** `all-MiniLM-L6-v2` is a strong general-purpose sentence encoder. At 384 dimensions it is fast enough to run inference in under 100ms on CPU, which matters for a pre-filter use case. It outperforms TF-IDF or bag-of-words on semantic tasks while remaining far cheaper than fine-tuning a full transformer.

**Balanced class weights:** Real prompt distributions are heavily skewed toward informational and task prompts. Without `class_weight="balanced"`, classifiers maximise accuracy by ignoring the adversarial class. Balanced weights ensure the adversarial precision/recall is meaningful.

**Three classifiers:** Each adds something. Logistic Regression is the interpretable baseline and achieved the best performance, suggesting the embedding space is already linearly separable. LinearSVC tends to perform well in high-dimensional spaces. MLP adds non-linear capacity but did not outperform the linear models on this dataset.

**Adversarial class ambiguity:** Noted explicitly in the model card and report. A creative prompt that happens to involve dark themes is not automatically adversarial; the classifier learns from patterns in intent framing, not topic alone.

---

## Limitations

- Adversarial classification is inherently uncertain. Use as a heuristic signal, not a security guarantee.
- Dataset skew: AdvBench prompts are more direct than real jailbreaks seen in production.
- Intent can be ambiguous — a prompt can simultaneously be `creative` and `task`. The model predicts the dominant intent.
- Model not updated as jailbreak patterns evolve.

---

## License

Apache 2.0
