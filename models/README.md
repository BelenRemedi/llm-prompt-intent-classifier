---
language: en
tags:
  - text-classification
  - sentence-transformers
  - prompt-classification
  - ai-safety
  - llm
license: apache-2.0
datasets:
  - belrem/llm-prompt-intent
---

# LLM Prompt Intent Classifier

Classifies user prompts sent to LLMs into four intent categories using
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
sentence embeddings and a **Logistic Regression** classification head.

## Labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | `creative` | Fiction, brainstorming, roleplay, poetry |
| 1 | `informational` | Factual questions, explanations, definitions |
| 2 | `task` | Code, translation, summarisation, editing |
| 3 | `adversarial` | Jailbreaks, prompt injection, manipulation |

## Classifier comparison

| Classifier | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| Logistic Regression | 0.8218 | 0.8222 | 0.8209 |
| Linear SVM | 0.7816 | 0.7824 | 0.7816 |
| MLP | 0.8103 | 0.8090 | 0.8100 |

## Best model: Logistic Regression

```
               precision    recall  f1-score   support

     creative       0.78      0.89      0.83        45
informational       0.84      0.77      0.80        48
         task       0.80      0.89      0.85        37
  adversarial       0.87      0.75      0.80        44

     accuracy                           0.82       174
    macro avg       0.82      0.83      0.82       174
 weighted avg       0.83      0.82      0.82       174

```

## Confusion matrix (best model)

```
                 Predicted →
                 creative  info  task  adversarial
creative             40     2     3            0
informational         3    37     5            3
task                  0     2    33            2
adversarial           8     3     0           33
```

## Inference

```python
from sentence_transformers import SentenceTransformer
import joblib
from huggingface_hub import hf_hub_download

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
clf_path = hf_hub_download(repo_id="belrem/llm-prompt-intent-classifier", filename="classifier.joblib")
clf = joblib.load(clf_path)

prompt = "Write a poem about the ocean."
vec = embedder.encode([prompt])
label_id = clf.predict(vec)[0]
labels = ["creative", "informational", "task", "adversarial"]
print(labels[label_id])  # → creative
```

## Limitations

- Adversarial prompts are the hardest class: sophisticated jailbreaks using
  creative or hypothetical framing may be misclassified as `creative` or `task`.
- Intent is inherently ambiguous — a prompt can be simultaneously creative and
  a task. The model predicts the dominant intent.
- Dataset skew: adversarial examples from AdvBench may not reflect real-world
  jailbreak distributions.
