"""
train.py
────────
Loads the prompt-intent dataset, encodes each prompt with
sentence-transformers/all-MiniLM-L6-v2, then trains and compares:

  1. Logistic Regression   (interpretable baseline)
  2. Linear SVM            (strong in high-dim spaces)
  3. MLP (512 → 256)       (non-linear capacity)

Outputs:
  models/classifier.joblib      — best model (by weighted F1)
  models/results.json           — metrics for all three
  models/README.md              — auto-generated HF model card

Usage:
  pip install sentence-transformers scikit-learn datasets joblib huggingface_hub
  python train.py [--push] [--from_hub]
"""

import argparse
import json
import os
import time

import joblib
import numpy as np
from datasets import load_from_disk, load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

LABEL_NAMES = ["creative", "informational", "task", "adversarial"]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed(texts: list[str], model: SentenceTransformer) -> np.ndarray:
    return model.encode(texts, batch_size=128, show_progress_bar=True, convert_to_numpy=True)


# ─── Classifiers ─────────────────────────────────────────────────────────────

def get_classifiers() -> dict:
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=5.0, random_state=42,
                                       class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(max_iter=2000, C=1.0, random_state=42,
                              class_weight="balanced")),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(512, 256),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42,
            )),
        ]),
    }


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "report": classification_report(y_test, y_pred, target_names=LABEL_NAMES),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_pred": y_pred.tolist(),
    }


def print_comparison(results: dict) -> None:
    w = 24
    header = f"{'Classifier':<{w}} {'Accuracy':>9} {'F1 macro':>9} {'F1 weighted':>12}"
    bar = "─" * len(header)
    print(f"\n{bar}\n{header}\n{bar}")
    for name, r in results.items():
        print(f"{name:<{w}} {r['accuracy']:>9.4f} {r['f1_macro']:>9.4f} {r['f1_weighted']:>12.4f}")
    print(bar)


# ─── Model card ──────────────────────────────────────────────────────────────

def make_model_card(best_name: str, best_result: dict, hf_model: str, hf_dataset: str,
                    all_results: dict) -> str:
    rows = "\n".join(
        f"| {n} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['f1_weighted']:.4f} |"
        for n, r in all_results.items()
    )
    cm = best_result["confusion_matrix"]
    return f"""---
language: en
tags:
  - text-classification
  - sentence-transformers
  - prompt-classification
  - ai-safety
  - llm
license: apache-2.0
datasets:
  - {hf_dataset}
---

# LLM Prompt Intent Classifier

Classifies user prompts sent to LLMs into four intent categories using
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
sentence embeddings and a **{best_name}** classification head.

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
{rows}

## Best model: {best_name}

```
{best_result['report']}
```

## Confusion matrix (best model)

```
                 Predicted →
                 creative  info  task  adversarial
creative       {cm[0][0]:>8}  {cm[0][1]:>4}  {cm[0][2]:>4}  {cm[0][3]:>11}
informational  {cm[1][0]:>8}  {cm[1][1]:>4}  {cm[1][2]:>4}  {cm[1][3]:>11}
task           {cm[2][0]:>8}  {cm[2][1]:>4}  {cm[2][2]:>4}  {cm[2][3]:>11}
adversarial    {cm[3][0]:>8}  {cm[3][1]:>4}  {cm[3][2]:>4}  {cm[3][3]:>11}
```

## Inference

```python
from sentence_transformers import SentenceTransformer
import joblib
from huggingface_hub import hf_hub_download

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
clf_path = hf_hub_download(repo_id="{hf_model}", filename="classifier.joblib")
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
"""


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/prompt_intent")
    parser.add_argument("--hf_dataset", default="belrem/llm-prompt-intent")
    parser.add_argument("--hf_model", default="belrem/llm-prompt-intent-classifier")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--from_hub", action="store_true")
    args = parser.parse_args()

    # 1. Load dataset
    print("Loading dataset …")
    if args.from_hub:
        ds = load_dataset(args.hf_dataset)
    else:
        ds = load_from_disk(args.data)

    train_texts = ds["train"]["text"]
    train_labels = np.array(ds["train"]["label"])
    test_texts = ds["test"]["text"]
    test_labels = np.array(ds["test"]["label"])
    print(f"  Train: {len(train_texts)}  Test: {len(test_texts)}")

    # Print class distribution
    for i, name in enumerate(LABEL_NAMES):
        n = (train_labels == i).sum()
        print(f"  [{name}] train={n}")

    # 2. Embed
    print(f"\nEmbedding with {EMBEDDING_MODEL} …")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    t0 = time.time()
    X_train = embed(train_texts, embedder)
    X_test = embed(test_texts, embedder)
    print(f"  Done in {time.time()-t0:.1f}s  shape={X_train.shape}")

    # 3. Train and evaluate
    classifiers = get_classifiers()
    results = {}
    trained = {}
    for name, clf in classifiers.items():
        print(f"\nTraining {name} …")
        t0 = time.time()
        clf.fit(X_train, train_labels)
        elapsed = time.time() - t0
        res = evaluate(clf, X_test, test_labels)
        results[name] = res
        trained[name] = clf
        print(f"  [{elapsed:.1f}s]  accuracy={res['accuracy']}  f1_macro={res['f1_macro']}")
        print(res["report"])

    print_comparison(results)

    # 4. Select best
    best_name = max(results, key=lambda n: results[n]["f1_weighted"])
    print(f"\nBest: {best_name}")

    # 5. Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(trained[best_name], "models/classifier.joblib")
    print("Saved → models/classifier.joblib")

    # Save all results (exclude y_pred for brevity)
    summary = {
        n: {k: v for k, v in r.items() if k not in ("y_pred",)}
        for n, r in results.items()
    }
    with open("models/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("models/embedding_model.txt", "w") as f:
        f.write(EMBEDDING_MODEL)

    card = make_model_card(best_name, results[best_name], args.hf_model,
                           args.hf_dataset, results)
    with open("models/README.md", "w") as f:
        f.write(card)

    # 6. Push to HF
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=args.hf_model, repo_type="model", exist_ok=True)
        for fname in ["classifier.joblib", "embedding_model.txt", "results.json", "README.md"]:
            api.upload_file(
                path_or_fileobj=f"models/{fname}",
                path_in_repo=fname,
                repo_id=args.hf_model,
            )
        print(f"\nModel pushed → https://huggingface.co/{args.hf_model}")


if __name__ == "__main__":
    main()
