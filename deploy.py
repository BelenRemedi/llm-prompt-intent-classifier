"""
deploy.py — push all three HF artifacts in one command.

Run after:
  python data/build_dataset.py   (produces data/processed/prompt_intent)
  python train.py                (produces models/)

Then:
  huggingface-cli login
  python deploy.py
"""

import os
from huggingface_hub import HfApi

USERNAME    = "belrem"
DATASET_REPO = f"{USERNAME}/llm-prompt-intent"
MODEL_REPO   = f"{USERNAME}/llm-prompt-intent-classifier"
SPACE_REPO   = f"{USERNAME}/llm-prompt-intent-demo"

api = HfApi()


def ensure(repo_id, repo_type):
   kwargs = {"repo_id": repo_id, "repo_type": repo_type, "exist_ok": True}
   if repo_type == "space":
      kwargs["space_sdk"] = "gradio"
   api.create_repo(**kwargs)


def push_dataset():
    print("\n── Dataset ──────────────────────────────────")
    from datasets import load_from_disk
    ds = load_from_disk("data/processed/prompt_intent")
    ds.push_to_hub(DATASET_REPO, private=False)
    print(f"  ✓ https://huggingface.co/datasets/{DATASET_REPO}")


def push_model():
    print("\n── Model ────────────────────────────────────")
    ensure(MODEL_REPO, "model")
    for fname in ["classifier.joblib", "embedding_model.txt", "results.json", "README.md"]:
        path = f"models/{fname}"
        if os.path.exists(path):
            api.upload_file(path_or_fileobj=path, path_in_repo=fname, repo_id=MODEL_REPO)
            print(f"  ✓ {fname}")
        else:
            print(f"  ⚠ missing {path}")
    print(f"  ✓ https://huggingface.co/{MODEL_REPO}")


def push_space():
    print("\n── Space ────────────────────────────────────")
    ensure(SPACE_REPO, "space")

    readme = f"""---
title: LLM Prompt Intent Classifier
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
---

Classifies LLM prompts into: **creative · informational · task · adversarial**

**Model:** [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO})
**Dataset:** [{DATASET_REPO}](https://huggingface.co/datasets/{DATASET_REPO})
"""
    with open("spaces/app/README.md", "w") as f:
        f.write(readme)

    for fname in ["app.py", "requirements.txt", "README.md"]:
        path = f"spaces/app/{fname}"
        if os.path.exists(path):
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=fname,
                repo_id=SPACE_REPO,
                repo_type="space",
            )
            print(f"  ✓ {fname}")
    print(f"  ✓ https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    push_dataset()
    push_model()
    push_space()
    print(f"""
🎉 Done!
   Dataset : https://huggingface.co/datasets/{DATASET_REPO}
   Model   : https://huggingface.co/{MODEL_REPO}
   Demo    : https://huggingface.co/spaces/{SPACE_REPO}
""")
