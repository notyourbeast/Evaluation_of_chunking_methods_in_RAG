from datasets import load_dataset
import json
import os

print("Loading NQ Open dataset...")

dataset = load_dataset("nq_open", split="train")

questions = []

for i, item in enumerate(dataset):
    
    if i >= 200:
        break

    questions.append({
        "question": item["question"],
        "answer": item["answer"][0] if item["answer"] else ""
    })

print("Collected:", len(questions))

os.makedirs("data/raw", exist_ok=True)

with open("data/raw/nq_questions.json", "w") as f:
    json.dump(questions, f, indent=2)

print("Saved to data/raw/nq_questions.json")
