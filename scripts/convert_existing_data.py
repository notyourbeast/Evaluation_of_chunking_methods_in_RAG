import json
import jsonlines
from pathlib import Path

nq_input  = Path("data/raw/nq_questions.json")
nq_output = Path("data/raw/nq_eval_queries.jsonl")

with open(nq_input) as f:
    raw = json.load(f)

selected = raw[:100]
converted = []
for i, item in enumerate(selected):
    converted.append({
        "question":     item["question"],
        "short_answer": item["answer"],
        "example_id":   str(i),
        "document_url": "",
    })

with jsonlines.open(nq_output, mode="w") as writer:
    writer.write_all(converted)

print(f"Done: {len(converted)} queries saved to {nq_output}")
