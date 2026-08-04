import json
import sys
from pathlib import Path
import argparse

import faiss
import ollama

from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    INDEX_OUTPUTS,
    METADATA_OUTPUTS,
    EMBEDDING_MODEL,
    TOP_K,
    OLLAMA_MODEL
)


QUERY_FILE = (
    "data/raw/nq_filtered_queries.jsonl"
)


OUTPUT_FILE = (
    "results/generation_results.json"
)



# ==========================================================
# Load queries
# ==========================================================

def load_queries():

    queries = []

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                queries.append(
                    json.loads(line)
                )

    return queries



# ==========================================================
# Load chunks
# ==========================================================

CHUNK_FILES = {

    "fixed":
        "data/processed/fixed_size/fixed_256_overlap32.json",

    "sentence":
        "data/processed/sentence_based/sentence_chunks.json",

    "semantic":
        "data/processed/semantic_based/semantic_chunks.json",

    "topic":
        "data/processed/topic_based/topic_chunks.json"

}



def load_chunks(strategy):

    with open(
        CHUNK_FILES[strategy],
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



# ==========================================================
# Evaluate generation
# ==========================================================

def evaluate(strategy):


    print(
        "Loading index..."
    )


    index = faiss.read_index(
        INDEX_OUTPUTS[strategy]
    )


    with open(
        METADATA_OUTPUTS[strategy],
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)



    chunks = load_chunks(
        strategy
    )


    if index.ntotal != len(chunks):

        raise ValueError(
            "FAISS index and chunk count mismatch"
        )



    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )



    queries = load_queries()



    results = []



    print(
        "Generating:",
        strategy
    )



    for count, item in enumerate(
        queries,
        start=1
    ):


        question = item["question"]



        query_embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )



        scores, indices = index.search(
            query_embedding,
            TOP_K
        )



        retrieved_contexts = []



        for idx in indices[0]:

            retrieved_contexts.append(
                chunks[idx]["text"]
            )



        context = "\n\n".join(
            retrieved_contexts
        )



        prompt = f"""
You are a Retrieval-Augmented Generation assistant.

Your answer must be based ONLY on the provided retrieved context.

Rules:
1. Do not use outside knowledge.
2. Do not infer missing facts.
3. If the answer is not explicitly supported by the retrieved context, respond exactly:

"I do not have enough information in the retrieved context."

Retrieved Context:

{context}


Question:

{question}


Answer:
"""



        response = ollama.chat(

            model=OLLAMA_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )



        answer = response["message"]["content"]



        results.append(
            {
                "strategy": strategy,
                "question_id": item["example_id"],
                "question": question,
                "ground_truth": item["short_answer"],
                "answer": answer,
                "contexts": retrieved_contexts
            }
        )



        if count % 10 == 0:

            print(
                "Processed:",
                count
            )



    return results



# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "fixed",
            "sentence",
            "semantic",
            "topic"
        ]
    )


    args = parser.parse_args()



    new_results = evaluate(
        args.strategy
    )



    output = Path(
        OUTPUT_FILE
    )



    if output.exists():

        with open(
            output,
            "r",
            encoding="utf-8"
        ) as f:

            all_results = json.load(f)

    else:

        all_results = []



    all_results.extend(
        new_results
    )



    output.parent.mkdir(
        exist_ok=True
    )



    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )



    print(
        "Saved:",
        OUTPUT_FILE
    )