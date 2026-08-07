import json
import sys
from pathlib import Path
import argparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)

import faiss
from sentence_transformers import SentenceTransformer

import tiktoken


# ==========================================================
# Configuration
# ==========================================================

from config import (
    INDEX_OUTPUTS,
    EMBEDDING_MODEL,
    TOP_K
)


QUERY_FILE = (
    "data/raw/nq_filtered_queries.jsonl"
)


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


OUTPUT_FILE = (
    "results/token_usage_results.csv"
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

def load_chunks(strategy):

    with open(
        CHUNK_FILES[strategy],
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)



# ==========================================================
# Token counter
# ==========================================================

def count_tokens(text):

    encoding = tiktoken.get_encoding(
        "cl100k_base"
    )

    return len(
        encoding.encode(text)
    )



# ==========================================================
# Evaluate
# ==========================================================

def evaluate(strategy):


    print()
    print("=" * 60)
    print(
        "Evaluating:",
        strategy
    )
    print("=" * 60)



    index = faiss.read_index(
        INDEX_OUTPUTS[strategy]
    )


    chunks = load_chunks(
        strategy
    )


    queries = load_queries()


    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )


    results = []



    for count, item in enumerate(
        queries,
        start=1
    ):


        question = item["question"]


        embedding = embedding_model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )


        scores, indices = index.search(
            embedding,
            TOP_K
        )


        retrieved_chunks = []


        for idx in indices[0]:

            retrieved_chunks.append(
                chunks[idx]["text"]
            )



        context = "\n\n".join(
            retrieved_chunks
        )


        results.append(
            {
                "strategy": strategy,

                "question_id":
                item["example_id"],

                "retrieved_chunks":
                len(retrieved_chunks),

                "context_tokens":
                count_tokens(context),

                "context_characters":
                len(context)
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



    results = evaluate(
        args.strategy
    )



    import pandas as pd


    df = pd.DataFrame(
        results
    )


    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )


    output = Path(
        OUTPUT_FILE
    )


    if output.exists():

        old = pd.read_csv(
            output
        )

        df = pd.concat(
            [
                old,
                df
            ],
            ignore_index=True
        )



    df.to_csv(
        output,
        index=False
    )


    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )