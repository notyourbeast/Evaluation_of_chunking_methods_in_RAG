import argparse
import json
import sys
from pathlib import Path
import csv

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    EMBEDDING_MODEL,
    INDEX_OUTPUTS,
    METADATA_OUTPUTS,
    TOP_K
)



# ==========================================================
# Files
# ==========================================================

QUERY_FILE = (
    "data/raw/nq_filtered_queries.jsonl"
)


OUTPUT_FILE = (
    "results/retrieval_results.csv"
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
# Load chunk texts
# ==========================================================

def load_chunks(strategy):

    with open(
        CHUNK_FILES[strategy],
        "r",
        encoding="utf-8"
    ) as f:

        chunks = json.load(f)



    chunk_lookup = {}


    for chunk in chunks:

        chunk_lookup[
            chunk["chunk_id"]
        ] = chunk["text"]


    return chunk_lookup



# ==========================================================
# Text normalization
# ==========================================================

def normalize_text(text):

    return (
        text
        .lower()
        .strip()
    )



# ==========================================================
# Evaluation
# ==========================================================

def evaluate(strategy):


    print(
        "Loading FAISS index..."
    )


    index = faiss.read_index(
        INDEX_OUTPUTS[strategy]
    )


    print(
        "Vectors:",
        index.ntotal
    )



    print(
        "\nLoading metadata..."
    )


    with open(
        METADATA_OUTPUTS[strategy],
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)



    if index.ntotal != len(metadata):

        raise ValueError(
            "Index and metadata mismatch"
        )



    print(
        "Loading chunk texts..."
    )


    chunk_lookup = load_chunks(
        strategy
    )


    print(
        "Chunks loaded:",
        len(chunk_lookup)
    )



    print(
        "\nLoading embedding model..."
    )


    model = SentenceTransformer(
        EMBEDDING_MODEL
    )



    queries = load_queries()


    print(
        "Queries loaded:",
        len(queries)
    )



    results = []



    for count, item in enumerate(
        queries,
        start=1
    ):


        question = item["question"]


        answer = normalize_text(
            item["short_answer"]
        )



        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )



        scores, indices = index.search(
            query_embedding,
            TOP_K
        )



        hit = 0

        first_rank = 0

        relevant_chunks = 0

        reciprocal_rank = 0



        for rank, idx in enumerate(
            indices[0],
            start=1
        ):


            chunk_id = metadata[idx]["chunk_id"]


            chunk_text = normalize_text(
                chunk_lookup[chunk_id]
            )



            if answer in chunk_text:

                relevant_chunks += 1

                hit = 1


                if first_rank == 0:

                    first_rank = rank

                    reciprocal_rank = 1 / rank



        precision_at_5 = (
            relevant_chunks / TOP_K
        )



        results.append(
            {
                "strategy": strategy,
                "question_id": item["example_id"],
                "question": question,
                "answer": answer,
                "hit": hit,
                "rank": first_rank,
                "precision_at_5": precision_at_5,
                "mrr": reciprocal_rank
            }
        )



        if count % 10 == 0:

            print(
                "Processed:",
                count
            )



    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )



    output = Path(
        OUTPUT_FILE
    )



    write_header = not output.exists()



    with open(
        output,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )


        if write_header:

            writer.writeheader()


        writer.writerows(
            results
        )



    precision = sum(
        r["precision_at_5"]
        for r in results
    ) / len(results)



    recall = sum(
        r["hit"]
        for r in results
    ) / len(results)



    mrr = sum(
        r["mrr"]
        for r in results
    ) / len(results)



    print("==============================")
    print("Strategy:", strategy)
    print("Precision@5:", precision)
    print("Recall@5:", recall)
    print("MRR@5:", mrr)
    print("==============================")



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


    evaluate(
        args.strategy
    )