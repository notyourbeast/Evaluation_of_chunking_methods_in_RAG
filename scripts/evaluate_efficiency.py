import argparse
import json
import sys
import time
import csv

from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    EMBEDDING_MODEL,
    INDEX_OUTPUTS,
    TOP_K
)



# ==========================================================
# Files
# ==========================================================

QUERY_FILE = (
    "data/raw/nq_filtered_queries.jsonl"
)


OUTPUT_FILE = (
    "results/efficiency_results.csv"
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

                item = json.loads(line)

                queries.append(
                    item["question"]
                )


    return queries



# ==========================================================
# Evaluate efficiency
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
        "Loading embedding model..."
    )


    model = SentenceTransformer(
        EMBEDDING_MODEL
    )



    queries = load_queries()


    print(
        "Queries loaded:",
        len(queries)
    )



    embedding_times = []

    search_times = []

    total_times = []



    for question in queries:


        start_total = time.perf_counter()



        # ------------------------------
        # Embedding time
        # ------------------------------

        start_embedding = time.perf_counter()


        query_embedding = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )


        end_embedding = time.perf_counter()



        embedding_time = (
            end_embedding
            -
            start_embedding
        )



        # ------------------------------
        # FAISS search time
        # ------------------------------

        start_search = time.perf_counter()


        index.search(
            query_embedding,
            TOP_K
        )


        end_search = time.perf_counter()



        search_time = (
            end_search
            -
            start_search
        )



        end_total = time.perf_counter()



        total_time = (
            end_total
            -
            start_total
        )



        embedding_times.append(
            embedding_time
        )


        search_times.append(
            search_time
        )


        total_times.append(
            total_time
        )



    result = {

        "strategy":
        strategy,

        "vectors":
        index.ntotal,

        "queries":
        len(queries),

        "avg_embedding_ms":
        (
            sum(embedding_times)
            /
            len(embedding_times)
        )
        *
        1000,


        "avg_faiss_search_ms":
        (
            sum(search_times)
            /
            len(search_times)
        )
        *
        1000,


        "avg_total_latency_ms":
        (
            sum(total_times)
            /
            len(total_times)
        )
        *
        1000
    }



    return result



# ==========================================================
# Main
# ==========================================================

def main(strategy):


    result = evaluate(
        strategy
    )


    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )


    output = Path(
        OUTPUT_FILE
    )


    exists = output.exists()



    with open(
        output,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:


        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )


        if not exists:

            writer.writeheader()


        writer.writerow(
            result
        )



    print("==============================")

    print(
        "Strategy:",
        strategy
    )

    print(
        "Vectors:",
        result["vectors"]
    )

    print(
        "Average embedding ms:",
        result["avg_embedding_ms"]
    )

    print(
        "Average FAISS search ms:",
        result["avg_faiss_search_ms"]
    )

    print(
        "Average total latency ms:",
        result["avg_total_latency_ms"]
    )

    print("==============================")



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


    main(
        args.strategy
    )