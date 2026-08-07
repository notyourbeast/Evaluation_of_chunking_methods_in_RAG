import pandas as pd
import faiss
from pathlib import Path


# ==========================================================
# Files
# ==========================================================

RETRIEVAL_FILE = (
    "results/retrieval_results.csv"
)

EFFICIENCY_FILE = (
    "results/efficiency_results.csv"
)

TOKEN_FILE = (
    "results/token_usage_results.csv"
)


OUTPUT_FILE = (
    "results/final_results_summary.csv"
)



# ==========================================================
# FAISS indexes
# ==========================================================

INDEX_FILES = {

    "fixed":
        "models/fixed_faiss.index",

    "sentence":
        "models/sentence_faiss.index",

    "semantic":
        "models/semantic_faiss.index",

    "topic":
        "models/topic_faiss.index"

}



# ==========================================================
# Main
# ==========================================================

def main():


    # ------------------------------
    # Retrieval metrics
    # ------------------------------

    retrieval = pd.read_csv(
        RETRIEVAL_FILE
    )


    retrieval_summary = (
        retrieval
        .groupby("strategy")
        [
            [
                "precision_at_5",
                "mrr",
                "hit"
            ]
        ]
        .mean()
        .reset_index()
    )


    retrieval_summary = retrieval_summary.rename(
        columns={
            "hit":
            "recall_at_5"
        }
    )



    # ------------------------------
    # Latency
    # ------------------------------

    efficiency = pd.read_csv(
        EFFICIENCY_FILE
    )


    efficiency = (
        efficiency
        [
            [
                "strategy",
                "avg_total_latency_ms"
            ]
        ]
    )



    # ------------------------------
    # Token usage
    # ------------------------------

    token_usage = pd.read_csv(
        TOKEN_FILE
    )


    token_summary = (
        token_usage
        .groupby("strategy")
        [
            [
                "context_tokens",
                "context_characters"
            ]
        ]
        .mean()
        .reset_index()
    )


    token_summary = token_summary.rename(
        columns={
            "context_tokens":
            "avg_context_tokens",

            "context_characters":
            "avg_context_characters"
        }
    )



    # ------------------------------
    # Indexed chunks
    # ------------------------------

    vectors = []


    for strategy, file in INDEX_FILES.items():

        index = faiss.read_index(
            file
        )


        vectors.append(
            {
                "strategy":
                strategy,

                "indexed_chunks":
                index.ntotal
            }
        )


    vectors = pd.DataFrame(
        vectors
    )



    # ------------------------------
    # Merge
    # ------------------------------

    summary = (
        retrieval_summary

        .merge(
            efficiency,
            on="strategy"
        )

        .merge(
            token_summary,
            on="strategy"
        )

        .merge(
            vectors,
            on="strategy"
        )
    )



    # Order columns

    summary = summary[
        [
            "strategy",

            "indexed_chunks",

            "precision_at_5",

            "recall_at_5",

            "mrr",

            "avg_total_latency_ms",

            "avg_context_tokens",

            "avg_context_characters"
        ]
    ]



    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )



    summary.to_csv(
        OUTPUT_FILE,
        index=False
    )



    print()
    print(
        "Final Experiment Summary"
    )

    print(
        "=" * 70
    )

    print(summary)


    print()

    print(
        "Saved:",
        OUTPUT_FILE
    )



if __name__ == "__main__":

    main()