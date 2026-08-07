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
# Create summary
# ==========================================================

def main():


    # ----------------------------
    # Retrieval metrics
    # ----------------------------

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


    retrieval_summary = (
        retrieval_summary
        .rename(
            columns={
                "hit":
                "recall_at_5"
            }
        )
    )



    # ----------------------------
    # Efficiency metrics
    # ----------------------------

    efficiency = pd.read_csv(
        EFFICIENCY_FILE
    )


    efficiency = efficiency[
        [
            "strategy",
            "avg_total_latency_ms"
        ]
    ]



    # ----------------------------
    # Chunk counts
    # ----------------------------

    chunks = []


    for strategy, index_file in INDEX_FILES.items():


        index = faiss.read_index(
            index_file
        )


        chunks.append(
            {
                "strategy":
                strategy,

                "num_chunks":
                index.ntotal
            }
        )



    chunks = pd.DataFrame(
        chunks
    )



    # ----------------------------
    # Merge results
    # ----------------------------

    summary = (
        retrieval_summary
        .merge(
            efficiency,
            on="strategy"
        )
        .merge(
            chunks,
            on="strategy"
        )
    )



    # ----------------------------
    # Final column order
    # ----------------------------

    summary = summary[
        [
            "strategy",
            "num_chunks",
            "precision_at_5",
            "recall_at_5",
            "mrr",
            "avg_total_latency_ms"
        ]
    ]
    order = [
        "fixed",
        "sentence",
        "semantic",
        "topic"
    ]

    summary["strategy"] = pd.Categorical(
        summary["strategy"],
        categories=order,
        ordered=True
    )

    summary = summary.sort_values(
        "strategy"
    )

    summary["strategy"] = summary["strategy"].astype(str)


    # ----------------------------
    # Save
    # ----------------------------

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
        "=" * 60
    )

    print(summary)



    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )



if __name__ == "__main__":

    main()