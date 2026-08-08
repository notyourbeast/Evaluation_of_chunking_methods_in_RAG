import pandas as pd
from pathlib import Path


# ==========================================================
# Files
# ==========================================================

SUMMARY_FILE = (
    "results/final_results_summary.csv"
)

STATS_FILE = (
    "results/statistical_results.csv"
)


OUTPUT_DIR = (
    "results/tables"
)



# ==========================================================
# Formatting helpers
# ==========================================================

def round_numeric(df):

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=[
            "float64",
            "float32"
        ]
    ).columns


    df.loc[:, numeric_columns] = (
        df[numeric_columns]
        .round(4)
    )


    return df



# ==========================================================
# Main
# ==========================================================

def main():


    Path(
        OUTPUT_DIR
    ).mkdir(
        exist_ok=True
    )


    summary = pd.read_csv(
        SUMMARY_FILE
    )


    stats = pd.read_csv(
        STATS_FILE
    )



    # ======================================================
    # Table 1
    # Retrieval Performance
    # ======================================================

    retrieval = summary[
        [
            "strategy",
            "precision_at_5",
            "recall_at_5",
            "mrr"
        ]
    ]


    retrieval = round_numeric(
        retrieval
    )


    retrieval.to_csv(
        f"{OUTPUT_DIR}/table_retrieval_performance.csv",
        index=False
    )



    # ======================================================
    # Table 2
    # Efficiency Trade-offs
    # ======================================================

    efficiency = summary[
        [
            "strategy",
            "indexed_chunks",
            "avg_total_latency_ms",
            "avg_context_tokens",
            "avg_context_characters"
        ]
    ]


    efficiency = round_numeric(
        efficiency
    )


    efficiency.to_csv(
        f"{OUTPUT_DIR}/table_efficiency_tradeoffs.csv",
        index=False
    )



    # ======================================================
    # Table 3
    # Statistical Testing
    # ======================================================

    statistical = stats[
        [
            "strategy_a",
            "strategy_b",
            "metric",
            "hypothesis_group",
            "p_value",
            "bonferroni_alpha",
            "significant"
        ]
    ]


    statistical = round_numeric(
        statistical
    )


    statistical.to_csv(
        f"{OUTPUT_DIR}/table_statistical_tests.csv",
        index=False
    )



    print(
        "Created dissertation tables:"
    )


    for file in Path(
        OUTPUT_DIR
    ).glob(
        "*.csv"
    ):

        print(
            file
        )



if __name__ == "__main__":

    main()