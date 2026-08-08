import pandas as pd
import numpy as np

from scipy.stats import wilcoxon


# ==========================================================
# Files
# ==========================================================

RETRIEVAL_FILE = "results/retrieval_results.csv"
TOKEN_FILE = "results/token_usage_results.csv"

OUTPUT_FILE = "results/statistical_results.csv"


# ==========================================================
# Configuration
# ==========================================================

ALPHA = 0.05


RETRIEVAL_METRICS = [
    "precision_at_5",
    "recall_at_5",
    "mrr"
]


EFFICIENCY_METRICS = [
    "context_tokens"
]


STRATEGIES = [
    "fixed",
    "sentence",
    "semantic",
    "topic"
]


# ==========================================================
# Wilcoxon + effect size
# ==========================================================

def run_wilcoxon(a, b):

    differences = a - b

    differences = differences[
        differences != 0
    ]

    n = len(differences)


    if n < 5:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )


    try:

        result = wilcoxon(
            a,
            b,
            zero_method="wilcox"
        )


        W = result.statistic
        p = result.pvalue


        mean_W = (
            n * (n + 1)
        ) / 4


        std_W = np.sqrt(
            (
                n *
                (n + 1) *
                (2*n + 1)
            ) / 24
        )


        z = (
            W - mean_W
        ) / std_W


        effect_r = (
            abs(z)
            /
            np.sqrt(n)
        )


        return (
            W,
            z,
            effect_r,
            p
        )


    except Exception:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )



# ==========================================================
# Pairwise comparison
# ==========================================================

def compare(
    df,
    strategy_a,
    strategy_b,
    metric,
    group
):


    a = (
        df[
            df["strategy"] == strategy_a
        ]
        .sort_values(
            "question_id"
        )[metric]
        .reset_index(drop=True)
    )


    b = (
        df[
            df["strategy"] == strategy_b
        ]
        .sort_values(
            "question_id"
        )[metric]
        .reset_index(drop=True)
    )


    print(
        f"Running: {strategy_a} vs {strategy_b} {metric}"
    )


    W,z,effect,p = run_wilcoxon(
        a,
        b
    )


    return {

        "strategy_a":
            strategy_a,

        "strategy_b":
            strategy_b,

        "metric":
            metric,

        "hypothesis_group":
            group,

        "wilcoxon_W":
            W,

        "z_score":
            z,

        "effect_size_r":
            effect,

        "p_value":
            p

    }



# ==========================================================
# Main
# ==========================================================

def main():


    retrieval = pd.read_csv(
        RETRIEVAL_FILE
    )


    # raw retrieval uses hit
    if "recall_at_5" not in retrieval.columns:

        retrieval = retrieval.rename(
            columns={
                "hit":
                "recall_at_5"
            }
        )



    tokens = pd.read_csv(
        TOKEN_FILE
    )


    results = []



    # -------------------------
    # Retrieval comparisons
    # -------------------------

    for metric in RETRIEVAL_METRICS:


        for i in range(len(STRATEGIES)):

            for j in range(
                i+1,
                len(STRATEGIES)
            ):


                results.append(
                    compare(
                        retrieval,
                        STRATEGIES[i],
                        STRATEGIES[j],
                        metric,
                        "retrieval"
                    )
                )



    # -------------------------
    # Efficiency comparisons
    # -------------------------

    for metric in EFFICIENCY_METRICS:


        for i in range(len(STRATEGIES)):

            for j in range(
                i+1,
                len(STRATEGIES)
            ):


                results.append(
                    compare(
                        tokens,
                        STRATEGIES[i],
                        STRATEGIES[j],
                        metric,
                        "efficiency"
                    )
                )



    df = pd.DataFrame(
        results
    )


    # ==================================================
    # Separate Bonferroni correction
    # ==================================================

    retrieval_tests = (
        df["hypothesis_group"]
        ==
        "retrieval"
    )


    efficiency_tests = (
        df["hypothesis_group"]
        ==
        "efficiency"
    )


    df.loc[
        retrieval_tests,
        "bonferroni_alpha"
    ] = (
        ALPHA /
        retrieval_tests.sum()
    )


    df.loc[
        efficiency_tests,
        "bonferroni_alpha"
    ] = (
        ALPHA /
        efficiency_tests.sum()
    )


    df["significant"] = (
        df["p_value"]
        <
        df["bonferroni_alpha"]
    )



    df = df[
        [
            "strategy_a",
            "strategy_b",
            "metric",
            "hypothesis_group",
            "wilcoxon_W",
            "z_score",
            "effect_size_r",
            "p_value",
            "bonferroni_alpha",
            "significant"
        ]
    ]



    print()

    print(
        "="*30
    )

    print(
        "Statistical Results"
    )

    print(
        "="*30
    )

    print(df)



    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print()

    print(
        "Saved:",
        OUTPUT_FILE
    )



if __name__ == "__main__":
    main()