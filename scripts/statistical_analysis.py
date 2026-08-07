import pandas as pd
from scipy.stats import wilcoxon
from pathlib import Path


INPUT_FILE = (
    "results/retrieval_results.csv"
)


OUTPUT_FILE = (
    "results/statistical_results.csv"
)


ALPHA = 0.05


def compare(
    df,
    strategy_a,
    strategy_b
):

    a = df[
        df["strategy"] == strategy_a
    ].sort_values(
        "question_id"
    )


    b = df[
        df["strategy"] == strategy_b
    ].sort_values(
        "question_id"
    )


    results = []


    for metric in [
        "mrr",
        "hit"
    ]:

        stat, p = wilcoxon(
            a[metric],
            b[metric]
        )


        results.append(
            {
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
                "metric": metric,
                "wilcoxon_W": stat,
                "p_value": p
            }
        )


    return results



def main():

    df = pd.read_csv(
        INPUT_FILE
    )


    strategies = [
        "fixed",
        "sentence",
        "semantic",
        "topic"
    ]


    comparisons = []


    for i in range(
        len(strategies)
    ):

        for j in range(
            i + 1,
            len(strategies)
        ):

            comparisons.append(
                (
                    strategies[i],
                    strategies[j]
                )
            )



    all_results = []


    for a, b in comparisons:

        print(
            f"Running: {a} vs {b}"
        )


        all_results.extend(
            compare(
                df,
                a,
                b
            )
        )



    results = pd.DataFrame(
        all_results
    )


    # Bonferroni correction
    number_of_tests = len(results)


    results["bonferroni_alpha"] = (
        ALPHA / number_of_tests
    )


    results["significant"] = (
        results["p_value"]
        <
        results["bonferroni_alpha"]
    )



    Path(
        "results"
    ).mkdir(
        exist_ok=True
    )


    results.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print()
    print("==============================")
    print("Statistical Results")
    print("==============================")

    print(results)



    print()
    print(
        "Saved:",
        OUTPUT_FILE
    )



if __name__ == "__main__":

    main()