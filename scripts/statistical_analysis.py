import pandas as pd
from scipy.stats import wilcoxon


INPUT_FILE = (
    "results/retrieval_results.csv"
)


def compare(df, strategy_a, strategy_b):

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


    print()
    print("==============================")
    print(
        strategy_a,
        "vs",
        strategy_b
    )
    print("==============================")


    for metric in [
        "mrr",
        "hit"
    ]:

        stat, p = wilcoxon(
            a[metric],
            b[metric]
        )


        print(
            metric,
            "W:",
            stat,
            "p-value:",
            p
        )



def main():

    df = pd.read_csv(
        INPUT_FILE
    )


    comparisons = [
        ("sentence", "fixed"),
        ("semantic", "fixed"),
        ("topic", "fixed")
    ]


    for a, b in comparisons:

        compare(
            df,
            a,
            b
        )



if __name__ == "__main__":

    main()