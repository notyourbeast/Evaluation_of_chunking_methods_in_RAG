import json


QUERY_FILE = (
    "data/raw/nq_eval_queries.jsonl"
)

CORPUS_FILE = (
    "data/raw/wikipedia_articles.jsonl"
)

OUTPUT_FILE = (
    "data/raw/nq_filtered_queries.jsonl"
)



def main():

    with open(
        CORPUS_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        corpus = f.read().lower()



    kept = []



    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue


            item = json.loads(line)


            answer = item["short_answer"].lower()


            if answer in corpus:

                kept.append(item)



    print(
        "Original queries:",
        100
    )

    print(
        "Filtered queries:",
        len(kept)
    )



    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for item in kept:

            f.write(
                json.dumps(item)
                + "\n"
            )



    print(
        "Saved:",
        OUTPUT_FILE
    )



if __name__ == "__main__":

    main()