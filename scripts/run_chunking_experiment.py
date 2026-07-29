import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    PROCESSED_DATA,
    CHUNK_OUTPUTS
)


from chunkers.fixed_chunker import (
    create_fixed_chunks
)


from chunkers.sentence_chunker import (
    create_sentence_chunks
)


from chunkers.semantic_chunker import (
    create_semantic_chunks
)


from chunkers.topic_chunker import (
    create_topic_chunks
)



def load_documents(path):

    documents = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            if line.strip():

                documents.append(
                    json.loads(line)
                )

    return documents



def main(strategy):


    print("Loading processed corpus...")


    documents = load_documents(
        PROCESSED_DATA
    )


    print(
        "Articles loaded:",
        len(documents)
    )



    print(
        "Running strategy:",
        strategy
    )


    all_chunks = []



    # ======================================================
    # Topic-aware BERTopic
    # ======================================================

    if strategy == "topic":


        all_chunks = create_topic_chunks(
            documents
        )



    # ======================================================
    # Other chunking strategies
    # ======================================================

    else:


        if strategy == "fixed":

            chunk_function = create_fixed_chunks


        elif strategy == "sentence":

            chunk_function = create_sentence_chunks


        elif strategy == "semantic":

            chunk_function = create_semantic_chunks


        else:

            raise ValueError(
                f"Unsupported strategy: {strategy}"
            )



        for document in documents:


            chunks = chunk_function(
                document["text"],
                document["title"],
                document["doc_id"]
            )


            all_chunks.extend(
                chunks
            )



    print(
        "Total chunks:",
        len(all_chunks)
    )



    output_file = CHUNK_OUTPUTS[strategy]


    output_path = Path(
        output_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )



    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            all_chunks,
            f,
            indent=2,
            ensure_ascii=False
        )



    print(
        "Saved:",
        output_file
    )



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