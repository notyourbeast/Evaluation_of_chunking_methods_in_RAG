import argparse
import json
import sys
from pathlib import Path

import faiss

from sentence_transformers import SentenceTransformer


# ==========================================================
# Add project root
# ==========================================================

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
# Retrieval Test
# ==========================================================

def main(strategy):


    if strategy not in INDEX_OUTPUTS:

        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )


    index_file = INDEX_OUTPUTS[strategy]

    metadata_file = METADATA_OUTPUTS[strategy]



    # ======================================================
    # Load FAISS
    # ======================================================

    print("Loading FAISS index...")


    index = faiss.read_index(
        index_file
    )


    print(
        "Vectors:",
        index.ntotal
    )



    # ======================================================
    # Load Metadata
    # ======================================================

    print("\nLoading metadata...")


    with open(
        metadata_file,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)



    print(
        "Metadata records:",
        len(metadata)
    )



    if index.ntotal != len(metadata):

        raise ValueError(
            "FAISS vectors and metadata count do not match"
        )



    # ======================================================
    # Load embedding model
    # ======================================================

    print("\nLoading embedding model...")


    model = SentenceTransformer(
        EMBEDDING_MODEL
    )


    print(
        "Embedding model loaded"
    )



    # ======================================================
    # Query
    # ======================================================

    query = input(
        "\nEnter your question: "
    ).strip()


    if not query:

        raise ValueError(
            "Query cannot be empty"
        )



    # ======================================================
    # Encode query
    # ======================================================

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )



    # ======================================================
    # Search
    # ======================================================

    scores, indices = index.search(
        query_embedding,
        TOP_K
    )



    # ======================================================
    # Display results
    # ======================================================

    print("\n")
    print("=" * 60)
    print("Top Results")
    print("=" * 60)

    print(
        "Strategy:",
        strategy
    )


    for rank, idx in enumerate(
        indices[0],
        start=1
    ):


        print("=" * 60)

        print(
            "Rank:",
            rank
        )


        print(
            "Score:",
            scores[0][rank-1]
        )


        print(
            "Title:",
            metadata[idx]["title"]
        )


        print(
            "Chunk ID:",
            metadata[idx]["chunk_id"]
        )


        print(
            "Chunk method:",
            metadata[idx]["chunk_method"]
        )


        print(
            "Token count:",
            metadata[idx]["token_count"]
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