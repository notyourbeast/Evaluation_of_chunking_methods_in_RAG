import argparse
import os
import sys
from pathlib import Path

import numpy as np
import faiss


# ==========================================================
# Add project root
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.append(
    str(PROJECT_ROOT)
)


from config import (
    EMBEDDING_OUTPUTS,
    INDEX_OUTPUTS
)



# ==========================================================
# Build FAISS index
# ==========================================================

def main(strategy):


    if strategy not in EMBEDDING_OUTPUTS:

        raise ValueError(
            f"Unsupported strategy: {strategy}"
        )


    embedding_file = EMBEDDING_OUTPUTS[strategy]

    index_file = INDEX_OUTPUTS[strategy]



    # ======================================================
    # Load embeddings
    # ======================================================

    print("Loading embeddings...")


    embeddings = np.load(
        embedding_file
    ).astype("float32")


    print(
        "Embedding shape:",
        embeddings.shape
    )



    # ======================================================
    # Normalize
    # ======================================================

    print(
        "Normalizing embeddings..."
    )


    faiss.normalize_L2(
        embeddings
    )



    # ======================================================
    # Create FAISS index
    # ======================================================

    dimension = embeddings.shape[1]


    print(
        "Creating IndexFlatIP..."
    )


    index = faiss.IndexFlatIP(
        dimension
    )



    # ======================================================
    # Add vectors
    # ======================================================

    print(
        "Adding embeddings to index..."
    )


    index.add(
        embeddings
    )


    print(
        "Total vectors:",
        index.ntotal
    )



    # ======================================================
    # Save index
    # ======================================================

    os.makedirs(
        os.path.dirname(index_file),
        exist_ok=True
    )


    faiss.write_index(
        index,
        index_file
    )


    print()

    print("=" * 40)
    print("FAISS index created successfully")
    print("=" * 40)

    print(
        "Strategy:",
        strategy
    )

    print(
        "Saved:",
        index_file
    )



# ==========================================================
# CLI
# ==========================================================

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