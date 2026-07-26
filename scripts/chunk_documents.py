import json
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer


# ==========================================================
# Allow importing config.py from project root
# ==========================================================

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)


from config import (
    PROCESSED_DATA,
    FIXED_CHUNK_FILE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# ==========================================================
# Paths
# ==========================================================

INPUT_FILE = PROCESSED_DATA

OUTPUT_FILE = FIXED_CHUNK_FILE

OVERLAP = CHUNK_OVERLAP


# ==========================================================
# Load tokenizer
# ==========================================================

print("Loading tokenizer...")


tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================================
# Chunk creation
# ==========================================================

def create_chunks(text, title, doc_id):

    tokens = tokenizer.encode(
        text,
        add_special_tokens=False
    )


    chunks = []

    start = 0
    chunk_number = 0


    while start < len(tokens):

        end = start + CHUNK_SIZE


        chunk_tokens = tokens[start:end]


        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )


        chunks.append(
            {
                "doc_id": doc_id,
                "title": title,
                "source": "wikipedia",

                "chunk_id":
                    f"{doc_id}_chunk_{chunk_number:04d}",

                "chunk_method": "fixed",

                "chunk_size": CHUNK_SIZE,

                "chunk_overlap": OVERLAP,

                "start_token": start,

                "end_token": min(
                    end,
                    len(tokens)
                ),

                "token_count": len(chunk_tokens),

                "text": chunk_text
            }
        )


        chunk_number += 1

        start += CHUNK_SIZE - OVERLAP


    return chunks



# ==========================================================
# Load processed corpus
# ==========================================================

print("Loading processed corpus...")


articles = []


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            articles.append(
                json.loads(line)
            )


print(
    f"Articles loaded: {len(articles)}"
)



# ==========================================================
# Create chunks
# ==========================================================

all_chunks = []


for article in articles:

    chunks = create_chunks(
        article["text"],
        article["title"],
        article["doc_id"]
    )

    all_chunks.extend(chunks)



print(
    "Total chunks:",
    len(all_chunks)
)



# ==========================================================
# Save chunks
# ==========================================================

output_path = Path(OUTPUT_FILE)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    OUTPUT_FILE,
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
    OUTPUT_FILE
)