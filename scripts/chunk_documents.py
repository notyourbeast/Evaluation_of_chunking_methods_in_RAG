import json
import os
from transformers import AutoTokenizer


INPUT_FILE = "data/raw/wiki_articles.json"
OUTPUT_FILE = "data/processed/fixed_size/fixed_256_overlap32.json"

CHUNK_SIZE = 256
OVERLAP = 32


print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_chunks(text, title, article_id):

    # Tokenize without truncation.
    # Long documents are intentionally split manually
    # using our fixed chunking strategy.
    tokens = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False
    )["input_ids"]


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
                "article_id": article_id,
                "title": title,
                "source": "wikipedia",
                "chunk_id": f"article{article_id:03d}_chunk{chunk_number:03d}",
                "chunk_method": "fixed",
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": OVERLAP,
                "start_token": start,
                "end_token": min(end, len(tokens)),
                "token_count": len(chunk_tokens),
                "text": chunk_text
            }
        )


        chunk_number += 1

        start += CHUNK_SIZE - OVERLAP


    return chunks



print("Loading Wikipedia corpus...")


with open(INPUT_FILE, "r") as f:
    articles = json.load(f)


all_chunks = []


for article_id, article in enumerate(articles, start=1):

    title = article["title"]
    content = article["content"]


    chunks = create_chunks(
        content,
        title,
        article_id
    )


    all_chunks.extend(chunks)



print("Total chunks:", len(all_chunks))


os.makedirs(
    "data/processed",
    exist_ok=True
)


with open(
    OUTPUT_FILE,
    "w"
) as f:

    json.dump(
        all_chunks,
        f,
        indent=2
    )


print("Saved:", OUTPUT_FILE)