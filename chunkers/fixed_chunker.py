from transformers import AutoTokenizer


CHUNK_SIZE = 256
CHUNK_OVERLAP = 32


tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_fixed_chunks(text, title, doc_id):

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

                "chunk_overlap": CHUNK_OVERLAP,

                "start_token": start,

                "end_token": min(
                    end,
                    len(tokens)
                ),

                "token_count":
                len(chunk_tokens),

                "text": chunk_text
            }
        )


        chunk_number += 1

        start += CHUNK_SIZE - CHUNK_OVERLAP


    return chunks
