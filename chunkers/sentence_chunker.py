import spacy
from transformers import AutoTokenizer


CHUNK_SIZE = 256


nlp = spacy.load(
    "en_core_web_sm"
)


tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def create_sentence_chunks(text, title, doc_id):

    doc = nlp(text)


    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]


    chunks = []

    current_sentences = []
    current_tokens = 0

    chunk_number = 0
    start_token = 0


    for sentence in sentences:


        sentence_tokens = tokenizer.encode(
            sentence,
            add_special_tokens=False
        )


        sentence_length = len(sentence_tokens)


        if (
            current_tokens + sentence_length > CHUNK_SIZE
            and current_sentences
        ):


            chunk_text = " ".join(
                current_sentences
            )


            chunks.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "source": "wikipedia",

                    "chunk_id":
                    f"{doc_id}_sentence_chunk_{chunk_number:04d}",

                    "chunk_method":
                    "sentence",

                    "chunk_size":
                    CHUNK_SIZE,

                    "chunk_overlap":
                    0,

                    "start_token":
                    start_token,

                    "end_token":
                    start_token + current_tokens,

                    "token_count":
                    current_tokens,

                    "text":
                    chunk_text
                }
            )


            chunk_number += 1

            start_token += current_tokens


            current_sentences = []
            current_tokens = 0



        current_sentences.append(sentence)

        current_tokens += sentence_length



    # Save remaining sentences

    if current_sentences:


        chunk_text = " ".join(
            current_sentences
        )


        chunks.append(
            {
                "doc_id": doc_id,
                "title": title,
                "source": "wikipedia",

                "chunk_id":
                f"{doc_id}_sentence_chunk_{chunk_number:04d}",

                "chunk_method":
                "sentence",

                "chunk_size":
                CHUNK_SIZE,

                "chunk_overlap":
                0,

                "start_token":
                start_token,

                "end_token":
                start_token + current_tokens,

                "token_count":
                current_tokens,

                "text":
                chunk_text
            }
        )


    return chunks
