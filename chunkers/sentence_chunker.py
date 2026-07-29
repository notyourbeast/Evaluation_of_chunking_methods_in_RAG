import spacy
from transformers import AutoTokenizer


# ==========================================================
# Configuration
# ==========================================================

CHUNK_SIZE = 256

SAFE_SENTENCE_LIMIT = 240



# ==========================================================
# Models
# ==========================================================

nlp = spacy.load(
    "en_core_web_sm"
)


tokenizer = AutoTokenizer.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)



# ==========================================================
# Token utilities
# ==========================================================

def count_tokens(text):

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )



# ==========================================================
# Split extremely long sentences
# ==========================================================

def split_long_sentence(sentence):


    words = sentence.split()

    segments = []

    current_words = []


    for word in words:


        current_words.append(
            word
        )


        candidate = " ".join(
            current_words
        )


        if count_tokens(candidate) > SAFE_SENTENCE_LIMIT:


            current_words.pop()


            if current_words:

                segments.append(
                    " ".join(current_words)
                )


            current_words = [
                word
            ]



    if current_words:

        segments.append(
            " ".join(current_words)
        )


    return segments



# ==========================================================
# Save chunk
# ==========================================================

def save_chunk(
    chunks,
    doc_id,
    title,
    chunk_number,
    start_token,
    token_count,
    text
):


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
            start_token + token_count,

            "token_count":
            token_count,

            "text":
            text
        }
    )



# ==========================================================
# Sentence-based Chunk Creation
# ==========================================================

def create_sentence_chunks(text, title, doc_id):


    doc = nlp(text)



    raw_sentences = [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]



    sentences = []


    for sentence in raw_sentences:

        sentences.extend(
            split_long_sentence(sentence)
        )



    chunks = []

    current_sentences = []

    current_tokens = 0

    chunk_number = 0

    start_token = 0



    for sentence in sentences:


        sentence_length = count_tokens(
            sentence
        )



        if (
            current_sentences
            and current_tokens + sentence_length > CHUNK_SIZE
        ):


            save_chunk(
                chunks,
                doc_id,
                title,
                chunk_number,
                start_token,
                current_tokens,
                " ".join(current_sentences)
            )


            chunk_number += 1

            start_token += current_tokens


            current_sentences = []

            current_tokens = 0



        current_sentences.append(
            sentence
        )


        current_tokens += sentence_length



    if current_sentences:


        save_chunk(
            chunks,
            doc_id,
            title,
            chunk_number,
            start_token,
            current_tokens,
            " ".join(current_sentences)
        )



    return chunks