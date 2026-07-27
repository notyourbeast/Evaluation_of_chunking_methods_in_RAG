import spacy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# Configuration
# ==========================================================

CHUNK_SIZE = 256

MIN_CHUNK_SIZE = 100

SIMILARITY_THRESHOLD = 0.25

MODEL_MAX_TOKENS = 256

# Safety margin for MiniLM tokenizer
SAFE_SENTENCE_LIMIT = 120



# ==========================================================
# Models
# ==========================================================

nlp = spacy.load(
    "en_core_web_sm"
)


embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


embedding_model.max_seq_length = MODEL_MAX_TOKENS


tokenizer = embedding_model.tokenizer



# ==========================================================
# Token utilities
# ==========================================================

def count_tokens(text):

    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=MODEL_MAX_TOKENS
    )

    return len(
        encoded["input_ids"]
    )



# ==========================================================
# Split long sentences
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
            f"{doc_id}_semantic_chunk_{chunk_number:04d}",

            "chunk_method": "semantic",

            "chunk_size": CHUNK_SIZE,

            "chunk_overlap": 0,

            "start_token": start_token,

            "end_token":
            start_token + token_count,

            "token_count": token_count,

            "text": text
        }
    )



# ==========================================================
# Semantic Chunk Creation
# ==========================================================

def create_semantic_chunks(text, title, doc_id):


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


    if not sentences:

        return []



    # ======================================================
    # Validate embedding inputs
    # ======================================================

    for sentence in sentences:

        length = count_tokens(
            sentence
        )


        if length > MODEL_MAX_TOKENS:

            raise ValueError(
                f"Sentence exceeds MiniLM limit: {length}"
            )



    # ======================================================
    # Generate embeddings
    # ======================================================

    sentence_embeddings = embedding_model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False
    )



    chunks = []

    current_sentences = []

    chunk_number = 0

    start_token = 0

    current_tokens = 0



    # ======================================================
    # Semantic grouping
    # ======================================================

    for i, sentence in enumerate(sentences):


        sentence_tokens = count_tokens(
            sentence
        )


        boundary = False


        if i > 0:

            similarity = cosine_similarity(
                [
                    sentence_embeddings[i - 1]
                ],
                [
                    sentence_embeddings[i]
                ]
            )[0][0]


            if similarity < SIMILARITY_THRESHOLD:

                boundary = True



        # Semantic boundary
        if (
            boundary
            and current_tokens >= MIN_CHUNK_SIZE
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



        # ==================================================
        # Enforce maximum chunk size
        # ==================================================

        if (
            current_tokens + sentence_tokens > CHUNK_SIZE
            and current_sentences
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


        current_tokens += sentence_tokens



    # ======================================================
    # Remaining chunk
    # ======================================================

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