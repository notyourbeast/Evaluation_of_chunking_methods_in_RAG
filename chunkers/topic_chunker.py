import spacy

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from sentence_transformers import SentenceTransformer



# ==========================================================
# Configuration
# ==========================================================

CHUNK_SIZE = 256

MIN_CHUNK_SIZE = 100

MODEL_MAX_TOKENS = 256

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

    current = []


    for word in words:

        current.append(word)

        candidate = " ".join(current)


        if count_tokens(candidate) > SAFE_SENTENCE_LIMIT:

            current.pop()


            if current:

                segments.append(
                    " ".join(current)
                )


            current = [word]


    if current:

        segments.append(
            " ".join(current)
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
            f"{doc_id}_topic_chunk_{chunk_number:04d}",

            "chunk_method": "topic",

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
# BERTopic Topic-Aware Chunking
# ==========================================================

def create_topic_chunks(documents):


    print("Preparing sentences for BERTopic...")


    sentence_records = []

    texts = []



    for document in documents:


        doc = nlp(
            document["text"]
        )


        sentences = []


        for sent in doc.sents:

            clean = sent.text.strip()


            if clean:

                sentences.extend(
                    split_long_sentence(clean)
                )



        for sentence in sentences:


            sentence_records.append(
                {
                    "doc_id": document["doc_id"],
                    "title": document["title"],
                    "sentence": sentence
                }
            )


            texts.append(
                sentence
            )



    print(
        "Total sentences:",
        len(texts)
    )



    # ======================================================
    # Sentence embeddings
    # ======================================================

    print(
        "Generating sentence embeddings..."
    )


    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=True
    )



    # ======================================================
    # BERTopic
    # ======================================================

    print(
        "Running BERTopic..."
    )


    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )


    hdbscan_model = HDBSCAN(
        min_cluster_size=10,
        metric="euclidean",
        cluster_selection_method="eom"
    )


    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=None,
        verbose=False
    )


    topics, _ = topic_model.fit_transform(
        texts,
        embeddings
    )


    print(
        "Topics found:",
        len(set(topics))
    )



    # ======================================================
    # Group sentences by document
    # ======================================================

    grouped = {}


    for record, topic in zip(
        sentence_records,
        topics
    ):


        doc_id = record["doc_id"]


        if doc_id not in grouped:

            grouped[doc_id] = []


        grouped[doc_id].append(
            {
                "title": record["title"],
                "sentence": record["sentence"],
                "topic": topic
            }
        )



    # ======================================================
    # Create chunks
    # ======================================================

    print(
        "Creating topic chunks..."
    )


    all_chunks = []



    for doc_id, sentences in grouped.items():


        chunk_number = 0

        current = []

        current_tokens = 0

        current_topic = None

        start_token = 0



        for item in sentences:


            sentence_tokens = count_tokens(
                item["sentence"]
            )



            topic_changed = (
                current_topic is not None
                and item["topic"] != current_topic
            )



            should_split = (

                current

                and current_tokens >= MIN_CHUNK_SIZE

                and topic_changed

            ) or (

                current

                and current_tokens + sentence_tokens > CHUNK_SIZE

            )



            if should_split:


                save_chunk(
                    all_chunks,
                    doc_id,
                    current[0]["title"],
                    chunk_number,
                    start_token,
                    current_tokens,
                    " ".join(
                        x["sentence"]
                        for x in current
                    )
                )


                chunk_number += 1

                start_token += current_tokens


                current = []

                current_tokens = 0



            current.append(
                item
            )


            current_tokens += sentence_tokens


            current_topic = item["topic"]



        if current:


            save_chunk(
                all_chunks,
                doc_id,
                current[0]["title"],
                chunk_number,
                start_token,
                current_tokens,
                " ".join(
                    x["sentence"]
                    for x in current
                )
            )



    return all_chunks