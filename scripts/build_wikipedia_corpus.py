import wikipedia
import json
import random
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
TARGET_ARTICLES = 1000
OUTPUT_FILE = "data/raw/wiki_corpus.jsonl"

# Broad domain seed topics (NOT AI biased)
SEED_TOPICS = [
    "Science", "History", "Geography", "Physics",
    "Mathematics", "Biology", "Chemistry",
    "Politics", "Economics", "Technology",
    "Literature", "Art", "Music", "Philosophy"
]

# -----------------------------
# HELPERS
# -----------------------------
def get_random_article():
    try:
        title = wikipedia.random(1)
        page = wikipedia.page(title, auto_suggest=False)

        content = page.content

        # filter very small / junk pages
        if len(content) < 1000:
            return None

        return {
            "title": page.title,
            "url": page.url,
            "content": content
        }

    except Exception:
        return None


def is_duplicate(title, seen):
    return title.lower() in seen


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def build_corpus():
    seen_titles = set()
    dataset = []

    print("Building Wikipedia corpus...")

    with open(OUTPUT_FILE, "w") as f:

        with tqdm(total=TARGET_ARTICLES) as pbar:

            while len(dataset) < TARGET_ARTICLES:

                article = get_random_article()

                if article is None:
                    continue

                title = article["title"]

                if is_duplicate(title, seen_titles):
                    continue

                seen_titles.add(title.lower())

                # write immediately (IMPORTANT: no RAM buildup)
                f.write(json.dumps(article) + "\n")

                dataset.append(article)
                pbar.update(1)

    print(f"\nDONE: {len(dataset)} articles saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_corpus()
