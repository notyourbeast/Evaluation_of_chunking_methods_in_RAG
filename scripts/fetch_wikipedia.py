import wikipedia
import json
import os

TOPICS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Neural Network",
    "Large Language Model",
    "Information Retrieval",
    "Natural Language Processing",
    "Vector Database",
    "Deep Learning",
]

OUTPUT_FILE = "data/raw/wiki_articles.json"

os.makedirs("data/raw", exist_ok=True)

articles = []

for topic in TOPICS:
    try:
        page = wikipedia.page(topic)
        articles.append({
            "title": page.title,
            "content": page.content
        })
        print(f"Downloaded: {topic}")
    except Exception as e:
        print(f"Skipped: {topic}", e)

with open(OUTPUT_FILE, "w") as f:
    json.dump(articles, f, indent=2)

print("Saved to", OUTPUT_FILE)