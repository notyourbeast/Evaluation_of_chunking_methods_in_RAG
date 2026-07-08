import argparse, json, time, logging
from pathlib import Path
import jsonlines, wikipediaapi
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DOMAIN_TITLES = [
    "Artificial intelligence", "Machine learning", "Artificial neural network",
    "Large language model", "Information retrieval",
    "Natural language processing", "Deep learning",
]
DIVERSE_TITLES = [
    "Human evolution", "Photosynthesis", "Black hole", "DNA",
    "General relativity", "Quantum mechanics", "Periodic table",
    "Vaccination", "World War II", "French Revolution",
    "Roman Empire", "Cold War", "American Civil War",
    "Ancient Egypt", "Renaissance", "Diabetes mellitus",
    "Cancer", "Penicillin", "Internet", "Alan Turing",
    "Apollo 11", "Amazon River", "Mount Everest",
    "Great Barrier Reef", "Sahara", "William Shakespeare",
    "Ludwig van Beethoven", "Leonardo da Vinci", "Hamlet",
    "Jane Austen", "Industrial Revolution", "Democracy",
    "Capitalism", "United Nations", "Karl Marx",
    "Mongol Empire", "Age of Enlightenment", "HIV/AIDS",
    "Higgs boson", "Space Shuttle Challenger disaster",
    "Yellowstone National Park", "CRISPR", "Alzheimer's disease",
]

def fetch_article(wiki, title):
    page = wiki.page(title)
    if not page.exists():
        log.warning(f"Not found: '{title}'"); return None
    if "may refer to" in page.summary[:200].lower():
        log.warning(f"Disambiguation, skipping: '{title}'"); return None
    text = page.text.strip()
    words = len(text.split())
    if words < 500:
        log.warning(f"Too short ({words} words): '{title}'"); return None
    return {"title": page.title, "url": page.fullurl, "text": text,
            "word_count": words, "char_count": len(text), "licence": "CC BY-SA 4.0"}

def main(n_articles, output_path, delay=0.5):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    wiki = wikipediaapi.Wikipedia(language="en",
        user_agent="RAGChunkingResearch/1.0 (MSc dissertation, Dublin Business School)")
    all_titles = DOMAIN_TITLES + DIVERSE_TITLES[:n_articles - len(DOMAIN_TITLES)]
    log.info(f"Fetching {len(all_titles)} articles")
    records, failed = [], []
    for title in tqdm(all_titles, desc="Downloading"):
        r = fetch_article(wiki, title)
        if r: records.append(r); log.info(f"  OK: {r['title']} ({r['word_count']:,} words)")
        else: failed.append(title)
        time.sleep(delay)
    with jsonlines.open(output, mode="w") as writer:
        writer.write_all(records)
    print(f"\n── Done ──────────────────────────────────────────")
    print(f"  Saved  : {len(records)} articles")
    print(f"  Failed : {len(failed)} {failed if failed else ''}")
    print(f"  Words  : {sum(r['word_count'] for r in records):,} total")
    print(f"  Output : {output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_articles", type=int, default=50)
    parser.add_argument("--output", type=str, default="data/raw/wikipedia_articles.jsonl")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    main(args.n_articles, args.output, args.delay)
