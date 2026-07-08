import argparse
import json
import re
import logging
from pathlib import Path
import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def clean_text(raw_text):
    text = raw_text
    # Remove References/External links sections and everything after
    earliest = len(text)
    for pattern in [r"(?i)^=+\s*(references|bibliography|further reading|external links|see also|notes|footnotes)\s*=+.*"]:
        for match in re.finditer(pattern, text, re.MULTILINE):
            if match.start() < earliest:
                earliest = match.start()
    text = text[:earliest]
    # Remove section headers markup but keep title text
    text = re.sub(r"^=+\s*(.+?)\s*=+\s*$", r"\1", text, flags=re.MULTILINE)
    # Remove citation markers like [1], [2]
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalise whitespace within lines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()

def main(input_path, output_path):
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    raw_records = list(jsonlines.open(input_p))
    log.info(f"Loaded {len(raw_records)} raw articles")

    processed, skipped = [], []

    for i, record in enumerate(tqdm(raw_records, desc="Preprocessing")):
        # Handle both 'text' and 'content' field names
        raw_text = record.get("text") or record.get("content", "")
        cleaned = clean_text(raw_text)
        word_count = len(cleaned.split())

        if word_count < 300:
            log.warning(f"Skipping '{record['title']}' — {word_count} words after cleaning")
            skipped.append(record["title"])
            continue

        processed.append({
            "doc_id":     f"wiki_{i+1:04d}",
            "title":      record["title"],
            "url":        record.get("url", ""),
            "text":       cleaned,
            "word_count": word_count,
            "char_count": len(cleaned),
            "licence":    record.get("licence", "CC BY-SA 4.0"),
        })

    with jsonlines.open(output_p, mode="w") as writer:
        writer.write_all(processed)

    print(f"\n── Preprocessing complete ─────────────────────────────")
    print(f"  Processed : {len(processed)} documents")
    print(f"  Skipped   : {len(skipped)}")
    print(f"  Total words: {sum(d['word_count'] for d in processed):,}")
    print(f"  Avg words  : {sum(d['word_count'] for d in processed) // max(len(processed),1):,}")
    print(f"  Output     : {output_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="data/raw/wikipedia_articles.jsonl")
    parser.add_argument("--output", type=str, default="data/processed/wikipedia_processed.jsonl")
    args = parser.parse_args()
    main(args.input, args.output)
