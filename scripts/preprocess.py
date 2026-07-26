import argparse
import re
import logging
from pathlib import Path

import jsonlines
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger(__name__)


def clean_text(raw_text):

    text = raw_text

    # Remove unwanted trailing sections
    sections_to_remove = [
        "references",
        "bibliography",
        "further reading",
        "external links",
        "see also",
        "notes",
        "footnotes"
    ]

    for section in sections_to_remove:

        pattern = (
            rf"(?im)^=+\s*{section}\s*=+.*"
        )

        match = re.search(pattern, text)

        if match:
            text = text[:match.start()]


    # Remove wiki headings
    text = re.sub(
        r"^=+\s*(.+?)\s*=+\s*$",
        r"\1",
        text,
        flags=re.MULTILINE
    )


    # Remove citation markers

    text = re.sub(
        r"\[\d+\]",
        "",
        text
    )

    text = re.sub(
        r"\[citation needed\]",
        "",
        text,
        flags=re.IGNORECASE
    )


    # Remove excessive whitespace

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    lines = []

    for line in text.split("\n"):

        cleaned = re.sub(
            r"[ \t]+",
            " ",
            line
        ).strip()

        if cleaned:
            lines.append(cleaned)


    return "\n".join(lines)



def main(input_path, output_path):

    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    processed = []
    skipped = []


    with jsonlines.open(input_path) as reader:

        records = list(reader)


    log.info(
        f"Loaded {len(records)} raw articles"
    )


    for idx, record in enumerate(
        tqdm(records, desc="Cleaning")
    ):

        text = record.get("text", "")

        if not text:
            skipped.append(record.get("title"))
            continue


        cleaned = clean_text(text)


        words = len(
            cleaned.split()
        )


        if words < 300:

            log.warning(
                f"Skipping {record['title']} ({words} words)"
            )

            skipped.append(
                record["title"]
            )

            continue


        processed.append(
            {
                "doc_id": f"wiki_{idx+1:04d}",
                "title": record["title"],
                "url": record.get("url", ""),
                "text": cleaned,
                "word_count": words,
                "char_count": len(cleaned),
                "licence": record.get(
                    "licence",
                    "CC BY-SA 4.0"
                )
            }
        )


    with jsonlines.open(
        output_path,
        mode="w"
    ) as writer:

        writer.write_all(processed)



    print("\n==============================")
    print("Preprocessing complete")
    print("==============================")
    print(
        f"Processed : {len(processed)}"
    )
    print(
        f"Skipped   : {len(skipped)}"
    )
    print(
        f"Words     : {sum(x['word_count'] for x in processed):,}"
    )
    print(
        f"Output    : {output_path}"
    )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--input",
        default="data/raw/wikipedia_articles.jsonl"
    )


    parser.add_argument(
        "--output",
        default="data/processed/wikipedia_processed.jsonl"
    )


    args = parser.parse_args()


    main(
        args.input,
        args.output
    )