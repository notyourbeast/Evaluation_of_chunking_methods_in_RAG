import argparse
import time
import logging
import random
from pathlib import Path

import jsonlines
import wikipediaapi
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

log = logging.getLogger(__name__)


CATEGORY_FILE = "data/raw/article_list.txt"


def load_categories(category_file):

    path = Path(category_file)

    if not path.exists():
        raise FileNotFoundError(
            f"Category file not found: {category_file}"
        )

    with open(path, "r", encoding="utf-8") as f:

        categories = [
            line.strip()
            for line in f
            if line.strip()
        ]

    return categories



def get_category_articles(wiki, category, limit=50):

    titles = []

    try:

        category_page = wiki.page(
            f"Category:{category}"
        )

        if not category_page.exists():
            log.warning(
                f"Category not found: {category}"
            )
            return titles


        for title, item in category_page.categorymembers.items():

            # Ignore nested categories
            if title.startswith("Category:"):
                continue

            titles.append(title)

            if len(titles) >= limit:
                break


    except Exception as e:

        log.warning(
            f"Failed category {category}: {e}"
        )


    return titles



def collect_article_titles(wiki, categories, target):

    all_titles = []

    for category in categories:

        log.info(
            f"Collecting articles from {category}"
        )

        titles = get_category_articles(
            wiki,
            category,
            limit=50
        )

        all_titles.extend(titles)


    # remove duplicates
    all_titles = list(
        dict.fromkeys(all_titles)
    )


    return all_titles[:target]



def safe_page_fetch(page, retries=5):

    """
    Safely fetch Wikipedia page data.
    Handles temporary API failures and invalid responses.
    """

    for attempt in range(retries):

        try:

            if page.exists():

                return page

            return None


        except Exception as e:

            wait_time = (
                (attempt + 1) * 5
                + random.uniform(0, 2)
            )

            log.warning(
                f"Wikipedia API error: {e}. "
                f"Retrying in {wait_time:.1f}s "
                f"({attempt + 1}/{retries})"
            )

            time.sleep(wait_time)


    log.error(
        f"Failed after {retries} retries"
    )

    return None



def fetch_article(wiki, title):

    page = wiki.page(title)


    page = safe_page_fetch(page)


    if page is None:

        log.warning(
            f"Not found or unavailable: {title}"
        )

        return None



    if "may refer to" in page.summary[:200].lower():

        log.warning(
            f"Disambiguation skipped: {title}"
        )

        return None



    text = page.text.strip()

    words = len(text.split())


    if words < 500:

        log.warning(
            f"Too short ({words} words): {title}"
        )

        return None



    return {

        "title": page.title,
        "url": page.fullurl,
        "text": text,
        "word_count": words,
        "char_count": len(text),
        "licence": "CC BY-SA 4.0"

    }



def main(n_articles, output_path, delay):


    output = Path(output_path)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    wiki = wikipediaapi.Wikipedia(

        language="en",

        user_agent=
        "RAGChunkingResearch/1.0 (MSc Dissertation, Dublin Business School)"

    )


    categories = load_categories(
        CATEGORY_FILE
    )


    log.info(
        f"Loaded {len(categories)} categories"
    )


    titles = collect_article_titles(

        wiki,

        categories,

        n_articles

    )


    log.info(
        f"Collected {len(titles)} article titles"
    )


    records = []

    failed = []


    for title in tqdm(
        titles,
        desc="Downloading"
    ):


        article = fetch_article(
            wiki,
            title
        )


        if article:

            records.append(article)

            log.info(

                f"OK: {article['title']} "
                f"({article['word_count']:,} words)"

            )


        else:

            failed.append(title)



        time.sleep(delay)



    with jsonlines.open(
        output,
        mode="w"
    ) as writer:

        writer.write_all(records)



    print("\n==========================================")

    print("Wikipedia corpus completed")

    print("==========================================")

    print(
        f"Requested : {len(titles)}"
    )

    print(
        f"Downloaded: {len(records)}"
    )

    print(
        f"Failed    : {len(failed)}"
    )

    print(
        f"Words     : {sum(r['word_count'] for r in records):,}"
    )

    print(
        f"Output    : {output}"
    )




if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--n_articles",

        type=int,

        default=500

    )


    parser.add_argument(

        "--output",

        type=str,

        default="data/raw/wikipedia_articles.jsonl"

    )


    parser.add_argument(

        "--delay",

        type=float,

        default=2.0

    )


    args = parser.parse_args()


    main(

        args.n_articles,

        args.output,

        args.delay

    )