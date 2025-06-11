# -*- coding: utf-8 -*-
"""ETL script for tweet sentiment analysis.

Usage:
    python etl.py --source csv --input data/raw/tweets.csv
    python etl.py --source api

The script cleans tweets, filters French tweets and stores the
resulting DataFrame as Parquet and in a SQLite database.
"""

import argparse
import logging
import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd
import tweepy
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
from dotenv import load_dotenv


# Regular expressions for cleaning
URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#\w+")
# simple emoji pattern covering a broad range of unicode emoji characters
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
    flags=re.UNICODE,
)


load_dotenv()
nlp = spacy.load("fr_core_news_sm", disable=["parser", "ner"])


def clean_text(text: str) -> str:
    """Clean and normalize tweet text."""
    text = text or ""
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = HASHTAG_PATTERN.sub("", text)
    text = EMOJI_PATTERN.sub("", text)
    text = re.sub(r"[!?\.]{2,}", " ", text)
    text = text.lower()

    doc = nlp(text)
    tokens = [t.lemma_ for t in doc if t.is_alpha and t.lemma_ not in STOP_WORDS]
    return " ".join(tokens)


def load_from_csv(path: Path) -> pd.DataFrame:
    logging.info("Loading CSV %s", path)
    return pd.read_csv(path)


def load_from_api() -> pd.DataFrame:
    """Load tweets from Twitter API v2 using Tweepy."""
    bearer = os.getenv("TWITTER_BEARER_TOKEN")
    if not bearer:
        raise RuntimeError("TWITTER_BEARER_TOKEN is not set")

    client = tweepy.Client(bearer_token=bearer, wait_on_rate_limit=True)
    query = "lang:fr -is:retweet"
    response = client.search_recent_tweets(
        query=query,
        tweet_fields=["id", "text", "created_at", "lang"],
        max_results=100,
    )
    tweets = response.data or []
    data = [tweet.data for tweet in tweets]
    return pd.DataFrame(data)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Filter French tweets and add a ``clean_text`` column."""
    df = df[df.get("lang") == "fr"].copy()
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    return df


def save(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "tweets.parquet"
    db_path = out_dir / "tweets.db"

    logging.info("Saving to %s", parquet_path)
    df.to_parquet(parquet_path, index=False)

    logging.info("Saving to %s", db_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("tweets", conn, if_exists="replace", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tweet ETL")
    parser.add_argument("--source", choices=["csv", "api"], required=True)
    parser.add_argument("--input", help="Input CSV path for csv source")
    parser.add_argument("--output", default="data/clean", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    start = time.time()

    if args.source == "csv":
        if not args.input:
            raise ValueError("--input is required when source=csv")
        df = load_from_csv(Path(args.input))
    else:
        df = load_from_api()

    df = transform(df)
    save(df, Path(args.output))

    elapsed = time.time() - start
    logging.info("ETL finished in %.2f seconds", elapsed)


if __name__ == "__main__":
    main()
