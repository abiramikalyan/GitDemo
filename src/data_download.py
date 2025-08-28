import argparse
import os
from typing import Optional, Tuple

import pandas as pd
from datasets import load_dataset

from .utils import ensure_dir


AMAZON_DATASETS = {
    "clothing2018": {
        "reviews": ("McAuley-Lab/Amazon-Reviews-2018", "raw_review_data/Clothing_Shoes_and_Jewelry_5"),
        "meta": ("McAuley-Lab/Amazon-Reviews-2018", "raw_meta_data/Clothing_Shoes_and_Jewelry"),
    }
}


def _standardize_reviews(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "reviewerID": "user_id",
        "asin": "item_id",
        "overall": "rating",
        "reviewText": "review_text",
        "summary": "summary",
        "unixReviewTime": "time",
    }
    df = df.rename(columns=col_map)
    keep_cols = ["user_id", "item_id", "rating", "review_text", "summary", "time"]
    df = df[keep_cols]
    return df


def _standardize_meta(df: pd.DataFrame) -> pd.DataFrame:
    # Meta fields can vary; we keep common ones when present
    possible_cols = {
        "asin": "item_id",
        "title": "title",
        "description": "description",
        "category": "category",
        "categories": "categories",
        "brand": "brand",
    }
    present = {src: dst for src, dst in possible_cols.items() if src in df.columns}
    df = df.rename(columns=present)
    keep_cols = list(set(present.values()))
    if "item_id" not in keep_cols:
        # If item_id (asin) missing, create from existing index (rare)
        df["item_id"] = df.index.astype(str)
        keep_cols = ["item_id"] + keep_cols
    df = df[keep_cols]
    return df


def download_amazon_reviews_and_meta(dataset_key: str, out_dir: str, max_rows: Optional[int] = None, max_meta: Optional[int] = None) -> Tuple[str, str]:
    info = AMAZON_DATASETS[dataset_key]
    reviews_ds_id, reviews_config = info["reviews"]
    meta_ds_id, meta_config = info["meta"]

    print(f"Loading reviews: {reviews_ds_id} / {reviews_config}")
    ds_reviews = load_dataset(reviews_ds_id, reviews_config, split="train")
    df_reviews = ds_reviews.to_pandas()
    df_reviews = _standardize_reviews(df_reviews)
    if max_rows is not None:
        df_reviews = df_reviews.head(max_rows)

    print(f"Loading metadata: {meta_ds_id} / {meta_config}")
    ds_meta = load_dataset(meta_ds_id, meta_config, split="train")
    df_meta = ds_meta.to_pandas()
    df_meta = _standardize_meta(df_meta)
    if max_meta is not None:
        df_meta = df_meta.head(max_meta)

    ensure_dir(out_dir)
    reviews_path = os.path.join(out_dir, f"{dataset_key}_reviews.parquet")
    meta_path = os.path.join(out_dir, f"{dataset_key}_meta.parquet")
    df_reviews.to_parquet(reviews_path, index=False)
    df_meta.to_parquet(meta_path, index=False)

    print(f"Saved reviews to {reviews_path} with {len(df_reviews)} rows")
    print(f"Saved metadata to {meta_path} with {len(df_meta)} rows")
    return reviews_path, meta_path


essage = """
This script downloads Amazon Clothing reviews (2018) and metadata via Hugging Face Datasets.
Outputs Parquet files with standardized columns.
"""


def main():
    parser = argparse.ArgumentParser(description=essage)
    parser.add_argument("--dataset", type=str, default="clothing2018")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit number of review rows")
    parser.add_argument("--max_meta", type=int, default=None, help="Limit number of metadata rows")
    args = parser.parse_args()

    if args.dataset not in AMAZON_DATASETS:
        raise ValueError(f"Unknown dataset {args.dataset}")

    download_amazon_reviews_and_meta(args.dataset, args.out_dir, args.max_rows, args.max_meta)


if __name__ == "__main__":
    main()
