"""
Run this script AFTER executing the training notebook (multimodal_counterfeit_run.ipynb)
to save the Doc2Vec model into the artifacts/ directory.

Usage (from the notebook, add a new cell at the end):

    import joblib, sys
    sys.path.insert(0, 'path/to/counterfeit_service')
    exec(open('save_d2v_model.py').read())

Or run from command line after the notebook has populated d2v_model in the kernel:

    python save_d2v_model.py --from-notebook

The script expects `d2v_model` to be available in the calling namespace.
If called as a standalone script it will attempt to re-train Doc2Vec on ozon_train.csv
(requires the data files to be present).
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
DATA_DIR = Path(__file__).parent.parent / "data"
SEED = 42


def save_existing_model(d2v_model):
    """Save an already-trained Doc2Vec model object to artifacts/."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    out_path = ARTIFACTS_DIR / "d2v_model.pkl"
    joblib.dump(d2v_model, out_path)
    logger.info("Saved Doc2Vec model → %s", out_path)


def retrain_and_save():
    """
    Re-train Doc2Vec with the same parameters used in the notebook and save it.
    Requires ozon_train.csv in the data directory.
    """
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from sklearn.model_selection import train_test_split

    logger.info("Loading data from %s", DATA_DIR)
    train_full = pd.read_csv(DATA_DIR / "ozon_train.csv")

    # Seller-based split — same logic as in notebook
    y = (train_full["resolution"] != "Нет нарушений").astype(int)
    mixed_sellers = (
        train_full.groupby("SellerId")["resolution"]
        .apply(lambda x: x.nunique() > 1)
    )
    mixed = mixed_sellers[mixed_sellers].index
    mask_mixed = train_full["SellerId"].isin(mixed)
    idx_mixed = train_full.index[mask_mixed]
    idx_clean = train_full.index[~mask_mixed]

    _, test_idx_mixed = train_test_split(idx_mixed, test_size=0.3, stratify=y[idx_mixed], random_state=SEED)
    train_idx = train_full.index.difference(test_idx_mixed)
    train_df = train_full.loc[train_idx].copy()

    # Build text same as notebook
    train_df["text"] = (
        train_df["name_rus"].fillna("") + " " +
        train_df["description"].fillna("") + " " +
        train_df["brand_name"].fillna("")
    )

    logger.info("Training Doc2Vec on %d documents...", len(train_df))
    tagged = [
        TaggedDocument(words=text.lower().split(), tags=[str(i)])
        for i, text in enumerate(train_df["text"])
    ]

    d2v_model = Doc2Vec(
        vector_size=200,
        window=5,
        min_count=2,
        dm=1,
        epochs=20,
        seed=SEED,
        workers=4,
    )
    d2v_model.build_vocab(tagged)
    logger.info("Vocab size: %d", len(d2v_model.wv))
    d2v_model.train(tagged, total_examples=d2v_model.corpus_count, epochs=d2v_model.epochs)
    logger.info("Doc2Vec training done.")

    save_existing_model(d2v_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save Doc2Vec model to artifacts/")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Re-train Doc2Vec from scratch using ozon_train.csv",
    )
    args = parser.parse_args()

    if args.retrain:
        retrain_and_save()
    else:
        print(
            "Use --retrain to re-train Doc2Vec from the raw data.\n"
            "Or call save_existing_model(d2v_model) from inside your notebook\n"
            "after the model has been trained.\n\n"
            "Example notebook cell:\n"
            "  import joblib\n"
            "  joblib.dump(d2v_model, '../counterfeit_service/artifacts/d2v_model.pkl')\n"
            "  print('Saved!')"
        )
