"""Shared utilities for team_runs/ experiments.

Functions
---------
normalize_emb_df(emb_df, expected_dim, prefix, id_col)
    Normalize an embeddings dataframe into a (ItemID + prefix_0..prefix_{D-1}) form.
    Handles three input formats:
      (a) single 'embedding' column with list/ndarray/string values,
      (b) N already-flat columns,
      (c) id-column named 'image_name' with '<ItemID>.png' values.
r_at_p90(y_true, y_score)
    Recall at the maximum recall point where precision >= 0.9 (0.0 if unreachable).
avg_precision_metric(y_true, y_score)
    Thin wrapper around sklearn.metrics.average_precision_score with the same
    (y_true, y_score) signature so it can plug into bootstrap_diff.
evaluate(y_true, y_proba, name)
    Print + return {name, roc_auc, pr_auc, r_at_p90} for one model.
bootstrap_diff(y_true, p_a, p_b, metric_fn, n, seed)
    Paired bootstrap CI for metric(p_b) - metric(p_a) at the 95% level (n=1000 default).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Embeddings normalization
# ---------------------------------------------------------------------------

def _parse_embedding_value(v):
    """Convert a single embedding value into a 1-D numpy float array."""
    if isinstance(v, np.ndarray):
        return v.astype(np.float32, copy=False)
    if isinstance(v, (list, tuple)):
        return np.asarray(v, dtype=np.float32)
    if isinstance(v, str):
        # tolerate "[0.1, 0.2, ...]" strings produced by csv round-trips
        s = v.strip().lstrip('[').rstrip(']')
        return np.fromstring(s, sep=',', dtype=np.float32)
    raise TypeError(f"Unsupported embedding value type: {type(v).__name__}")


def normalize_emb_df(emb_df: pd.DataFrame,
                     expected_dim: int,
                     prefix: str,
                     id_col: str = 'ItemID') -> pd.DataFrame:
    """Return a DataFrame with columns [ItemID, prefix_0, ..., prefix_{expected_dim-1}].

    Parameters
    ----------
    emb_df : input DataFrame in one of three accepted shapes (see module docstring).
    expected_dim : required embedding dimensionality; mismatch raises AssertionError.
    prefix : column-name prefix for the flat output columns (e.g. 'clip', 'text').
    id_col : name of the id column we want in the output (default 'ItemID').
             If the input has 'image_name' instead, it is converted: '10.png' -> 10.
    """
    df = emb_df.copy()

    # --- (c) image_name -> ItemID
    if id_col not in df.columns:
        if 'image_name' in df.columns:
            df['ItemID'] = (
                df['image_name'].astype(str)
                .str.replace(r'\.png$', '', regex=True)
                .str.replace(r'\.jpg$', '', regex=True)
                .astype(np.int64)
            )
            df = df.drop(columns=['image_name'])
        else:
            raise KeyError(
                f"Neither '{id_col}' nor 'image_name' present in columns: {list(df.columns)}"
            )

    # --- (a) single 'embedding' column with list/array/string values
    if 'embedding' in df.columns and df.shape[1] == 2:
        parsed = np.stack([_parse_embedding_value(v) for v in df['embedding'].values])
        flat_cols = [f'{prefix}_{i}' for i in range(parsed.shape[1])]
        out = pd.DataFrame(parsed, columns=flat_cols, index=df.index)
        out['ItemID'] = df['ItemID'].astype(np.int64).values
        out = out[['ItemID'] + flat_cols]
    else:
        # --- (b) N flat columns + ItemID
        non_id_cols = [c for c in df.columns if c != 'ItemID']
        flat_cols = [f'{prefix}_{i}' for i in range(len(non_id_cols))]
        rename_map = dict(zip(non_id_cols, flat_cols))
        out = df.rename(columns=rename_map)
        out['ItemID'] = out['ItemID'].astype(np.int64)
        out = out[['ItemID'] + flat_cols]

    assert out['ItemID'].is_unique, (
        f"Duplicate ItemIDs after normalization (n_dups={out['ItemID'].duplicated().sum()})"
    )
    actual_dim = out.shape[1] - 1
    assert actual_dim == expected_dim, (
        f"Dim mismatch: expected {expected_dim}, got {actual_dim}"
    )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def r_at_p90(y_true, y_score) -> float:
    """Maximum recall on the precision-recall curve where precision >= 0.9.

    Returns 0.0 if no threshold achieves precision >= 0.9.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= 0.9
    if not mask.any():
        return 0.0
    return float(recall[mask].max())


def avg_precision_metric(y_true, y_score) -> float:
    """Wrapper around average_precision_score so it has the (y, p) signature."""
    return float(average_precision_score(y_true, y_score))


def evaluate(y_true, y_proba, name: str) -> dict:
    """Compute ROC-AUC, PR-AUC, R@P90 and pretty-print them."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    roc = float(roc_auc_score(y_true, y_proba))
    pr = float(average_precision_score(y_true, y_proba))
    r90 = r_at_p90(y_true, y_proba)
    print(f"{name:<25} | ROC-AUC={roc:.4f} | PR-AUC={pr:.4f} | R@P90={r90:.4f}")
    return {'name': name, 'roc_auc': roc, 'pr_auc': pr, 'r_at_p90': r90}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_diff(y_true,
                   p_a,
                   p_b,
                   metric_fn=r_at_p90,
                   n: int = 1000,
                   seed: int = 42) -> dict:
    """Paired bootstrap 95% CI for metric(p_b) - metric(p_a).

    On each of `n` iterations a single set of resampled indices is applied to
    y_true, p_a, and p_b — so the two predictions are compared on identical
    samples (paired comparison). Iterations where the resampled y_true has no
    positives are skipped (logged via 'n_effective').
    """
    y_true = np.asarray(y_true)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    assert y_true.shape == p_a.shape == p_b.shape, (
        f"Shape mismatch: y={y_true.shape}, p_a={p_a.shape}, p_b={p_b.shape}"
    )

    rng = np.random.default_rng(seed)
    n_samples = len(y_true)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        yt = y_true[idx]
        if yt.sum() == 0:
            continue
        diffs.append(metric_fn(yt, p_b[idx]) - metric_fn(yt, p_a[idx]))

    diffs = np.asarray(diffs)
    lo = float(np.quantile(diffs, 0.025))
    hi = float(np.quantile(diffs, 0.975))
    return {
        'delta_mean': float(diffs.mean()),
        'ci_low': lo,
        'ci_high': hi,
        'significant': bool((lo > 0) or (hi < 0)),
        'n_effective': int(len(diffs)),
        'metric': metric_fn.__name__,
    }
