"""
Business-logic layer for H-FGAT backend.

All functions accept a `bundle` dict (from loader.load_artifacts()) and
return plain Python dicts/lists — no Flask or HTTP concerns here.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch


# ─── Shared helpers ────────────────────────────────────────────────────────────

def get_item_info(item_id: str, item_meta: Dict) -> dict:
    """Return enriched item metadata dict for a given item_id string."""
    meta = item_meta.get(str(item_id), {})
    return {
        "item_id": str(item_id),
        "title": meta.get("title", ""),
        "category": meta.get("category", None),
        "image_url": meta.get("image_url", ""),
    }


def enrich_outfit(outfit_id: str, bundle: Dict) -> dict:
    """Return outfit dict with its items enriched with metadata."""
    item_ids: List[str] = bundle["outfit_items"].get(str(outfit_id), [])
    return {
        "outfit_id": str(outfit_id),
        "items": [get_item_info(iid, bundle["item_meta"]) for iid in item_ids],
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ─── Service: Personalized Outfit Recommendation ───────────────────────────────

def recommend_outfits(
    user_id: str,
    bundle: Dict,
    top_k: int = 10,
    exclude_seen: bool = False,
) -> Optional[List[dict]]:
    """
    Return the top-k outfits recommended for a given user based on
    cosine similarity between user embedding and outfit embeddings.

    Returns None if user_id is not in the training set.
    """
    if user_id not in bundle["user2idx"]:
        return None

    uidx = bundle["user2idx"][user_id]
    user_vec = bundle["user_emb"][uidx: uidx + 1]           # [1, D]
    scores = torch.matmul(user_vec, bundle["outfit_emb"].T).squeeze(0)  # [N_outfits]

    if exclude_seen:
        seen_outfits = set(
            bundle["history_df"]
            .loc[bundle["history_df"]["user_id"] == user_id, "outfit_id"]
            .tolist()
        )
        for oid_str, oidx in bundle["outfit2idx"].items():
            if oid_str in seen_outfits:
                scores[oidx] = -1e9

    k = min(top_k, scores.numel())
    vals, idxs = torch.topk(scores, k=k)

    results = []
    for rank, (score, oidx) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        oid = bundle["idx2outfit"][oidx]
        outfit = enrich_outfit(oid, bundle)
        outfit["rank"] = rank
        outfit["score"] = round(score, 4)
        results.append(outfit)

    return results


# ─── Service: Compatibility Scoring + FITB Suggestion ─────────────────────────

def score_outfit_compatibility(
    item_ids: List[str],
    bundle: Dict,
) -> Tuple[Optional[dict], List[dict]]:
    """
    Score the compatibility of a set of items.

    Returns:
        (compatibility_result, valid_items_info)
        compatibility_result is None when no valid item_ids are recognised.
    """
    valid_indices = [
        bundle["item2idx"][str(iid)]
        for iid in item_ids
        if str(iid) in bundle["item2idx"]
    ]
    if not valid_indices:
        return None, []

    padded = (valid_indices[:8] + [-1] * 8)[:8]
    x = torch.tensor([padded], dtype=torch.long)

    with torch.no_grad():
        raw = bundle["model"].score_compatibility(bundle["item_emb"], x).item()

    prob = _sigmoid(raw)
    label = "Compatible" if prob >= 0.5 else "Not Compatible"

    valid_items_info = [
        get_item_info(bundle["idx2item"][idx], bundle["item_meta"])
        for idx in valid_indices
    ]

    result = {
        "raw_score": round(raw, 4),
        "compatibility_prob": round(prob, 4),
        "label": label,
        "valid_items": valid_items_info,
    }
    return result, valid_items_info


def suggest_completing_items(
    item_ids: List[str],
    bundle: Dict,
    top_k: int = 5,
    chunk_size: int = 2048,
) -> List[dict]:
    """
    Fill-in-the-Blank (FITB): given k items already in an outfit, find the
    top_k items from the full catalog that best complete the outfit.

    Strategy: For every candidate item not already in the outfit, build an
    augmented outfit [current_items + candidate] and run score_compatibility
    in one batched forward pass (chunked to control peak memory usage).
    """
    valid_indices = [
        bundle["item2idx"][str(iid)]
        for iid in item_ids
        if str(iid) in bundle["item2idx"]
    ]
    if not valid_indices:
        return []

    n_items = bundle["item_emb"].shape[0]
    current_set = set(valid_indices)
    candidates = [i for i in range(n_items) if i not in current_set]

    if not candidates:
        return []

    # Keep up to 7 current items to leave one slot for each candidate
    base = valid_indices[:7]

    # Build batch tensor [N_candidates, 8]
    rows = []
    for c in candidates:
        row = (base + [c])[:8]
        if len(row) < 8:
            row = row + [-1] * (8 - len(row))
        rows.append(row)

    x = torch.tensor(rows, dtype=torch.long)  # [N_candidates, 8]

    # Chunked inference to avoid OOM on CPU
    scores_list = []
    with torch.no_grad():
        for start in range(0, len(candidates), chunk_size):
            chunk = x[start: start + chunk_size]
            scores_list.append(
                bundle["model"].score_compatibility(bundle["item_emb"], chunk)
            )
    all_scores = torch.cat(scores_list)  # [N_candidates]

    k = min(top_k, len(candidates))
    top_vals, top_idxs = torch.topk(all_scores, k=k)

    results = []
    for val, idx in zip(top_vals.tolist(), top_idxs.tolist()):
        item_idx = candidates[idx]
        item_id = bundle["idx2item"][item_idx]
        info = get_item_info(str(item_id), bundle["item_meta"])
        info["score"] = round(val, 4)
        info["compatibility_prob"] = round(_sigmoid(val), 4)
        results.append(info)

    return results


# ─── Service: Similar Outfits ──────────────────────────────────────────────────

def find_similar_outfits(
    outfit_id: str,
    bundle: Dict,
    top_k: int = 10,
) -> Optional[List[dict]]:
    """
    Return top_k outfits most similar to the given outfit based on cosine
    similarity of outfit embeddings. The query outfit itself is excluded.

    Returns None if outfit_id is not in the model.
    """
    if outfit_id not in bundle["outfit2idx"]:
        return None

    oidx = bundle["outfit2idx"][outfit_id]
    q = bundle["outfit_emb"][oidx: oidx + 1]               # [1, D]
    scores = torch.matmul(q, bundle["outfit_emb"].T).squeeze(0)  # [N_outfits]
    scores[oidx] = -1e9                                     # exclude self

    k = min(top_k, max(1, scores.numel() - 1))
    vals, idxs = torch.topk(scores, k=k)

    results = []
    for rank, (score, j) in enumerate(zip(vals.tolist(), idxs.tolist()), start=1):
        oid = bundle["idx2outfit"][j]
        outfit = enrich_outfit(oid, bundle)
        outfit["rank"] = rank
        outfit["similarity"] = round(score, 4)
        results.append(outfit)

    return results


# ─── Service: Item Catalog ─────────────────────────────────────────────────────

def list_items(
    bundle: Dict,
    page: int = 1,
    limit: int = 50,
    search: Optional[str] = None,
    category: Optional[int] = None,
) -> dict:
    """
    Return a paginated, optionally filtered slice of the item catalog.
    """
    df = bundle["item_df"].copy()

    if search:
        mask = df["title"].str.contains(search, case=False, na=False)
        df = df[mask]

    if category is not None:
        df = df[df["category"].astype(str) == str(category)]

    total = len(df)
    total_pages = max(1, math.ceil(total / limit))
    page = max(1, min(page, total_pages))

    start = (page - 1) * limit
    end = start + limit
    slice_df = df.iloc[start:end]

    items = [
        {
            "item_id": str(row["item_id"]),
            "title": row.get("title", ""),
            "category": row.get("category", None),
            "image_url": row.get("image_url", ""),
        }
        for _, row in slice_df.iterrows()
    ]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "items": items,
    }


# ─── Service: User Interaction History ────────────────────────────────────────

def get_user_history(
    user_id: str,
    bundle: Dict,
    page: int = 1,
    limit: int = 20,
) -> Optional[dict]:
    """
    Return paginated outfit interaction history for a given user,
    with each outfit enriched with its item details.

    Returns None if user_id has no interaction history.
    """
    df = bundle["history_df"]
    user_rows = df[df["user_id"] == user_id]

    if user_rows.empty:
        return None

    outfit_ids = user_rows["outfit_id"].tolist()
    total = len(outfit_ids)
    total_pages = max(1, math.ceil(total / limit))
    page = max(1, min(page, total_pages))

    start = (page - 1) * limit
    end = start + limit
    page_outfit_ids = outfit_ids[start:end]

    histories = [enrich_outfit(oid, bundle) for oid in page_outfit_ids]

    return {
        "user_id": user_id,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "histories": histories,
    }


def list_users(
    bundle: Dict,
    page: int = 1,
    limit: int = 50,
) -> dict:
    """
    Return a paginated list of all known user IDs with their outfit interaction count.
    """
    df = bundle["history_df"]
    counts = df.groupby("user_id")["outfit_id"].count().reset_index()
    counts.columns = ["user_id", "outfit_count"]
    counts = counts.sort_values("outfit_count", ascending=False).reset_index(drop=True)

    total = len(counts)
    total_pages = max(1, math.ceil(total / limit))
    page = max(1, min(page, total_pages))

    start = (page - 1) * limit
    end = start + limit
    users = counts.iloc[start:end].to_dict(orient="records")

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "users": users,
    }


# ─── Service: Outfit Catalog ───────────────────────────────────────────────────

def list_outfits(
    bundle: Dict,
    page: int = 1,
    limit: int = 20,
) -> dict:
    """
    Return a paginated list of all outfits with their items enriched.
    """
    outfit_ids = list(bundle["outfit_items"].keys())
    total = len(outfit_ids)
    total_pages = max(1, math.ceil(total / limit))
    page = max(1, min(page, total_pages))

    start = (page - 1) * limit
    end = start + limit
    page_outfits = [enrich_outfit(oid, bundle) for oid in outfit_ids[start:end]]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": total_pages,
        "outfits": page_outfits,
    }
