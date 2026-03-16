"""
Artifact loader for H-FGAT backend.

Loads the trained model checkpoint and all pre-computed embeddings once at
startup and keeps them in a process-level singleton so every Flask request
reuses the same in-memory bundle without reloading from disk.

Expected directory layout (relative to this file):
    artifacts/
        model.pt
        exported_embeddings/
            user_embeddings.pt
            outfit_embeddings.pt
            item_embeddings.pt
            user2idx.json
            outfit2idx.json
            item2idx.json
            outfit_items.json
            train_uo_sub.csv
    data/
        item_sub.csv          ← item catalog with image_url
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from model import HFGATDetailed

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
EXPORT_DIR = ARTIFACT_DIR / "exported_embeddings"
DATA_DIR = BASE_DIR / "data"

# Process-level singleton — populated on first call to load_artifacts()
_bundle: Optional[Dict] = None


def load_artifacts() -> Dict:
    """
    Load and return the shared artifact bundle.
    Thread-safe for read-only access after the first call.
    """
    global _bundle
    if _bundle is not None:
        return _bundle

    logger.info("Loading H-FGAT artifacts from disk …")

    # ── Model ──────────────────────────────────────────────────────────────
    model_path = ARTIFACT_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"model.pt not found at {model_path}. "
            "Copy the trained checkpoint into artifacts/ before starting the server."
        )

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = HFGATDetailed(
        image_dim=cfg["image_dim"],
        text_dim=cfg["text_dim"],
        cat_dim=cfg["cat_dim"],
        num_users=cfg["num_users"],
        num_outfits=cfg["num_outfits"],
        num_items=cfg["num_items"],
        embed_dim=cfg["embed_dim"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded (embed_dim=%d).", cfg["embed_dim"])

    # ── Pre-computed embeddings ────────────────────────────────────────────
    user_emb = torch.load(EXPORT_DIR / "user_embeddings.pt", map_location="cpu", weights_only=False)
    outfit_emb = torch.load(EXPORT_DIR / "outfit_embeddings.pt", map_location="cpu", weights_only=False)
    item_emb = torch.load(EXPORT_DIR / "item_embeddings.pt", map_location="cpu", weights_only=False)
    logger.info(
        "Embeddings loaded: users=%d, outfits=%d, items=%d.",
        user_emb.shape[0], outfit_emb.shape[0], item_emb.shape[0],
    )

    # ── Index mappings ─────────────────────────────────────────────────────
    user2idx: Dict[str, int] = json.loads((EXPORT_DIR / "user2idx.json").read_text("utf-8"))
    outfit2idx: Dict[str, int] = json.loads((EXPORT_DIR / "outfit2idx.json").read_text("utf-8"))
    item2idx: Dict[str, int] = json.loads((EXPORT_DIR / "item2idx.json").read_text("utf-8"))

    idx2user: Dict[int, str] = {v: k for k, v in user2idx.items()}
    idx2outfit: Dict[int, str] = {v: k for k, v in outfit2idx.items()}
    idx2item: Dict[int, str] = {v: k for k, v in item2idx.items()}

    # ── Outfit → item list mapping ─────────────────────────────────────────
    outfit_items: Dict[str, list] = json.loads(
        (EXPORT_DIR / "outfit_items.json").read_text("utf-8")
    )

    # ── Item metadata with image URLs (from item_sub.csv) ──────────────────
    item_sub_path = DATA_DIR / "item_sub.csv"
    if item_sub_path.exists():
        item_df = pd.read_csv(item_sub_path, dtype={"item_id": str})
        # Build lookup dict: item_id_str → {category, image_url, title}
        item_meta: Dict[str, dict] = item_df.set_index("item_id").to_dict(orient="index")
        logger.info("Item catalog loaded: %d items.", len(item_df))
    else:
        item_df = pd.DataFrame(columns=["item_id", "category", "image_url", "title"])
        item_meta = {}
        logger.warning("item_sub.csv not found — item metadata will be empty.")

    # ── User–outfit interaction history ───────────────────────────────────
    history_path = EXPORT_DIR / "train_uo_sub.csv"
    if history_path.exists():
        history_df = pd.read_csv(history_path, dtype={"user_id": str, "outfit_id": str})
        logger.info("Interaction history loaded: %d rows.", len(history_df))
    else:
        history_df = pd.DataFrame(columns=["user_id", "outfit_id"])
        logger.warning("train_uo_sub.csv not found — history will be empty.")

    _bundle = {
        "model": model,
        "user_emb": F.normalize(user_emb.float(), p=2, dim=-1),
        "outfit_emb": F.normalize(outfit_emb.float(), p=2, dim=-1),
        "item_emb": F.normalize(item_emb.float(), p=2, dim=-1),
        "user2idx": user2idx,
        "outfit2idx": outfit2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2outfit": idx2outfit,
        "idx2item": idx2item,
        "outfit_items": outfit_items,
        "item_meta": item_meta,
        "item_df": item_df,
        "history_df": history_df,
    }

    logger.info("All artifacts ready.")
    return _bundle
