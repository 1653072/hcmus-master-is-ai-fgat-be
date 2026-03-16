"""
H-FGAT Backend API — Flask entry point.

Start locally:
    python app.py

Production (Render / gunicorn):
    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
"""

import logging
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from loader import load_artifacts
from services import (
    find_similar_outfits,
    get_item_info,
    get_user_history,
    list_items,
    list_outfits,
    list_users,
    recommend_outfits,
    score_outfit_compatibility,
    suggest_completing_items,
)

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow all origins (Vercel FE)

# ─── Warm-up: load artifacts once at import time (gunicorn workers each load) ──
try:
    bundle = load_artifacts()
    logger.info("Artifact bundle ready.")
except Exception as exc:
    logger.critical("Failed to load artifacts: %s", exc)
    bundle = None


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _bundle_required():
    """Return an error response if artifacts failed to load."""
    if bundle is None:
        return jsonify({"error": "Model artifacts not loaded. Check server logs."}), 503
    return None


def _bad_request(msg: str):
    return jsonify({"error": msg}), 400


def _not_found(msg: str):
    return jsonify({"error": msg}), 404


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """
    GET /health
    Returns server + model status.
    """
    return jsonify({
        "status": "ok",
        "model": "H-FGAT",
        "artifacts_loaded": bundle is not None,
    })


# ─── POST /recommend ───────────────────────────────────────────────────────────

@app.post("/recommend")
def recommend():
    """
    POST /recommend

    Request body (JSON):
        {
            "user_id":       "844126",   // required
            "top_k":         10,         // optional, default 10, max 50
            "exclude_seen":  false       // optional, default false
        }

    Response:
        {
            "user_id": "844126",
            "total": 10,
            "recommendations": [
                {
                    "rank": 1,
                    "outfit_id": "1277",
                    "score": 0.9234,
                    "items": [
                        { "item_id": "2243", "title": "...", "category": 3, "image_url": "..." },
                        ...
                    ]
                },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    user_id = str(body.get("user_id", "")).strip()
    if not user_id:
        return _bad_request("user_id is required.")

    top_k = min(int(body.get("top_k", 10)), 50)
    exclude_seen = bool(body.get("exclude_seen", False))

    results = recommend_outfits(user_id, bundle, top_k=top_k, exclude_seen=exclude_seen)
    if results is None:
        return _not_found(f"user_id '{user_id}' not found in the model.")

    return jsonify({
        "user_id": user_id,
        "total": len(results),
        "recommendations": results,
    })


# ─── POST /suggest-outfit-compatibility ────────────────────────────────────────

@app.post("/suggest-outfit-compatibility")
def suggest_outfit_compatibility():
    """
    POST /suggest-outfit-compatibility

    Scores the compatibility of a set of items AND suggests the best items
    to fill in the remaining outfit slots (FITB — Fill in the Blank).

    Request body (JSON):
        {
            "item_ids":       ["1001", "1002", "1003"],  // required, 1–8 items
            "suggest_top_k":  5                          // optional, default 5, max 20
        }

    Response:
        {
            "compatibility": {
                "raw_score":           0.85,
                "compatibility_prob":  0.70,
                "label":               "Compatible",
                "valid_items": [
                    { "item_id": "1001", "title": "...", "category": 3, "image_url": "..." },
                    ...
                ]
            },
            "suggested_items": [
                {
                    "item_id": "5678", "title": "...", "category": 10,
                    "image_url": "...", "score": 0.92, "compatibility_prob": 0.71
                },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    item_ids = body.get("item_ids", [])

    if not isinstance(item_ids, list) or len(item_ids) == 0:
        return _bad_request("item_ids must be a non-empty list.")
    if len(item_ids) > 8:
        return _bad_request("item_ids supports a maximum of 8 items per outfit.")

    item_ids = [str(iid).strip() for iid in item_ids]
    suggest_top_k = min(int(body.get("suggest_top_k", 5)), 20)

    compat_result, _ = score_outfit_compatibility(item_ids, bundle)
    if compat_result is None:
        return _bad_request("None of the provided item_ids are recognised in the model.")

    suggested = suggest_completing_items(item_ids, bundle, top_k=suggest_top_k)

    return jsonify({
        "compatibility": compat_result,
        "suggested_items": suggested,
    })


# ─── POST /similar-outfits ─────────────────────────────────────────────────────

@app.post("/similar-outfits")
def similar_outfits():
    """
    POST /similar-outfits

    Request body (JSON):
        {
            "outfit_id": "1277",  // required
            "top_k":     10       // optional, default 10, max 50
        }

    Response:
        {
            "outfit_id": "1277",
            "outfit_items": [ { "item_id": "...", ... }, ... ],
            "similar_outfits": [
                {
                    "rank": 1,
                    "outfit_id": "2345",
                    "similarity": 0.92,
                    "items": [ ... ]
                },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    body = request.get_json(silent=True) or {}
    outfit_id = str(body.get("outfit_id", "")).strip()
    if not outfit_id:
        return _bad_request("outfit_id is required.")

    top_k = min(int(body.get("top_k", 10)), 50)

    results = find_similar_outfits(outfit_id, bundle, top_k=top_k)
    if results is None:
        return _not_found(f"outfit_id '{outfit_id}' not found in the model.")

    item_ids = bundle["outfit_items"].get(outfit_id, [])
    query_items = [get_item_info(iid, bundle["item_meta"]) for iid in item_ids]

    return jsonify({
        "outfit_id": outfit_id,
        "outfit_items": query_items,
        "similar_outfits": results,
    })


# ─── GET /list-items ───────────────────────────────────────────────────────────

@app.get("/list-items")
def list_items_route():
    """
    GET /list-items

    Query parameters:
        page     (int)    default 1
        limit    (int)    default 50, max 200
        search   (str)    keyword filter on item title (case-insensitive)
        category (int)    filter by category ID

    Response:
        {
            "page": 1,
            "limit": 50,
            "total": 12877,
            "total_pages": 258,
            "items": [
                { "item_id": "0", "title": "...", "category": 3, "image_url": "..." },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    page = max(1, int(request.args.get("page", 1)))
    limit = min(int(request.args.get("limit", 50)), 200)
    search = request.args.get("search", None) or None
    category_raw = request.args.get("category", None) or None
    category = int(category_raw) if category_raw is not None else None

    result = list_items(bundle, page=page, limit=limit, search=search, category=category)
    return jsonify(result)


# ─── GET /list-user-histories ──────────────────────────────────────────────────

@app.get("/list-user-histories")
def list_user_histories():
    """
    GET /list-user-histories

    Query parameters:
        user_id  (str)    required — the user whose outfit history to retrieve
        page     (int)    default 1
        limit    (int)    default 20, max 100

    Response:
        {
            "user_id": "844126",
            "page": 1,
            "limit": 20,
            "total": 2,
            "total_pages": 1,
            "histories": [
                {
                    "outfit_id": "1277",
                    "items": [ { "item_id": "...", ... }, ... ]
                },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    user_id = str(request.args.get("user_id", "")).strip()
    if not user_id:
        return _bad_request("user_id query parameter is required.")

    page = max(1, int(request.args.get("page", 1)))
    limit = min(int(request.args.get("limit", 20)), 100)

    result = get_user_history(user_id, bundle, page=page, limit=limit)
    if result is None:
        return _not_found(f"No interaction history found for user_id '{user_id}'.")

    return jsonify(result)


# ─── GET /list-users ───────────────────────────────────────────────────────────

@app.get("/list-users")
def list_users_route():
    """
    GET /list-users

    Returns a paginated list of all users (sorted by outfit interaction count).

    Query parameters:
        page   (int)  default 1
        limit  (int)  default 50, max 200

    Response:
        {
            "page": 1,
            "limit": 50,
            "total": 30000,
            "total_pages": 600,
            "users": [
                { "user_id": "844126", "outfit_count": 5 },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    page = max(1, int(request.args.get("page", 1)))
    limit = min(int(request.args.get("limit", 50)), 200)

    result = list_users(bundle, page=page, limit=limit)
    return jsonify(result)


# ─── GET /list-outfits ─────────────────────────────────────────────────────────

@app.get("/list-outfits")
def list_outfits_route():
    """
    GET /list-outfits

    Returns a paginated list of all outfits with their items enriched.

    Query parameters:
        page   (int)  default 1
        limit  (int)  default 20, max 100

    Response:
        {
            "page": 1,
            "limit": 20,
            "total": 5725,
            "total_pages": 287,
            "outfits": [
                {
                    "outfit_id": "2",
                    "items": [ { "item_id": "2243", "title": "...", "category": 3, "image_url": "..." }, ... ]
                },
                ...
            ]
        }
    """
    err = _bundle_required()
    if err:
        return err

    page = max(1, int(request.args.get("page", 1)))
    limit = min(int(request.args.get("limit", 20)), 100)

    result = list_outfits(bundle, page=page, limit=limit)
    return jsonify(result)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
