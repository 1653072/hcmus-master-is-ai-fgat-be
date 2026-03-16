# H-FGAT Backend v1

REST API for the **Hierarchical Fashion Graph Attention Network (H-FGAT)** recommender system.  
Deployed on **Render** · Consumed by a **Vercel** front-end.

---

## System Architecture

```
┌────────────────────────────────────────────────┐
│              Vercel Frontend (Next.js)          │
│   Pages: Recommend / Compatibility / Browse     │
└────────────────┬───────────────────────────────┘
                 │  HTTPS (CORS enabled)
                 ▼
┌────────────────────────────────────────────────┐
│           Render Web Service (Flask + gunicorn) │
│                                                 │
│  app.py ──► routes ──► services.py             │
│                             │                   │
│                        loader.py                │
│                             │                   │
│              ┌──────────────┼──────────────┐    │
│          model.pt    embeddings/      item_sub  │
│        (HFGATDetailed)  *.pt + *.json   .csv   │
└────────────────────────────────────────────────┘
```

### Inference pipeline (no re-training at serve time)

1. All heavy feature extraction (ResNet-50 image + BERT-Chinese text) ran **offline** during training.
2. At startup the server loads **pre-computed embeddings** (`user_emb`, `outfit_emb`, `item_emb`) from `.pt` files into RAM.
3. Every API call does only fast, CPU-bound operations: matrix multiplications, top-k, and a 2-layer MLP — typically **< 200 ms** per request.

---

## Project Structure

```
backend_v1/
├── app.py                        # Flask entry point + all route definitions
├── model.py                      # HFGATDetailed model class (inference only)
├── loader.py                     # Artifact loader (singleton, loaded once at startup)
├── services.py                   # Business logic (recommend, FITB, similar, catalog)
│
├── artifacts/                    # Committed model artifacts (all < 100 MB)
│   ├── model.pt                  # Full checkpoint (weights + config)
│   └── exported_embeddings/
│       ├── user_embeddings.pt    # [N_users,  128]
│       ├── outfit_embeddings.pt  # [N_outfits, 128]
│       ├── item_embeddings.pt    # [N_items,  128]
│       ├── user2idx.json
│       ├── outfit2idx.json
│       ├── item2idx.json
│       ├── outfit_items.json     # outfit_id → [item_id, ...]
│       └── train_uo_sub.csv      # user–outfit interaction history
│
├── data/
│   └── item_sub.csv              # Item catalog: item_id, category, image_url, title
│
├── requirements.txt
├── render.yaml                   # Render deployment config
├── .gitignore
└── README.md
```

---

## Local Development

### 1. Install dependencies

```bash
cd backend_v1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
python app.py
# Server starts at http://localhost:5000
```

---

## Render Deployment

1. Push `backend_v1/` as the root of a new **public GitHub repository**.
2. Go to [render.com](https://render.com) → **New Web Service** → connect the repo.
3. Render auto-detects `render.yaml` and configures the service.
4. Set **Build Command**: `pip install -r requirements.txt`  
   Set **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
5. After deploy, your base URL will be `https://<your-app>.onrender.com`.

> **Note on cold starts**: Render free tier spins down after 15 min of inactivity.  
> The first request after a cold start takes ~30–60 s while artifacts load into RAM.  
> Use a service like [cron-job.org](https://cron-job.org) to ping `/health` every 10 min to keep it warm.

---

## API Reference

**Base URL** (local): `http://localhost:5000`  
**Base URL** (Render): `https://<your-app>.onrender.com`

---

### GET /health

Health check — confirms the server is up and artifacts are loaded.

```bash
curl https://<your-app>.onrender.com/health
```

**Response `200`**
```json
{
  "status": "ok",
  "model": "H-FGAT",
  "artifacts_loaded": true
}
```

---

### POST /recommend

Personalized outfit recommendations for a user based on learned user–outfit embeddings.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `user_id` | string | ✅ | — | Must exist in training set |
| `top_k` | int | ❌ | `10` | Max `50` |
| `exclude_seen` | bool | ❌ | `false` | Skip outfits the user already interacted with |

```bash
curl -X POST https://<your-app>.onrender.com/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "844126",
    "top_k": 5,
    "exclude_seen": false
  }'
```

**Response `200`**
```json
{
  "user_id": "844126",
  "total": 5,
  "recommendations": [
    {
      "rank": 1,
      "outfit_id": "1277",
      "score": 0.9234,
      "items": [
        { "item_id": "2243", "title": "...", "category": 3, "image_url": "http://..." },
        { "item_id": "5098", "title": "...", "category": 10, "image_url": "http://..." }
      ]
    }
  ]
}
```

**Error responses**
- `400` — `user_id` missing
- `404` — `user_id` not in model
- `503` — artifacts not loaded

---

### POST /suggest-outfit-compatibility

Scores how well a set of items work together **and** suggests the top items that would best complete the outfit *(Fill-in-the-Blank / FITB)*.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `item_ids` | string[] | ✅ | — | 1–8 item IDs |
| `suggest_top_k` | int | ❌ | `5` | Max `20` |

```bash
curl -X POST https://<your-app>.onrender.com/suggest-outfit-compatibility \
  -H "Content-Type: application/json" \
  -d '{
    "item_ids": ["2243", "5098", "4258"],
    "suggest_top_k": 5
  }'
```

**Response `200`**
```json
{
  "compatibility": {
    "raw_score": 0.8512,
    "compatibility_prob": 0.7007,
    "label": "Compatible",
    "valid_items": [
      { "item_id": "2243", "title": "...", "category": 3, "image_url": "http://..." }
    ]
  },
  "suggested_items": [
    {
      "item_id": "7891",
      "title": "...",
      "category": 16,
      "image_url": "http://...",
      "score": 0.9102,
      "compatibility_prob": 0.7131
    }
  ]
}
```

**Compatibility labels**
- `"Compatible"` — `compatibility_prob >= 0.5`
- `"Not Compatible"` — `compatibility_prob < 0.5`

**Error responses**
- `400` — `item_ids` missing / empty / > 8 items / none recognised

---

### POST /similar-outfits

Find outfits most similar to a given outfit using cosine similarity of outfit embeddings.

| Field | Type | Required | Default | Notes |
|---|---|---|---|---|
| `outfit_id` | string | ✅ | — | Must exist in training set |
| `top_k` | int | ❌ | `10` | Max `50` |

```bash
curl -X POST https://<your-app>.onrender.com/similar-outfits \
  -H "Content-Type: application/json" \
  -d '{
    "outfit_id": "1277",
    "top_k": 5
  }'
```

**Response `200`**
```json
{
  "outfit_id": "1277",
  "outfit_items": [
    { "item_id": "2243", "title": "...", "category": 3, "image_url": "http://..." }
  ],
  "similar_outfits": [
    {
      "rank": 1,
      "outfit_id": "2345",
      "similarity": 0.9214,
      "items": [
        { "item_id": "6789", "title": "...", "category": 10, "image_url": "http://..." }
      ]
    }
  ]
}
```

---

### GET /list-items

Paginated item catalog with optional search and category filter.

| Param | Type | Default | Notes |
|---|---|---|---|
| `page` | int | `1` | |
| `limit` | int | `50` | Max `200` |
| `search` | string | — | Case-insensitive substring match on `title` |
| `category` | int | — | Filter by category ID |

```bash
# All items, page 1
curl "https://<your-app>.onrender.com/list-items?page=1&limit=20"

# Search + category filter
curl "https://<your-app>.onrender.com/list-items?search=裙&category=10&limit=10"
```

**Response `200`**
```json
{
  "page": 1,
  "limit": 20,
  "total": 12877,
  "total_pages": 644,
  "items": [
    { "item_id": "0", "title": "...", "category": 3, "image_url": "http://..." }
  ]
}
```

---

### GET /list-user-histories

Outfit interaction history for a specific user.

| Param | Type | Required | Default |
|---|---|---|---|
| `user_id` | string | ✅ | — |
| `page` | int | ❌ | `1` |
| `limit` | int | ❌ | `20` (max `100`) |

```bash
curl "https://<your-app>.onrender.com/list-user-histories?user_id=844126&page=1&limit=10"
```

**Response `200`**
```json
{
  "user_id": "844126",
  "page": 1,
  "limit": 10,
  "total": 2,
  "total_pages": 1,
  "histories": [
    {
      "outfit_id": "1277",
      "items": [
        { "item_id": "2243", "title": "...", "category": 3, "image_url": "http://..." }
      ]
    }
  ]
}
```

---

### GET /list-users

Paginated list of all users sorted by interaction count (useful for FE dropdowns).

| Param | Type | Default | Notes |
|---|---|---|---|
| `page` | int | `1` | |
| `limit` | int | `50` | Max `200` |

```bash
curl "https://<your-app>.onrender.com/list-users?page=1&limit=50"
```

**Response `200`**
```json
{
  "page": 1,
  "limit": 50,
  "total": 30000,
  "total_pages": 600,
  "users": [
    { "user_id": "844126", "outfit_count": 5 }
  ]
}
```

---

### GET /list-outfits

Paginated list of all outfits with their items enriched.

| Param | Type | Default | Notes |
|---|---|---|---|
| `page` | int | `1` | |
| `limit` | int | `20` | Max `100` |

```bash
curl "https://<your-app>.onrender.com/list-outfits?page=1&limit=10"
```

**Response `200`**
```json
{
  "page": 1,
  "limit": 10,
  "total": 5725,
  "total_pages": 573,
  "outfits": [
    {
      "outfit_id": "2",
      "items": [
        { "item_id": "2243", "title": "...", "category": 3, "image_url": "http://..." }
      ]
    }
  ]
}
```

---

## Postman Collection (Quick Setup)

1. Create a new **Collection** named `H-FGAT Backend v1`.
2. Set a **Collection Variable**: `baseUrl = https://<your-app>.onrender.com`
3. Add requests using `{{baseUrl}}/recommend` etc.

Recommended test order:
1. `GET {{baseUrl}}/health` — verify server is up
2. `GET {{baseUrl}}/list-users?limit=10` — grab a real `user_id`
3. `POST {{baseUrl}}/recommend` with that `user_id`
4. `GET {{baseUrl}}/list-items?limit=5` — grab real `item_id` values
5. `POST {{baseUrl}}/suggest-outfit-compatibility` with those `item_ids`
6. `GET {{baseUrl}}/list-outfits?limit=5` — grab a real `outfit_id`
7. `POST {{baseUrl}}/similar-outfits` with that `outfit_id`
8. `GET {{baseUrl}}/list-user-histories?user_id=<id>` — verify history

---

## Category ID Reference

| ID | Category |
|---|---|
| 3 | Jewelry / Accessories |
| 10 | Dresses |
| 16 | Pants |
| 43 | Leggings / Base layer |
| *(others)* | See `item_sub.csv` for full mapping |

---

## Notes for FE Integration (Vercel / Next.js)

- All responses are `application/json` with UTF-8 encoding.
- CORS is enabled for **all origins** — no extra headers needed from Vercel.
- `image_url` values point to the AliCDN (`gw.alicdn.com`) — display directly in `<img src="...">`.
- Item/outfit/user IDs are always returned as **strings** — even numeric ones like `"0"`, `"1277"`.
- For the compatibility endpoint, send `item_ids` as an array of strings: `["2243", "5098"]`.
