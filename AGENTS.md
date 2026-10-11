# AGENTS.md

Text-to-image search: use Google **EmbeddingGemma 2** to project text and images into the same vector space, then search by cosine similarity. Python, **CPU-only**.

## Environment & commands

This project uses `.venv`; run all Python commands with `.venv/bin/python` (or `source .venv/bin/activate` first).

```bash
# Install dependencies (requirements.txt already pins CPU torch, no CUDA)
.venv/bin/python -m pip install -r requirements.txt

# Build / update index (incremental by default: only encode new images)
.venv/bin/python build_index.py --images ./photos

# Force a full rebuild, or prune deleted photos
.venv/bin/python build_index.py --images ./photos --force-full
.venv/bin/python build_index.py --images ./photos --prune

# Start the persistent server (model loaded once)
.venv/bin/python server.py --index img_index.npy --meta images_meta.json --port 8000

# Search (defaults to HTTP; add --local to run locally, which reloads the model)
.venv/bin/python search_images.py --query "a white cat outdoors" --top 5
```

## Architecture (three roles)

- `build_index.py` — scans a folder → EmbeddingGemma 2 encoding → faiss `IndexFlatIP` + JSON metadata. Auto-POSTs `/reload` to the server when done.
- `server.py` — Flask persistent service, model loaded once. Endpoints: `GET /health`, `POST /search` (`{"query","top"}`), `POST /embed` (`{"text"}`), `POST /reload` (swaps only the faiss index, doesn't reload the model).
- `search_images.py` — client. Defaults to connecting to the HTTP server; falls back to `--local` execution when unreachable.

Data flow: text query → model encodes to 768 dims (can truncate with `--dim` to 512/256/128) → L2 normalize → faiss inner-product match. Index file `img_index.npy`, metadata `images_meta.json` (with `model`/`dim`/`images[]`).

## Key conventions & pitfalls

- **CPU-only**: torch/torchvision are pinned with `+cpu` in `requirements.txt`. Don't introduce a CUDA build when adding dependencies; the `--cpu` flag forces CPU.
- **Incremental update is the default**: when an old index + metadata are detected, only new images are encoded. A dimension mismatch (`old_dim != args.dim`) auto-falls back to a full rebuild.
- **faiss can't delete in place**: after removing a photo from `./photos`, you must use `--prune` (full rebuild) to move it out of faiss and metadata. Search has a safety check that skips paths not on disk (even if deleted after the server started).
- **HEIC/HEIF**: optional decoding needs `pillow-heif`; when not installed, skip and print a WARN without aborting.
- **Deferred imports**: numpy / sentence_transformers are imported only when encoding actually happens, so the "no new images" branch saves ~8s of import time. Preserve this convention when changing it.
- **reload swaps only the index**: `/reload` (and `_load_index`) don't touch `MODEL`, to avoid a timeout on build's reload POST. Model reloading uses `_load` (at server startup).
- **Comments/docs in English**: existing code and README are now in English; keep new content consistent.

## Files

| File | Purpose |
|------|---------|
| `build_index.py` | Scan, encode, write faiss index + metadata (incremental) |
| `server.py` | Persistent server, model loaded once |
| `search_images.py` | Text query client (HTTP / --local) |
| `requirements.txt` | CPU torch + faiss-cpu + flask + requests |
| `README.md` | Full usage docs (don't duplicate here; sync with README when changing details) |
