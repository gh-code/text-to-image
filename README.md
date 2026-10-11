# Text-to-Image Search (EmbeddingGemma 2)

[English](README.md) | [繁體中文](README_zh.md)

Collect the images in a folder, build a vector index with Google's **EmbeddingGemma 2** (`google/embeddinggemma-2`), then search for the most similar images with a text query.

The model projects text and images into the same 768-dimensional vector space, so you can compare cosine similarity directly.

## Architecture

```
┌──────────────┐   HTTP POST /search    ┌─────────────────────────┐
│ search_images │ ────────────────────▶ │  server.py (persistent) │
│   client     │ ◀──────────────────── │  model loaded once      │
└──────────────┘   JSON results         │  faiss index in memory  │
                                         └─────────────────────────┘
```

Search connects over HTTP to the persistent server, where the model is loaded once (saving ~4s + re-encoding per query).

## Installation

This environment has no GPU, so it uses the CPU build of torch:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` pins the CPU build of torch/torchvision to avoid downloading large CUDA packages.

## Building the index

```bash
.venv/bin/python build_index.py \
    --images ./photos \
    --index img_index.npy \
    --meta images_meta.json
```

- `--images`: folder to scan (recursive, supports jpg/jpeg/png/webp/gif/bmp/heic). HEIC from phones is decoded with `pillow-heif`.
- `--index`: faiss index output file.
- `--meta`: metadata (image path list), used at search time.
- `--dim 256`: truncate the vector with Matryoshka to shrink the index (768/512/256/128).
- `--cpu`: force CPU (auto-detected by default).

The first run downloads the model (~1.4 GB); afterwards it is cached locally.

### Incremental update

`build_index.py` **updates incrementally by default**: when it detects an old index and metadata, it only encodes the "new" images and appends them to the faiss index. Old photos are not re-run.

```bash
# After adding photos, just run it (auto-encodes only new images)
python build_index.py --images ./photos

# When dimensions mismatch or you want a full rebuild, force it
python build_index.py --images ./photos --force-full
```

### Pruning deleted photos

faiss indices cannot delete rows in place, so after removing a photo from the folder you **need to rebuild the index once** to move it out of faiss and metadata.

```bash
# --prune: full rebuild, keeping only files still on disk (auto-prunes deleted photos)
python build_index.py --images ./photos --prune
```

A safety check is added at search time: even if metadata still holds a path to an image, the result will automatically skip it as long as the file is not found on disk. So deleting a photo after the server starts will never return a broken path.

### Auto-syncing the persistent server

After building the index, `build_index.py` **automatically POSTs `/reload` to notify the running server to reload the faiss index from disk**, so you don't need to restart the server manually.

```bash
# Add photos (server automatically receives new images)
python build_index.py --images ./photos

# Rebuild after deleting photos (server automatically removes them)
python build_index.py --images ./photos --prune
```

Reload swaps only the faiss index, not the model, and returns instantly. If the server isn't running (or you changed `--index`/`--meta`), build prints a hint and you can POST manually:

```bash
curl -X POST http://127.0.0.1:8000/reload
```

## Starting the persistent server (background)

```bash
.venv/bin/python server.py \
    --index img_index.npy --meta images_meta.json --port 8000
```

- The model is loaded once; every subsequent query goes over HTTP.
- Health check: `GET /health` → `{"status":"ok","loaded":true}`
- Stop: press `Ctrl+C` in the terminal.

## Searching images (via the persistent server)

```bash
.venv/bin/python search_images.py \
    --query "a white cat outdoors" \
    --top 5
```

- `--query`: query text.
- `--server http://host:port`: server address (default `http://127.0.0.1:8000`).
- `--top N`: number of results to return (default 5).

Results are both printed and written to `search_results.json`.
(If you don't want to use a server, add `--local` to run locally, but the model is reloaded each time.)

## File reference

| File | Purpose |
|------|---------|
| `build_index.py` | Scans the folder, produces vectors, writes the faiss index |
| `server.py` | Persistent server (model loaded once, serves HTTP /search and /embed) |
| `search_images.py` | Text query client (defaults to HTTP, can add --local) |
| `requirements.txt` | Dependencies (CPU torch/torchvision + faiss-cpu + flask + requests) |

## Notes

- The index and metadata are stored separately; just re-run `build_index.py` after adding images.
- Query text and images share the same vector space, so both English and Chinese work.