#!/usr/bin/env python3
# Author: Gary Huang
# Copyright 2026 Gary Huang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build a vector index of images from a folder (text-to-image search) using EmbeddingGemma 2.

Supports incremental updates: if an old index and metadata already exist, only the
"new" images are encoded and appended to the faiss index.

Usage:
    # First build (full)
    python build_index.py --images ./photos

    # Add photos afterwards, incremental update (auto-detected)
    python build_index.py --images ./photos

    # Force a full rebuild (ignore the old index)
    python build_index.py --images ./photos --force-full

    # Prune deleted photos (full rebuild, keep only files still on disk)
    python build_index.py --images ./photos --prune

Scans every image in the folder (jpg/jpeg/png/webp/gif/bmp).
Each image produces a 768-dim vector, stored as a faiss index + json metadata.

HEIC / HEIF (the default phone camera format) is optional: it requires the extra
`pillow-heif` package. When not installed, HEIC files are skipped with a warning.
"""

import argparse
import datetime
import json
import os
import sys

# numpy and sentence_transformers (which pulls in torch) import slowly, so we defer
# them until encoding actually happens. This lets the "no new images" branch skip
# importing entirely and save ~8s of import time.
# import numpy as np
# from sentence_transformers import SentenceTransformer

# HEIC / HEIF (the default phone camera format) is optional: needs pillow-heif.
# When not installed we don't abort; collect_images warns and skips HEIC files.
_HEIF_READY = False
try:
    import pillow_heif

    if not getattr(pillow_heif, "_registered", False):
        pillow_heif.register_heif_opener()
        pillow_heif._registered = True
    _HEIF_READY = True
except Exception:
    pass


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heics"}


def log(msg):
    """Print a timestamped message so you can see how long each step takes."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def collect_images(root):
    """Recursively collect all image paths in a folder, returning a sorted list."""
    paths = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(dirpath, name))
    return sorted(paths)


def load_existing(index_path, meta_path):
    """Load the old faiss index and metadata, returning (index, images, dim); None if missing."""
    if not (os.path.exists(index_path) and os.path.exists(meta_path)):
        return None
    import faiss

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    index = faiss.read_index(index_path)
    return index, meta["images"], meta["dim"]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a vector index of images with EmbeddingGemma 2")
    ap.add_argument("--images", required=True, help="image folder to scan")
    ap.add_argument("--index", default="img_index.npy", help="faiss index output filename")
    ap.add_argument("--meta", default="images_meta.json", help="metadata output filename")
    ap.add_argument("--model", default="google/embeddinggemma-2", help="embedding model ID")
    ap.add_argument("--dim", type=int, default=768, help="vector dimension (default 768)")
    ap.add_argument("--batch", type=int, default=16, help="number of images per batch")
    ap.add_argument("--cpu", action="store_true", help="force CPU usage (saves VRAM)")
    ap.add_argument("--force-full", action="store_true", help="force a full rebuild, ignoring the old index")
    ap.add_argument("--prune", action="store_true", help="full rebuild and prune deleted photos (keep only files on disk)")
    ap.add_argument("--server", default="http://127.0.0.1:8000/reload", help="POST to this URL after building, to notify the running server to reload")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.images):
        log(f"[ERROR] Folder not found: {args.images}")
        sys.exit(1)

    images = collect_images(args.images)

    # HEIC / HEIF is optional: warn if detected but pillow-heif is not installed.
    heic_files = [p for p in images if os.path.splitext(p)[1].lower() in (".heic", ".heics")]
    if heic_files and not _HEIF_READY:
        log(f"[WARN] Detected {len(heic_files)} HEIC/HEIF image(s), but pillow-heif is not installed. Skipping.")
        log("[WARN] To support them, run: pip install pillow-heif")

    if not images:
        log(f"[ERROR] No images in folder: {args.images}")
        sys.exit(1)

    # --prune: full rebuild, keep only files still on disk (prune deleted photos)
    if args.prune:
        missing = [p for p in images if not os.path.exists(p)]
        if missing:
            log(f"[INFO] Pruning {len(missing)} deleted image(s)")
            for p in missing:
                log(f"        removing {p}")
        images = [p for p in images if os.path.exists(p)]

    existing = None if (args.force_full or args.prune) else load_existing(args.index, args.meta)
    incremental = existing is not None

    if incremental:
        old_index, old_images, old_dim = existing
        log(f"[INFO] Detected old index with {len(old_images)} image(s)")
        if old_dim != args.dim:
            log(f"[WARN] Dimension mismatch (old {old_dim} vs new {args.dim}), forcing full rebuild")
            incremental = False
        else:
            new_images = [p for p in images if p not in set(old_images)]
            log(f"[INFO] Added {len(new_images)} new, skipping {len(old_images)} (already exist, not re-encoded)")
            if not new_images:
                # No new images: no need to load the model, just re-save index/metadata (unchanged) and notify server
                log("[INFO] No new images, skipping encoding")
                import faiss
                faiss.write_index(existing[0], args.index)
                meta = {
                    "model": args.model,
                    "dim": existing[2],
                    "images": old_images,
                }
                with open(args.meta, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                log(f"[DONE] Index unchanged: {args.index} ({existing[0].ntotal} images)")
                log(f"[DONE] Metadata: {args.meta}")
                notify_server(args, existing[0].ntotal)
                return
    else:
        new_images = images
        log(f"[INFO] Found {len(images)} image(s) (full mode)")

    device = "cpu" if args.cpu else None
    log(f"[INFO] Loading model {args.model} (device={device or 'auto'}) ...")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=device)

    import faiss

    if incremental and new_images:
        # Encode only the new images, append to the existing index
        emb_dim = args.dim
        embeddings = model.encode(
            new_images,
            batch_size=args.batch,
            normalize_embeddings=True,   # L2 normalize so cosine == dot product
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")[:, :emb_dim]
        index = existing[0]
        index.add(embeddings)
    else:
        log("[INFO] Generating vectors ...")
        embeddings = model.encode(
            images,
            batch_size=args.batch,
            normalize_embeddings=True,   # L2 normalize so cosine == dot product
            show_progress_bar=True,
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        emb_dim = min(args.dim, embeddings.shape[1])
        if emb_dim < embeddings.shape[1]:
            log(f"[INFO] Truncating vectors to {emb_dim} dims (Matryoshka)")
        embeddings = embeddings[:, :emb_dim]

        index = faiss.IndexFlatIP(emb_dim)  # inner product == cosine (already normalized)
        index.add(embeddings)

    faiss.write_index(index, args.index)

    meta = {
        "model": args.model,
        "dim": emb_dim if not incremental else existing[2],
        # Incremental mode: metadata must include old + new images, matching the faiss vector count.
        "images": (old_images + new_images) if incremental else images,
    }
    with open(args.meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log(f"[DONE] Index saved: {args.index} ({index.ntotal} images)")
    log(f"[DONE] Metadata: {args.meta}")

    notify_server(args, index.ntotal)


def notify_server(args, n):
    """POST to the server's /reload endpoint so it reloads the index from disk (including --prune effects)."""
    try:
        import requests

        r = requests.post(args.server, timeout=5)
        if r.ok:
            loaded = r.json().get("loaded", "?")
            log(f"[INFO] Notified server to reload index ({loaded} images)")
        else:
            log(f"[WARN] Server reload returned error: {r.status_code}")
    except Exception as e:
        log(f"[INFO] Did not notify server ({e}). If the server is running, you can POST {args.server} manually")


if __name__ == "__main__":
    main()
