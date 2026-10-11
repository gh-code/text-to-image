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
"""Persistent vector search server.

The model is loaded once; every subsequent query goes over HTTP without reloading
(saves ~4s + re-encoding per request).

Start:
    python server.py --index img_index.npy --meta images_meta.json --port 8000

API:
    GET /health                       -> {"status": "ok"}
    POST /search  {"query": "...", "top": 5}
                                          -> {"results": [{"path":..., "score":...}, ...]}
    POST /embed   {"text": "..."}       -> {"embedding": [768 numbers]}
"""

import argparse
import json
import os
import threading

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)


INDEX_PATH = None
META_PATH = None
MODEL = None
FAISS = None
IMAGES = None
DIM = None
_LOCK = threading.Lock()


def _load(model_name, index_path, meta_path):
    global MODEL, FAISS, IMAGES, DIM
    from sentence_transformers import SentenceTransformer

    print(f"[server] Loading model {model_name} ...", flush=True)
    MODEL = SentenceTransformer(model_name)

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    DIM = meta["dim"]
    IMAGES = meta["images"]

    FAISS = __import__("faiss").read_index(index_path)
    print(f"[server] Index loaded, {FAISS.ntotal} images total", flush=True)


def _load_index():
    # Reload only the faiss index + metadata, not the model (fast, avoids build_index reload timeout).
    global FAISS, IMAGES, DIM
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    DIM = meta["dim"]
    IMAGES = meta["images"]
    FAISS = __import__("faiss").read_index(INDEX_PATH)
    print(f"[server] Index reloaded, {FAISS.ntotal} images total", flush=True)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "loaded": MODEL is not None})


@app.route("/reload", methods=["POST"])
def reload_index():
    # Called automatically by build_index.py after building, reloads faiss from disk (including --prune).
    # Swaps only the index, not the model, returns instantly (avoids build_index reload timeout).
    with _LOCK:
        try:
            _load_index()
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify({"status": "ok", "loaded": FAISS.ntotal})


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    query = data["query"]
    top = int(data.get("top", 5))

    with _LOCK:
        q_emb = MODEL.encode([query], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")[:, :DIM]
        scores, idxs = FAISS.search(q_emb, min(top, len(IMAGES)))

    # Filter out files deleted after the server started (path no longer exists)
    results = []
    for r, (s, i) in enumerate(zip(scores[0], idxs[0]), 1):
        path = IMAGES[i]
        if not os.path.exists(path):
            continue
        results.append({"rank": len(results) + 1, "path": path, "score": float(s)})

    return jsonify({"query": query, "results": results})


@app.route("/embed", methods=["POST"])
def embed():
    data = request.get_json(force=True)
    text = data["text"]
    with _LOCK:
        emb = MODEL.encode([text], normalize_embeddings=True)
    return jsonify({"embedding": np.asarray(emb, dtype="float32").tolist()[0]})


def main():
    global INDEX_PATH, META_PATH
    ap = argparse.ArgumentParser(description="Persistent vector search server")
    ap.add_argument("--index", default="img_index.npy")
    ap.add_argument("--meta", default="images_meta.json")
    ap.add_argument("--model", default="google/embeddinggemma-2")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    global MODEL_NAME
    INDEX_PATH, META_PATH = args.index, args.meta
    MODEL_NAME = args.model
    _load(args.model, args.index, args.meta)

    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
