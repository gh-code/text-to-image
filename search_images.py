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
"""Text-to-image search.

Connects to the persistent server (server.py) by default, where the model is loaded once,
so there's no need to reload it per query.
Usage:
    python search_images.py --query "a white cat outdoors"        # connect to server :8000
    python search_images.py --query "dog" --server localhost:9000
    python search_images.py --query "dog" --local                 # run locally (loads the model)

Query text and images share the same vector space, so we compare cosine similarity directly.
"""

import argparse
import json
import sys

import requests


def search_via_server(url, query, top):
    r = requests.post(f"{url}/search", json={"query": query, "top": top}, timeout=120)
    r.raise_for_status()
    return r.json()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Text-to-image search (EmbeddingGemma 2)")
    ap.add_argument("--query", required=True, help="query text")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="persistent server address")
    ap.add_argument("--top", type=int, default=5, help="number of results to return")
    ap.add_argument("--local", action="store_true", help="skip the server and run locally")
    args = ap.parse_args(argv)

    if not args.local:
        try:
            resp = search_via_server(args.server, args.query, args.top)
        except requests.exceptions.ConnectionError:
            # Server not running: fall back to local execution (loads the model, slower)
            print(f"[WARN] Could not connect to server {args.server}, falling back to local execution")
            args.local = True
        else:
            results = resp["results"]
    if args.local:
        # Run locally (loads the model, slower)
        import os
        import numpy as np
        from sentence_transformers import SentenceTransformer

        with open("images_meta.json", encoding="utf-8") as f:
            meta = json.load(f)
        model = SentenceTransformer(meta["model"])
        q_emb = np.asarray(model.encode([args.query], normalize_embeddings=True), dtype="float32")[:, :meta["dim"]]
        import faiss
        index = faiss.read_index("img_index.npy")
        scores, idxs = index.search(q_emb, min(args.top, len(meta["images"])))
        results = []
        for r, (s, i) in enumerate(zip(scores[0], idxs[0]), 1):
            path = meta["images"][i]
            if not os.path.exists(path):
                continue
            results.append({"rank": len(results) + 1, "path": path, "score": float(s)})

    print(f"Query: {args.query}")
    print("=" * 60)
    for res in results:
        print(f"{res['rank']}. [cosine={res['score']:.4f}] {res['path']}")
    print("=" * 60)

    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump({"query": args.query, "results": results}, f, ensure_ascii=False, indent=2)
    print("[DONE] Results written to search_results.json")


if __name__ == "__main__":
    main()
