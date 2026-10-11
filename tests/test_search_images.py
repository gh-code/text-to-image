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
"""Tests for search_images.py client.

Two paths:
  * server mode, with requests.post mocked (no real server needed).
  * local mode, using a deterministic fake model + a real faiss index.
"""

import json
import os

import faiss
import numpy as np
import pytest

import search_images


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def test_search_via_server_parses_response(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        return FakeResponse({"query": "dog", "results": [
            {"rank": 1, "path": "/a.png", "score": 0.9},
            {"rank": 2, "path": "/b.png", "score": 0.5},
        ]})

    monkeypatch.setattr(search_images.requests, "post", fake_post)
    result = search_images.search_via_server("http://127.0.0.1:8000", "dog", 5)
    assert captured["url"] == "http://127.0.0.1:8000/search"
    assert len(result["results"]) == 2


def test_local_mode_runs_with_fake_model(image_dir, fake_model, monkeypatch, tmp_path):
    # Build a real index + metadata in the temp dir.
    monkeypatch.chdir(tmp_path)

    paths = sorted(str(p) for p in __import__("build_index").collect_images(image_dir))
    emb = fake_model.encode(paths, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(tmp_path / "img_index.npy"))

    meta = {"model": "fake", "dim": emb.shape[1], "images": paths}
    with open(tmp_path / "images_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    # Replace the real SentenceTransformer with our fake for local mode.
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *a, **k: fake_model)

    search_images.main(["--query", "dog", "--top", "5", "--local"])

    out = json.loads((tmp_path / "search_results.json").read_text())
    assert out["query"] == "dog"
    assert len(out["results"]) > 0
    assert all("path" in r and "score" in r for r in out["results"])
