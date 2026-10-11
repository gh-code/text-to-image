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
"""Tests for server.py Flask routes.

Uses the real faiss index + a deterministic fake model (no torch download).
Exercises /health, /search (including the deleted-path safety filter), and
/reload.
"""

import json
import os

import faiss
import numpy as np
import pytest

import server


@pytest.fixture()
def client(image_dir, fake_model):
    """A Flask test client with a loaded index built from image_dir."""
    server.MODEL = fake_model

    paths = sorted(str(p) for p in build_index_collect(image_dir))
    emb = fake_model.encode(paths, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    server.FAISS = index
    server.IMAGES = paths
    server.DIM = emb.shape[1]

    app = server.app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def build_index_collect(image_dir):
    import build_index
    return build_index.collect_images(image_dir)


def test_health_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["loaded"] is True


def test_search_returns_ranked_results(client):
    resp = client.post("/search", json={"query": "a white cat outdoors", "top": 3})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["query"] == "a white cat outdoors"
    assert len(data["results"]) <= 3
    for rank, res in enumerate(data["results"], start=1):
        assert res["rank"] == rank
        assert "path" in res and "score" in res


def test_search_filters_deleted_paths(client, image_dir):
    # Simulate a photo deleted after the server started: remove it from disk so
    # os.path.exists() returns False. FAISS and IMAGES stay consistent (same
    # length), which is what the production /search filter relies on.
    deleted = next(p for p in server.IMAGES if os.path.basename(p) == "b.png")
    os.remove(deleted)

    resp = client.post("/search", json={"query": "x", "top": 10})
    data = resp.get_json()
    assert all("b.png" not in r["path"] for r in data["results"])


def test_reload_swaps_index(client, image_dir):
    # Add a brand-new fake entry and reload from disk to confirm /reload works.
    new_index = faiss.IndexFlatIP(server.DIM)
    fake_vec = np.zeros((1, server.DIM), dtype="float32")
    new_index.add(fake_vec)
    faiss.write_index(new_index, str(image_dir / "reload_idx.bin"))

    # Point META_PATH/INDEX_PATH at temp files so _load_index can read them.
    meta_path = str(image_dir / "reload_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"dim": server.DIM, "images": ["/nonexistent/fake.png"]}, f)
    server.INDEX_PATH = str(image_dir / "reload_idx.bin")
    server.META_PATH = meta_path

    resp = client.post("/reload")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert data["loaded"] == 1


def test_embed_returns_vector(client):
    resp = client.post("/embed", json={"text": "hello"})
    assert resp.status_code == 200
    emb = resp.get_json()["embedding"]
    assert isinstance(emb, list)
    assert len(emb) == server.DIM
