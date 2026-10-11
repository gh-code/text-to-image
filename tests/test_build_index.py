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
"""Tests for build_index.py logic.

Covers the pure helpers (collect_images, load_existing) and the full/incremental
build paths using a deterministic fake model (no torch / no model download).
"""

import json
import os

import faiss
import numpy as np
import pytest

import build_index


def test_collect_images_is_recursive_and_sorted(image_dir):
    result = build_index.collect_images(str(image_dir))
    names = sorted(os.path.basename(p) for p in result)
    assert names == ["a.png", "b.png", "c.png"]
    # sorted() is deterministic regardless of filesystem order
    assert result == sorted(result)


def test_collect_images_skips_non_images(image_dir):
    (image_dir / "notes.txt").write_text("ignore me")
    result = build_index.collect_images(str(image_dir))
    assert all(p.endswith(".png") for p in result)


def test_load_existing_returns_none_when_missing(tmp_path):
    assert build_index.load_existing(str(tmp_path / "nope.npy"), str(tmp_path / "nope.json")) is None


def test_load_existing_roundtrip(image_dir, fake_model):
    index_path = str(image_dir / "idx.bin")
    meta_path = str(image_dir / "meta.json")

    # Build a real faiss index + metadata the same way build_index does.
    paths = sorted(str(p) for p in build_index.collect_images(image_dir))
    emb = fake_model.encode(paths, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, index_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"model": "fake", "dim": emb.shape[1], "images": paths}, f)

    loaded = build_index.load_existing(index_path, meta_path)
    assert loaded is not None
    _, images, dim = loaded
    assert images == paths
    assert dim == emb.shape[1]


def test_full_build_writes_index_and_meta(image_dir, fake_model, monkeypatch, tmp_path):
    # Replace the real SentenceTransformer with our deterministic fake.
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *a, **k: fake_model)

    index_path = str(tmp_path / "img_index.npy")
    meta_path = str(tmp_path / "images_meta.json")

    build_index.main(["--images", str(image_dir),
                      "--index", index_path, "--meta", meta_path, "--cpu"])

    assert os.path.exists(index_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["images"] == sorted(str(p) for p in build_index.collect_images(image_dir))
    assert meta["dim"] == fake_model.dim


def test_incremental_no_new_images_is_idempotent(image_dir, fake_model, monkeypatch, tmp_path):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", lambda *a, **k: fake_model)
    index_path = str(tmp_path / "img_index.npy")
    meta_path = str(tmp_path / "images_meta.json")

    build_index.main(["--images", str(image_dir),
                      "--index", index_path, "--meta", meta_path, "--cpu"])

    # Re-run with no new images: should not re-encode or change the index.
    build_index.main(["--images", str(image_dir),
                      "--index", index_path, "--meta", meta_path, "--cpu"])

    reloaded = build_index.load_existing(index_path, meta_path)
    assert reloaded[0].ntotal == len(build_index.collect_images(image_dir))
