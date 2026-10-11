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
"""Shared pytest fixtures for the image_search test suite.

These tests are designed to run in CI without downloading the real
EmbeddingGemma 2 model (~1.4 GB). Instead of sentence_transformers/torch,
we inject a deterministic fake model that mimics the tiny slice of the API
that build_index.py / server.py actually use:

    model.encode(inputs, batch_size=..., normalize_embeddings=True) -> np.ndarray

faiss-cpu is a real dependency (small, pure-ish C), so we use it for
genuine index/search behaviour. torch is never imported here, keeping the
CI job fast and dependency-light.
"""

import sys
import types

import numpy as np
import pytest


def _inject_fake_sentence_transformers_module():
    """Register a stub ``sentence_transformers`` module in sys.modules.

    Tests use ``monkeypatch.setattr("sentence_transformers.SentenceTransformer", ...)``
    to swap in the fake model. ``monkeypatch.setattr`` imports the target module
    first, which fails in CI because sentence_transformers/torch are never
    installed. Injecting a stub module here lets the import succeed while keeping
    CI dependency-light; build_index/server/search use deferred imports and never
    touch this stub at runtime.
    """
    if "sentence_transformers" not in sys.modules:
        module = types.ModuleType("sentence_transformers")

        class _StubSentenceTransformer:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "sentence_transformers.SentenceTransformer must be monkeypatched in tests"
                )

        module.SentenceTransformer = _StubSentenceTransformer
        sys.modules["sentence_transformers"] = module


_inject_fake_sentence_transformers_module()


class FakeModel:
    """Deterministic stand-in for sentence_transformers.SentenceTransformer.

    Encodes each input string into a fixed-size vector derived from its bytes,
    so identical inputs always produce identical vectors and different inputs
    generally differ. This is enough to exercise index building, incremental
    appends, and search ranking without any ML backend.
    """

    def __init__(self, dim=768):
        self.dim = dim

    def encode(self, inputs, batch_size=16, normalize_embeddings=False, show_progress_bar=False):
        vectors = []
        for text in inputs:
            seed = 0.0
            # Derive a stable pseudo-vector from the string bytes.
            for byte in str(text).encode("utf-8"):
                seed += byte * 0.5
            vec = np.full(self.dim, -1.0, dtype="float32")
            for i in range(min(self.dim, max(1, len(str(text))))):
                vec[i] = ((seed + i * 3.7) % 50.0) - 25.0
            vectors.append(vec)
        out = np.asarray(vectors, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


@pytest.fixture()
def fake_model():
    """A deterministic model with a small embedding dimension."""
    return FakeModel(dim=64)


@pytest.fixture()
def image_dir(tmp_path):
    """A temp folder with a few real (tiny) PNG images in nested subfolders."""
    from PIL import Image

    files = {
        "a.png": (255, 0, 0),
        "sub/b.png": (0, 255, 0),
        "sub/c.png": (0, 0, 255),
    }
    for rel, color in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (4, 4), color).save(path)
    return tmp_path
