# 文搜圖 (EmbeddingGemma 2)

[English](README.md) | [繁體中文](README_zh.md)

蒐集指定資料夾的圖片，用 Google 的 **EmbeddingGemma 2**（`google/embeddinggemma-2`）建立向量索引，之後可用文字查詢最相似的圖片。

模型把文字與圖片投影到同一個 768 維向量空間，所以可以直接比對 cosine 相似度。

## 架構

```
┌──────────────┐   HTTP POST /search    ┌─────────────────────────┐
│ search_images │ ────────────────────▶ │  server.py（常駐）       │
│   client     │ ◀──────────────────── │  模型只載入一次          │
└──────────────┘   JSON results         │  faiss 索引在記憶體      │
                                         └─────────────────────────┘
```

搜尋走 HTTP 連常駐伺服器，模型只載入一次（省掉每次 ~4 秒 + 重新編碼）。

## 安裝

本環境無 GPU，使用 CPU 版 torch：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

`requirements.txt` 已指定 CPU 版 torch/torchvision，避免下載大量 CUDA 套件。

## 建立索引

```bash
.venv/bin/python build_index.py \
    --images ./photos \
    --index img_index.npy \
    --meta images_meta.json
```

- `--images`：要掃描的資料夾（遞迴，支援 jpg/jpeg/png/webp/gif/bmp/heic）。手機拍的 HEIC 用 `pillow-heif` 解碼。
- `--index`：faiss 索引輸出檔。
- `--meta`：metadata（圖片路徑清單），搜尋時用。
- `--dim 256`：可用 Matryoshka 截斷向量縮小索引（768/512/256/128）。
- `--cpu`：強制 CPU（預設自動偵測）。

首次執行會下載模型（約 1.4 GB），之後快取在本機。

### 增量更新

`build_index.py` **預設增量更新**：偵測到舊索引與 metadata 時，只編碼「新增」的圖片再 append 到 faiss 索引，舊照片不會重跑。

```bash
# 新增照片後直接執行（自動只編碼新圖）
python build_index.py --images ./photos

# 維度不符或要完全重建時，強制全量
python build_index.py --images ./photos --force-full
```

### 剪除已刪除的照片

faiss 索引無法在原地刪除列，所以照片從資料夾移除後，**需要重建一次索引**才能把它移出 faiss 與 metadata。

```bash
# --prune：全量重建，只保留磁碟上還存在的檔案（自動剪除已刪除的照片）
python build_index.py --images ./photos --prune
```

搜尋時已加一道安全檢查：即使 metadata 裡還留有某張圖的路徑，只要磁碟上找不到該檔案，搜尋結果就會自動跳過它。所以伺服器啟動後才刪除照片也不會回傳失效路徑。

### 自動同步常駐伺服器

`build_index.py` 建完索引後，**會自動 POST `/reload` 通知常駐 server 從磁碟重載 faiss 索引**，不用手動重啟 server。

```bash
# 新增照片（server 自動收到新圖）
python build_index.py --images ./photos

# 刪除照片後重建（server 自動移除）
python build_index.py --images ./photos --prune
```

reload 只換 faiss 索引、不重載模型，秒回。若 server 沒在跑（或改過 `--index`/`--meta`），build 會印出提示，可手動 POST：

```bash
curl -X POST http://127.0.0.1:8000/reload
```

## 啟動常駐伺服器（背景）

```bash
.venv/bin/python server.py \
    --index img_index.npy --meta images_meta.json --port 8000
```

- 模型只載入一次，之後每次查詢走 HTTP。
- 健康檢查：`GET /health` → `{"status":"ok","loaded":true}`
- 停止：在終端按 `Ctrl+C`。

## 搜尋圖片（連常駐伺服器）

```bash
.venv/bin/python search_images.py \
    --query "一隻在戶外的白貓" \
    --top 5
```

- `--query`：查詢文字。
- `--server http://host:port`：伺服器位址（預設 `http://127.0.0.1:8000`）。
- `--top N`：回傳結果數量（預設 5）。

結果同時印出並寫入 `search_results.json`。
（若不想用伺服器，可加 `--local` 本機直接跑，但每次會重新載入模型。）

## 檔案說明

| 檔案 | 用途 |
|------|------|
| `build_index.py` | 掃描資料夾、產生向量、寫入 faiss 索引 |
| `server.py` | 常駐伺服器（模型只載入一次，提供 HTTP /search、/embed） |
| `search_images.py` | 文字查詢 client（預設走 HTTP，可加 --local） |
| `requirements.txt` | 依賴（CPU 版 torch/torchvision + faiss-cpu + flask + requests） |

## 備註

- 索引與 metadata 分開存放，新增圖片後重新執行 `build_index.py` 即可。
- 查詢文字與圖片共用同一向量空間，中英文皆可。
