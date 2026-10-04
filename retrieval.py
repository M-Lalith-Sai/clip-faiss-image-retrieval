"""CLIP + FAISS semantic image retrieval.

Images are encoded with OpenAI CLIP (ViT-B/32). The L2-normalised embeddings are
stored in a FAISS inner-product index, so the score is cosine similarity.
A natural-language query is encoded with the CLIP text encoder and matched
against the index.

Usage:
    python retrieval.py --build --images_dir images
    python retrieval.py --query "fluorescent microscopy image of neurons" --topk 5
"""
import argparse
import json
import os
from functools import lru_cache
from pathlib import Path

import clip
import faiss
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DEFAULT_MODEL = "ViT-B/32"


def list_images(root):
    """Return the paths of all supported images under `root` (recursive)."""
    return sorted(str(p) for p in Path(root).rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def _resolve_device(device=None):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=2)
def _load_cached(model_name, device):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def load_model(model_name=DEFAULT_MODEL, device=None):
    """Load CLIP once per process; repeated calls reuse the same model."""
    return _load_cached(model_name, _resolve_device(device))


def _normalise(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def encode_images(paths, model, preprocess, device, desc="Encoding images"):
    """Encode image files to L2-normalised float32 embeddings (one row per image)."""
    embeddings = []
    with torch.no_grad():
        for path in tqdm(paths, desc=desc):
            img = Image.open(path).convert("RGB")
            x = preprocess(img).unsqueeze(0).to(device)
            emb = model.encode_image(x).cpu().numpy().astype("float32")
            embeddings.append(_normalise(emb)[0])
    return np.vstack(embeddings).astype("float32")


def _save(index, meta, index_path, meta_path):
    faiss.write_index(index, index_path)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def build_index(images_dir, index_path="index.faiss", meta_path="meta.json",
                model_name=DEFAULT_MODEL, device=None):
    """Encode every image in `images_dir` and write a fresh FAISS index + metadata."""
    device = _resolve_device(device)
    model, preprocess = load_model(model_name, device)

    image_paths = list_images(images_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    embeddings = encode_images(image_paths, model, preprocess, device)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    meta = [{"path": p} for p in image_paths]
    _save(index, meta, index_path, meta_path)
    print(f"Indexed {len(meta)} images into {index_path}")


def query_index(query, index_path="index.faiss", meta_path="meta.json",
                model_name=DEFAULT_MODEL, topk=5, device=None):
    """Return the `topk` most similar images for a text query as dicts with score and path."""
    device = _resolve_device(device)
    model, _ = load_model(model_name, device)

    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    with torch.no_grad():
        tokens = clip.tokenize([query]).to(device)
        text_emb = model.encode_text(tokens).cpu().numpy().astype("float32")
    text_emb = _normalise(text_emb)

    scores, idxs = index.search(text_emb, min(topk, index.ntotal))
    return [
        {"score": float(s), "image_url": meta[i]["path"]}
        for s, i in zip(scores[0], idxs[0])
        if i >= 0  # FAISS pads with -1 when fewer than topk results exist
    ]


def search_images(query, topk=5):
    """Frontend-friendly wrapper used by the Streamlit app."""
    return query_index(query, topk=topk)


def add_dataset(dataset_dir, index_path="index.faiss", meta_path="meta.json",
                model_name=DEFAULT_MODEL, device=None):
    """Add new images to an existing index without re-encoding what is already indexed.

    Images whose path is already in the metadata are skipped, so indexing the same
    folder twice does not create duplicate entries.
    """
    device = _resolve_device(device)
    model, preprocess = load_model(model_name, device)

    if os.path.exists(index_path) and os.path.exists(meta_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "r") as f:
            meta = json.load(f)
    else:
        index, meta = None, []

    known = {m["path"] for m in meta}
    new_paths = [p for p in list_images(dataset_dir) if p not in known]
    if not new_paths:
        print("No new images to add.")
        return

    embeddings = encode_images(new_paths, model, preprocess, device, desc="Encoding new dataset")
    if index is None:
        index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    meta.extend({"path": p} for p in new_paths)
    _save(index, meta, index_path, meta_path)
    print(f"Added {len(new_paths)} new images to index")


def visualize(results, out="query_results.png"):
    """Save the retrieved images side by side, titled with their similarity scores."""
    n = len(results)
    fig, axs = plt.subplots(1, n, figsize=(3 * n, 3), squeeze=False)
    for ax, r in zip(axs[0], results):
        ax.imshow(Image.open(r["image_url"]))
        ax.axis("off")
        ax.set_title(f"{r['score']:.3f}", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Saved visualization -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="CLIP + FAISS semantic image retrieval")
    p.add_argument("--build", action="store_true", help="build the index from --images_dir")
    p.add_argument("--images_dir", default="images")
    p.add_argument("--index_path", default="index.faiss")
    p.add_argument("--meta_path", default="meta.json")
    p.add_argument("--query", default=None, help="text query to search for")
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    if args.build:
        build_index(args.images_dir, args.index_path, args.meta_path)
    elif args.query:
        res = query_index(args.query, args.index_path, args.meta_path, topk=args.topk)
        print("Top results:")
        for r in res:
            print(f"{r['score']:.4f}\t{r['image_url']}")
        if res:
            visualize(res)
    else:
        p.print_help()
