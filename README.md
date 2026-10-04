# CLIP + FAISS Semantic Retrieval of Scientific Images

Search microscopy images with plain-English queries such as *"fluorescent microscopy image of dense neuron cluster"*. Images are embedded with OpenAI CLIP (ViT-B/32), indexed with FAISS, and served through a small Streamlit app. No domain-specific training is involved: retrieval is zero-shot.

This is the code behind the paper listed under [Citation](#citation).

## How it works

```
images ──► CLIP image encoder ──► L2-normalised 512-d embeddings ──► FAISS IndexFlatIP ─┐
                                                                                         ├─► top-k images
text query ──► CLIP text encoder ──► L2-normalised 512-d embedding ─────────────────────┘
```

- Embeddings are normalised, so the inner-product score is cosine similarity.
- `IndexFlatIP` is an exact (brute-force) search. That is fine at the scale used here; an approximate index would be the next step for much larger collections.
- Scores are only meaningful for ranking within a query. CLIP text-image similarities are low in absolute terms (roughly 0.25 to 0.32 in the logged demo runs), so they are not calibrated probabilities.

## Quick start

```bash
git clone https://github.com/M-Lalith-Sai/clip-faiss-image-retrieval.git
cd clip-faiss-image-retrieval
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put your images in `images/` (any nesting; `.png .jpg .jpeg .tif .tiff`), then:

```bash
# 1. Build the index (writes index.faiss and meta.json)
python retrieval.py --build --images_dir images

# 2. Query from the command line (also saves query_results.png)
python retrieval.py --query "fluorescent microscopy image of bipolar neurons" --topk 5

# 3. Or use the web app
streamlit run app.py
```

The app lets you search the index and upload new images, which are added to the existing index without re-encoding what is already there (already-indexed paths are skipped).

More example queries are in [docs/example_queries.md](docs/example_queries.md).

## Data

This repository contains no image data and no prebuilt index. The index used in the paper experiments covered 9,722 images:

| Set | Images | Notes |
|---|---|---|
| Histopathology cell tiles | 9,431 | Not redistributed here. |
| Fluorescent neuronal cells | 291 | The public *Fluorescent Neuronal Cells* dataset (283 images, CC BY-SA 4.0; Morelli et al., 2021, [doi:10.1038/s41598-021-01929-5](https://doi.org/10.1038/s41598-021-01929-5)) plus 8 additional neuron crops. |

If you reuse the neuron images, follow that dataset's licence and cite its authors.

## Evaluation and limitations

- Evaluation so far is **qualitative**: natural-language queries about apoptosis, mitosis and neuron morphology, inspected by eye. No retrieval metrics (precision@k, recall@k) are reported, and that is the main open item.
- CLIP was not trained on microscopy images. Ranking quality for fine-grained biological concepts has not been measured against expert labels and should not be treated as diagnostic.
- Exact search scales linearly with collection size.
- The metadata stores image paths as given at indexing time, so an index must be rebuilt (or its paths kept valid) if the images move.

## Project layout

```
retrieval.py   index building, querying, incremental dataset add, CLI
app.py         Streamlit interface
docs/          example queries
```

## Citation

M. Lalith Sai, L. Manish, G. Varshitha Raj and M. Supriya, "CLIP-Based Semantic Retrieval of Scientific Images," *International Research Journal of Modernization in Engineering Technology and Science (IRJMETS)*, vol. 8, no. 4, April 2026. DOI: [10.56726/IRJMETS93807](https://doi.org/10.56726/IRJMETS93807).

## Acknowledgements

Built on [OpenAI CLIP](https://github.com/openai/CLIP) and [FAISS](https://github.com/facebookresearch/faiss).

## License

Released under the [MIT License](LICENSE). The licence covers this code only; image datasets keep their own licences (see [Data](#data)).
