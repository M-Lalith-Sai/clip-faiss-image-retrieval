import os

import streamlit as st
from PIL import Image

from retrieval import add_dataset, search_images

st.set_page_config(page_title="Scientific Image Search", layout="wide")

st.title("CLIP-Based Scientific Image Retrieval")
st.write("Search microscopy images using natural language.")

# -----------------------------
# Dataset upload
# -----------------------------
st.header("Upload a new dataset")

uploaded_files = st.file_uploader(
    "Upload microscopy images",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=True,
)

dataset_folder = "datasets/new_dataset"
os.makedirs(dataset_folder, exist_ok=True)

if uploaded_files:
    for file in uploaded_files:
        # basename only: never trust a client-supplied file name as a path
        file_path = os.path.join(dataset_folder, os.path.basename(file.name))
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
    st.success(f"{len(uploaded_files)} images uploaded successfully.")

if st.button("Index uploaded dataset"):
    with st.spinner("Indexing images..."):
        add_dataset(dataset_folder)
    st.success("Dataset indexed.")

st.divider()

# -----------------------------
# Search
# -----------------------------
st.header("Search images")

query = st.text_input("Enter your search query")
topk = st.slider("Number of results", 1, 10, 5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a search query.")
    else:
        with st.spinner("Searching images..."):
            try:
                results = search_images(query, topk=topk)
            except FileNotFoundError:
                st.error("No index found. Build one first: python retrieval.py --build --images_dir images")
                results = []

        if not results:
            st.warning("No results found.")
        else:
            st.subheader("Results")
            for col, r in zip(st.columns(len(results)), results):
                try:
                    col.image(Image.open(r["image_url"]), use_container_width=True)
                    col.caption(f"Similarity: {r['score']:.3f}")
                except Exception:
                    col.write("Image could not be loaded")
