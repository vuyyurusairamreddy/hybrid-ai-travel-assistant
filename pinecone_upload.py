import json
import time
from typing import List, Dict, Tuple
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config
import os

DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

def build_semantic_text(node: Dict) -> str:
    parts = []
    if node.get("name"):
        parts.append(f"Name: {node['name']}")
    if node.get("type"):
        parts.append(f"Type: {node['type']}")
    if node.get("description"):
        parts.append(f"Description: {node['description']}")
    if node.get("city"):
        parts.append(f"City: {node['city']}")
    if node.get("region"):
        parts.append(f"Region: {node['region']}")
    if node.get("tags"):
        parts.append("Tags: " + ", ".join(node["tags"]))
    if node.get("semantic_text"):
        parts.append(node["semantic_text"])
    return " | ".join(parts)

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]

def main():
    if not os.path.exists(DATA_FILE):
        print(f" Missing {DATA_FILE}")
        return

    print(" Loading embedding model...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print(" Connecting to Pinecone...")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    print(" Ensuring index exists...")
    existing = [idx.name for idx in pc.list_indexes()]
    if config.PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.PINECONE_VECTOR_DIM,
            metric=config.PINECONE_METRIC,
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
        print(" Waiting for index to be ready...")
        time.sleep(15)

    index = pc.Index(config.PINECONE_INDEX_NAME)

    print(" Loading dataset...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items: List[Tuple[str, str, Dict]] = []
    for n in nodes:
        if "id" not in n:
            continue
        text = build_semantic_text(n).strip()
        if not text:
            continue
        meta = {
            "id": n["id"],
            "name": n.get("name", ""),
            "type": n.get("type", "Unknown"),
            "city": n.get("city", n.get("region", "")),
            "tags": (n.get("tags") or [])[:6],
        }
        items.append((n["id"], text, meta))

    print(f" Preparing {len(items)} vectors...")
    upserted = 0
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Upserting"):
        ids = [b[0] for b in batch]
        texts = [b[1] for b in batch]
        metas = [b[2] for b in batch]
        embs = model.encode(texts, show_progress_bar=False)
        vectors = [{"id": _id, "values": emb.tolist(), "metadata": meta} for _id, emb, meta in zip(ids, embs, metas)]
        index.upsert(vectors=vectors)
        upserted += len(vectors)
        time.sleep(0.05)

    print(f"Upserted {upserted} vectors to index '{config.PINECONE_INDEX_NAME}'")

    stats = index.describe_index_stats()
    print(" Index stats:", stats)

if __name__ == "__main__":
    main()
