import time
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
import config

from utils.embedding_cache import EmbeddingCache
from utils.graph_utils import fetch_neighbors, fetch_shortest_paths

print(" Initializing Hybrid AI Travel Assistant...")

# Embeddings
model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
cache = EmbeddingCache() if config.CACHE_ENABLED else None

# Pinecone
pc = Pinecone(api_key=config.PINECONE_API_KEY)
index = pc.Index(config.PINECONE_INDEX_NAME)

# Neo4j
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def embed(text: str) -> List[float]:
    if cache:
        e = cache.get(text, config.EMBEDDING_MODEL_NAME)
        if e is not None:
            return e
    v = model.encode(text).tolist()
    if cache:
        cache.set(text, config.EMBEDDING_MODEL_NAME, v)
    return v

def pinecone_query(query: str, top_k: int) -> List[Dict]:
    vec = embed(query)
    res = index.query(vector=vec, top_k=top_k, include_metadata=True)
    return res.get("matches", []) or []

def format_vector_context(matches: List[Dict]) -> str:
    lines = []
    for i, m in enumerate(matches, 1):
        meta = m.get("metadata", {})
        name = meta.get("name", "Unknown")
        _id = meta.get("id", m.get("id"))
        etype = meta.get("type", "Unknown")
        city = meta.get("city") or ""
        score = m.get("score", 0.0)
        tags = ", ".join(meta.get("tags", [])[:3])
        lines.append(
            f"{i}. {name} (ID: {_id}) | Type: {etype} | City: {city} | Score: {score:.3f} | Tags: {tags}"
        )
    return "\n".join(lines)

def build_messages(user_query: str, matches: List[Dict], graph_facts: List[Dict]) -> List[Dict]:
    vec_ctx = format_vector_context(matches) if matches else "No vector matches."
    graph_lines = []
    for f in graph_facts[: min(15, len(graph_facts))]:
        graph_lines.append(
            f"({f['id']}) {f.get('name','')} [{f.get('type','')}] via {f['rel']} | {f.get('city','')}"
        )
    graph_ctx = "\n".join(graph_lines) if graph_lines else "No graph neighbors."

    system = (
        "You are an expert Vietnam travel assistant.\n"
        "Use the provided semantic matches and knowledge graph relationships to craft accurate, structured answers.\n"
        "If generating itineraries, organize by days and reference node IDs where relevant."
    )
    user = (
        f"User Query: \"{user_query}\"\n\n"
        "--- Semantic Matches ---\n"
        f"{vec_ctx}\n\n"
        "--- Graph Context ---\n"
        f"{graph_ctx}\n\n"
        "Return a coherent, specific answer with actionable tips."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]

def call_perplexity(messages: List[Dict]) -> str:
    url = f"{config.PERPLEXITY_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.PERPLEXITY_MODEL,
        "messages": messages,
        "max_tokens": config.MAX_TOKENS,
        "temperature": config.TEMPERATURE,
        "top_p": config.TOP_P,
        "stream": False,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=45)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def run_once(user_query: str) -> str:
    t0 = time.time()
    matches = pinecone_query(user_query, top_k=config.TOP_K_VECTOR)
    node_ids = [m.get("metadata", {}).get("id", m.get("id")) for m in matches if m.get("id")]
    graph_facts = []
    with driver.session() as session:
        for nid in node_ids:
            graph_facts += fetch_neighbors(session, nid, limit=config.GRAPH_MAX_RELS_PER_NODE)
        if len(node_ids) >= 2:
            _ = fetch_shortest_paths(session, node_ids[0], node_ids[1], max_len=3, limit=2)
    messages = build_messages(user_query, matches, graph_facts)
    ans = call_perplexity(messages)
    print(f"  Total time: {time.time() - t0:.2f}s")
    return ans

def interactive():
    print("=" * 64)
    print(" HYBRID AI VIETNAM TRAVEL ASSISTANT (Perplexity + Pinecone + Neo4j)")
    print("=" * 64)
    print("Type 'exit' to quit.\n")
    try:
        while True:
            q = input("Enter your travel question: ").strip()
            if not q:
                continue
            if q.lower() in {"exit", "quit"}:
                break
            try:
                out = run_once(q)
                print("\n" + "-" * 64)
                print(out)
                print("-" * 64 + "\n")
            except Exception as e:
                print(f" Error: {e}")
    finally:
        driver.close()

if __name__ == "__main__":
    interactive()
