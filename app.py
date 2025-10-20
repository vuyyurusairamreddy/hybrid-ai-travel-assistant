import streamlit as st
import time
import requests
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from neo4j import GraphDatabase
import config
from utils.embedding_cache import EmbeddingCache
from utils.graph_utils import fetch_neighbors, fetch_shortest_paths

# Page configuration
st.set_page_config(
    page_title="Vietnam Travel Assistant",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .response-box {
        background: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_data" not in st.session_state:
    st.session_state.response_data = None

# Initialize connections
@st.cache_resource
def init_connections():
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    cache = EmbeddingCache() if config.CACHE_ENABLED else None
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    index = pc.Index(config.PINECONE_INDEX_NAME)
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    return model, cache, index, driver

model, cache, index, driver = init_connections()

# Helper functions
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

def call_gemini(messages: List[Dict]) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent"
    
    gemini_messages = []
    for msg in messages:
        if msg["role"] == "system":
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": msg["content"]}]
            })
        elif msg["role"] == "user":
            gemini_messages.append({
                "role": "user",
                "parts": [{"text": msg["content"]}]
            })
        else:
            gemini_messages.append({
                "role": "model",
                "parts": [{"text": msg["content"]}]
            })
    
    payload = {
        "contents": gemini_messages,
        "generationConfig": {
            "temperature": config.TEMPERATURE,
            "topP": config.TOP_P,
            "maxOutputTokens": config.MAX_TOKENS,
        }
    }
    
    headers = {"Content-Type": "application/json"}
    params = {"key": config.GEMINI_API_KEY}
    
    r = requests.post(url, json=payload, headers=headers, params=params, timeout=45)
    r.raise_for_status()
    j = r.json()
    
    if "candidates" in j and len(j["candidates"]) > 0:
        if "content" in j["candidates"][0] and "parts" in j["candidates"][0]["content"]:
            if len(j["candidates"][0]["content"]["parts"]) > 0:
                return j["candidates"][0]["content"]["parts"][0]["text"]
    
    raise Exception(f"Unexpected Gemini response: {j}")

def run_once(user_query: str):
    t0 = time.time()
    
    # Vector search
    matches = pinecone_query(user_query, top_k=config.TOP_K_VECTOR)
    node_ids = [m.get("metadata", {}).get("id", m.get("id")) for m in matches if m.get("id")]
    
    # Graph search
    graph_facts = []
    with driver.session() as session:
        for nid in node_ids:
            graph_facts += fetch_neighbors(session, nid, limit=config.GRAPH_MAX_RELS_PER_NODE)
        if len(node_ids) >= 2:
            _ = fetch_shortest_paths(session, node_ids[0], node_ids[1], max_len=3, limit=2)
    
    # LLM call
    messages = build_messages(user_query, matches, graph_facts)
    ans = call_gemini(messages)
    elapsed = time.time() - t0
    
    return {
        "query": user_query,
        "answer": ans,
        "matches": matches,
        "graph_facts": graph_facts,
        "elapsed_time": elapsed,
        "vector_results_count": len(matches),
        "graph_results_count": len(graph_facts)
    }

# Sidebar
with st.sidebar:
    st.title(" Settings")
    top_k = st.slider("Vector Results (TOP_K)", 3, 15, config.TOP_K_VECTOR)
    max_tokens = st.slider("Max Response Tokens", 300, 1500, config.MAX_TOKENS)
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, config.TEMPERATURE, 0.1)
    
    st.divider()
    st.markdown("###  System Info")
    st.metric("LLM Provider", config.LLM_PROVIDER.upper())
    st.metric("Embedding Model", "MiniLM-L6")
    st.metric("Vector Dimension", config.PINECONE_VECTOR_DIM)

# Main content
st.markdown("<h1 class='main-header'>🌴 Vietnam Travel Assistant 🌴</h1>", unsafe_allow_html=True)
st.markdown("*Powered by Gemini AI + Pinecone + Neo4j*")

# Search bar
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input(
        "Ask me anything about Vietnam travel!",
        placeholder="E.g., What are the best beaches near Da Nang?",
        label_visibility="collapsed"
    )

with col2:
    search_button = st.button(" Search", use_container_width=True)

# Process query
if search_button and user_query:
    with st.spinner(" Searching beaches, hotels, and more..."):
        try:
            response_data = run_once(user_query)
            st.session_state.response_data = response_data
            st.session_state.chat_history.append({
                "query": user_query,
                "answer": response_data["answer"]
            })
        except Exception as e:
            st.error(f" Error: {str(e)}")

# Display results
if st.session_state.response_data:
    data = st.session_state.response_data
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(" Response Time", f"{data['elapsed_time']:.2f}s")
    with col2:
        st.metric(" Vector Matches", data['vector_results_count'])
    with col3:
        st.metric(" Graph Connections", data['graph_results_count'])
    with col4:
        st.metric(" LLM", config.GEMINI_MODEL.split('-')[1])
    
    st.divider()
    
    # Main answer
    st.markdown("###  Answer")
    st.markdown(f"""
    <div class='response-box'>
    {data['answer']}
    </div>
    """, unsafe_allow_html=True)
    
    # Expandable sections
    with st.expander(" Vector Search Results", expanded=False):
        if data['matches']:
            for i, match in enumerate(data['matches'], 1):
                meta = match.get('metadata', {})
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i}. {meta.get('name', 'Unknown')}**")
                    st.caption(f"Type: {meta.get('type', 'N/A')} | City: {meta.get('city', 'N/A')}")
                    if meta.get('tags'):
                        st.caption(f"Tags: {', '.join(meta['tags'][:3])}")
                with col2:
                    st.metric("Score", f"{match.get('score', 0):.3f}")
        else:
            st.info("No vector results found")
    
    with st.expander(" Knowledge Graph Context", expanded=False):
        if data['graph_facts']:
            for fact in data['graph_facts'][:10]:
                st.write(f"• **{fact.get('name', 'N/A')}** ({fact.get('type', 'N/A')})")
                st.caption(f"Connected via: {fact.get('rel', 'N/A')} | City: {fact.get('city', 'N/A')}")
        else:
            st.info("No graph results found")
    
    st.divider()
    
    # Copy button for answer
    st.markdown("### Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Copy Answer"):
            st.success(" Answer copied to clipboard!")
    with col2:
        if st.button(" Clear Results"):
            st.session_state.response_data = None
            st.session_state.chat_history = []
            st.rerun()

# Chat history
if st.session_state.chat_history:
    st.divider()
    st.markdown("###  Chat History")
    
    for i, chat in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
        with st.expander(f"Query {len(st.session_state.chat_history) - i + 1}: {chat['query'][:50]}..."):
            st.write(chat['answer'])

# Footer
st.divider()
st.markdown("""
---
<div style='text-align: center'>
<p> Vietnam Travel Assistant | Powered by <strong>Gemini</strong> + <strong>Pinecone</strong> + <strong>Neo4j</strong></p>
<p><small>Made with  for travel enthusiasts</small></p>
</div>
""", unsafe_allow_html=True)