# Hybrid AI Travel Assistant

A sophisticated multi-stage retrieval-augmented generation (RAG) system for Vietnam travel queries, combining semantic search (Pinecone), knowledge graph reasoning (Neo4j), and advanced LLM reasoning (Google Gemini or Perplexity).

**Choose your interface:**
-  **Streamlit Web UI** - Rich, interactive web dashboard
-  **CLI** - Command-line chat interface

##  Features

- **Dual Interface**: Web UI (Streamlit) + Command-line chat
- **Semantic Search**: Local embeddings (`all-MiniLM-L6-v2`) with disk-backed caching
- **Knowledge Graph**: Neo4j relationships for contextual travel connections
- **Advanced Reasoning**: Google Gemini or Perplexity LLM with large context windows
- **Flexible LLM**: Switch between Gemini (free) or Perplexity (premium) with one config change
- **Performance**: Conservative TOP_K and relation limits for stable latency
- **Reliability**: Error handling, consistent timeouts, and stats logging

##  Architecture

```
User Query
    ↓
Vector Embeddings (SentenceTransformers)
    ↓
Pinecone Semantic Search ← Embedding Cache
    ↓
Neo4j Graph Context (Neighbors + Shortest Paths)
    ↓
Gemini/Perplexity LLM with Grounded Reasoning
    ↓
Structured Travel Response
```

##  Prerequisites

- Python 3.8+
- Neo4j database instance (free at neo4j.com)
- Pinecone account (free tier available)
- Google Gemini API key (free) OR Perplexity API key (paid)

##  Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Credentials

Create a `.env` file in your project root:

```env
NEO4J_URI=neo4j+s://your-neo4j-uri
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

GEMINI_API_KEY=your-gemini-key
# OR
PERPLEXITY_API_KEY=pplx_your_key

PINECONE_API_KEY=pcsk_your_key
```

### 3. Get API Keys

**Google Gemini (Free):**
- Go to https://ai.google.dev/
- Click "Get API Key"
- Create a new key and copy it

**Perplexity (Paid):**
- Go to https://www.perplexity.ai/
- Sign up and navigate to API settings
- Generate an API key

**Pinecone (Free tier):**
- Go to https://www.pinecone.io/
- Sign up and create an index

**Neo4j (Free):**
- Go to https://neo4j.com/
- Create a free cloud instance

### 4. Prepare Dataset

Create `vietnam_travel_dataset.json`:

```json
[
  {
    "id": "attraction_001",
    "name": "Ha Long Bay",
    "type": "Attraction",
    "description": "UNESCO World Heritage site with limestone karsts",
    "city": "Ha Long",
    "region": "Northern",
    "tags": ["beach", "scenic", "unesco"],
    "connections": [
      {"relation": "Located_In", "target": "hanoi_city"}
    ]
  },
  {
    "id": "hotel_001",
    "name": "Luxury Resort",
    "type": "Hotel",
    "description": "5-star beachfront hotel with WiFi",
    "city": "Da Nang",
    "region": "Central",
    "tags": ["luxury", "beachfront", "wifi"],
    "connections": [
      {"relation": "Located_In", "target": "danang_city"},
      {"relation": "Near", "target": "beach_001"}
    ]
  }
]
```

### 5. Load Data

**Initialize Neo4j:**

```bash
python load_to_neo4j.py
```

Output:
```
100%|████████| 150/150 [00:05<00:00, 28.45it/s]
100%|████████| 150/150 [00:03<00:00, 45.23it/s]
 Neo4j load complete.
```

**Upload to Pinecone:**

```bash
python pinecone_upload.py
```

Output:
```
 Loading embedding model...
 Connecting to Pinecone...
 Ensuring index exists...
100%|████████| 5/5 [00:12<00:00, 2.40s/batch]
Upserted 150 vectors to index 'vietnam-travel'
```

---

##  Running the Assistant

### Option 1: Web UI (Streamlit) 

**Start the web dashboard:**

```bash
streamlit run app.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

**Features:**
-  Beautiful interactive web interface
-  Adjustable settings (TOP_K, MAX_TOKENS, Temperature)
-  Real-time metrics (response time, vector matches, graph connections)
-  Chat history with expandable queries
-  Detailed result exploration (vector search + graph context)
-  Copy & export answers

**Using the Web UI:**

1. Open http://localhost:8501 in your browser
2. Adjust settings in the left sidebar:
   - Vector Results (TOP_K): 3-15
   - Max Response Tokens: 300-1500
   - Temperature (Creativity): 0.0-1.0

3. Type your question in the search bar:
   ```
   "What are the best beaches near Da Nang?"
   ```

4. Click "Search" button

5. View results:
   - Main answer displayed prominently
   - Response time & metrics
   - Expandable sections for vector search results
   - Knowledge graph context
   - Chat history for previous queries

---

### Option 2: Command-Line Interface (CLI) 

**Start the CLI chat:**

```bash
python hybrid_chat.py
```

**Output:**
```
================================================================
 HYBRID AI VIETNAM TRAVEL ASSISTANT (GEMINI)
================================================================
Type 'exit' to quit.

Enter your travel question: 
```

**Using the CLI:**

```
Enter your travel question: What are the best beaches near Da Nang?

  Total time: 2.45s
--------------------------------------------------------------------
Based on the semantic matches and knowledge graph context, here are 
the top beaches near Da Nang:

1. My Khe Beach - Known for its pristine sandy shores and clear waters.
   Perfect for swimming and water sports.

2. Non Nuoc Beach - A quieter alternative with stunning limestone 
   backdrop and local seafood restaurants.

Recommended itinerary:
- Day 1: Explore My Khe Beach in the morning (30 min walk from city)
- Day 2: Visit Non Nuoc Beach and nearby attractions
--------------------------------------------------------------------

Enter your travel question: 
```

Type `exit` or `quit` to close the assistant.

---

##  Example Queries

Try these queries in either interface:

```
"What are the best beaches near Da Nang?"
"Recommend budget hotels in Hanoi with WiFi"
"How do I get from Ho Chi Minh City to Ha Long Bay?"
"Best restaurants in Hanoi for street food"
"Family-friendly attractions in Da Nang"
"Luxury resorts with spa in Ho Chi Minh City"
"Transportation options from airport to city center"
"What's the best time to visit Vietnam?"
```

---

##  Configuration

Update parameters in `config.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `LLM_PROVIDER` | `"gemini"` | Choose `"gemini"` or `"perplexity"` |
| `GEMINI_MODEL` | `"gemini-1.5-flash"` | Use `"gemini-1.5-pro"` for better quality |
| `PERPLEXITY_MODEL` | `"sonar-pro"` | Perplexity model selection |
| `TOP_K_VECTOR` | `7` | Number of semantic matches from Pinecone |
| `GRAPH_MAX_RELS_PER_NODE` | `10` | Limit relationships per node |
| `MAX_TOKENS` | `2000` | Response length limit |
| `TEMPERATURE` | `0.3` | LLM sampling (0=deterministic, 1=creative) |
| `TOP_P` | `0.9` | Nucleus sampling parameter |
| `CACHE_ENABLED` | `True` | Enable embedding cache |

### Switch Between LLMs

**Use Gemini (Free):**
```python
LLM_PROVIDER = "gemini"
GEMINI_API_KEY = "your-key"
```

**Use Perplexity (Premium):**
```python
LLM_PROVIDER = "perplexity"
PERPLEXITY_API_KEY = "pplx_your_key"
```

---

##  Project Structure

```
.
├── app.py                   # Streamlit web UI
├── hybrid_chat.py           # CLI chat interface
├── config.py                # Configuration & credentials
├── load_to_neo4j.py         # Neo4j data loader
├── pinecone_upload.py       # Pinecone uploader
├── visualize_graph.py       # Graph visualization
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create this)
├── .gitignore              # Git ignore file
├── README.md               # This file
└── utils/
    ├── __init__.py
    ├── embedding_cache.py   # Disk-backed cache
    └── graph_utils.py       # Neo4j query helpers
```

---

##  Web UI Features

### Dashboard Metrics
-  Response time tracking
-  Vector search result counts
-  Knowledge graph connection counts
-  LLM model information

### Settings Sidebar
-  Adjustable TOP_K slider (3-15)
-  Max tokens slider (300-1500)
-  Temperature slider (0.0-1.0)
-  System information panel

### Result Exploration
-  Main answer with formatted display
-  Expandable vector search results with scores
-  Knowledge graph context section
-  Copy & export functionality
-  Full chat history with previous queries

---

##  CLI Features

### Simple Interactive Chat
-  Direct text input/output
-  Response time display
-  Clean formatted output
-  Continuous conversation loop
-  Easy exit (type 'exit' or 'quit')

---

##  Optional: Visualize the Graph

```bash
python visualize_graph.py
```

Opens `neo4j_viz.html` in your browser showing the knowledge graph network.

---

##  Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `401 Unauthorized` (Gemini) | Check your GEMINI_API_KEY is correct and enabled |
| `401 Unauthorized` (Perplexity) | Verify PERPLEXITY_API_KEY starts with `pplx_` |
| `ConnectionFailure` (Neo4j) | Verify URI and credentials in .env or config.py |
| `FileNotFoundError: vietnam_travel_dataset.json` | Create the dataset file with proper structure |
| `Index does not exist` (Pinecone) | Run `pinecone_upload.py` to create the index |
| `Unexpected Gemini response` | Increase MAX_TOKENS to 2000+ in config.py |
| `Streamlit port already in use` | Run on different port: `streamlit run app.py --server.port 8502` |
| `Neo4j SyntaxError on shortestPath` | Already fixed in graph_utils.py (depth hardcoded to 3) |

---

##  Performance Tips

- **Embedding Cache**: Disk-backed caching reduces recomputation for repeated queries
- **Conservative Limits**: TOP_K=7 and graph relations capped for predictable latency
- **Model Selection**: Use `gemini-1.5-flash` for speed, `gemini-1.5-pro` for quality
- **Async Queries**: Future optimization - parallelize Pinecone + Neo4j lookups
- **Query Expansion**: Future - parse user intent for better retrieval
- **Web UI**: Better for exploration and iterative queries
- **CLI**: Better for automation and batch processing

---

##  Security

**Never commit credentials!** Always use `.env` file:

```bash
# Add to .gitignore
.env
.cache/
__pycache__/
*.pyc
neo4j_viz.html
```

Load credentials in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
```

---

##  Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `neo4j` | Graph database |
| `pinecone-client` | Vector database |
| `sentence-transformers` | Local embeddings |
| `requests` | HTTP API calls |
| `diskcache` | Persistent caching |
| `torch`, `transformers` | ML foundations |

Full list in `requirements.txt`

---

##  Complete Workflow Example

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create .env file with your credentials
cat > .env << EOF
NEO4J_URI=neo4j+s://your-uri
NEO4J_PASSWORD=your-password
GEMINI_API_KEY=your-gemini-key
PINECONE_API_KEY=your-pinecone-key
EOF

# 3. Prepare vietnam_travel_dataset.json

# 4. Load to Neo4j
python load_to_neo4j.py

# 5. Upload to Pinecone
python pinecone_upload.py

# 6a. Run Web UI (Streamlit)
streamlit run app.py

# 6b. OR Run CLI
python hybrid_chat.py

# 7. (Optional) Visualize graph
python visualize_graph.py
```

---

##  Deployment

### Local Development
```bash
streamlit run app.py
```

### Production Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Go to https://share.streamlit.io/
3. Connect your GitHub repo
4. Deploy with secrets from .env

### Docker Deployment
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py"]
```

---

##  Future Enhancements

- [ ] Async parallelization of Pinecone + Neo4j queries
- [ ] Query expansion based on intent parsing
- [ ] Multi-turn conversation memory
- [ ] BM25 + vector search hybrid
- [ ] Response caching layer
- [ ] User feedback loop for ranking
- [ ] Multi-language support
- [ ] Mobile app

---

##  Interface Comparison

| Feature | Streamlit Web | CLI |
|---------|---------------|-----|
| **User Experience** | Rich, visual | Simple, text-based |
| **Settings Adjustment** | Easy sliders | Manual config edits |
| **Result Visualization** | Expandable cards | Plain text |
| **Chat History** | Built-in UI | Manual tracking |
| **Copy/Export** | One-click buttons | Manual selection |
| **Automation** | Less suitable | Ideal for scripts |
| **Mobile-friendly** | Yes | No |
| **Learning Curve** | Beginner-friendly | Minimal |
| **Deployment** | Cloud-ready | Server/CLI |

---

##  License

Internal use only.

##  Support

For issues:
1. Check the **Troubleshooting** section
2. Verify all API keys are correct
3. Ensure dataset is properly formatted
4. Check error messages and logs

---

**Happy traveling! **