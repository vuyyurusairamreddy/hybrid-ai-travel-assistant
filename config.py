# config.py - Configuration for Hybrid AI Travel Assistant (Perplexity + Pinecone + Neo4j)

# Neo4j
NEO4J_URI = "neo4j+s://80cd8916.databases.neo4j.io"           # Or neo4j://… if preferred
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "5-45BoNOPR7zSHWaFUiEdxT-AFPGWypB_FgIhzUokJU12"

# Perplexity API (sonar / sonar-pro)
# Endpoint uses OpenAI-compatible Chat Completions interface, POST /chat/completions
#PERPLEXITY_API_KEY = ""
#PERPLEXITY_BASE_URL = "https://api.perplexity.ai"
#PERPLEXITY_MODEL = "sonar-pro"                # sonar-pro for deeper reasoning and longer context
# LLM Selection: 'gemini' or 'perplexity'
LLM_PROVIDER = "gemini"  # Change to "perplexity" if needed

# Gemini API (Google)
GEMINI_API_KEY = "AIzaSyCYxLXe3U5w0gqj4tgXb0QLHd-g5qeb5wA12"
GEMINI_MODEL = "gemini-2.5-flash"  # or "gemini-1.5-pro" for better quality

# Pinecone
PINECONE_API_KEY = "pcsk_XCdXx_Ryo5h46oFN1SfQdvy4VduPFdd6LcAnH9DNxfk8r7965mvVAUEyW4Ttd9kP7gtRj12ac"
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_CLOUD = "aws"                        # "aws" or "gcp"
PINECONE_REGION = "us-east-1"                 # e.g., us-east-1 for AWS
PINECONE_METRIC = "cosine"

# Embeddings (free, local)
# all-MiniLM-L6-v2 => 384-dim embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_VECTOR_DIM = 384

# Retrieval / generation params
TOP_K_VECTOR = 7
GRAPH_MAX_RELS_PER_NODE = 10
MAX_TOKENS = 1500
TEMPERATURE = 0.3
TOP_P = 0.9

# Cache
CACHE_ENABLED = True
CACHE_DIR = ".cache"
