# Improvements

## Overview
- Migrated chat LLM to Perplexity `sonar-pro` using the OpenAI-compatible Chat Completions interface at `/chat/completions` for robust multi-step reasoning and large context windows.  
- Switched to free local embeddings via `sentence-transformers/all-MiniLM-L6-v2` (384-dim), eliminating embedding API costs.  
- Upgraded Pinecone client to v3 and fixed serverless spec (cloud/region), with robust index bootstrapping and batch upserts.  

## Retrieval & Reasoning
- Vector stage improved by constructing richer semantic texts from node properties (name/type/desc/city/tags).  
- Graph stage fetches neighbors and shortest paths to inject relational context.  
- Prompt organized with explicit “Semantic Matches” and “Graph Context” sections to guide reasoning with grounded references.  

## Performance & Reliability
- Added disk-backed embedding cache to reduce repeated computation.  
- Clean error handling around HTTP calls and DB sessions; consistent timeouts and stats logs.  
- Kept TOP_K conservative and relations capped per node for stable latency.  

## Next Steps (Optional)
- Async parallelization for Pinecone + Neo4j lookups.  
- Query expansion based on intent parsing (locations, duration, preferences).  
- Web UI for richer UX and logging.  
