from typing import List, Dict
from neo4j import Session

def fetch_neighbors(session: Session, node_id: str, limit: int = 10) -> List[Dict]:
    q = """
    MATCH (n:Entity {id: $id})-[r]-(m:Entity)
    RETURN type(r) as rel, m.id as id, m.name as name, m.type as type,
           m.description as description, m.city as city, m.tags as tags
    LIMIT $limit
    """
    recs = session.run(q, id=node_id, limit=limit)
    out = []
    for r in recs:
        out.append({
            "rel": r["rel"],
            "id": r["id"],
            "name": r["name"],
            "type": r["type"],
            "description": (r["description"] or "")[:300] if "description" in r else "",
            "city": r.get("city"),
            "tags": r.get("tags") or [],
        })
    return out

def fetch_shortest_paths(session: Session, id1: str, id2: str, max_len: int = 3, limit: int = 2) -> List[Dict]:
    """
    Find shortest paths between two entities.
    Max depth is hardcoded to avoid Neo4j parameter binding issues with shortestPath.
    """
    try:
        q = """
        MATCH p=shortestPath((a:Entity {id:$id1})-[*..3]-(b:Entity {id:$id2}))
        RETURN p LIMIT $limit
        """
        recs = session.run(q, id1=id1, id2=id2, limit=limit)
        paths = []
        for r in recs:
            p = r["p"]
            paths.append({"length": len(p.relationships)})
        return paths
    except Exception as e:
        # If shortest path fails, return empty list gracefully
        print(f"  Warning: Could not fetch shortest path between {id1} and {id2}: {e}")
        return []