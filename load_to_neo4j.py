import json
from typing import Dict
from neo4j import GraphDatabase
from tqdm import tqdm
import config

DATA_FILE = "vietnam_travel_dataset.json"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def create_constraints():
    with driver.session() as session:
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

def upsert_node(session, node: Dict):
    labels = [node.get("type", "Unknown"), "Entity"]
    label_str = ":" + ":".join(labels)
    props = {k: v for k, v in node.items() if k != "connections"}
    session.run(
        f"MERGE (n{label_str} {{id: $id}}) SET n += $props",
        id=node["id"],
        props=props,
    )

def create_relationship(session, source_id: str, rel: Dict):
    # rel = {"relation": "Located_In", "target": "xyz"}
    session.run(
        """
        MATCH (a:Entity {id:$src}), (b:Entity {id:$dst})
        MERGE (a)-[r:`%s`]->(b)
        """ % rel["relation"],
        src=source_id,
        dst=rel["target"],
    )

def main():
    create_constraints()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with driver.session() as session:
        for node in tqdm(nodes, desc="Upserting nodes"):
            upsert_node(session, node)

        for node in tqdm(nodes, desc="Creating relationships"):
            source_id = node["id"]
            for rel in node.get("connections", []):
                create_relationship(session, source_id, rel)

    driver.close()
    print(" Neo4j load complete.")

if __name__ == "__main__":
    main()
