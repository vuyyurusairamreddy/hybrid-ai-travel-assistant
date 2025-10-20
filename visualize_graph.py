from neo4j import GraphDatabase
import config
import json
import webbrowser
import os

OUTPUT_HTML = "neo4j_viz_simple.html"
LIMIT = 500

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def fetch_edges(limit=500):
    q = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.id as a_id, a.name as a_name, labels(a) as a_labels, "
        "b.id as b_id, b.name as b_name, labels(b) as b_labels, type(r) as rel "
        "LIMIT $limit"
    )
    with driver.session() as session:
        return list(session.run(q, limit=limit))

def build_simple_viz(rows, output_html=OUTPUT_HTML):
    """Create a simple D3.js visualization"""
    
    # Build nodes and links
    nodes = {}
    links = []
    
    node_colors = {
        "City": "#FF6B6B",
        "Attraction": "#4ECDC4",
        "Hotel": "#45B7D1",
        "Activity": "#FFA07A",
        "Entity": "#95E1D3"
    }
    
    for rec in rows:
        # Source node
        a_id = rec["a_id"]
        if a_id not in nodes:
            a_type = rec["a_labels"][0] if rec["a_labels"] else "Entity"
            nodes[a_id] = {
                "id": a_id,
                "name": rec["a_name"] or a_id,
                "type": a_type,
                "color": node_colors.get(a_type, node_colors["Entity"])
            }
        
        # Target node
        b_id = rec["b_id"]
        if b_id not in nodes:
            b_type = rec["b_labels"][0] if rec["b_labels"] else "Entity"
            nodes[b_id] = {
                "id": b_id,
                "name": rec["b_name"] or b_id,
                "type": b_type,
                "color": node_colors.get(b_type, node_colors["Entity"])
            }
        
        # Add link
        links.append({
            "source": a_id,
            "target": b_id,
            "type": rec["rel"]
        })
    
    # Convert nodes dict to list
    nodes_list = list(nodes.values())
    
    # Create HTML with embedded D3.js
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Vietnam Travel Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background: #f5f5f5;
        }}
        
        #graph {{
            width: 100%;
            height: 900px;
            background: white;
            border: 2px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .info {{
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .legend {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
        }}
        
        .links line {{
            stroke: #999;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }}
        
        .nodes circle {{
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
        }}
        
        .nodes circle:hover {{
            stroke-width: 4px;
            stroke: #333;
        }}
        
        .node-labels {{
            pointer-events: none;
            font-size: 11px;
            font-weight: bold;
            fill: #333;
        }}
        
        .link-labels {{
            font-size: 9px;
            fill: #666;
            pointer-events: none;
        }}
        
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            pointer-events: none;
            font-size: 12px;
            display: none;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h1> Vietnam Travel Knowledge Graph</h1>
        <p><strong>Nodes:</strong> {len(nodes_list)} | <strong>Relationships:</strong> {len(links)}</p>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #FF6B6B;"></div>
                <span>City</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #4ECDC4;"></div>
                <span>Attraction</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #45B7D1;"></div>
                <span>Hotel</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #FFA07A;"></div>
                <span>Activity</span>
            </div>
        </div>
        <p style="margin-top: 10px; color: #666;"> Drag nodes to rearrange | Hover for details | Zoom with mouse wheel</p>
    </div>
    
    <div id="graph"></div>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const data = {{
            nodes: {json.dumps(nodes_list)},
            links: {json.dumps(links)}
        }};
        
        const width = document.getElementById('graph').offsetWidth;
        const height = 900;
        
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", (event) => {{
                container.attr("transform", event.transform);
            }}));
        
        const container = svg.append("g");
        
        const tooltip = d3.select("#tooltip");
        
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(30));
        
        const link = container.append("g")
            .attr("class", "links")
            .selectAll("line")
            .data(data.links)
            .enter().append("line");
        
        const linkLabels = container.append("g")
            .attr("class", "link-labels")
            .selectAll("text")
            .data(data.links)
            .enter().append("text")
            .text(d => d.type);
        
        const node = container.append("g")
            .attr("class", "nodes")
            .selectAll("circle")
            .data(data.nodes)
            .enter().append("circle")
            .attr("r", 8)
            .attr("fill", d => d.color)
            .on("mouseover", function(event, d) {{
                tooltip
                    .style("display", "block")
                    .html(`<strong>${{d.name}}</strong><br>Type: ${{d.type}}<br>ID: ${{d.id}}`);
            }})
            .on("mousemove", function(event) {{
                tooltip
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            }})
            .on("mouseout", function() {{
                tooltip.style("display", "none");
            }})
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        const labels = container.append("g")
            .attr("class", "node-labels")
            .selectAll("text")
            .data(data.nodes)
            .enter().append("text")
            .text(d => d.name.length > 20 ? d.name.substring(0, 17) + "..." : d.name)
            .attr("dx", 12)
            .attr("dy", 4);
        
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            linkLabels
                .attr("x", d => (d.source.x + d.target.x) / 2)
                .attr("y", d => (d.source.y + d.target.y) / 2);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            labels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
    </script>
</body>
</html>
"""
    
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print(f" Visualization saved to {output_html}")
    return output_html

def main():
    print(" Fetching graph data from Neo4j...")
    rows = fetch_edges(LIMIT)
    print(f" Retrieved {len(rows)} relationships")
    
    if not rows:
        print("  No data found in Neo4j. Please run load_to_neo4j.py first.")
        driver.close()
        return
    
    print(" Building visualization...")
    output_file = build_simple_viz(rows)
    
    # Open in browser
    abs_path = os.path.abspath(output_file)
    webbrowser.open(f'file://{abs_path}')
    print(f" Opening in browser...")
    
    driver.close()
    print(" Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        driver.close()
    except Exception as e:
        print(f" Error: {e}")
        import traceback
        traceback.print_exc()
        driver.close()