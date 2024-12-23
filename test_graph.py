# node_graph_visualizer.py

import os
from pyvis.network import Network
import streamlit as st
import networkx as nx
from jinja2 import Template


# Node class definition
class Node:
    def __init__(
        self,
        id,
        name,
        role_desc,
        data,
        data_id,
        previous_data,
        requires_opt,
        param_type,
        gradients,
    ):
        self.id = id
        self.name = name
        self.role_desc = role_desc
        self.data = data
        self.data_id = data_id
        self.previous_data = previous_data
        self.requires_opt = requires_opt
        self.param_type = param_type
        self.gradients = gradients

    def get_gradients_names(self):
        return self.gradients.split(", ") if self.gradients else []


# Function to generate individual HTML pages for each node
def generate_node_html(node, output_dir="node_pages"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{output_dir}/{node.name}.html"
    # from_param = Parameter("from", "From Parameter")
    # dummy_gradients = Gradient("dummy", "Dummy Gradient")
    with open(filename, "w") as file:
        file.write(
            f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{node.name}</title>
        </head>
        <body>
            <h1>Details for Node: {node.name}</h1>
            <p><b>ID:</b> {node.id}</p>
            <p><b>Role:</b> {node.role_desc}</p>
            <p><b>Data:</b> {node.data}</p>
            <p><b>Data ID:</b> {node.data_id}</p>
            <p><b>Previous Value:</b> {node.previous_data}</p>
            <p><b>Requires Optimization:</b> {node.requires_opt}</p>
            <p><b>Type:</b> {node.param_type}</p>
            <p><b>Gradients:</b> {', '.join(node.get_gradients_names())}</p>
        </body>
        </html>
        """
        )
    print(f"Generated HTML for node: {node.name} at {filename}")


# Function to create the main graph with clickable links to individual node pages
def create_graph_with_links(
    nodes, edges, output_dir="node_pages", main_file="graph.html"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    net = Network(height="750px", width="100%", directed=True)
    net.toggle_physics(True)
    net.template = Template(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis-network.min.css" rel="stylesheet" />
        <style>
            #tooltip {
                display: none;
                position: absolute;
                max-width: 300px;
                border: 1px solid #ccc;
                padding: 10px;
                background: white;
                z-index: 1000;
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
            #tooltip button {
                display: block;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div id="tooltip">
            <div id="tooltip-content"></div>
            <button onclick="document.getElementById('tooltip').style.display='none'">Close</button>
        </div>
        <div id="mynetwork" style="height: {{ height }};"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({{ nodes | safe }});
            var edges = new vis.DataSet({{ edges | safe }});
            var container = document.getElementById('mynetwork');
            var data = { nodes: nodes, edges: edges };
            var options = {{ options | safe }};
            var network = new vis.Network(container, data, options);

            // Handle node click to open a link
            network.on("click", function (params) {
                if (params.nodes.length > 0) {
                    const nodeId = params.nodes[0];
                    const node = nodes.get(nodeId);
                    if (node.url) {
                        window.open(node.url, '_blank');
                    }
                }
            });
        </script>
    </body>
    </html>
    """
    )

    for node in nodes:
        # Generate individual HTML pages for each node
        generate_node_html(node, output_dir)

        # Add node to the main graph with link to its HTML page
        net.add_node(
            node.id,
            label=node.name,
            title=f"<a href='{output_dir}/{node.name}.html' target='_blank'>Open Details</a>",
            shape="dot",
            size=15,
            url=f"{output_dir}/{node.name}.html",  # Add the URL here
        )

    for edge in edges:
        net.add_edge(edge[0].id, edge[1].id)

    net.show(main_file)
    print(f"Generated main graph HTML at {main_file}")


# Function to create a Streamlit app for interactive graph exploration
def create_interactive_streamlit_app(nodes, edges):
    G = nx.DiGraph()
    for node in nodes:
        G.add_node(node.id, node_obj=node)
    G.add_edges_from([(edge[0].id, edge[1].id) for edge in edges])

    st.title("Interactive Graph Visualization")
    st.sidebar.title("Node Selector")
    selected_node_name = st.sidebar.selectbox(
        "Select a node", [node.name for node in nodes]
    )

    net = Network(height="500px", width="100%", directed=True)
    net.template = Template(
        """
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis-network.min.css" rel="stylesheet" />
        <style>
            #tooltip {
                display: none;
                position: absolute;
                max-width: 300px;
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ccc;
                padding: 10px;
                background: white;
                z-index: 1000;
                font-family: Arial, sans-serif;
                font-size: 12px;
                line-height: 1.5;
            }
        </style>
    </head>
    <body>
        <div id="tooltip"></div>
        <div id="mynetwork" style="height: {{ height }};"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({{ nodes | safe }});
            var edges = new vis.DataSet({{ edges | safe }});
            var container = document.getElementById('mynetwork');
            var data = { nodes: nodes, edges: edges };
            var options = {{ options | safe }};
            var network = new vis.Network(container, data, options);

            // Tooltip functionality
            const tooltip = document.getElementById('tooltip');

            network.on("hoverNode", function (params) {
                const node = nodes.get(params.node);
                tooltip.innerHTML = node.title;
                tooltip.style.display = "block";
                tooltip.style.left = params.event.pointer.DOM.x + "px";
                tooltip.style.top = params.event.pointer.DOM.y + "px";
            });

            network.on("blurNode", function () {
                // Keep tooltip visible for persistence
                tooltip.style.display = "block";
            });

            // Hide tooltip only on outside click
            document.addEventListener("click", function (event) {
                if (!tooltip.contains(event.target) && event.target.id !== 'tooltip') {
                    tooltip.style.display = "none";
                }
            });
        </script>
    </body>
    </html>
    """
    )

    for node in nodes:
        net.add_node(node.id, label=node.name)
    for edge in edges:
        net.add_edge(edge[0].id, edge[1].id)

    net.save_graph("graph.html")
    st.components.v1.html(open("graph.html", "r").read(), height=550)

    if selected_node_name:
        selected_node = next(node for node in nodes if node.name == selected_node_name)
        st.subheader(f"Details for Node: {selected_node.name}")
        st.write(f"**ID**: {selected_node.id}")
        st.write(f"**Role**: {selected_node.role_desc}")
        st.write(f"**Data**: {selected_node.data}")
        st.write(f"**Data ID**: {selected_node.data_id}")
        st.write(f"**Previous Value**: {selected_node.previous_data}")
        st.write(f"**Requires Optimization**: {selected_node.requires_opt}")
        st.write(f"**Type**: {selected_node.param_type}")
        st.write(f"**Gradients**: {', '.join(selected_node.get_gradients_names())}")


if __name__ == "__main__":
    # Dummy data
    dummy_nodes = [
        Node(1, "Node1", "Input", "Value1", "D1", "Prev1", "Yes", "Type1", "Grad1"),
        Node(2, "Node2", "Process", "Value2", "D2", "Prev2", "No", "Type2", "Grad2"),
        Node(3, "Node3", "Output", "Value3", "D3", "Prev3", "Yes", "Type3", "Grad3"),
    ]

    dummy_edges = [(dummy_nodes[0], dummy_nodes[1]), (dummy_nodes[1], dummy_nodes[2])]

    # Test HTML generation
    create_graph_with_links(dummy_nodes, dummy_edges)

    # Uncomment the following line to test the Streamlit app
    # create_interactive_streamlit_app(dummy_nodes, dummy_edges)
