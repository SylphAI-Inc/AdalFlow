from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.database import PostgreSQL
from diagrams.generic.storage import Storage

def generate_architecture():
    """Generate architecture diagram for AdalFlow project."""
    
    graph_attr = {
        "fontsize": "30",
        "bgcolor": "white",
        "splines": "ortho",
        "pad": "0.5"
    }
    
    node_attr = {
        "fontsize": "14"
    }
    
    with Diagram(
        "AdalFlow Architecture",
        show=False,
        direction="TB",
        graph_attr=graph_attr,
        node_attr=node_attr,
        filename="architecture",
        outformat="png"
    ):
        with Cluster("Core"):
            core = Python("Core Engine")
            
        with Cluster("Data Processing"):
            datasets = Python("Datasets")
            optim = Python("Optimization")
            eval_comp = Python("Evaluation")
            
        with Cluster("Infrastructure"):
            database = PostgreSQL("Database")
            tracing = Storage("Tracing")
            
        with Cluster("Components"):
            components = Python("Components")
            utils = Python("Utils")

        # Core connections
        core >> Edge(color="darkgreen") >> datasets
        core >> Edge(color="darkgreen") >> optim
        core >> Edge(color="darkgreen") >> eval_comp
        core >> Edge(color="darkblue") >> components
        
        # Infrastructure connections
        components >> Edge(color="red") >> database
        datasets >> Edge(color="red") >> database
        
        # Tracing connections
        optim >> Edge(color="orange") >> tracing
        eval_comp >> Edge(color="orange") >> tracing
        
        # Utils connections
        utils >> Edge(style="dotted") >> components
        utils >> Edge(style="dotted") >> core

if __name__ == "__main__":
    generate_architecture()