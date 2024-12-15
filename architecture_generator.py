from diagrams import Diagram, Cluster
from diagrams.programming.language import Python
from diagrams.programming.framework import FastAPI
from diagrams.onprem.database import PostgreSQL
from diagrams.generic.storage import Storage

def generate_architecture():
    """Generate architecture diagram for AdalFlow project."""
    
    with Diagram("AdalFlow Architecture", show=False, direction="TB"):
        with Cluster("Core Components"):
            core = Python("Core Engine")
            datasets = Python("Datasets")
            optim = Python("Optimization")
            eval_comp = Python("Evaluation")
        
        with Cluster("Infrastructure"):
            database = PostgreSQL("Database")
            tracing = Storage("Tracing")
            
        with Cluster("Components"):
            components = Python("Components")
            utils = Python("Utils")

        # Connect components
        core >> datasets
        core >> optim
        core >> eval_comp
        core >> components
        core >> database
        core >> tracing
        core >> utils
        
        # Component interactions
        components >> database
        optim >> tracing
        eval_comp >> tracing
        datasets >> database

if __name__ == "__main__":
    generate_architecture()