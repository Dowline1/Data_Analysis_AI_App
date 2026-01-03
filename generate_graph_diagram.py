"""
Generate LangGraph workflow visualization diagram.

This script creates a visual representation of the multi-agent workflow
showing all nodes, edges, and conditional routing.
"""

import os
from src.graph.workflow import create_workflow

def generate_diagram():
    """Generate and save the workflow diagram."""
    
    # Create the workflow
    workflow = create_workflow()
    
    # Generate PNG diagram
    try:
        # Get the graph visualization
        png_data = workflow.get_graph().draw_mermaid_png()
        
        # Save to file
        output_path = "docs/workflow_diagram.png"
        os.makedirs("docs", exist_ok=True)
        
        with open(output_path, "wb") as f:
            f.write(png_data)
        
        print(f"Workflow diagram saved to: {output_path}")
        
    except Exception as e:
        print(f"Error generating PNG: {e}")
        print("Generating Mermaid code instead...")
        
        # Fallback to mermaid code
        mermaid = workflow.get_graph().draw_mermaid()
        
        output_path = "docs/workflow_diagram.mmd"
        os.makedirs("docs", exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(mermaid)
        
        print(f"Mermaid diagram code saved to: {output_path}")
        print("\nYou can visualize this at: https://mermaid.live/")

if __name__ == "__main__":
    generate_diagram()
