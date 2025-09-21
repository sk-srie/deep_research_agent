"""
Test script to verify the ML knowledge base setup.
"""

import sys
import os
from rich.console import Console
from rich.panel import Panel

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

console = Console()

def test_ml_knowledge_base():
    """Test if the ML knowledge base is properly set up."""
    console.print("[bold blue]Testing ML Knowledge Base Setup...[/bold blue]")
    
    try:
        from src.query_handler import DeepResearcherAgent
        
        # Initialize agent with ML knowledge base
        agent = DeepResearcherAgent(
            collection_name="ml_knowledge_base",
            persist_directory="./chroma_db_ml"
        )
        
        # Check if knowledge base is loaded
        info = agent.get_system_info()
        doc_count = info['vector_store_info'].get('document_count', 0)
        
        if doc_count == 0:
            console.print(Panel(
                "[yellow]ML Knowledge Base not found![/yellow]\n\n"
                "To set up the knowledge base, run:\n"
                "  python embed_ml_knowledge.py",
                title="Setup Required",
                border_style="yellow"
            ))
            return False
        
        console.print(f"[green]✓ ML Knowledge Base loaded ({doc_count} documents)[/green]")
        
        # Test a simple query
        console.print("[blue]Testing with sample query...[/blue]")
        result = agent.handle_query("What is machine learning?")
        
        console.print(f"[green]✓ Query processed successfully![/green]")
        console.print(f"[green]✓ Confidence: {result.confidence_score:.3f}[/green]")
        console.print(f"[green]✓ Documents retrieved: {result.retrieved_documents_count}[/green]")
        
        console.print(Panel(
            "[bold green]ML Knowledge Base is ready![/bold green]\n\n"
            "You can now use:\n"
            "  python main.py --query 'What is machine learning?'\n"
            "  python main.py  # Interactive mode",
            title="Setup Complete",
            border_style="green"
        ))
        
        return True
        
    except Exception as e:
        console.print(f"[red]✗ Error: {e}[/red]")
        return False

if __name__ == "__main__":
    test_ml_knowledge_base()
