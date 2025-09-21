"""
Quick script to embed the ML knowledge base.

This script specifically processes the ML knowledge base file and creates embeddings.
"""

import sys
import os
from rich.console import Console
from rich.panel import Panel

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.query_handler import DeepResearcherAgent

# Initialize Rich console
console = Console()


def main():
    """Main function to embed the ML knowledge base."""
    console.print(Panel(
        "[bold blue]ML Knowledge Base Embedding[/bold blue]\n\n"
        "This script will embed the ML knowledge base for immediate use.",
        title="ML Knowledge Base Setup",
        border_style="blue"
    ))
    
    try:
        # Initialize agent
        console.print("[blue]Initializing Deep Researcher Agent...[/blue]")
        agent = DeepResearcherAgent(
            collection_name="ml_knowledge_base",
            persist_directory="./chroma_db_ml"
        )
        
        # Load ML knowledge base
        console.print("[blue]Loading ML knowledge base...[/blue]")
        with open('data/ml_basics_knowledge_base.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into documents
        documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        # Create metadata
        topics = [
            "Machine Learning Fundamentals",
            "Supervised Learning", 
            "Unsupervised Learning",
            "Reinforcement Learning",
            "Feature Engineering",
            "Model Evaluation and Validation",
            "Bias and Variance",
            "Overfitting and Underfitting",
            "Cross-Validation",
            "Hyperparameter Tuning",
            "Ensemble Methods",
            "Data Preprocessing",
            "Model Deployment"
        ]
        
        metadatas = []
        ids = []
        
        for i, doc in enumerate(documents):
            metadatas.append({
                "topic": topics[i] if i < len(topics) else f"ML Topic {i+1}",
                "source": "ml_basics_knowledge_base",
                "document_type": "educational_content",
                "domain": "machine_learning"
            })
            ids.append(f"ml_topic_{i+1}")
        
        # Store documents
        console.print(f"[blue]Embedding {len(documents)} ML documents...[/blue]")
        agent.load_documents(documents, metadatas, ids)
        
        # Get info
        info = agent.get_system_info()
        doc_count = info['vector_store_info'].get('document_count', 0)
        
        console.print(Panel(
            f"[bold green]ML Knowledge Base Embedded Successfully![/bold green]\n\n"
            f"Documents: {doc_count}\n"
            f"Collection: ml_knowledge_base\n"
            f"Directory: ./chroma_db_ml\n\n"
            "You can now use the ML knowledge base:\n"
            "  python main.py --collection ml_knowledge_base --persist-dir ./chroma_db_ml",
            title="Embedding Complete",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
