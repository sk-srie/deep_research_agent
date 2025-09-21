"""
Create embeddings for all text files in the data folder.

This script processes all text files in the data directory, generates embeddings,
and stores them in Chroma DB for use with the Deep Researcher Agent.
"""

import os
import sys
from pathlib import Path
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.embeddings import EmbeddingGenerator
from src.vectorstore import ChromaVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


def load_text_files(data_dir: str = "data"):
    """
    Load all text files from the data directory.
    
    Args:
        data_dir (str): Directory containing text files
    
    Returns:
        tuple: (documents, metadatas, ids)
    """
    console.print(f"[bold blue]Loading text files from {data_dir}/...[/bold blue]")
    
    documents = []
    metadatas = []
    ids = []
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        console.print(f"[red]✗ Data directory {data_dir} not found[/red]")
        return documents, metadatas, ids
    
    # Find all text files
    text_files = list(data_path.glob("*.txt")) + list(data_path.glob("*.md"))
    
    if not text_files:
        console.print(f"[yellow]No text files found in {data_dir}[/yellow]")
        return documents, metadatas, ids
    
    console.print(f"[green]Found {len(text_files)} text files[/green]")
    
    for file_path in text_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if not content:
                console.print(f"[yellow]Skipping empty file: {file_path.name}[/yellow]")
                continue
            
            # Split content into documents (by double newlines or sections)
            file_docs = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
            
            # If no double newlines, treat entire file as one document
            if len(file_docs) == 1 and '\n' in content:
                # Try splitting by single newlines if content is long
                if len(content) > 500:
                    file_docs = [doc.strip() for doc in content.split('\n') if doc.strip()]
            
            # Add each document from the file
            for i, doc in enumerate(file_docs):
                if len(doc) > 50:  # Only include substantial documents
                    documents.append(doc)
                    metadatas.append({
                        "source_file": file_path.name,
                        "document_index": i,
                        "file_path": str(file_path),
                        "document_type": "text_content",
                        "domain": "general_knowledge"
                    })
                    ids.append(f"{file_path.stem}_{i}")
            
            console.print(f"[green]✓ Loaded {len(file_docs)} documents from {file_path.name}[/green]")
            
        except Exception as e:
            console.print(f"[red]✗ Error loading {file_path.name}: {e}[/red]")
    
    console.print(f"[green]✓ Total documents loaded: {len(documents)}[/green]")
    return documents, metadatas, ids


def create_embeddings_and_store(documents, metadatas, ids, 
                               collection_name="documents", 
                               persist_directory="./chroma_db",
                               embedding_model="all-MiniLM-L6-v2"):
    """
    Create embeddings for documents and store them in Chroma DB.
    
    Args:
        documents: List of documents
        metadatas: List of metadata dictionaries
        ids: List of document IDs
        collection_name: Name of Chroma collection
        persist_directory: Directory to persist Chroma DB
        embedding_model: Name of embedding model to use
    """
    console.print(f"[bold blue]Creating embeddings and storing in Chroma DB...[/bold blue]")
    
    try:
        # Initialize embedding generator
        console.print(f"[blue]Initializing embedding model: {embedding_model}[/blue]")
        embedding_generator = EmbeddingGenerator(embedding_model)
        
        # Initialize vector store
        console.print(f"[blue]Initializing Chroma DB at: {persist_directory}[/blue]")
        vector_store = ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
        
        # Generate embeddings and store
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing documents...", total=len(documents))
            
            # Store documents in batches for better performance
            batch_size = 10
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                # Store batch
                vector_store.store_embeddings(batch_docs, batch_metadatas, batch_ids)
                
                # Update progress
                progress.update(task, advance=len(batch_docs))
                progress.update(task, description=f"Processed {min(i+batch_size, len(documents))}/{len(documents)} documents")
        
        # Get collection info
        info = vector_store.get_collection_info()
        console.print(f"[green]✓ Embeddings created and stored successfully![/green]")
        console.print(f"[green]✓ Total documents in collection: {info.get('document_count', 0)}[/green]")
        
        return vector_store
        
    except Exception as e:
        console.print(f"[red]✗ Error creating embeddings: {e}[/red]")
        return None


def test_embeddings(vector_store, test_queries=None):
    """
    Test the created embeddings with sample queries.
    
    Args:
        vector_store: ChromaVectorStore instance
        test_queries: List of test queries
    """
    if test_queries is None:
        test_queries = [
            "What is machine learning?",
            "Explain artificial intelligence",
            "How does deep learning work?",
            "What are neural networks?",
            "Describe data preprocessing"
        ]
    
    console.print(f"\n[bold blue]Testing embeddings with {len(test_queries)} sample queries...[/bold blue]")
    
    results_table = Table(title="Embedding Test Results")
    results_table.add_column("Query", style="cyan")
    results_table.add_column("Documents Found", style="green")
    results_table.add_column("Top Similarity", style="yellow")
    results_table.add_column("Status", style="white")
    
    for query in test_queries:
        try:
            results = vector_store.retrieve_from_chroma(query, n_results=3)
            
            if results:
                top_similarity = max([r.get('similarity', 0) for r in results])
                status = "✓ Good" if top_similarity > 0.7 else "⚠ Fair" if top_similarity > 0.5 else "✗ Poor"
                results_table.add_row(
                    query[:50] + "..." if len(query) > 50 else query,
                    str(len(results)),
                    f"{top_similarity:.3f}",
                    status
                )
            else:
                results_table.add_row(query[:50] + "..." if len(query) > 50 else query, "0", "0.000", "✗ No results")
                
        except Exception as e:
            results_table.add_row(query[:50] + "..." if len(query) > 50 else query, "0", "0.000", f"✗ Error: {str(e)[:20]}")
    
    console.print(results_table)


def main():
    """Main function to create embeddings for all data files."""
    console.print(Panel(
        "[bold blue]Deep Researcher Agent - Embedding Creation[/bold blue]\n\n"
        "This script will:\n"
        "1. Load all text files from the data/ directory\n"
        "2. Generate embeddings using sentence-transformers\n"
        "3. Store embeddings in Chroma DB\n"
        "4. Test the embeddings with sample queries",
        title="Embedding Creation Process",
        border_style="blue"
    ))
    
    # Check if data directory exists
    if not os.path.exists("data"):
        console.print("[red]✗ Data directory not found![/red]")
        console.print("[yellow]Please ensure the 'data' directory exists with text files[/yellow]")
        return 1
    
    # Load text files
    documents, metadatas, ids = load_text_files("data")
    
    if not documents:
        console.print("[red]✗ No documents to process[/red]")
        return 1
    
    # Create embeddings and store
    vector_store = create_embeddings_and_store(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        collection_name="data_documents",
        persist_directory="./chroma_db_data"
    )
    
    if vector_store is None:
        console.print("[red]✗ Failed to create embeddings[/red]")
        return 1
    
    # Test embeddings
    test_embeddings(vector_store)
    
    console.print(Panel(
        "[bold green]Embedding Creation Complete![/bold green]\n\n"
        "Your embeddings are now ready to use:\n\n"
        "Collection: data_documents\n"
        "Directory: ./chroma_db_data\n"
        f"Documents: {len(documents)}\n\n"
        "To use with the Deep Researcher Agent:\n"
        "  python main.py --collection data_documents --persist-dir ./chroma_db_data\n\n"
        "Or use the quick start script:\n"
        "  python run_ml_agent.py --collection data_documents --persist-dir ./chroma_db_data",
        title="Setup Complete",
        border_style="green"
    ))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
