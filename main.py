"""
Deep Researcher Agent - Main Entry Point

This is the main entry point for the Deep Researcher Agent application.
It provides a CLI interface for loading documents and processing research queries.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import logging
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.query_handler import DeepResearcherAgent, QueryConfig, QueryResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()


def load_documents_from_file(file_path: str) -> List[str]:
    """Load documents from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines or other delimiters
        documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        
        if not documents:
            # If no double newlines, split by single newlines
            documents = [line.strip() for line in content.split('\n') if line.strip()]
        
        return documents
    except Exception as e:
        console.print(f"[red]Error loading file {file_path}: {e}[/red]")
        return []


def display_query_result(result: QueryResult) -> None:
    """Display query result in a formatted way."""
    # Display summary
    console.print(Panel(
        f"[bold blue]Query:[/bold blue] {result.query}\n\n"
        f"[bold green]Confidence Score:[/bold green] {result.confidence_score:.3f}\n"
        f"[bold green]Documents Retrieved:[/bold green] {result.retrieved_documents_count}\n"
        f"[bold green]Timestamp:[/bold green] {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
        title="Research Summary",
        border_style="blue"
    ))
    
    # Display final summary
    console.print(Panel(
        result.final_summary,
        title="Executive Summary",
        border_style="green"
    ))


def interactive_mode(agent: DeepResearcherAgent) -> None:
    """Run the agent in interactive mode."""
    console.print(Panel(
        "[bold blue]Deep Researcher Agent - Interactive Mode[/bold blue]\n\n"
        "Commands:\n"
        "• Type your research query to get started\n"
        "• Type 'history' to see query history\n"
        "• Type 'export <format>' to export last result (markdown/pdf)\n"
        "• Type 'info' to see system information\n"
        "• Type 'quit' to exit",
        title="Welcome",
        border_style="blue"
    ))
    
    last_result: Optional[QueryResult] = None
    
    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Enter your research query[/bold cyan]")
            
            if query.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            elif query.lower() == 'history':
                history = agent.get_query_history()
                if history:
                    console.print(f"\n[bold]Query History ({len(history)} queries):[/bold]")
                    for i, result in enumerate(history[-5:], 1):  # Show last 5
                        console.print(f"{i}. {result.query} (Confidence: {result.confidence_score:.3f})")
                else:
                    console.print("[yellow]No query history available[/yellow]")
                continue
            
            elif query.lower().startswith('export '):
                if not last_result:
                    console.print("[red]No previous result to export[/red]")
                    continue
                
                format_type = query.split(' ', 1)[1].lower()
                if format_type in ['markdown', 'md']:
                    output_path = f"research_report_{last_result.timestamp.strftime('%Y%m%d_%H%M%S')}.md"
                    agent.export_to_markdown(last_result, output_path)
                    console.print(f"[green]Report exported to: {output_path}[/green]")
                elif format_type == 'pdf':
                    output_path = f"research_report_{last_result.timestamp.strftime('%Y%m%d_%H%M%S')}.pdf"
                    agent.export_to_pdf(last_result, output_path)
                    console.print(f"[green]Report exported to: {output_path}[/green]")
                else:
                    console.print("[red]Supported formats: markdown, pdf[/red]")
                continue
            
            elif query.lower() == 'info':
                info = agent.get_system_info()
                console.print(Panel(
                    f"[bold]Embedding Model:[/bold] {info['embedding_model']}\n"
                    f"[bold]Summarization Model:[/bold] {info['summarization_model']}\n"
                    f"[bold]Collection Name:[/bold] {info['collection_name']}\n"
                    f"[bold]Documents in Store:[/bold] {info['vector_store_info'].get('document_count', 0)}\n"
                    f"[bold]Query History:[/bold] {info['query_history_count']} queries",
                    title="System Information",
                    border_style="green"
                ))
                continue
            
            # Process the query
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Processing query...", total=None)
                
                try:
                    result = agent.handle_query(query)
                    last_result = result
                    progress.update(task, description="Query processed successfully!")
                    
                except Exception as e:
                    console.print(f"[red]Error processing query: {e}[/red]")
                    continue
            
            # Display results
            display_query_result(result)
            
            # Ask for refinement
            if Confirm.ask("\n[bold yellow]Would you like to refine this query?[/bold yellow]"):
                refinement = Prompt.ask("[bold cyan]Enter your refinement[/bold cyan]")
                try:
                    refined_result = agent.interactive_query_refinement(result, refinement)
                    last_result = refined_result
                    console.print("\n[bold green]Refined Results:[/bold green]")
                    display_query_result(refined_result)
                except Exception as e:
                    console.print(f"[red]Error in refinement: {e}[/red]")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Unexpected error: {e}[/red]")


def main():
    """Main function to run the Deep Researcher Agent."""
    parser = argparse.ArgumentParser(
        description="Deep Researcher Agent - AI-powered research assistant"
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Research query to process"
    )
    
    parser.add_argument(
        "--load", "-l",
        type=str,
        help="Load documents from file or directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file for results (markdown or pdf)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of reasoning steps (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize agent
        console.print("[bold blue]Initializing Deep Researcher Agent...[/bold blue]")
        agent = DeepResearcherAgent()
        
        # Load documents if specified
        if args.load:
            console.print(f"[bold blue]Loading documents from: {args.load}[/bold blue]")
            documents = load_documents_from_file(args.load)
            if documents:
                agent.load_documents(documents)
                console.print(f"[green]Successfully loaded {len(documents)} documents[/green]")
            else:
                console.print("[yellow]No documents loaded[/yellow]")
        
        # Process query or enter interactive mode
        if args.query:
            # Single query mode
            console.print(f"[bold blue]Processing query: {args.query}[/bold blue]")
            
            config = QueryConfig(max_reasoning_steps=args.max_steps)
            result = agent.handle_query(args.query, config)
            
            # Display results
            display_query_result(result)
            
            # Export if specified
            if args.output:
                if args.output.endswith('.pdf'):
                    agent.export_to_pdf(result, args.output)
                else:
                    agent.export_to_markdown(result, args.output)
                console.print(f"[green]Results exported to: {args.output}[/green]")
        
        else:
            # Interactive mode
            interactive_mode(agent)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Unexpected error in main")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())