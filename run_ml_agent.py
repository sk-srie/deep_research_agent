"""
Quick start script for the Deep Researcher Agent with ML Knowledge Base.

This script provides an easy way to run the agent with the ML knowledge base
without needing to remember the collection and directory parameters.
"""

import sys
import os
import argparse
from rich.console import Console
from rich.panel import Panel

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.query_handler import DeepResearcherAgent, QueryConfig

# Initialize Rich console
console = Console()


def main():
    """Main function to run the ML agent."""
    parser = argparse.ArgumentParser(
        description="Deep Researcher Agent with ML Knowledge Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ml_agent.py                                    # Interactive mode
  python run_ml_agent.py --query "What is machine learning?" # Single query
  python run_ml_agent.py --query "Explain overfitting" --export report.pdf
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="ML research query to process"
    )
    
    parser.add_argument(
        "--export", "-e",
        type=str,
        help="Export results to file (markdown or pdf)"
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum number of reasoning steps (default: 5)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize agent with ML knowledge base
        console.print("[bold blue]Initializing Deep Researcher Agent with ML Knowledge Base...[/bold blue]")
        
        agent = DeepResearcherAgent(
            collection_name="ml_basics_kb",
            persist_directory="./chroma_db_ml"
        )
        
        # Check if knowledge base is loaded
        info = agent.get_system_info()
        doc_count = info['vector_store_info'].get('document_count', 0)
        
        if doc_count == 0:
            console.print(Panel(
                "[red]ML Knowledge Base not found![/red]\n\n"
                "Please run the setup script first:\n"
                "  python setup_knowledge_base.py",
                title="Setup Required",
                border_style="red"
            ))
            return 1
        
        console.print(f"[green]✓ ML Knowledge Base loaded ({doc_count} documents)[/green]")
        
        if args.query:
            # Single query mode
            console.print(f"[bold blue]Processing query: {args.query}[/bold blue]")
            
            config = QueryConfig(max_reasoning_steps=args.max_steps)
            result = agent.handle_query(args.query, config)
            
            # Display results
            console.print(Panel(
                f"[bold blue]Query:[/bold blue] {result.query}\n\n"
                f"[bold green]Confidence Score:[/bold green] {result.confidence_score:.3f}\n"
                f"[bold green]Documents Retrieved:[/bold green] {result.retrieved_documents_count}\n"
                f"[bold green]Reasoning Steps:[/bold green] {len(result.reasoning_steps)}",
                title="Research Summary",
                border_style="blue"
            ))
            
            console.print(Panel(
                result.final_summary,
                title="Executive Summary",
                border_style="green"
            ))
            
            # Export if specified
            if args.export:
                if args.export.endswith('.pdf'):
                    agent.export_to_pdf(result, args.export)
                else:
                    agent.export_to_markdown(result, args.export)
                console.print(f"[green]✓ Results exported to: {args.export}[/green]")
        
        else:
            # Interactive mode
            console.print(Panel(
                "[bold blue]Deep Researcher Agent - ML Knowledge Base Mode[/bold blue]\n\n"
                "Ask questions about machine learning topics:\n"
                "• What is machine learning?\n"
                "• Explain supervised vs unsupervised learning\n"
                "• What is overfitting?\n"
                "• Describe the bias-variance tradeoff\n"
                "• What are ensemble methods?\n"
                "• How does cross-validation work?\n"
                "• What is feature engineering?\n"
                "• Explain hyperparameter tuning\n\n"
                "Commands:\n"
                "• Type your ML question to get started\n"
                "• Type 'history' to see query history\n"
                "• Type 'export <format>' to export last result\n"
                "• Type 'info' to see system information\n"
                "• Type 'quit' to exit",
                title="ML Knowledge Base Ready",
                border_style="blue"
            ))
            
            # Import and run interactive mode
            from main import interactive_mode
            interactive_mode(agent)
    
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
