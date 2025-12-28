"""Command-line interface for the AI assistant."""
import argparse
import json
from typing import Optional, List

from .app import AIAssistant
from .config import config

def main(args: Optional[List[str]] = None) -> None:
    """Run the CLI.
    
    Args:
        args: Command-line arguments (for testing)
    """
    parser = argparse.ArgumentParser(description="AI Assistant CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start an interactive chat session")
    chat_parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="User ID (default: default)"
    )
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument(
        "query",
        type=str,
        help="The query to process"
    )
    query_parser.add_argument(
        "--user",
        type=str,
        default="default",
        help="User ID (default: default)"
    )
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old memories")
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Initialize the assistant
    assistant = AIAssistant()
    
    # Handle commands
    if parsed_args.command == "chat":
        _run_chat(assistant, parsed_args.user)
    elif parsed_args.command == "query":
        _run_query(assistant, parsed_args.query, parsed_args.user)
    elif parsed_args.command == "stats":
        _show_stats(assistant, parsed_args.format)
    elif parsed_args.command == "cleanup":
        _run_cleanup(assistant)
    else:
        parser.print_help()

def _run_chat(assistant: AIAssistant, user_id: str) -> None:
    """Run an interactive chat session."""
    print(f"AI Assistant {config['app.name']} v{config['app.version']}")
    print("Type 'exit' or 'quit' to end the session.")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            # Process the query
            result = assistant.process_query(user_input, user_id)
            print(f"\nAssistant: {result['response']}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}\n")

def _run_query(assistant: AIAssistant, query: str, user_id: str) -> None:
    """Run a single query and print the result."""
    try:
        result = assistant.process_query(query, user_id)
        print(f"Response: {result['response']}")
    except Exception as e:
        print(f"Error: {str(e)}")

def _show_stats(assistant: AIAssistant, format: str = "text") -> None:
    """Show memory statistics."""
    try:
        stats = assistant.get_memory_stats()
        
        if format == "json":
            print(json.dumps(stats, indent=2))
        else:
            print("\n=== Memory Statistics ===")
            print(f"Total memories: {stats.get('total_memories', 0)}")
            print("\nBy type:")
            for mem_type, count in stats.get('count_by_type', {}).items():
                print(f"  - {mem_type}: {count}")
                
            print(f"\nWith embeddings: {stats.get('with_embedding', 0)}")
            print(f"Without embeddings: {stats.get('without_embedding', 0)}")
            
            if 'oldest_memory' in stats and 'newest_memory' in stats:
                print(f"\nOldest memory: {stats['oldest_memory']}")
                print(f"Newest memory: {stats['newest_memory']}")
                
            print(f"\nTotal associations: {stats.get('total_associations', 0)}")
            
    except Exception as e:
        print(f"Error getting statistics: {str(e)}")

def _run_cleanup(assistant: AIAssistant) -> None:
    """Run memory cleanup and show results."""
    try:
        print("Cleaning up old memories...")
        result = assistant.cleanup_old_memories()
        
        print("\n=== Cleanup Results ===")
        print(f"Total memories found: {result.get('total_found', 0)}")
        print(f"Memories deleted: {result.get('deleted', 0)}")
        print(f"Associations removed: {result.get('associations_removed', 0)}")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()
