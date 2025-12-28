"""Test script for the AI Assistant with improved test cases and output formatting."""
from src.app import AIAssistant
import time
from typing import Dict, Any

def format_response(response: Dict[str, Any]) -> str:
    """Format the assistant's response for better readability."""
    if not isinstance(response, dict):
        return str(response)
    
    # Get the main response text
    response_text = response.get('response', 'No response generated')
    
    # Add context if available
    if 'context' and isinstance(response.get('context'), dict):
        context = response['context']
        if 'relevant_memories' in context and context['relevant_memories']:
            response_text += "\n\n[Context from memory]"
            for i, memory in enumerate(context['relevant_memories'], 1):
                if isinstance(memory, dict):
                    content = memory.get('content', '').strip()
                    if content:
                        response_text += f"\n- {content[:100]}{'...' if len(content) > 100 else ''}"
    
    return response_text

def test_text_queries(assistant: AIAssistant, queries: list[str]) -> None:
    """Test the assistant with a list of text queries."""
    print("\n" + "="*50)
    print("Starting Test Session")
    print("="*50)
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Test {i}/{len(queries)}]")
        print(f"You: {query}")
        
        try:
            start_time = time.time()
            response = assistant.process_query(query)
            response_time = time.time() - start_time
            
            print(f"\nAssistant (in {response_time:.2f}s):")
            print("-" * 30)
            print(format_response(response))
            print("-" * 30)
            
            # Add a small delay between queries
            time.sleep(1)
            
        except Exception as e:
            print(f"\nError processing query: {str(e)}")
            import traceback
            traceback.print_exc()
            break

def main():
    """Main test function."""
    test_queries = [
        "What's the weather like today?",
        "Can you explain quantum computing in simple terms?",
        "Suggest three interesting facts about ancient Egypt",
        "How would you solve a Rubik's cube?",
        "What are some good programming practices?",
        "Tell me a joke about technology",
        "What's the latest news about artificial intelligence?"
    ]
    
    try:
        print("Initializing AI Assistant...")
        assistant = AIAssistant()
        print("Assistant initialized successfully!")
        
        test_text_queries(assistant, test_queries)
        
    except Exception as e:
        print(f"Failed to initialize assistant: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())