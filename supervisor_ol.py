
# supervisor.py

from utils_ol import initialize_llm, get_llm_response

# Fixed Income Classification Instructions
CLASSIFICATION_INSTRUCTIONS = """Instruction: You are a supervisor tasked with managing the chat messages with queries about Fixed Income instruments, an AI assistant.
Your task is to classify the incoming questions. 
Depending on your answer, question will be routed to the right team, so your task is crucial for our team. 
There are 3 possible question types: 
- QuoteRequest - questions related to price, bid, offer for a given Fixed Income security.
- BondRequest - questions related to information about a Fixed Income security.
- GENERAL - general questions
Return in the output only one word (QuoteRequest, BondRequest or  GENERAL).

Examples:
Chat Message: 8.3774% HDB Financial Apr 26 INE756I07ER5 Qtm: 1 Cr Offer please
Output:QuoteRequest
Chat Message: can you please share a brief note about AP state bonds?
Output:BondRequest
Chat Message: How is the track record of profitability?
Output:BondRequest

Chat Message: """

def create_prompt(user_input: str) -> str:
    """Create the complete prompt by combining the instruction and user input."""
    return CLASSIFICATION_INSTRUCTIONS + user_input

def main():
    """Main function to run the chatbot."""
    print("Fixed Income Query Classification Bot (type 'exit' to quit)")
    print("-" * 50)
    
    # Initialize LLM
    try:
        initialize_llm()
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        print("Falling back to direct classification without initialization")
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter your query: ").strip()
            
            # Check for exit command
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            # Skip empty inputs
            if not user_input:
                continue
            
            # Create prompt and get response
            prompt = create_prompt(user_input)
            response = get_llm_response(prompt)
            
            # Print response
            print(f"\nClassification: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nGENERAL")  # Fallback response for any unexpected errors
            continue

if __name__ == "__main__":
    main()