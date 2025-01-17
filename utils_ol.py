# utils.py

import sys
from typing import Optional
import ollama

# Centralized model configuration
LLM_MODEL = 'smollm:latest'  # Change this to switch models 

def initialize_llm() -> None:
    """Initialize the LLM model."""
    try:
        print(f"Pulling {LLM_MODEL} model... This might take a few minutes on first run.")
        ollama.pull(LLM_MODEL)
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing LLM: {str(e)}")
        sys.exit(1)

def get_llm_response(prompt: str) -> Optional[str]:
    """Get response from LLM with error handling."""
    try:
        response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        response_text = response['message']['content'].strip()
        
        # If the response contains 'Output:', extract just that part
        if 'Output:' in response_text:
            response_text = response_text.split('Output:')[-1].strip()
            
        # If still no clear classification, look for the three valid types
        valid_types = ['QuoteRequest', 'BondRequest', 'GENERAL']
        for type_ in valid_types:
            if type_ in response_text:
                return type_
                
        return "GENERAL"  # Default fallback
    except Exception as e:
        print(f"Error getting LLM response: {str(e)}")
        return "GENERAL"  # Fallback response in case of error