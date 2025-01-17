# utils.py

import sys
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Centralized model configuration
MODEL_ID = "HuggingFaceTB/SmolLM-1.7B-Instruct"  # Change this to any HF model you want to use
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LLMHandler:
    def __init__(self):
        self.pipe = None

    def initialize_llm(self) -> None:
        """Initialize the LLM model."""
        try:
            print(f"Loading model {MODEL_ID}... This might take a few minutes on first run.")
            self.pipe = pipeline(
                "text-generation",
                model=MODEL_ID,
                device=DEVICE,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            )
            print("Model initialized successfully!")
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}")
            sys.exit(1)

    def get_llm_response(self, prompt: str) -> Optional[str]:
        """Get response from LLM with error handling."""
        try:
            # Ensure pipeline is initialized
            if self.pipe is None:
                self.initialize_llm()

            # Generate response using pipeline
            response = self.pipe(
                prompt,
                max_length=len(prompt) + 100,  # Adjust based on your needs
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
            
            response_text = response[0]['generated_text']
            
            # Remove the prompt from the response
            if response_text.startswith(prompt):
                response_text = response_text[len(prompt):].strip()
            
            # If the response contains 'Output:', extract just that part
            if 'Output:' in response_text:
                response_text = response_text.split('Output:')[-1].strip()
            
            # Look for valid classification types
            valid_types = ['QuoteRequest', 'BondRequest', 'GENERAL']
            for type_ in valid_types:
                if type_ in response_text:
                    return type_
            
            return "GENERAL"  # Default fallback
            
        except Exception as e:
            print(f"Error getting LLM response: {str(e)}")
            return "GENERAL"  # Fallback response in case of error

