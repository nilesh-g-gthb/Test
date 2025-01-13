# supervisor.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import logging
from utils import get_llm, AgentState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """Classify as QuoteRequest, BondRequest, or GENERAL:

Rules:
QuoteRequest = price/bid/offer questions
BondRequest = bond info questions
GENERAL = other questions

Input: {message}
Answer with one word only:"""

class Supervisor:
    def __init__(self, llm):
        logger.info("Initializing Supervisor...")
        self.prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
        self.chain = self.prompt | llm
        logger.info("Supervisor initialized successfully")

    def __call__(self, state: AgentState) -> AIMessage:
        last_message = state['messages'][-1]
        output = self.invoke(last_message)
        last_message["message_type"] = output
        return {'messages': []}

    def invoke(self, message: str) -> str:
        try:
            logger.info(f"Processing message: {message}")
            # Pre-classify based on keywords
            message_lower = message.lower()
            if any(word in message_lower for word in ['price', 'bid', 'offer', 'quote', 'rate']):
                return "QuoteRequest"
            elif any(word in message_lower for word in ['about', 'information', 'details', 'track record', 'rating']):
                return "BondRequest"
            
            # If no keywords match, try the model
            output = self.chain.invoke({"message": message})
            cleaned_output = output.content.strip().upper()
            
            if "QUOTE" in cleaned_output:
                return "QuoteRequest"
            elif "BOND" in cleaned_output:
                return "BondRequest"
            return "GENERAL"
            
        except Exception as e:
            logger.error(f"Error in invoke method: {e}")
            return "GENERAL"

def tester():
    print("Starting Fixed Income Query Classifier...")
    try:
        print("Initializing classifier...")
        agent = Supervisor(get_llm())
        print("Classifier initialized successfully!")
        print("\nCategories: QuoteRequest, BondRequest, GENERAL")
        print("\nExample queries:")
        print("- 'What's the price of HDB bonds?'")
        print("- 'Tell me about AP state bonds'")
        print("- 'Show bid for 8.3774% HDB Financial Apr 26'")
        
        while True:
            try:
                user_input = input("\nEnter your query (Ctrl+C to exit): ")
                if not user_input.strip():
                    continue
                print("Processing query...")
                classification = agent.invoke(user_input)
                print(f"Classification: {classification}")
            except KeyboardInterrupt:
                print("\nExiting the program...")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Defaulting to GENERAL classification")
    except Exception as e:
        print(f"Error initializing the classifier: {e}")
        print("Please check if the model is properly downloaded.")

if __name__ == "__main__":
    tester()
