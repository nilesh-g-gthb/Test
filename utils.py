# utils.py
from langchain_core.messages import BaseMessage
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from typing import TypedDict, Annotated
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMessage(TypedDict):
    msg: BaseMessage
    sender: str
    message_type: str 
    answer: str

def reducer(a: list, b: list | str) -> list:
    if isinstance(b, list):
        return (a + b)[-10:]
    else:
        return a[-10:]

class AgentState(TypedDict):
    messages: Annotated[list[ChatMessage], reducer]

def get_llm():
    logger.info("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-360M-Instruct",
        padding_side='left',
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceTB/SmolLM-360M-Instruct",
        trust_remote_code=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=3,        # Very short output - just enough for one word
        do_sample=False,         # Deterministic output
        temperature=None,        # Remove temperature
        num_return_sequences=1,
        early_stopping=False,    # Disable early stopping
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    logger.info("Creating LangChain wrapper...")
    llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))
    return llm
