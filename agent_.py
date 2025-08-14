from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.memory import Memory
from llama_index.core.chat_engine import SimpleChatEngine
import os
from dotenv import load_dotenv                                                    # load variables from .env file otherwise Github dont allow to push sensitive info like API keys..
load_dotenv() 
ME_M0_API_KEY = os.getenv("MeM0_API_KEY")                                         # Memory API key
MODEL_NAME = "llama-3.3-70b-versatile"
API_KEY = os.getenv("GROQ_API_KEY")                                               #Groq API key
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
KNOWLEDGE_BASE = "./budget_data/"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
OUTPUT_TOKENS = 512

def get_llm(model_name, api_key):
    return Groq(model=model_name, api_key=api_key, temperature=0.5)

def initialize_settings():
    Settings.llm = get_llm(MODEL_NAME, API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_output = OUTPUT_TOKENS
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# context = {"user_id": "test_user_1"}  #userask que from agent
# # memory_from_client = Memory.from_client(
# #     context=context,
# #     api_key=os.environ["MeM0_API_KEY"],
# #     search_msg_limit=5,       #default is 5
# # )
# memory_from_client = Memory(
#     context=context,  # initial context
#     # If you want to track conversation history, this is handled internally
# )

context = {"user_id": "test_user_1"}

memory_from_client = Memory(
    context=context,
    token_limit=512  # required in latest version
)

def multiply(a, b):
    return a * b

def add(a, b):
    return a + b

multiplication_tool = FunctionTool.from_defaults(fn=multiply)
addition_tool = FunctionTool.from_defaults(fn=add)

initialize_settings()

def load_index(folder_path): #Load documents as RAG because agent uses this and give answers
    documents = SimpleDirectoryReader(folder_path).load_data()
    index = VectorStoreIndex(
        documents, 
        embed_model=Settings.embed_model, 
        llm=Settings.llm
    )
    index.storage_context.persist()
    return index.as_query_engine()

query_engine = load_index(KNOWLEDGE_BASE)
budget_tool = QueryEngineTool.from_defaults(
    query_engine,
    name="canadian_budget_2023",
    description="A RAG tool engine with some basic facts regarding the 2023 Canadian federal budget"
)

agent = SimpleChatEngine.from_defaults(
    tools=[budget_tool, multiplication_tool, addition_tool],
    llm=Settings.llm,
    memory=memory_from_client,
    verbose=True,
)

# react_agent = ReActAgent.from_tools(
#     tools=[budget_tool, multiplication_tool, addition_tool], #budget_tool is the document tool and otehrs are function tools
#     llm=Settings.llm,
#     memory=memory_from_client,
#     verbose=True,
# )

react_agent = SimpleChatEngine.from_defaults(
    tools=[budget_tool, multiplication_tool, addition_tool],
    llm=Settings.llm,
    memory=memory_from_client,
    verbose=True,
)

# print("----- First prompt (introducing name) ------")
response1 = agent.chat("Hi, my name is Savandi")
print(response1)

# print("\n----- Second prompt (memory recall) ------")
response2= react_agent.chat("What is my name")
print(response2)

react_agent.chat("My mother is Chandani")

response3= react_agent.chat("What is my mothers name")
print(response3)


# print("----- First prompt (introducing name) ------")
response1 = agent.chat("Hi, my name is Savandi")
print(response1)

response4 = agent.chat("What is the theree times of total amount of the 2023canadian federal budget")
print(response4)

#dont need to train agents, they are already trained by frameworks such as LlamaIndex,  LangChain, ReAct Agent, Chat agent etc.