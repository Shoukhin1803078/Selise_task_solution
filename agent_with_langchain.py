import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
)

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        print("Generating answer ...")

def get_llm():
    return AzureChatOpenAI(
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        api_version=os.environ["AZURE_OPENAI_CHAT_API_VERSION"],
        temperature=0,
        callbacks=[MyCustomHandler()]
    )

# Tools 
@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents for a given query."""
    print("Retrive chunks ...")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=EMBEDDING_MODEL
    )
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    return "\n\n".join([d.page_content for d in docs])




# Agent 
model = get_llm()
tools = [retrieve_documents]


SYSTEM_PROMPT = """You are a helpful AI assistant with RAG capabilities.
When you receive a user question:
1. Retrieve relevant documents using the `retrieve_documents` tool.
2. CRITIC / REFLECTION: deeply reflect on the retrieved documents. Ask yourself:
   - Do these documents contain the answer?
   - Is there any missing information?
   - Is the query ambiguous?
3. If the information is missing, try to search again with a different query.
4. Once you have enough information, generate the FINAL ANSWER.
"""

# Retrieval, reasoning, and answer generation loop automatically.
agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
