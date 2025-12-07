import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def ingest_data():
    pdf_path = "./data/Md Al Amin Tokder.pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Created {len(all_splits)} chunks.")

    print("Creating embeddings and storing in Chroma...")
   
    if not os.environ.get("AZURE_OPENAI_API_KEY"):
        print("Error: AZURE_OPENAI_API_KEY not found in environment variables.")
        return

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
        openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    )

    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Ingestion complete. Vector store saved to ./chroma_db")

if __name__ == "__main__":
    ingest_data()
