# Selise_task_solution

# Mini Agentic RAG System

A lightweight **Mini Agentic RAG (Retrieval-Augmented Generation) system** built in Python. This project allows you to ingest documents into a vector database and query them using either **LangChain** or **LangGraph** agents.

---

## Project Structure

.
├── README.md
├── agent_with_langchain.py
├── agent_with_langgraph.py
├── chroma_db/
├── data/
│ └── Md Al Amin Tokder.pdf
├── ingest.py
├── main.py
└── requirements.txt

---


- **agent_with_langchain.py** – LangChain-based agent implementation.  
- **agent_with_langgraph.py** – LangGraph-based agent implementation.  
- **chroma_db/** – Chroma vector database storage.  
- **data/** – Folder containing documents to be ingested.  
- **ingest.py** – Script to ingest documents into the vector database.  
- **main.py** – Main program to run the RAG system.  
- **requirements.txt** – Python dependencies.  

---

## Setup Instructions

```bash
python -m venv myvenv
source venv/bin/activate (For MAC)
source .venv\Scripts\activate (For Windows PC)

pip install -r requirements.txt
python ingest.py
python main.py
```

## Notes
Before running, you can set the mode in main.py:
WAY = "langchain"  
